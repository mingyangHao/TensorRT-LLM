# DSpark Attention Performance and Acceptance Validation

This runbook compares the current one-warp DSpark Attention kernel with the
acceptance-preserving packed-warp tactic. It covers both kernel-only performance
and the complete DSpark module in context-only, generation-only, and end-to-end
workloads.

The optimized tactic is accepted only when all of the following conditions are
true:

1. Kernel output and rolling KV cache are bitwise equal to the one-warp
   baseline.
2. Complete-model output tokens are identical under greedy decoding.
3. DSpark accepted-token, draft-token, request, and acceptance-length counters
   are exactly equal.
4. The median end-to-end result improves on the target workload.

Performance is not a correctness waiver. If acceptance changes, reject the
tactic even when its kernel or end-to-end timing is faster.

## Tactics under test

The current PR behavior remains the default and is the baseline.

| Variant | `TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA` | `TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP` |
| --- | ---: | ---: |
| Baseline | 1 | 0 |
| Candidate | 2 | 1 |

Set both variables before the process starts. The CuteDSL compilation cache
includes both values, so each combination compiles to a distinct kernel.

## Test environment

Use the same exclusive Blackwell node for both variants. Keep the following
items unchanged:

- TensorRT LLM commit and container
- target and DSpark checkpoint files
- dataset, tokenizer, sampler options, and request order
- TP, EP, attention-DP, EPLB, CUDA graph, KV cache, and scheduler configuration
- GPU clocks, power limit, and application clocks
- maximum batch size, maximum token count, concurrency, and request count

Do not compare results collected on different nodes. Run at least three
measured repetitions per variant and report the median. Alternate the order
(`baseline`, `candidate`, `candidate`, `baseline`) when drift is visible.

The kernel test requires SM100/SM103 and a working CuteDSL installation. The
complete DSpark test normally requires the same eight-GPU TP8/EP8 environment
used by the production workload.

## 1. Bitwise kernel test

Run the committed correctness test first:

```bash
python3 -m pytest -q \
  tests/unittest/_torch/speculative/test_dspark_cute_dsl_attention.py \
  -k 'candidate_cuda_graph_replay_is_bitwise_equal or candidate_stateful_trace_preserves_exact_acceptance_length'
```

This covers draft lengths 4, 5, and 6, partially filled windows, full windows, and
wrapped windows. It compares output and the complete KV backing storage byte by
byte, exercises CUDA graph replay with changing inputs, and passes the resulting
draft IDs through the production DSpark strict-acceptance implementation. The
accepted-token tensor, every per-request accepted length, and the integer AL
numerator and denominator must be identical.

The following standalone microbenchmark uses the production shape: BF16,
128 query heads, head dimension 512, a 128-row rolling window, and a strided
per-stage cache view. It excludes JIT compilation and reports the median of
seven samples.

```bash
python3 - <<'PY'
import os
import statistics

import torch

from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
    cute_dsl_dspark_attention,
)


DEVICE = "cuda"
BATCH = 64
HEADS = 128
HEAD_DIM = 512
WINDOW = 128
STAGES = 3
WARMUP = 20
ITERS = 100
REPEATS = 7

TACTICS = {
    "baseline": ("1", "0"),
    "candidate": ("2", "1"),
}


def set_tactic(name):
    warps, dynamic = TACTICS[name]
    os.environ["TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA"] = warps
    os.environ["TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP"] = dynamic


def invoke(q, main_kv, block_kv, cache, slots, start_pos, sink):
    return cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        cache,
        slots,
        start_pos,
        sink,
        HEAD_DIM**-0.5,
    )


def time_one(name, args):
    set_tactic(name)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        invoke(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / ITERS


for block in (4, 5, 6):
    for position in (7, 63, 127, 390):
        torch.manual_seed(10000 + block * 1000 + position)
        q = torch.randn(
            BATCH,
            block,
            HEADS,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        main_kv = torch.randn(
            BATCH, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        block_kv = torch.randn(
            BATCH, block, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        cache_storage = torch.randn(
            BATCH,
            STAGES,
            WINDOW,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        slots = torch.arange(BATCH, device=DEVICE, dtype=torch.int64)
        start_pos = torch.full(
            (BATCH,), position, device=DEVICE, dtype=torch.int64
        )
        sink = torch.randn(HEADS, device=DEVICE, dtype=torch.float32)

        cache_storages = {
            "baseline": cache_storage.clone(),
            "candidate": cache_storage.clone(),
        }
        # Slice after cloning so both caches retain the production stride
        # inherited from [batch, stage, window, head_dim].
        caches = {name: storage[:, 1] for name, storage in cache_storages.items()}
        args = {
            name: (q, main_kv, block_kv, cache, slots, start_pos, sink)
            for name, cache in caches.items()
        }

        outputs = {}
        for name in TACTICS:
            set_tactic(name)
            outputs[name] = invoke(*args[name])
            for _ in range(WARMUP):
                invoke(*args[name])
        torch.cuda.synchronize()

        torch.testing.assert_close(
            outputs["candidate"], outputs["baseline"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            caches["candidate"], caches["baseline"], rtol=0, atol=0
        )

        samples = {name: [] for name in TACTICS}
        for repeat in range(REPEATS):
            order = (
                ("baseline", "candidate")
                if repeat % 2 == 0
                else ("candidate", "baseline")
            )
            for name in order:
                samples[name].append(time_one(name, args[name]))

        baseline_us = statistics.median(samples["baseline"])
        candidate_us = statistics.median(samples["candidate"])
        gain = (baseline_us - candidate_us) / baseline_us * 100.0
        print(
            f"block={block} position={position:3d} "
            f"baseline={baseline_us:9.3f} us "
            f"candidate={candidate_us:9.3f} us "
            f"gain={gain:+7.3f}% bitwise_equal=PASS"
        )
PY
```

A positive `gain` means that the candidate is faster. Pay particular attention
to positions 127 and 390: both execute a full rolling window and are more
representative of steady-state generation than positions 7 and 63.

Reference B200 batch-64 measurements from development are shown below. They are
not release numbers; rerun the script in the target environment.

| Block | Position | Baseline (us) | Candidate (us) | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 7 | 165.396 | 117.489 | +28.965% |
| 5 | 63 | 633.065 | 601.933 | +4.918% |
| 5 | 127 | 1184.801 | 1156.680 | +2.374% |
| 6 | 7 | 205.401 | 148.445 | +27.730% |
| 6 | 63 | 762.712 | 726.016 | +4.811% |
| 6 | 127 | 1421.179 | 1388.282 | +2.315% |

## 2. Prepare the complete DSpark A/B test

Start from the production DSpark YAML rather than creating a reduced model
configuration. The configuration must select DSpark and point
`speculative_model` at the checkpoint containing the `mtp.*` draft weights:

```yaml
speculative_config:
  decoding_type: DSpark
  speculative_model: /path/to/DeepSeek-V4-Pro-DSpark
  # Must equal dspark_block_size in the checkpoint config.json.
  max_draft_len: 5
```

Do not override `target_layer_ids`, `mask_token_id`, `block_size`,
`markov_rank`, or `markov_head_type` unless intentionally testing a checkpoint
change. TensorRT LLM reads these values from the checkpoint, and they must be
identical in both runs.

Use greedy decoding for a deterministic acceptance comparison:

```bash
cat > /tmp/dspark_greedy.yml <<'EOF'
temperature: 0.0
top_k: 1
seed: 0
EOF
```

Prepare each dataset once and reuse the same files for every run. These three
workloads answer different questions:

| Workload | ISL | OSL | Purpose |
| --- | ---: | ---: | --- |
| `ctx_only` | 1024 | 1 | Context-path control; DSpark generation work is negligible. |
| `gen_only` | 1 | 512 | Isolates steady-state generation and DSpark cost. |
| `e2e` | 1024 | 512 | Measures the user-visible workload. |

```bash
export MODEL=/path/to/DeepSeek-V4-Pro-DSpark
export BENCH_CONFIG=/path/to/production_dspark.yml
export DATASET_DIR=/path/to/dspark_ab_datasets
export RUN_ROOT=/path/to/dspark_ab_results
export NUM_REQUESTS=512
export DSPARK_DRAFT_LEN=5

mkdir -p "$DATASET_DIR" "$RUN_ROOT"

trtllm-bench --model "$MODEL" --model_path "$MODEL" prepare-dataset \
  --output "$DATASET_DIR/ctx_only.jsonl" token-norm-dist \
  --input-mean 1024 --input-stdev 0 \
  --output-mean 1 --output-stdev 0 \
  --num-requests "$NUM_REQUESTS"

trtllm-bench --model "$MODEL" --model_path "$MODEL" prepare-dataset \
  --output "$DATASET_DIR/gen_only.jsonl" token-norm-dist \
  --input-mean 1 --input-stdev 0 \
  --output-mean 512 --output-stdev 0 \
  --num-requests "$NUM_REQUESTS"

trtllm-bench --model "$MODEL" --model_path "$MODEL" prepare-dataset \
  --output "$DATASET_DIR/e2e.jsonl" token-norm-dist \
  --input-mean 1024 --input-stdev 0 \
  --output-mean 512 --output-stdev 0 \
  --num-requests "$NUM_REQUESTS"
```

Synthetic prompts give stable performance numbers. For an application-level AL
gate, repeat `gen_only` and `e2e` with one fixed, pre-tokenized production or
GSM8K dataset. Never compare AL from two independently generated datasets.

## 3. Run the whole DSpark module

Adjust the four capacity values to the production configuration. They must stay
identical across variants and repetitions.

```bash
export TP=8
export EP=8
export MAX_BATCH_SIZE=64
export MAX_NUM_TOKENS=384
export CONCURRENCY=1024
export WARMUP_REQUESTS=8

run_one() {
  variant=$1
  phase=$2
  repeat=$3
  max_seq_len=$4
  output_dir="$RUN_ROOT/$variant/$phase/r$repeat"
  mkdir -p "$output_dir"

  case "$variant" in
    baseline)
      export TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA=1
      export TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP=0
      ;;
    candidate)
      export TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA=2
      export TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP=1
      ;;
    *)
      echo "unknown variant: $variant" >&2
      return 2
      ;;
  esac

  # Per-rank metric export duplicates rank-0 speculative counters. Keep it off
  # so the acceptance parser below has one authoritative rank-0 record.
  export TLLM_METRICS_ALL_RANKS=0

  trtllm-bench --model "$MODEL" --model_path "$MODEL" throughput \
    --backend pytorch \
    --dataset "$DATASET_DIR/$phase.jsonl" \
    --config "$BENCH_CONFIG" \
    --sampler_options /tmp/dspark_greedy.yml \
    --custom_tokenizer deepseek_v4 \
    --tp "$TP" --ep "$EP" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_num_tokens "$MAX_NUM_TOKENS" \
    --max_seq_len "$max_seq_len" \
    --concurrency "$CONCURRENCY" \
    --num_requests "$NUM_REQUESTS" \
    --warmup "$WARMUP_REQUESTS" \
    --eos_id -1 \
    --report_json "$output_dir/report.json" \
    --iteration_log "$output_dir/iterations.jsonl" \
    --output_json "$output_dir/outputs.json" \
    --request_json "$output_dir/requests.json" \
    2>&1 | tee "$output_dir/run.log"
}

set -euo pipefail
for repeat in 1 2 3; do
  if (( repeat % 2 == 1 )); then
    variants=(baseline candidate)
  else
    variants=(candidate baseline)
  fi
  for variant in "${variants[@]}"; do
    run_one "$variant" ctx_only "$repeat" 1025
    run_one "$variant" gen_only "$repeat" 513
    run_one "$variant" e2e "$repeat" 1536
  done
done
```

Check the logs before using any measurement:

```bash
rg "DSparkWorker initialized" "$RUN_ROOT"/*/*/*/run.log
rg "DSpark Attention enabled" "$RUN_ROOT"/*/*/*/run.log
```

The candidate log must contain `warps_per_cta=2` and
`dynamic_context_loop=True`; the baseline must contain `warps_per_cta=1` and
`dynamic_context_loop=False`. Missing DSpark log lines make the run invalid.

## 4. Enforce exact output and DSpark AL

Keep the two acceptance checks separate:

1. The candidate and baseline must have identical integer acceptance counters
   for every paired run. This is the exact kernel-preservation gate.
2. On the canonical DSpark generation workload, the baseline AL must also
   reproduce the model-level reference for the configured draft length,
   rounded to two decimal places. Do not apply these values to a different
   prompt-length, concurrency, or end-to-end workload.

| Draft length | Model AL reference |
| ---: | ---: |
| 4 | 3.72 |
| 5 | 4.11 |
| 6 | 4.32 |

These model references are not expected values for the synthetic kernel unit
test. That test deliberately exercises every accepted-prefix length and only
requires the candidate and one-warp paths to match exactly. Do not tune a
synthetic target-token distribution to manufacture one of the model AL values.
When running the model-level gate, use a checkpoint whose configured
`dspark_block_size` equals the selected draft length, and set
`max_draft_len` to the same value. TensorRT LLM intentionally rejects a
`block_size` override that differs from the checkpoint. For example, the
current DeepSeek-V4-Pro-DSpark checkpoint has `dspark_block_size=5`, so only
the DL=5 reference applies to it; DL=4 and DL=6 require their corresponding
checkpoints. Keep the workload and scheduler configuration fixed.

First compare complete output-token JSON objects. This is a stronger end-to-end
check than comparing AL alone.

```bash
python3 - "$RUN_ROOT" <<'PY'
import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1])
for phase in ("ctx_only", "gen_only", "e2e"):
    baseline = sorted((root / "baseline" / phase).glob("r*/outputs.json"))
    candidate = sorted((root / "candidate" / phase).glob("r*/outputs.json"))
    if not baseline or len(baseline) != len(candidate):
        raise SystemExit(f"{phase}: missing or unpaired output files")
    for lhs, rhs in zip(baseline, candidate, strict=True):
        with lhs.open(encoding="utf-8") as f:
            baseline_outputs = json.load(f)
        with rhs.open(encoding="utf-8") as f:
            candidate_outputs = json.load(f)
        if baseline_outputs != candidate_outputs:
            raise SystemExit(f"{phase}: output mismatch: {lhs} != {rhs}")
    print(f"{phase}: exact output equality PASS ({len(baseline)} repetitions)")
PY
```

Then aggregate the authoritative attention-DP rank-0 iteration rows. TensorRT
LLM defines the per-iteration acceptance length as

```text
(numAcceptedTokens + numRequestsWithDraftTokens)
------------------------------------------------
          numRequestsWithDraftTokens
```

The comparison below uses integer counters and a rational number, so it does
not hide a mismatch behind floating-point rounding. It also compares draft
tokens and rejects missing, malformed, duplicated all-rank, or empty stats.

```bash
python3 - "$RUN_ROOT" <<'PY'
import ast
from fractions import Fraction
import json
import os
import pathlib
import sys


def parse_line(line, path, line_number):
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(line)
        except (SyntaxError, ValueError) as error:
            raise RuntimeError(
                f"{path}:{line_number}: cannot parse iteration row"
            ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}:{line_number}: iteration row is not an object")
    return value


def collect(path):
    rows = 0
    spec_rows = 0
    accepted = 0
    drafted = 0
    requests = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = parse_line(line, path, line_number)
            if "rank" in row:
                raise RuntimeError(
                    f"{path}:{line_number}: all-rank metrics are enabled; "
                    "rerun with TLLM_METRICS_ALL_RANKS=0"
                )
            if row.get("attentionDpRank", 0) != 0:
                # ADP fanout copies rank-0 specDecodingStats to the other
                # rank rows. Counting those rows would duplicate AL counters.
                continue
            rows += 1
            spec = row.get("specDecodingStats")
            if not spec or spec.get("numDraftTokens", 0) == 0:
                continue

            required = (
                "numAcceptedTokens",
                "numDraftTokens",
                "numRequestsWithDraftTokens",
                "acceptanceLength",
            )
            missing = [key for key in required if key not in spec]
            if missing:
                raise RuntimeError(
                    f"{path}:{line_number}: missing speculative fields {missing}"
                )

            row_accepted = int(spec["numAcceptedTokens"])
            row_drafted = int(spec["numDraftTokens"])
            row_requests = int(spec["numRequestsWithDraftTokens"])
            if row_requests <= 0 or not 0 <= row_accepted <= row_drafted:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid speculative counters: {spec}"
                )

            expected_row_al = Fraction(
                row_accepted + row_requests, row_requests
            )
            logged_row_al = float(spec["acceptanceLength"])
            if logged_row_al != float(expected_row_al):
                raise RuntimeError(
                    f"{path}:{line_number}: acceptanceLength={logged_row_al} "
                    f"does not match counters ({float(expected_row_al)})"
                )

            accepted += row_accepted
            drafted += row_drafted
            requests += row_requests
            spec_rows += 1

    if rows == 0:
        raise RuntimeError(f"{path}: no authoritative rank-0 rows")
    if spec_rows == 0 or drafted == 0 or requests == 0:
        raise RuntimeError(f"{path}: no DSpark generation statistics")

    return {
        "accepted": accepted,
        "drafted": drafted,
        "requests": requests,
        "al": Fraction(accepted + requests, requests),
        "acceptance_rate": Fraction(accepted, drafted),
        "spec_rows": spec_rows,
    }


root = pathlib.Path(sys.argv[1])
draft_len = int(os.environ["DSPARK_DRAFT_LEN"])
al_references = {
    4: Fraction(372, 100),
    5: Fraction(411, 100),
    6: Fraction(432, 100),
}
if draft_len not in al_references:
    raise SystemExit(f"no canonical AL reference for draft length {draft_len}")

for phase in ("gen_only", "e2e"):
    baseline = sorted((root / "baseline" / phase).glob("r*/iterations.jsonl"))
    candidate = sorted((root / "candidate" / phase).glob("r*/iterations.jsonl"))
    if not baseline or len(baseline) != len(candidate):
        raise SystemExit(f"{phase}: missing or unpaired iteration logs")

    for lhs, rhs in zip(baseline, candidate, strict=True):
        baseline_stats = collect(lhs)
        candidate_stats = collect(rhs)
        exact_fields = ("accepted", "drafted", "requests", "al")
        mismatches = {
            key: (baseline_stats[key], candidate_stats[key])
            for key in exact_fields
            if baseline_stats[key] != candidate_stats[key]
        }
        if mismatches:
            raise SystemExit(
                f"{phase}: DSpark acceptance mismatch for {lhs} vs {rhs}: "
                f"{mismatches}"
            )

        if phase == "gen_only":
            reference = al_references[draft_len]
            half_cent = Fraction(1, 200)
            if not reference - half_cent <= baseline_stats["al"] < reference + half_cent:
                raise SystemExit(
                    f"{phase}: DL={draft_len} model AL reference mismatch: "
                    f"actual={float(baseline_stats['al']):.9f}, "
                    f"expected={float(reference):.2f} after two-decimal rounding"
                )

        print(
            f"{phase} {lhs.parent.name}: exact DSpark acceptance PASS; "
            f"accepted={baseline_stats['accepted']} "
            f"drafted={baseline_stats['drafted']} "
            f"requests={baseline_stats['requests']} "
            f"AL={float(baseline_stats['al']):.9f} "
            f"AR={float(baseline_stats['acceptance_rate']):.9f}"
        )
PY
```

`ctx_only` is intentionally excluded from the AL parser because it contains no
meaningful DSpark generation iterations. It must still pass exact output
comparison and serves as a control: an attention-kernel tactic should not
materially change context-only performance.

## 5. Report end-to-end performance

Use the benchmark report JSON rather than the wall time of the shell command.
The following script prints the median throughput and latency over all paired
repetitions:

```bash
python3 - "$RUN_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys


def load_reports(root, variant, phase):
    paths = sorted((root / variant / phase).glob("r*/report.json"))
    if not paths:
        raise RuntimeError(f"no reports for {variant}/{phase}")
    values = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            report = json.load(f)
        perf = report["performance"]
        values.append(
            (
                float(perf["system_output_throughput_tok_s"]),
                float(perf["total_latency_ms"]),
            )
        )
    return values


root = pathlib.Path(sys.argv[1])
print("phase       base_tok/s    cand_tok/s  throughput_gain   latency_gain")
for phase in ("ctx_only", "gen_only", "e2e"):
    baseline = load_reports(root, "baseline", phase)
    candidate = load_reports(root, "candidate", phase)
    if len(baseline) != len(candidate):
        raise SystemExit(f"{phase}: unpaired reports")
    base_tps = statistics.median(value[0] for value in baseline)
    cand_tps = statistics.median(value[0] for value in candidate)
    base_ms = statistics.median(value[1] for value in baseline)
    cand_ms = statistics.median(value[1] for value in candidate)
    throughput_gain = (cand_tps / base_tps - 1.0) * 100.0
    latency_gain = (base_ms - cand_ms) / base_ms * 100.0
    print(
        f"{phase:10s} {base_tps:12.2f} {cand_tps:12.2f} "
        f"{throughput_gain:+15.3f}% {latency_gain:+13.3f}%"
    )
PY
```

Positive throughput and latency gains both mean improvement. Treat `gen_only`
as the DSpark-sensitive micro-E2E result and `e2e` as the release decision.
`ctx_only` should be statistically flat and helps identify unrelated run-to-run
noise.

Fill in this table in the PR description:

| Workload | Baseline output tok/s | Candidate output tok/s | Throughput gain | Baseline latency | Candidate latency | Latency gain | Output exact | DSpark AL exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Context only | | | | | | | PASS/FAIL | N/A |
| Generation only | | | | | | | PASS/FAIL | PASS/FAIL |
| End to end | | | | | | | PASS/FAIL | PASS/FAIL |

## Optional Nsight Systems attribution

Use Nsight Systems after the normal A/B is stable. Profile the same measured
request range and command for both variants, then export the CUDA kernel
summary:

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output baseline_cuda_gpu_kern_sum \
  baseline.nsys-rep

nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output candidate_cuda_gpu_kern_sum \
  candidate.nsys-rep

rg -i "dspark|DSparkAttentionKernel" *_cuda_gpu_kern_sum.csv
```

The kernel summary answers how much GPU time DSpark Attention consumes and
whether the optimized kernel reduced it. It does not replace the report-JSON
end-to-end comparison: overlap can turn a kernel-time reduction into a smaller
or zero user-visible gain.

## Failure conditions

Reject the candidate when any of these conditions occurs:

- kernel output or rolling KV cache differs by one bit;
- complete-model output tokens differ;
- accepted tokens, draft tokens, drafted-request count, or AL differs;
- iteration logs are missing, malformed, empty, or collected with duplicated
  all-rank speculative counters;
- the DSpark worker or the expected attention tactic is absent from the log;
- JIT compilation or model warmup is included in measured timing;
- A/B uses different commits, checkpoints, datasets, configuration, hardware,
  or clocks;
- only kernel time improves while the target end-to-end workload regresses.
