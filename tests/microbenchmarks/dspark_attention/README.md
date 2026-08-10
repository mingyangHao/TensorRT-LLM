# DSpark Attention Performance and Acceptance Validation

This runbook compares the current one-query-per-warp DSpark Attention path with
the exact query-packed kernel and inverse-RoPE epilogue fusion. The optimized
operator is still named `dspark_attn`; it is not an HCA attention variant.

Performance never waives correctness. Do not use an end-to-end speedup from a
run that fails the same-process bitwise and acceptance-length gates below.

## Variants

| Variant | Dynamic context | Queries/warp | Min blocks/MP | Fuse inverse RoPE |
| --- | ---: | ---: | ---: | ---: |
| Current PR | `0` | `1` | `0` | `0` |
| Candidate | `1` | `auto` | `auto` | `1` |

Both variants use `TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA=1`. The remaining
environment variables are:

```bash
export TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP=1
export TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP=auto
export TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP=auto
export TRTLLM_DSPARK_ATTENTION_FUSE_INVERSE_ROPE=1
```

`auto` is based on a B200 rank-local B1-B8 sweep. It resolves as follows:

| Draft length | Rank-local batch | Query tile |
| ---: | --- | ---: |
| 4 | B2, B4 | 1 |
| 4 | other | 2 |
| 5 | B1-B5 | 1 |
| 5 | B6+ | 5 |
| 6 | B1 | 1 |
| 6 | B2-B6 | 2 |
| 6 | B7+ | 6 |

For DL4 query-tile 2, `min_blocks_per_mp=16` is used at B3, B5, B7, and B32;
other cases use zero. Explicit integer environment values override the policy.

## 1. Strict correctness and exact AL

Run on SM100/SM103 with CuTe DSL available:

```bash
python3 -m pytest -q \
  tests/unittest/_torch/speculative/test_dspark_cute_dsl_attention.py \
  -k 'inverse_rope_epilogue_is_bitwise_equal or candidate_cuda_graph_replay_is_bitwise_equal or candidate_stateful_trace_preserves_exact_acceptance_length'
```

The tests cover DL=4/5/6, partially filled/full/wrapped 128-row windows,
production strided stage-cache storage, dynamic batch changes, slot reuse, and
CUDA Graph replay. They require all of the following:

1. The candidate BF16 output is byte-for-byte equal to current attention plus
   the standalone inverse-RoPE kernel.
2. The complete rolling-KV backing storage is byte-for-byte equal.
3. Every draft ID, accepted-token tensor, and per-request accepted length is
   equal.
4. The integer AL numerator and denominator are equal, not merely a rounded
   floating-point AL value.

The fused epilogue explicitly rounds the attention result to BF16 before RoPE,
matching the removed kernel boundary. It does not change QK, online-softmax, PV,
or inverse-RoPE arithmetic order.

### Canonical model AL

The application-level reference values supplied for the canonical workload are:

| Draft length | AL reference |
| ---: | ---: |
| 4 | 3.72 |
| 5 | 4.11 |
| 6 | 4.32 |

These values are not targets for the synthetic stateful unit test. Each model
run must use the matching checkpoint `dspark_block_size`, the same fixed prompt
set, greedy decoding, and the same scheduler settings. Compare the raw integer
accepted/drafted/request counters before rounding AL.

Full-model runs in the current benchmark stack are not reproducible across
separate process launches: even two baseline launches can produce different
tokens and AL. Therefore, a cross-process candidate/baseline mismatch alone
cannot be attributed to this kernel. The release correctness gate is the
same-process stateful test above; model AL is an additional environment/checkpoint
health check. If model-level exactness is required, first demonstrate exact
baseline-vs-baseline replay in that environment or run both variants inside one
deterministic process harness.

## 2. Kernel-chain performance

The committed benchmark times the execution chain that actually changed:

- `current_pr`: one query/warp attention plus standalone inverse RoPE;
- `packed`: auto query packing plus standalone inverse RoPE;
- `fused`: auto query packing and the in-kernel inverse-RoPE epilogue.

It uses BF16, 128 heads, head dimension 512, RoPE dimension 64, a 128-row
window, a production-strided stage cache, and position 390. It verifies output
and cache bits before timing. Each variant is captured in a CUDA Graph; seven
samples are taken in alternating order so Python overhead and one-sided thermal
drift are excluded.

```bash
python3 tests/microbenchmarks/dspark_attention/benchmark.py \
  --blocks 4 5 6 \
  --batches 1 2 3 4 5 6 7 8 \
  --position 390 \
  --warmup 20 \
  --iterations 200 \
  --repeats 7 \
  --json-out /tmp/dspark-attention.json
```

Positive gain means the candidate is faster. Reference measurements from an
exclusive B200 node are:

| DL | Rank-local batch | Current PR (us) | Packed (us) | Fused (us) | Fused gain vs PR | RoPE fusion increment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 112.635 | 83.499 | 77.870 | +30.865% | +6.742% |
| 5 | 8 | 173.176 | 111.833 | 106.463 | +38.523% | +4.801% |
| 6 | 8 | 177.013 | 125.385 | 118.747 | +32.916% | +5.294% |

All 24 DL×B1-B8 cases passed bitwise comparison. Across those cases, inverse
RoPE fusion added 3.5%-11.2% on top of query packing. Rerun on the target
machine; these are development references, not release guarantees.

For throughput batches, query packing keeps independent FP32 softmax state for
each query while loading their common MQA KV row once. The fused epilogue then
removes one BF16 intermediate write, one BF16 read, and one standalone kernel
launch.

## 3. End-to-end A/B

Use one exclusive 8×B200 node for both variants. Build the source locally into
one immutable container/SquashFS image, then start two clean containers from
that same image on the allocation. Do not overlay source or install packages on
the compute node. Record the image digest in the result directory.

Keep these identical:

- TensorRT LLM commit, image, target and DSpark checkpoints;
- pre-tokenized requests, request order, tokenizer, sampler, and seed;
- TP/EP/attention-DP/EPLB, CUDA Graph, KV cache, and scheduler configuration;
- max batch, max tokens, concurrency, request count, clocks, and power limit.

Use greedy decoding (`temperature=0`, `top_k=1`) and three datasets:

| Workload | ISL | OSL | Purpose |
| --- | ---: | ---: | --- |
| Context only | 1024 | 1 | Control; DSpark generation work is negligible. |
| Generation only | 1 | 512 | DSpark-sensitive workload. |
| End to end | 1024 | 512 | Release/user-visible workload. |

Run candidate first and baseline second so filesystem page-cache effects favor
the baseline. For a release decision, repeat at least three pairs and alternate
order on each pair. Report medians rather than the best run.

The environment delta must be exactly:

```bash
# current PR
TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP=0
TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP=1
TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP=0
TRTLLM_DSPARK_ATTENTION_FUSE_INVERSE_ROPE=0

# candidate
TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP=1
TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP=auto
TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP=auto
TRTLLM_DSPARK_ATTENTION_FUSE_INVERSE_ROPE=1
```

Calculate gains as:

```text
throughput gain = candidate_output_tok_s / baseline_output_tok_s - 1
latency gain    = (baseline_latency_ms - candidate_latency_ms) / baseline_latency_ms
```

Also compare generation GPU-forward sum/p50/p95 from iteration logs. A raw wall
time is not a valid kernel comparison when the two model launches take different
acceptance paths. Context-only should remain within measured baseline noise.

## 4. Nsight attribution

After the normal A/B is stable, capture the same request range with Nsight
Systems and export the CUDA-kernel summary:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output baseline_cuda_gpu_kern_sum baseline.nsys-rep
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output candidate_cuda_gpu_kern_sum candidate.nsys-rep
rg -i 'dspark|DSparkAttentionKernel|DSparkRMSNormRoPEKernel' \
  *_cuda_gpu_kern_sum.csv
```

The candidate trace should show no standalone inverse-RoPE launch after
DSpark Attention. Kernel time attribution does not replace end-to-end results:
overlap and a small module share can attenuate a large kernel-local speedup.

## Failure conditions

Reject the candidate if any of these occurs:

- one output or cache bit differs in the strict same-process tests;
- any accepted token, accepted length, or exact AL counter differs there;
- the expected candidate/baseline environment is absent from logs;
- JIT compilation or warmup is included in measured timing;
- A/B uses different images, checkpoints, requests, configuration, or hardware;
- only kernel time improves while a statistically valid target E2E workload regresses.
