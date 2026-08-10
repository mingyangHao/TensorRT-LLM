# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the exact DSpark attention + inverse-RoPE execution chain."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
    cute_dsl_dspark_attention,
    cute_dsl_dspark_attention_rope,
)
from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
    cute_dsl_dspark_rmsnorm_rope,
)
from tensorrt_llm._torch.models.dspark.attention import precompute_dspark_freqs_cis


def set_tactic(*, packed: bool) -> None:
    os.environ["TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA"] = "1"
    os.environ["TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP"] = "1" if packed else "0"
    os.environ["TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP"] = "auto" if packed else "1"
    os.environ["TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP"] = "auto" if packed else "0"


def capture(function) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    output = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph], warmup: int, iterations: int, repeats: int
) -> dict[str, tuple[float, float]]:
    for graph in graphs.values():
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name in graphs}
    names = list(graphs)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1_000.0 / iterations)
    return {
        name: (statistics.median(values), statistics.pstdev(values))
        for name, values in samples.items()
    }


def benchmark_case(
    *, batch: int, block: int, position: int, warmup: int, iterations: int, repeats: int
) -> dict[str, object]:
    torch.manual_seed(20260809 + batch * 10 + block)
    device = torch.device("cuda")
    heads, head_dim, rope_dim, window = 128, 512, 64, 128
    scale = head_dim**-0.5

    q = torch.randn(batch, block, heads, head_dim, dtype=torch.bfloat16, device=device)
    main_kv = torch.randn(batch, head_dim, dtype=torch.bfloat16, device=device)
    block_kv = torch.randn(batch, block, head_dim, dtype=torch.bfloat16, device=device)
    cache_storage = torch.randn(batch, 3, window, head_dim, dtype=torch.bfloat16, device=device)
    slots = torch.arange(batch, dtype=torch.int64, device=device)
    start_pos = torch.full((batch,), position, dtype=torch.int64, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device) * 0.1
    weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    positions = start_pos.unsqueeze(1) + 1 + torch.arange(block, device=device)
    freqs_cis = precompute_dspark_freqs_cis(
        rope_dim, int(positions.max().item()) + 1, device=device
    )
    inverse_rope_freqs = torch.view_as_real(freqs_cis[positions]).contiguous()

    cache_storages = {name: cache_storage.clone() for name in ("current_pr", "packed", "fused")}
    caches = {name: storage[:, 1] for name, storage in cache_storages.items()}
    expected_batch_stride = 3 * window * head_dim
    assert all(cache.stride(0) == expected_batch_stride for cache in caches.values())

    def attention_then_rope(name: str, *, packed: bool) -> torch.Tensor:
        set_tactic(packed=packed)
        output = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            caches[name],
            slots,
            start_pos,
            sink,
            scale,
        )
        return cute_dsl_dspark_rmsnorm_rope(
            output,
            weight,
            inverse_rope_freqs.flatten(0, 1),
            heads,
            rope_dim,
            0.0,
            False,
            False,
            True,
        )

    def fused() -> torch.Tensor:
        set_tactic(packed=True)
        return cute_dsl_dspark_attention_rope(
            q,
            main_kv,
            block_kv,
            caches["fused"],
            slots,
            start_pos,
            sink,
            inverse_rope_freqs,
            rope_dim,
            scale,
        )

    current_pr = attention_then_rope("current_pr", packed=False)
    packed = attention_then_rope("packed", packed=True)
    candidate = fused()
    torch.testing.assert_close(
        packed.view(torch.uint8), current_pr.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        candidate.view(torch.uint8), current_pr.view(torch.uint8), rtol=0, atol=0
    )
    for name in ("packed", "fused"):
        torch.testing.assert_close(
            caches[name].view(torch.uint8),
            caches["current_pr"].view(torch.uint8),
            rtol=0,
            atol=0,
        )

    graphs = {}
    retained_outputs = []
    for name, function in (
        ("current_pr", lambda: attention_then_rope("current_pr", packed=False)),
        ("packed", lambda: attention_then_rope("packed", packed=True)),
        ("fused", fused),
    ):
        graph, output = capture(function)
        graphs[name] = graph
        retained_outputs.append(output)
    measurements = time_graphs(graphs, warmup, iterations, repeats)
    assert len(retained_outputs) == len(graphs)

    current_us = measurements["current_pr"][0]
    packed_us = measurements["packed"][0]
    fused_us = measurements["fused"][0]
    result = {
        "batch": batch,
        "block": block,
        "position": position,
        "bitwise_equal": True,
        "current_pr_us": current_us,
        "packed_us": packed_us,
        "fused_us": fused_us,
        "packed_gain_percent": (current_us - packed_us) / current_us * 100.0,
        "fused_gain_vs_current_pr_percent": (current_us - fused_us) / current_us * 100.0,
        "rope_fusion_incremental_gain_percent": (packed_us - fused_us) / packed_us * 100.0,
        "stddev_us": {name: values[1] for name, values in measurements.items()},
    }
    print(
        f"DL={block} B={batch:2d} bitwise=PASS "
        f"current_pr={current_us:8.3f}us packed={packed_us:8.3f}us "
        f"fused={fused_us:8.3f}us packed_gain={result['packed_gain_percent']:+7.3f}% "
        f"fused_gain={result['fused_gain_vs_current_pr_percent']:+7.3f}% "
        f"fusion_incremental={result['rope_fusion_incremental_gain_percent']:+7.3f}%",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", nargs="+", type=int, default=(4, 5, 6))
    parser.add_argument("--batches", nargs="+", type=int, default=tuple(range(1, 9)))
    parser.add_argument("--position", type=int, default=390)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    results = [
        benchmark_case(
            batch=batch,
            block=block,
            position=args.position,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for block in args.blocks
        for batch in args.batches
    ]
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
