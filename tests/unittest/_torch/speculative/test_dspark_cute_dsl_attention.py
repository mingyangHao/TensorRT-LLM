# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU correctness tests for the fused DSpark CuteDSL attention op."""

import types

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.models.dspark.attention import (
    dspark_sparse_attn,
    get_dspark_topk_idxs_batched,
)
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE or not is_sm_100f(),
    reason="DSpark CuteDSL attention requires an SM100-family CUDA GPU",
)


def _make_inputs(seed: int = 0, batch: int = 2, block: int = 5):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    heads, head_dim, window = 24, 512, 128
    q = torch.randn(batch, block, heads, head_dim, device=device, dtype=torch.bfloat16)
    main_kv = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    block_kv = torch.randn(batch, block, head_dim, device=device, dtype=torch.bfloat16)

    # A real DSpark stage window is a strided view of
    # [max_batch, num_stages, window, head_dim]. Exercise that contract here.
    cache_storage = torch.randn(
        max(4, batch + 1), 3, window, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = cache_storage[:, 1]
    slots = torch.arange(batch - 1, -1, -1, device=device, dtype=torch.long)
    start_pos = torch.arange(batch, device=device, dtype=torch.long) * 199 + 1
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    return q, main_kv, block_kv, kv_cache, slots, start_pos, sink


def _reference(q, main_kv, block_kv, kv_cache, slots, start_pos, sink):
    cache = kv_cache.clone()
    window = cache.shape[1]
    cache[slots, start_pos % window] = main_kv
    kv_full = torch.cat([cache[slots], block_kv], dim=1)
    topk = get_dspark_topk_idxs_batched(window, q.shape[1], start_pos)
    return dspark_sparse_attn(q, kv_full, sink, topk, q.shape[-1] ** -0.5), cache


def _set_tactic(
    monkeypatch,
    *,
    warps_per_cta: int,
    dynamic_context_loop: bool,
    queries_per_warp: int | str = 1,
    min_blocks_per_mp: int | str = 0,
) -> None:
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA", str(warps_per_cta))
    monkeypatch.setenv(
        "TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP", str(int(dynamic_context_loop))
    )
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP", str(queries_per_warp))
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP", str(min_blocks_per_mp))


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare tensor storage bits, including signed zero and NaN payloads."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def _make_strict_acceptance_worker(block: int, monkeypatch):
    config = types.SimpleNamespace(
        max_draft_len=block,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
    )
    worker = DSparkWorker(config, Mapping())
    worker.force_num_accepted_tokens = 0.0
    target_tokens = {}

    def fixed_target_sampler(_logits, _metadata, _num_contexts, _batch_size):
        return target_tokens["value"].reshape(-1)

    monkeypatch.setattr(worker, "_sample_tokens_for_batch", fixed_target_sampler)
    return worker, target_tokens


def _strict_accept(
    worker,
    target_holder,
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, block = draft_tokens.shape
    target_holder["value"] = target_tokens
    logits = torch.empty(batch * (block + 1), 1, device=draft_tokens.device)
    metadata = types.SimpleNamespace(is_cuda_graph=False)
    return worker._sample_and_accept_draft_tokens_base(
        logits,
        draft_tokens,
        num_contexts=0,
        batch_size=batch,
        spec_metadata=metadata,
    )


def _target_trace(draft_tokens: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed target tokens with every possible accepted-prefix length."""
    batch, block = draft_tokens.shape
    rows = torch.arange(batch, device=draft_tokens.device)
    accepted_drafts = (rows + step) % (block + 1)
    target_tokens = torch.empty(batch, block + 1, dtype=torch.int32, device=draft_tokens.device)
    target_tokens[:, :block] = draft_tokens
    target_tokens[:, block] = (rows + step) % 32

    rejected = accepted_drafts < block
    rejected_rows = rows[rejected]
    rejected_cols = accepted_drafts[rejected]
    target_tokens[rejected_rows, rejected_cols] = (
        draft_tokens[rejected_rows, rejected_cols] + 1
    ) % 32
    # The production acceptance path reports sequence lengths as int32. Keep
    # the expected contract equally strict so dtype drift also fails the test.
    return target_tokens, (accepted_drafts + 1).to(torch.int32)


@pytest.mark.parametrize(
    ("block", "batch", "expected"),
    (
        (4, 1, 2),
        (4, 2, 1),
        (4, 3, 2),
        (4, 4, 1),
        (4, 5, 2),
        (5, 5, 1),
        (5, 6, 5),
        (6, 1, 1),
        (6, 2, 2),
        (6, 6, 2),
        (6, 7, 6),
    ),
)
def test_dspark_attention_auto_query_tile(monkeypatch, block, batch, expected):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _get_dspark_attention_queries_per_warp,
    )

    monkeypatch.delenv("TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP", raising=False)
    assert _get_dspark_attention_queries_per_warp(block, batch) == expected

    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP", "auto")
    assert _get_dspark_attention_queries_per_warp(block, batch) == expected


@pytest.mark.parametrize(
    ("block", "batch", "query_tile", "expected"),
    (
        (4, 3, 2, 16),
        (4, 5, 2, 16),
        (4, 7, 2, 16),
        (4, 8, 2, 0),
        (5, 7, 5, 0),
        (6, 6, 2, 0),
    ),
)
def test_dspark_attention_auto_min_blocks(monkeypatch, block, batch, query_tile, expected):
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _get_dspark_attention_min_blocks_per_mp,
    )

    monkeypatch.delenv("TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP", raising=False)
    assert _get_dspark_attention_min_blocks_per_mp(block, batch, query_tile) == expected

    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP", "auto")
    assert _get_dspark_attention_min_blocks_per_mp(block, batch, query_tile) == expected


def test_cute_dsl_dspark_attention_matches_reference():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    inputs = _make_inputs()
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
    expected, expected_cache = _reference(*inputs)

    actual = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        sink,
        q.shape[-1] ** -0.5,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


def test_cute_dsl_dspark_attention_cuda_graph_replay():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    inputs = _make_inputs(3)
    q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
    scale = q.shape[-1] ** -0.5

    # Compile/JIT before capture. The replay must launch only the cached kernel.
    cute_dsl_dspark_attention(*inputs, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cute_dsl_dspark_attention(*inputs, scale)

    main_kv.copy_(torch.randn_like(main_kv))
    block_kv.copy_(torch.randn_like(block_kv))
    expected, expected_cache = _reference(
        q, main_kv, block_kv, kv_cache.clone(), slots, start_pos, sink
    )
    graph.replay()

    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(kv_cache, expected_cache, rtol=0, atol=0)


@pytest.mark.parametrize("block", (4, 5, 6))
def test_dspark_attention_inverse_rope_epilogue_is_bitwise_equal(monkeypatch, block):
    """Fused epilogue must match attention then standalone inverse RoPE bitwise."""
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention,
        cute_dsl_dspark_attention_rope,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )
    from tensorrt_llm._torch.models.dspark.attention import precompute_dspark_freqs_cis

    q, main_kv, block_kv, kv_cache, slots, _, sink = _make_inputs(
        seed=91 + block, batch=3, block=block
    )
    start_pos = torch.tensor([3, 127, 390], device=q.device, dtype=torch.int64)
    rope_dim = 64
    positions = start_pos.unsqueeze(1) + 1 + torch.arange(block, device=q.device)
    freqs_cis = precompute_dspark_freqs_cis(
        rope_dim, int(positions.max().item()) + 1, device=q.device
    )
    inverse_rope_freqs = torch.view_as_real(freqs_cis[positions]).contiguous()
    scale = q.shape[-1] ** -0.5

    baseline_cache = kv_cache.clone()
    _set_tactic(monkeypatch, warps_per_cta=1, dynamic_context_loop=False)
    baseline = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        baseline_cache,
        slots,
        start_pos,
        sink,
        scale,
    )
    baseline = cute_dsl_dspark_rmsnorm_rope(
        baseline,
        torch.ones(q.shape[-1], device=q.device, dtype=q.dtype),
        inverse_rope_freqs.flatten(0, 1),
        q.shape[2],
        rope_dim,
        0.0,
        False,
        False,
        True,
    )

    candidate_cache = kv_cache.clone()
    _set_tactic(
        monkeypatch,
        warps_per_cta=1,
        dynamic_context_loop=True,
        queries_per_warp="auto",
        min_blocks_per_mp="auto",
    )
    candidate = cute_dsl_dspark_attention_rope(
        q,
        main_kv,
        block_kv,
        candidate_cache,
        slots,
        start_pos,
        sink,
        inverse_rope_freqs,
        rope_dim,
        scale,
    )

    _assert_bitwise_equal(candidate, baseline)
    _assert_bitwise_equal(candidate_cache, baseline_cache)


@pytest.mark.parametrize("block", (4, 5, 6))
def test_candidate_cuda_graph_replay_is_bitwise_equal(monkeypatch, block):
    """Candidate graph replay must preserve the one-warp state trajectory."""
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    q, main_kv, block_kv, kv_cache, slots, _, sink = _make_inputs(
        seed=17 + block, batch=3, block=block
    )
    scale = q.shape[-1] ** -0.5
    start_pos = torch.tensor([3, 127, 390], device=q.device, dtype=torch.int64)
    initial_cache = kv_cache.clone()
    baseline_cache = initial_cache.clone()
    candidate_cache = initial_cache.clone()

    _set_tactic(
        monkeypatch,
        warps_per_cta=1,
        dynamic_context_loop=True,
        queries_per_warp="auto",
        min_blocks_per_mp="auto",
    )
    warmup_cache = initial_cache.clone()
    cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        warmup_cache,
        slots,
        start_pos,
        sink,
        scale,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            candidate_cache,
            slots,
            start_pos,
            sink,
            scale,
        )

    # Capture executes the graph once. Reset the persistent state so both paths
    # start the replay trace from the same cache contents.
    candidate_cache.copy_(initial_cache)
    positions = ([3, 127, 390], [4, 128, 391], [126, 255, 511], [127, 256, 512])
    for replay, values in enumerate(positions):
        torch.manual_seed(1700 + block * 10 + replay)
        q.copy_(torch.randn_like(q))
        main_kv.copy_(torch.randn_like(main_kv))
        block_kv.copy_(torch.randn_like(block_kv))
        start_pos.copy_(torch.tensor(values, device=q.device, dtype=start_pos.dtype))
        _set_tactic(monkeypatch, warps_per_cta=1, dynamic_context_loop=False)
        baseline = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            baseline_cache,
            slots,
            start_pos,
            sink,
            scale,
        )
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise_equal(candidate, baseline)
        _assert_bitwise_equal(candidate_cache, baseline_cache)


def test_cute_dsl_dspark_attention_compiles_once_across_batch_sizes():
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        _compile_fused_dspark_attention,
        cute_dsl_dspark_attention,
    )

    _compile_fused_dspark_attention.cache_clear()
    for batch in (1, 3):
        inputs = _make_inputs(4 + batch, batch=batch)
        q, main_kv, block_kv, kv_cache, slots, start_pos, sink = inputs
        expected, _ = _reference(*inputs)
        actual = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            kv_cache,
            slots,
            start_pos,
            sink,
            q.shape[-1] ** -0.5,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    cache_info = _compile_fused_dspark_attention.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1


@pytest.mark.parametrize("block", (4, 5, 6))
@pytest.mark.parametrize(
    ("warps_per_cta", "dynamic_context_loop"),
    ((2, False), (4, False), (8, False), (1, True), (2, True)),
)
def test_cute_dsl_dspark_attention_tactic_is_bitwise_equal(
    monkeypatch, block, warps_per_cta, dynamic_context_loop
):
    """Packing independent head warps must not change DSpark acceptance math."""
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import cute_dsl_dspark_attention

    q, main_kv, block_kv, kv_cache, slots, _, sink = _make_inputs(
        seed=1000 + block + warps_per_cta + int(dynamic_context_loop),
        batch=3,
        block=block,
    )
    # Exercise a partially filled row, a full window, and a wrapped window.
    start_pos = torch.tensor([3, 127, 390], device=q.device, dtype=torch.int64)
    scale = q.shape[-1] ** -0.5

    baseline_cache = kv_cache.clone()
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA", "1")
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP", "0")
    baseline = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        baseline_cache,
        slots,
        start_pos,
        sink,
        scale,
    )

    packed_cache = kv_cache.clone()
    monkeypatch.setenv("TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA", str(warps_per_cta))
    monkeypatch.setenv(
        "TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP", str(int(dynamic_context_loop))
    )
    packed = cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        packed_cache,
        slots,
        start_pos,
        sink,
        scale,
    )

    torch.testing.assert_close(packed, baseline, rtol=0, atol=0)
    torch.testing.assert_close(packed_cache, baseline_cache, rtol=0, atol=0)


@pytest.mark.parametrize("block", (4, 5, 6))
def test_candidate_stateful_trace_preserves_exact_acceptance_length(monkeypatch, block):
    """Production-shaped stateful A/B must preserve every exact AL counter.

    The scheduler trace is fixed before either tactic runs. Both tactics consume
    identical tensors and independently evolve strided rolling-window caches.
    Draft proposals derived from the attention result are verified by the same
    strict-acceptance implementation used by DSparkWorker. The test compares the
    integer AL numerator and denominator rather than rounded floating-point AL.
    """
    from tensorrt_llm._torch.custom_ops.dspark_attention_custom_op import (
        cute_dsl_dspark_attention,
        cute_dsl_dspark_attention_rope,
    )
    from tensorrt_llm._torch.custom_ops.dspark_rmsnorm_rope_custom_op import (
        cute_dsl_dspark_rmsnorm_rope,
    )
    from tensorrt_llm._torch.models.dspark.attention import precompute_dspark_freqs_cis

    torch.manual_seed(20260809 + block)
    device = torch.device("cuda")
    capacity, stages, window = 67, 3, 128
    heads, head_dim = 128, 512
    rope_dim = 64
    scale = head_dim**-0.5

    cache_storage = torch.randn(
        capacity,
        stages,
        window,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    baseline_storage = cache_storage.clone()
    candidate_storage = cache_storage.clone()
    baseline_cache = baseline_storage[:, 1]
    candidate_cache = candidate_storage[:, 1]
    assert not baseline_cache.is_contiguous()

    sink = torch.randn(heads, dtype=torch.float32, device=device) * 0.1
    rope_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    freqs_cis = precompute_dspark_freqs_cis(rope_dim, 2048, device=device)
    slot_order = torch.randperm(capacity, device=device)
    position_boundaries = torch.tensor(
        [0, 1, 7, 63, 126, 127, 128, 255, 390, 511, 1023],
        dtype=torch.int64,
        device=device,
    )
    position_by_slot = position_boundaries[
        torch.arange(capacity, device=device) % position_boundaries.numel()
    ].clone()
    batch_trace = (1, 8, 32, 64, 16, 4, 64, 32)

    worker, target_holder = _make_strict_acceptance_worker(block, monkeypatch)
    baseline_totals = dict(accepted=0, drafted=0, requests=0)
    candidate_totals = dict(accepted=0, drafted=0, requests=0)

    for step, batch in enumerate(batch_trace):
        slots = torch.roll(slot_order, shifts=step * 7)[:batch].contiguous()
        if step in (3, 6):
            # Reset two physical rows before reusing them for new logical
            # requests, matching DSparkWorker's slot lifecycle.
            reused = slots[-min(2, batch) :]
            baseline_cache[reused].zero_()
            candidate_cache[reused].zero_()
            position_by_slot[reused] = position_boundaries[step]

        start_pos = position_by_slot[slots].contiguous()
        q = torch.randn(
            batch,
            block,
            heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        main_kv = torch.randn(batch, head_dim, dtype=torch.bfloat16, device=device)
        block_kv = torch.randn(
            batch,
            block,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        positions = start_pos.unsqueeze(1) + 1 + torch.arange(block, device=device)
        inverse_rope_freqs = torch.view_as_real(freqs_cis[positions]).contiguous()

        _set_tactic(monkeypatch, warps_per_cta=1, dynamic_context_loop=False)
        baseline = cute_dsl_dspark_attention(
            q,
            main_kv,
            block_kv,
            baseline_cache,
            slots,
            start_pos,
            sink,
            scale,
        )
        baseline = cute_dsl_dspark_rmsnorm_rope(
            baseline,
            rope_weight,
            inverse_rope_freqs.flatten(0, 1),
            heads,
            rope_dim,
            0.0,
            False,
            False,
            True,
        )
        _set_tactic(
            monkeypatch,
            warps_per_cta=1,
            dynamic_context_loop=True,
            queries_per_warp="auto",
            min_blocks_per_mp="auto",
        )
        candidate = cute_dsl_dspark_attention_rope(
            q,
            main_kv,
            block_kv,
            candidate_cache,
            slots,
            start_pos,
            sink,
            inverse_rope_freqs,
            rope_dim,
            scale,
        )

        _assert_bitwise_equal(candidate, baseline)
        _assert_bitwise_equal(candidate_storage, baseline_storage)

        baseline_draft = baseline[:, :, 0, :32].float().argmax(dim=-1).to(torch.int32)
        candidate_draft = candidate[:, :, 0, :32].float().argmax(dim=-1).to(torch.int32)
        torch.testing.assert_close(candidate_draft, baseline_draft, rtol=0, atol=0)
        target_tokens, expected_lengths = _target_trace(baseline_draft, step)

        baseline_accepted, baseline_lengths = _strict_accept(
            worker, target_holder, baseline_draft, target_tokens
        )
        candidate_accepted, candidate_lengths = _strict_accept(
            worker, target_holder, candidate_draft, target_tokens
        )
        torch.testing.assert_close(candidate_accepted, baseline_accepted, rtol=0, atol=0)
        torch.testing.assert_close(candidate_lengths, baseline_lengths, rtol=0, atol=0)
        torch.testing.assert_close(baseline_lengths, expected_lengths, rtol=0, atol=0)

        for totals, lengths in (
            (baseline_totals, baseline_lengths),
            (candidate_totals, candidate_lengths),
        ):
            totals["accepted"] += int((lengths - 1).sum().item())
            totals["drafted"] += batch * block
            totals["requests"] += batch

        # DSpark advances its rolling position by target + accepted draft
        # tokens. Because the acceptance tensors are exact, the subsequent
        # cache trace is also identical by construction.
        position_by_slot[slots] += baseline_lengths.to(position_by_slot.dtype)

    expected_totals = {
        "accepted": sum(
            (row + step) % (block + 1)
            for step, batch in enumerate(batch_trace)
            for row in range(batch)
        ),
        "drafted": sum(batch_trace) * block,
        "requests": sum(batch_trace),
    }
    assert candidate_totals == baseline_totals == expected_totals
    assert 0 < expected_totals["accepted"] < expected_totals["drafted"]
    baseline_al = (
        baseline_totals["accepted"] + baseline_totals["requests"],
        baseline_totals["requests"],
    )
    candidate_al = (
        candidate_totals["accepted"] + candidate_totals["requests"],
        candidate_totals["requests"],
    )
    assert candidate_al == baseline_al


def test_dspark_attention_forward_batched_fused_matches_fallback(monkeypatch):
    import tensorrt_llm._torch.models.dspark.attention as dspark_attention

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, block, hidden = 1, 5, 64
    heads, head_dim, rope_dim = 24, 512, 64
    q_rank, groups, o_rank, window = 1024, 8, 32, 128

    def scaled_randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.02

    x = torch.randn(batch, block, hidden, device=device, dtype=dtype) * 0.1
    main_x = torch.randn(batch, 1, hidden, device=device, dtype=dtype) * 0.1
    start_pos = torch.tensor([5], device=device, dtype=torch.long)
    slots = torch.tensor([1], device=device, dtype=torch.long)
    kwargs = {
        "wq_a": scaled_randn(q_rank, hidden),
        "q_norm_w": torch.ones(q_rank, device=device, dtype=dtype),
        "wq_b": scaled_randn(heads * head_dim, q_rank),
        "wkv": scaled_randn(head_dim, hidden),
        "kv_norm_w": torch.ones(head_dim, device=device, dtype=dtype),
        "wo_a": scaled_randn(groups * o_rank, heads * head_dim // groups),
        "wo_b": scaled_randn(hidden, groups * o_rank),
        "attn_sink": torch.randn(heads, device=device, dtype=torch.float32) * 0.1,
        "n_heads": heads,
        "head_dim": head_dim,
        "rope_head_dim": rope_dim,
        "n_groups": groups,
        "o_lora_rank": o_rank,
        "window_size": window,
        "eps": 1e-6,
        "softmax_scale": head_dim**-0.5,
        "freqs_cis": dspark_attention.precompute_dspark_freqs_cis(rope_dim, 256, device=device),
        "persist": True,
    }
    cache_storage = torch.randn(3, 3, window, head_dim, device=device, dtype=dtype) * 0.1
    fused_cache = cache_storage[:, 1]
    fallback_cache = fused_cache.clone()
    calls = {"attention": 0, "rmsnorm_rope": 0}
    fused_attention = dspark_attention.cute_dsl_dspark_attention_rope
    fused_rmsnorm_rope = dspark_attention.cute_dsl_dspark_rmsnorm_rope

    def counted_attention(*args):
        calls["attention"] += 1
        return fused_attention(*args)

    def counted_rmsnorm_rope(*args):
        calls["rmsnorm_rope"] += 1
        return fused_rmsnorm_rope(*args)

    with monkeypatch.context() as patch:
        patch.setattr(
            dspark_attention,
            "cute_dsl_dspark_attention_rope",
            counted_attention,
        )
        patch.setattr(dspark_attention, "cute_dsl_dspark_rmsnorm_rope", counted_rmsnorm_rope)
        actual = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, fused_cache, slots, **kwargs
        )

    assert calls == {"attention": 1, "rmsnorm_rope": 4}

    with monkeypatch.context() as patch:
        patch.setattr(
            dspark_attention,
            "is_fused_dspark_attention_supported",
            lambda *args: False,
        )
        patch.setattr(
            dspark_attention,
            "is_fused_dspark_rmsnorm_rope_supported",
            lambda *args: False,
        )
        expected = dspark_attention.dspark_attention_forward_batched(
            x, main_x, start_pos, fallback_cache, slots, **kwargs
        )

    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=1e-2)
    torch.testing.assert_close(fused_cache, fallback_cache, rtol=2e-2, atol=2e-2)
