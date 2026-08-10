# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch custom op for fused DSpark rolling-window attention."""

import functools
import os

import cutlass
import cutlass.cute as cute
import torch

from ..._utils import get_sm_version, is_sm_100f
from ...logger import logger
from ..cute_dsl_kernels.blackwell.dspark_attention import DSparkAttentionKernel

_INDEX_DTYPE_TO_CUTLASS = {
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}

_VALID_WARPS_PER_CTA = (1, 2, 4, 8)


def _get_dspark_attention_warps_per_cta() -> int:
    """Return the acceptance-preserving head-warp packing knob."""
    value = int(os.environ.get("TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA", "1"))
    if value not in _VALID_WARPS_PER_CTA:
        raise ValueError(
            "TRTLLM_DSPARK_ATTENTION_WARPS_PER_CTA must be one of "
            f"{_VALID_WARPS_PER_CTA}; got {value}"
        )
    return value


def _get_dspark_attention_dynamic_context_loop() -> bool:
    value = os.environ.get("TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP", "1")
    if value not in ("0", "1"):
        raise ValueError(
            f"TRTLLM_DSPARK_ATTENTION_DYNAMIC_CONTEXT_LOOP must be 0 or 1; got {value}"
        )
    return value == "1"


def get_dspark_attention_fuse_inverse_rope() -> bool:
    value = os.environ.get("TRTLLM_DSPARK_ATTENTION_FUSE_INVERSE_ROPE", "1")
    if value not in ("0", "1"):
        raise ValueError(f"TRTLLM_DSPARK_ATTENTION_FUSE_INVERSE_ROPE must be 0 or 1; got {value}")
    return value == "1"


def _get_dspark_attention_min_blocks_per_mp(
    block_size: int, batch_size: int, queries_per_warp: int
) -> int:
    value = os.environ.get("TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP", "auto")
    if value == "auto":
        # DL4 qtile2 straddles a register-allocation threshold for a few odd
        # rank-local batch sizes. A 16-block launch bound improves those wave
        # shapes without spilling; it loses at B1/B2/B6/B8 and stays disabled
        # elsewhere unless explicitly requested for tuning.
        if block_size == 4 and queries_per_warp == 2 and batch_size in (3, 5, 7, 32):
            return 16
        return 0
    try:
        min_blocks = int(value)
    except ValueError as error:
        raise ValueError(
            f"TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP must be an integer; got {value}"
        ) from error
    if min_blocks not in (0, 8, 9, 10, 12, 16):
        raise ValueError(
            "TRTLLM_DSPARK_ATTENTION_MIN_BLOCKS_PER_MP must be auto, 0, 8, 9, 10, "
            "12, or 16; "
            f"got {min_blocks}"
        )
    return min_blocks


def _get_dspark_attention_queries_per_warp(block_size: int, batch_size: int) -> int:
    override = os.environ.get("TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP")
    if override is None or override == "auto":
        # Rank-local batches are small under attention DP and show sharp wave
        # boundaries, so dispatch uses the exhaustive B1-B8 B200 sweep instead
        # of a single global threshold. Both paths remain batch-symbolic.
        if block_size == 4:
            return 1 if batch_size in (2, 4) else 2
        if block_size == 5:
            return 5 if batch_size >= 6 else 1
        if block_size == 6:
            if batch_size == 1:
                return 1
            return 2 if batch_size <= 6 else 6
        return 1

    value = int(override)
    if value < 1:
        raise ValueError(f"TRTLLM_DSPARK_ATTENTION_QUERIES_PER_WARP must be positive; got {value}")
    return value


def is_fused_dspark_attention_supported(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
) -> bool:
    """Return whether the production DSpark shape can use the CuteDSL op."""
    if not is_sm_100f():
        return False
    if not all(t.is_cuda for t in (q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink)):
        return False
    if q.dtype != torch.bfloat16:
        return False
    if main_kv.dtype != q.dtype or block_kv.dtype != q.dtype or kv_cache.dtype != q.dtype:
        return False
    if attn_sink.dtype != torch.float32:
        return False
    if slots.dtype not in (torch.int32, torch.int64) or start_pos.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if slots.dtype != start_pos.dtype:
        return False
    if q.ndim != 4 or main_kv.ndim != 2 or block_kv.ndim != 3 or kv_cache.ndim != 3:
        return False
    head_dim = q.shape[-1]
    return (
        head_dim == 512
        and main_kv.shape == (q.shape[0], head_dim)
        and block_kv.shape == (q.shape[0], q.shape[1], head_dim)
        and kv_cache.shape[-1] == head_dim
        and attn_sink.shape == (q.shape[2],)
        and slots.shape == (q.shape[0],)
        and start_pos.shape == (q.shape[0],)
        and q.is_contiguous()
        and main_kv.is_contiguous()
        and block_kv.is_contiguous()
        and slots.is_contiguous()
        and start_pos.is_contiguous()
        and attn_sink.is_contiguous()
    )


def is_fused_dspark_attention_rope_supported(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    inverse_rope_dim: int,
) -> bool:
    """Return whether attention can fuse its exact inverse-RoPE epilogue."""
    return (
        is_fused_dspark_attention_supported(
            q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink
        )
        and inverse_rope_dim > 0
        and inverse_rope_dim <= q.shape[-1]
        and inverse_rope_dim % 64 == 0
        and inverse_rope_freqs.is_cuda
        and inverse_rope_freqs.dtype == torch.float32
        and inverse_rope_freqs.shape == (q.shape[0], q.shape[1], inverse_rope_dim // 2, 2)
        and inverse_rope_freqs.is_contiguous()
    )


@functools.cache
def _compile_fused_dspark_attention(
    block_size: int,
    num_heads: int,
    head_dim: int,
    window_size: int,
    cache_stride: tuple[int, ...],
    index_dtype: torch.dtype,
    softmax_scale: float,
    warps_per_cta: int,
    dynamic_context_loop: bool,
    queries_per_warp: int,
    min_blocks_per_mp: int,
    inverse_rope_dim: int,
):
    # Batch is deliberately symbolic: DSpark's block/head geometry is fixed by
    # the model, while the generation batch changes from iteration to iteration.
    # Each selected tactic therefore covers every eager and CUDA-graph batch
    # size that resolves to it; the automatic policy currently selects at most
    # two query-tile variants per draft length.
    batch_size = cute.sym_int()
    q_shape = (batch_size, block_size, num_heads, head_dim)
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, q_shape, stride_order=(3, 2, 1, 0)
    )
    main_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (batch_size, head_dim), stride_order=(1, 0)
    )
    block_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch_size, block_size, head_dim),
        stride_order=(2, 1, 0),
    )
    cache_fake = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), window_size, head_dim),
        stride=cache_stride,
    )
    index_cutlass_dtype = _INDEX_DTYPE_TO_CUTLASS[index_dtype]
    slots_fake = cute.runtime.make_fake_compact_tensor(
        index_cutlass_dtype, (batch_size,), stride_order=(0,)
    )
    start_fake = cute.runtime.make_fake_compact_tensor(
        index_cutlass_dtype, (batch_size,), stride_order=(0,)
    )
    sink_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (num_heads,), stride_order=(0,)
    )
    if inverse_rope_dim > 0:
        inverse_rope_freqs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (batch_size, block_size, inverse_rope_dim // 2, 2),
            stride_order=(3, 2, 1, 0),
        )
    else:
        # The non-RoPE specialization DCEs this argument. Reuse the sink's
        # one-dimensional ABI so the existing attention custom op allocates no
        # dummy tensor and retains its original runtime behavior.
        inverse_rope_freqs_fake = sink_fake
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, q_shape, stride_order=(3, 2, 1, 0)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DSparkAttentionKernel(
        window_size=window_size,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        warps_per_cta=warps_per_cta,
        dynamic_context_loop=dynamic_context_loop,
        queries_per_warp=queries_per_warp,
        min_blocks_per_mp=min_blocks_per_mp,
        inverse_rope_dim=inverse_rope_dim,
    )
    return cute.compile(
        kernel,
        q_fake,
        main_fake,
        block_fake,
        cache_fake,
        slots_fake,
        start_fake,
        sink_fake,
        inverse_rope_freqs_fake,
        output_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run fused DSpark cache update + sliding-window MQA attention."""
    if not is_fused_dspark_attention_supported(
        q, main_kv, block_kv, kv_cache, slots, start_pos, attn_sink
    ):
        raise ValueError(
            "cute_dsl_dspark_attention requires contiguous BF16 production DSpark tensors "
            f"with head_dim=512 on SM100/SM103; got SM {get_sm_version()}"
        )
    return _run_cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        attn_sink,
        attn_sink,
        0,
        softmax_scale,
    )


def _run_cute_dsl_dspark_attention(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    inverse_rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.empty_like(q)
    warps_per_cta = _get_dspark_attention_warps_per_cta()
    dynamic_context_loop = _get_dspark_attention_dynamic_context_loop()
    queries_per_warp = _get_dspark_attention_queries_per_warp(q.shape[1], q.shape[0])
    min_blocks_per_mp = _get_dspark_attention_min_blocks_per_mp(
        q.shape[1], q.shape[0], queries_per_warp
    )
    if q.shape[2] % warps_per_cta != 0:
        raise ValueError(
            "DSpark Attention num_heads must be divisible by warps_per_cta; "
            f"got num_heads={q.shape[2]}, warps_per_cta={warps_per_cta}"
        )
    if q.shape[1] % queries_per_warp != 0:
        raise ValueError(
            "DSpark Attention queries_per_warp must divide the draft block; "
            f"got block={q.shape[1]}, queries_per_warp={queries_per_warp}"
        )
    logger.info_once(
        "DSpark Attention enabled: implementation=dspark_attn, "
        f"warps_per_cta={warps_per_cta}, dynamic_context_loop={dynamic_context_loop}, "
        f"queries_per_warp={queries_per_warp}, "
        f"min_blocks_per_mp={min_blocks_per_mp}, "
        f"fuse_inverse_rope={inverse_rope_dim > 0}, "
        f"block={q.shape[1]}, "
        f"heads={q.shape[2]}, head_dim={q.shape[3]}",
        key=(
            f"dspark_attention|warps_per_cta={warps_per_cta}|"
            f"dynamic_context_loop={dynamic_context_loop}|block={q.shape[1]}"
            f"|queries_per_warp={queries_per_warp}"
            f"|min_blocks_per_mp={min_blocks_per_mp}"
            f"|inverse_rope_dim={inverse_rope_dim}"
        ),
    )
    compiled = _compile_fused_dspark_attention(
        q.shape[1],
        q.shape[2],
        q.shape[3],
        kv_cache.shape[1],
        tuple(kv_cache.stride()),
        slots.dtype,
        softmax_scale,
        warps_per_cta,
        dynamic_context_loop,
        queries_per_warp,
        min_blocks_per_mp,
        inverse_rope_dim,
    )
    compiled(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        attn_sink,
        inverse_rope_freqs,
        output,
    )
    return output


@torch.library.custom_op(
    "trtllm::cute_dsl_dspark_attention_rope",
    mutates_args=("kv_cache",),
    device_types="cuda",
)
def cute_dsl_dspark_attention_rope(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    inverse_rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run DSpark attention and its exact, BF16-rounded inverse-RoPE epilogue."""
    if not is_fused_dspark_attention_rope_supported(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        attn_sink,
        inverse_rope_freqs,
        inverse_rope_dim,
    ):
        raise ValueError(
            "cute_dsl_dspark_attention_rope requires the production DSpark "
            "attention contract and contiguous FP32 RoPE frequencies"
        )
    return _run_cute_dsl_dspark_attention(
        q,
        main_kv,
        block_kv,
        kv_cache,
        slots,
        start_pos,
        attn_sink,
        inverse_rope_freqs,
        inverse_rope_dim,
        softmax_scale,
    )


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention")
def _(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.register_fake("trtllm::cute_dsl_dspark_attention_rope")
def _(
    q: torch.Tensor,
    main_kv: torch.Tensor,
    block_kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    start_pos: torch.Tensor,
    attn_sink: torch.Tensor,
    inverse_rope_freqs: torch.Tensor,
    inverse_rope_dim: int,
    softmax_scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)
