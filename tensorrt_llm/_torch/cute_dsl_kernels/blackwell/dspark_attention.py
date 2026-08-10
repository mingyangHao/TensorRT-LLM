# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused DSpark rolling-window attention for Blackwell.

The DSpark draft attends to a small, fixed rolling window and the current draft
block. This kernel consumes the two sources separately so the hot path never
materializes a concatenated KV tensor or a gather-index tensor.
"""

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class DSparkAttentionKernel:
    """Warp-per-head, query-tiled MQA attention with an attention sink."""

    # A warp owns one request/head and one or more draft queries. Each lane
    # keeps 16 dimensions of every owned query in registers, avoiding a
    # shared-memory reduction and CTA barriers per attended token.
    log2_e = 1.4426950408889634

    def __init__(
        self,
        window_size: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scale: float,
        warps_per_cta: int = 1,
        dynamic_context_loop: bool = False,
        queries_per_warp: int = 1,
        min_blocks_per_mp: int = 0,
        inverse_rope_dim: int = 0,
    ):
        if warps_per_cta not in (1, 2, 4, 8):
            raise ValueError(
                "DSparkAttentionKernel warps_per_cta must be one of 1, 2, 4, or 8; "
                f"got {warps_per_cta}"
            )
        if num_heads % warps_per_cta != 0:
            raise ValueError(
                "DSparkAttentionKernel num_heads must be divisible by warps_per_cta; "
                f"got num_heads={num_heads}, warps_per_cta={warps_per_cta}"
            )
        if head_dim % cute.arch.WARP_SIZE != 0:
            raise ValueError(
                f"DSparkAttentionKernel head_dim must be divisible by {cute.arch.WARP_SIZE}; "
                f"got {head_dim}"
            )
        if queries_per_warp < 1 or block_size % queries_per_warp != 0:
            raise ValueError(
                "DSparkAttentionKernel queries_per_warp must be a positive divisor "
                f"of block_size; got block_size={block_size}, "
                f"queries_per_warp={queries_per_warp}"
            )
        if (
            inverse_rope_dim < 0
            or inverse_rope_dim > head_dim
            or inverse_rope_dim % (2 * cute.arch.WARP_SIZE) != 0
        ):
            raise ValueError(
                "DSparkAttentionKernel inverse_rope_dim must be a multiple of 64 "
                f"in [0, {head_dim}]; got {inverse_rope_dim}"
            )
        self.window_size = window_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.warps_per_cta = warps_per_cta
        self.num_threads = cute.arch.WARP_SIZE * warps_per_cta
        self.elements_per_thread = head_dim // cute.arch.WARP_SIZE
        self.softmax_scale = softmax_scale
        self.dynamic_context_loop = dynamic_context_loop
        self.queries_per_warp = queries_per_warp
        self.min_blocks_per_mp = min_blocks_per_mp
        self.inverse_rope_dim = inverse_rope_dim
        self.nope_dim = head_dim - inverse_rope_dim
        self.nope_elements_per_thread = self.nope_dim // cute.arch.WARP_SIZE
        self.rope_elements_per_thread = inverse_rope_dim // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        main_kv: cute.Tensor,
        block_kv: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        attn_sink: cute.Tensor,
        inverse_rope_freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.queries_per_warp > 1):
            self.fused_queries_kernel(
                q,
                main_kv,
                block_kv,
                kv_cache,
                slots,
                start_pos,
                attn_sink,
                inverse_rope_freqs,
                output,
            ).launch(
                grid=[
                    q.shape[0],
                    self.block_size // self.queries_per_warp,
                    self.num_heads // self.warps_per_cta,
                ],
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.min_blocks_per_mp,
            )
        else:
            self.kernel(
                q,
                main_kv,
                block_kv,
                kv_cache,
                slots,
                start_pos,
                attn_sink,
                inverse_rope_freqs,
                output,
            ).launch(
                grid=[q.shape[0], self.block_size, self.num_heads // self.warps_per_cta],
                block=[self.num_threads, 1, 1],
                stream=stream,
                min_blocks_per_mp=self.min_blocks_per_mp,
            )

    @cute.jit
    def _exp(self, value: cutlass.Float32):
        return cute.math.exp2(value * self.log2_e, fastmath=True)

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        main_kv: cute.Tensor,
        block_kv: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        attn_sink: cute.Tensor,
        inverse_rope_freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        request_idx, query_idx, head_group_idx = cute.arch.block_idx()
        if cutlass.const_expr(self.warps_per_cta == 1):
            # Keep the current PR's one-warp launch/index path as the exact A/B
            # baseline, including its generated instruction sequence.
            lane_idx, _, _ = cute.arch.thread_idx()
            warp_idx = cutlass.Int32(0)
            head_idx = head_group_idx
        else:
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            head_idx = head_group_idx * self.warps_per_cta + warp_idx

        slot = cutlass.Int32(slots[request_idx])
        position = cutlass.Int32(start_pos[request_idx])
        write_pos = position % self.window_size

        # Exactly one CTA persists the captured-context KV. Attention CTAs read
        # main_kv directly for write_pos, so no inter-CTA synchronization is
        # needed before the newly written row becomes visible on the next step.
        if query_idx == 0 and head_group_idx == 0 and warp_idx == 0:
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                kv_cache[slot, write_pos, dim] = main_kv[request_idx, dim]

        q_values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
        accum = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
        for item in cutlass.range_constexpr(self.elements_per_thread):
            dim = lane_idx + item * cute.arch.WARP_SIZE
            q_values[item] = cutlass.Float32(q[request_idx, query_idx, head_idx, dim])
            accum[item] = cutlass.Float32(0.0)

        running_max = -cutlass.Float32.inf
        running_sum = cutlass.Float32(0.0)

        # Physical cache order is irrelevant to attention. Before the window
        # fills, slot c is valid iff c <= start_pos; once full all rows are valid.
        context_rows = self.window_size
        if cutlass.const_expr(self.dynamic_context_loop):
            context_rows = position + 1
            if context_rows > self.window_size:
                context_rows = self.window_size
        for context_idx in cutlass.range(context_rows, unroll=1):
            is_valid = context_idx <= position
            if cutlass.const_expr(self.dynamic_context_loop):
                is_valid = True
            if is_valid:
                partial = cutlass.Float32(0.0)
                values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    dim = lane_idx + item * cute.arch.WARP_SIZE
                    value = cutlass.Float32(0.0)
                    if context_idx == write_pos:
                        value = cutlass.Float32(main_kv[request_idx, dim])
                    else:
                        value = cutlass.Float32(kv_cache[slot, context_idx, dim])
                    values[item] = value
                    partial += q_values[item] * value

                score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
                new_max = cute.arch.fmax(running_max, score)
                old_scale = cutlass.Float32(0.0)
                if running_max != -cutlass.Float32.inf:
                    old_scale = self._exp(running_max - new_max)
                weight = self._exp(score - new_max)
                running_sum = running_sum * old_scale + weight
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    accum[item] = accum[item] * old_scale + weight * values[item]
                running_max = new_max

        # The current draft block is non-causal: every query sees every block KV.
        for block_idx in cutlass.range_constexpr(self.block_size):
            partial = cutlass.Float32(0.0)
            values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                value = cutlass.Float32(block_kv[request_idx, block_idx, dim])
                values[item] = value
                partial += q_values[item] * value

            score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
            new_max = cute.arch.fmax(running_max, score)
            old_scale = self._exp(running_max - new_max)
            weight = self._exp(score - new_max)
            running_sum = running_sum * old_scale + weight
            for item in cutlass.range_constexpr(self.elements_per_thread):
                accum[item] = accum[item] * old_scale + weight * values[item]
            running_max = new_max

        sink_weight = self._exp(cutlass.Float32(attn_sink[head_idx]) - running_max)
        inv_denom = cutlass.Float32(1.0) / (running_sum + sink_weight)
        for item in cutlass.range_constexpr(self.nope_elements_per_thread):
            dim = lane_idx + item * cute.arch.WARP_SIZE
            output[request_idx, query_idx, head_idx, dim] = (accum[item] * inv_denom).to(
                output.element_type
            )

        # The production path immediately applies inverse RoPE to the last 64
        # dimensions. Preserve the former kernel boundary's BF16 rounding, then
        # exchange adjacent real/imaginary values within the warp. This removes
        # the intermediate output write/read without changing any QK, softmax,
        # PV, or RoPE arithmetic.
        for rope_item in cutlass.range_constexpr(self.rope_elements_per_thread):
            item = self.nope_elements_per_thread + rope_item
            dim = lane_idx + item * cute.arch.WARP_SIZE
            rounded = cutlass.Float32((accum[item] * inv_denom).to(output.element_type))
            partner = cute.arch.shuffle_sync_bfly(rounded, offset=1)
            pair = (dim - self.nope_dim) // 2
            cos = cutlass.Float32(inverse_rope_freqs[request_idx, query_idx, pair, 0])
            sin = cutlass.Float32(inverse_rope_freqs[request_idx, query_idx, pair, 1])
            rotated = cutlass.Float32(0.0)
            if lane_idx % 2 == 0:
                rotated = rounded * cos + partner * sin
            else:
                rotated = rounded * cos - partner * sin
            output[request_idx, query_idx, head_idx, dim] = rotated.to(output.element_type)

    @cute.kernel
    def fused_queries_kernel(
        self,
        q: cute.Tensor,
        main_kv: cute.Tensor,
        block_kv: cute.Tensor,
        kv_cache: cute.Tensor,
        slots: cute.Tensor,
        start_pos: cute.Tensor,
        attn_sink: cute.Tensor,
        inverse_rope_freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        request_idx, query_group_idx, head_group_idx = cute.arch.block_idx()
        if cutlass.const_expr(self.warps_per_cta == 1):
            lane_idx, _, _ = cute.arch.thread_idx()
            warp_idx = cutlass.Int32(0)
            head_idx = head_group_idx
        else:
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            head_idx = head_group_idx * self.warps_per_cta + warp_idx

        slot = cutlass.Int32(slots[request_idx])
        position = cutlass.Int32(start_pos[request_idx])
        write_pos = position % self.window_size

        if query_group_idx == 0 and head_group_idx == 0 and warp_idx == 0:
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                kv_cache[slot, write_pos, dim] = main_kv[request_idx, dim]

        # Every query in a DSpark draft block attends to exactly the same KV
        # rows. Keep independent query/softmax/output state in registers and
        # load each 512-wide KV row once. For each query, token traversal and
        # lane-local arithmetic order are identical to the one-query kernel.
        q_values = cute.make_rmem_tensor(
            (self.queries_per_warp, self.elements_per_thread), cutlass.Float32
        )
        accum = cute.make_rmem_tensor(
            (self.queries_per_warp, self.elements_per_thread), cutlass.Float32
        )
        running_max = cute.make_rmem_tensor((self.queries_per_warp,), cutlass.Float32)
        running_sum = cute.make_rmem_tensor((self.queries_per_warp,), cutlass.Float32)
        query_start = query_group_idx * self.queries_per_warp
        for local_query_idx in cutlass.range_constexpr(self.queries_per_warp):
            query_idx = query_start + local_query_idx
            running_max[local_query_idx] = -cutlass.Float32.inf
            running_sum[local_query_idx] = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                q_values[local_query_idx, item] = cutlass.Float32(
                    q[request_idx, query_idx, head_idx, dim]
                )
                accum[local_query_idx, item] = cutlass.Float32(0.0)

        context_rows = self.window_size
        if cutlass.const_expr(self.dynamic_context_loop):
            context_rows = position + 1
            if context_rows > self.window_size:
                context_rows = self.window_size
        for context_idx in cutlass.range(context_rows, unroll=1):
            is_valid = context_idx <= position
            if cutlass.const_expr(self.dynamic_context_loop):
                is_valid = True
            if is_valid:
                values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    dim = lane_idx + item * cute.arch.WARP_SIZE
                    value = cutlass.Float32(0.0)
                    if context_idx == write_pos:
                        value = cutlass.Float32(main_kv[request_idx, dim])
                    else:
                        value = cutlass.Float32(kv_cache[slot, context_idx, dim])
                    values[item] = value

                for query_idx in cutlass.range_constexpr(self.queries_per_warp):
                    partial = cutlass.Float32(0.0)
                    for item in cutlass.range_constexpr(self.elements_per_thread):
                        partial += q_values[query_idx, item] * values[item]

                    score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
                    new_max = cute.arch.fmax(running_max[query_idx], score)
                    old_scale = cutlass.Float32(0.0)
                    if running_max[query_idx] != -cutlass.Float32.inf:
                        old_scale = self._exp(running_max[query_idx] - new_max)
                    weight = self._exp(score - new_max)
                    running_sum[query_idx] = running_sum[query_idx] * old_scale + weight
                    for item in cutlass.range_constexpr(self.elements_per_thread):
                        accum[query_idx, item] = (
                            accum[query_idx, item] * old_scale + weight * values[item]
                        )
                    running_max[query_idx] = new_max

        # Keep this outer loop rolled to avoid multiplying the instruction
        # footprint by block_size twice; the inner query loop is deliberately
        # unrolled so the compiler can keep every query state in registers.
        for block_idx in cutlass.range(self.block_size, unroll=1):
            values = cute.make_rmem_tensor((self.elements_per_thread,), cutlass.Float32)
            for item in cutlass.range_constexpr(self.elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                values[item] = cutlass.Float32(block_kv[request_idx, block_idx, dim])

            for query_idx in cutlass.range_constexpr(self.queries_per_warp):
                partial = cutlass.Float32(0.0)
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    partial += q_values[query_idx, item] * values[item]

                score = cute.arch.warp_reduction_sum(partial) * self.softmax_scale
                new_max = cute.arch.fmax(running_max[query_idx], score)
                old_scale = self._exp(running_max[query_idx] - new_max)
                weight = self._exp(score - new_max)
                running_sum[query_idx] = running_sum[query_idx] * old_scale + weight
                for item in cutlass.range_constexpr(self.elements_per_thread):
                    accum[query_idx, item] = (
                        accum[query_idx, item] * old_scale + weight * values[item]
                    )
                running_max[query_idx] = new_max

        for local_query_idx in cutlass.range_constexpr(self.queries_per_warp):
            sink_weight = self._exp(
                cutlass.Float32(attn_sink[head_idx]) - running_max[local_query_idx]
            )
            inv_denom = cutlass.Float32(1.0) / (running_sum[local_query_idx] + sink_weight)
            query_idx = query_start + local_query_idx
            for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                dim = lane_idx + item * cute.arch.WARP_SIZE
                output[request_idx, query_idx, head_idx, dim] = (
                    accum[local_query_idx, item] * inv_denom
                ).to(output.element_type)

            for rope_item in cutlass.range_constexpr(self.rope_elements_per_thread):
                item = self.nope_elements_per_thread + rope_item
                dim = lane_idx + item * cute.arch.WARP_SIZE
                rounded = cutlass.Float32(
                    (accum[local_query_idx, item] * inv_denom).to(output.element_type)
                )
                partner = cute.arch.shuffle_sync_bfly(rounded, offset=1)
                pair = (dim - self.nope_dim) // 2
                cos = cutlass.Float32(inverse_rope_freqs[request_idx, query_idx, pair, 0])
                sin = cutlass.Float32(inverse_rope_freqs[request_idx, query_idx, pair, 1])
                rotated = cutlass.Float32(0.0)
                if lane_idx % 2 == 0:
                    rotated = rounded * cos + partner * sin
                else:
                    rotated = rounded * cos - partner * sin
                output[request_idx, query_idx, head_idx, dim] = rotated.to(output.element_type)
