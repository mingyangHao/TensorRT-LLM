# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4 fused q_a path: replace [kv_a_proj q-slice GEMM -> q_a_layernorm -> fp8_quant] with a
single fused fp8-out GEMM that emits (qr_fp8, qr_sf) directly for q_b_proj (and the indexer's wq_b).

Enabled by env TRTLLM_DSV4_FUSED_QA=1, off by default. The fp8-out GEMM kernels (2-CTA cluster + a
single-CTA small-M variant, raw PTX) are compiled into the TensorRT-LLM library and exposed as
torch.ops.trtllm.dsv4_fused_qa_fp8_out (SM100 family: B200 sm_100 + B300 sm_103); there is no runtime
JIT. gamma (q_a_layernorm weight) is folded offline into the q-slice weight; q_a_layernorm's per-token
1/rms factor is dropped.

Accuracy: dropping the 1/rms factor scales each token's qr by a positive per-token scalar d_t = rms(q).
For the attention branch this is near-exact -- q_b_proj is linear so q is scaled by the same d_t, and the
subsequent unweighted per-head q_b_layernorm (RMSNorm, has_weights=False) divides d_t back out by
scale-invariance. The only attention-path residuals are the O(rms_norm_eps) RMSNorm epsilon term and fp8
quantization. The genuine residual is the sparse indexer, which consumes qr BEFORE q_b_layernorm and so
sees the d_t-scaled qr: its top-k selection is rank-invariant to a positive per-token scale, but the fp8
quantization and emitted weights/scales must be measured.

Kernel I/O (byte-exact vs fp8_quantize_1x128_packed_ue8m0 on the gamma-folded, RMS-dropped input):
  fused_qa_fp8_out(A_fp8[M,K], A_sf, B_fp8[N,K], B_sf) -> (D_fp8[M,N], D_sf_packed[num_packed_sf_k, m_aligned])
  A_sf / B_sf: packed UE8M0 int32, physical [sf_k, lead] (data_ptr = buffer start). Requires M%4==0, N%4==0.
"""

import torch
import torch.nn.functional as F

# The fused fp8-out GEMM is the registered op torch.ops.trtllm.dsv4_fused_qa_fp8_out (compiled into
# libth_common; see cpp/tensorrt_llm/thop/dsv4FusedQaOp.cpp). Its fake/meta is registered in
# tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py for graph capture / torch.compile.


# ---------------------------------------------------------------------------
# Weight prep (offline / lazy at first forward). All UE8M0, matching deep_gemm.
# ---------------------------------------------------------------------------
def _ceil_log2_pow2_e8m0(amax: torch.Tensor):
    """amax [..] -> (e8m0 byte uint8, quant_scale=2^e fp32) for smallest 2^e >= amax/448 (no saturation)."""
    s = (amax.float() / 448.0).clamp_min(1e-10)
    e = torch.ceil(torch.log2(s))  # smallest integer e with 2^e >= s
    byte = (e + 127.0).clamp(0, 255).to(torch.uint8)
    return byte, torch.exp2(e)


def requant_128x128_ue8m0(w_bf16: torch.Tensor):
    """w [N,K] bf16 -> (fp8 [N,K], e8m0 byte [nb, kb])  (128x128 block UE8M0, deep_gemm weight format)."""
    N, K = w_bf16.shape
    nb, kb = (N + 127) // 128, (K + 127) // 128
    wp = F.pad(w_bf16.float(), (0, kb * 128 - K, 0, nb * 128 - N))
    amax = wp.view(nb, 128, kb, 128).abs().amax(dim=(1, 3))  # [nb, kb]
    byte, scale = _ceil_log2_pow2_e8m0(amax)  # [nb, kb]
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    fp8 = (w_bf16.float() / scale_full).to(torch.float8_e4m3fn)
    return fp8, byte


def weight_scale_128x128_to_fused_sfb(byte_nbkb: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """e8m0 [nb,kb] -> kernel sfb [sfb_k, N] int32 (broadcast 128x128 -> per-N, pack 4 K-blocks/uint32)."""
    nb, kb = byte_nbkb.shape
    sfb_k = (kb + 3) // 4
    per_n = byte_nbkb.repeat_interleave(128, 0)[:N].to(torch.int32)  # [N, kb]
    packed = torch.zeros((N, sfb_k), dtype=torch.int32, device=byte_nbkb.device)
    for j in range(kb):
        packed[:, j // 4] |= (per_n[:, j] & 0xFF) << ((j % 4) * 8)
    return packed.t().contiguous()  # [sfb_k, N]


def dequant_weight_128x128(w_fp8: torch.Tensor, byte_nbkb: torch.Tensor) -> torch.Tensor:
    """fp8 [N,K] + e8m0 BYTE [nb,kb] -> bf16 [N,K] (scale = 2^(byte-127))."""
    N, K = w_fp8.shape
    scale = torch.exp2(byte_nbkb.float() - 127.0)
    scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    return (w_fp8.float() * scale_full).bfloat16()


def dequant_weight_block_float(w_fp8: torch.Tensor, scale_nbkb: torch.Tensor) -> torch.Tensor:
    """fp8 [N,K] + FLOAT 128x128 block scale [nb,kb] -> bf16 [N,K]. This is the layout
    FP8BlockScalesLinearMethod stores (scale value = data_amax/448, UE8M0 powers of 2 as float32) --
    distinct from the e8m0-byte form ``dequant_weight_128x128`` consumes."""
    N, K = w_fp8.shape
    scale_full = scale_nbkb.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    return (w_fp8.float() * scale_full).bfloat16()


def deep_gemm_nt_out(
    a_fp8: torch.Tensor,
    a_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype,
    M_valid: int,
    disable_ue8m0_cast: bool = True,
) -> torch.Tensor:
    """deep_gemm.fp8_gemm_nt with a PRE-quantized activation (a_fp8 [Mfull,K], a_sf packed-UE8M0 view),
    against an fp8 module weight + its scale. Returns out[:M_valid] [.., N]. Used to feed q_b_proj /
    indexer.wq_b the fused qr fp8 directly (skip their internal _fp8_quantize)."""
    from tensorrt_llm import deep_gemm

    Mfull, N = a_fp8.shape[0], weight.shape[0]
    out = a_fp8.new_empty((Mfull, N), dtype=out_dtype)
    deep_gemm.fp8_gemm_nt(
        (a_fp8, a_sf), (weight, weight_scale), out, disable_ue8m0_cast=disable_ue8m0_cast
    )
    return out[:M_valid]


def dequant_qr_to_bf16(qr_fp8: torch.Tensor, qr_sf_view: torch.Tensor, K: int) -> torch.Tensor:
    """fp8 [M,K] + packed-UE8M0 a_sf view [M, num_packed] -> bf16 [M,K] (graph-capturable)."""
    num_kb = (K + 127) // 128
    kb = torch.arange(num_kb, device=qr_fp8.device)
    packed = qr_sf_view[:, kb // 4].to(torch.int64)
    byte = (packed >> ((kb % 4) * 8).to(torch.int64)) & 0xFF
    scale = torch.exp2(byte.float() - 127.0)
    return (qr_fp8.float() * scale.repeat_interleave(128, dim=1)[:, :K]).bfloat16()


def realign_packed_sf_for_deepgemm(sf_packed: torch.Tensor, m: int) -> torch.Tensor:
    """Re-lay a packed-UE8M0 activation scale into the layout deep_gemm requires for an ``m``-row GEMM.

    The fused kernel emits its output scale as physical ``[num_packed, m_aligned(Mpad)]`` -- M padded to
    the kernel's cluster granularity (256 for 2-CTA, 128 for single-CTA). deep_gemm consumes an activation
    scale as an ``[m, num_packed]`` view and asserts its leading-dim stride equals
    ``get_tma_aligned_size(m, elt)`` for the REAL row count ``m`` -- but the kernel's padding leaves that
    stride at ``m_aligned(Mpad)``, which differs whenever ``Mpad != align(m, 4)`` (e.g. decode m=16 padded
    to 128). Copy the real-m columns into a freshly aligned buffer so the resulting transposed view is
    deep_gemm-valid. Packed int32 values are preserved exactly.
    """
    from tensorrt_llm.quantization.utils.fp8_utils import get_tma_aligned_size

    num_packed = sf_packed.shape[0]
    aligned = get_tma_aligned_size(m, sf_packed.element_size())
    buf = sf_packed.new_zeros((num_packed, aligned))
    buf[:, :m] = sf_packed[:, :m]
    view = buf[:, :m].t()  # [m, num_packed], stride (1, aligned)
    assert view.stride(0) == 1 and view.stride(1) == aligned
    return view


def run_fused_qa_qpath(
    hidden_bf16: torch.Tensor,
    W_q_fp8: torch.Tensor,
    W_q_sfb: torch.Tensor,
    single_cta: bool = False,
):
    """Compute the gamma-folded q-slice as fp8 directly. Pads M to the kernel's cluster granularity
    (256 for the 2-CTA variant, 128 for the single-CTA small-M variant — required for correctness),
    runs the fused fp8-out GEMM, slices back.

    Returns (qr_fp8 [M, q_lora_rank], qr_sf_view [M, num_packed] re-laid to deep_gemm's real-M TMA-aligned
    layout, hidden_fp8 [Mpad, K], hidden_sf) — the quantized (padded) hidden is returned for reuse by the
    kv GEMM (which runs on Mpad rows, so its scale stays Mpad-aligned and needs no realign).
    """
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import _fp8_quantize_1x128_ue8m0

    M = hidden_bf16.shape[0]
    pad = 128 if single_cta else 256
    Mpad = (M + pad - 1) // pad * pad
    hidden_p = hidden_bf16 if Mpad == M else F.pad(hidden_bf16, (0, 0, 0, Mpad - M))
    hp_fp8, hp_sf = _fp8_quantize_1x128_ue8m0(hidden_p, tactic=0)
    qr_fp8_p, qr_sf_p = torch.ops.trtllm.dsv4_fused_qa_fp8_out(
        hp_fp8, hp_sf, W_q_fp8, W_q_sfb, single_cta
    )
    qr_fp8 = qr_fp8_p[:M].contiguous()
    # The kernel pads M to its cluster granularity, so qr_sf_p's leading-dim stride is m_aligned(Mpad).
    # q_b_proj's deep_gemm runs on the real-M qr_fp8 and requires the A-scale stride TMA-aligned to the
    # real M -- realign so the [M, num_packed] view has stride (1, get_tma_aligned_size(M)).
    qr_sf_view = realign_packed_sf_for_deepgemm(qr_sf_p, M)
    return qr_fp8, qr_sf_view, hp_fp8, hp_sf, Mpad


def build_fused_qa_weights(
    kv_a_weight_fp8: torch.Tensor,
    kv_a_weight_scale: torch.Tensor,
    gamma: torch.Tensor,
    q_lora_rank: int,
):
    """Split kv_a_proj weight at q_lora_rank, fold gamma into the q-slice (drop RMS), and produce both
    the fused-op B operands (W_q_folded fp8 + sfb) and the bf16 kv-slice GEMM weight.

    kv_a_weight_scale is FP8BlockScalesLinearMethod's FLOAT32 128x128 block scale [nb, kb] for the full
    [N_full, K] weight (values = data_amax/448, UE8M0 powers of 2). The q-slice is dequantized with that
    float scale; the kv-slice scale is transformed into deep_gemm's packed layout for the kv GEMM.
    Returns dict with W_q fp8/sfb (for the fused op) and W_kvrope fp8 + deep_gemm-ready scale.
    """
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    N_full, K = kv_a_weight_fp8.shape
    q_nb = q_lora_rank // 128
    # --- q-slice: dequant with the FLOAT block scale, fold gamma per row, requant to UE8M0 (fused op B) ---
    W_q_fp8 = kv_a_weight_fp8[:q_lora_rank]
    W_q_scale = kv_a_weight_scale[:q_nb]  # FLOAT32 [q_nb, kb] block scale (UE8M0 powers of 2)
    W_q_bf16 = dequant_weight_block_float(W_q_fp8, W_q_scale)
    W_q_folded = (W_q_bf16.float() * gamma.float().unsqueeze(1)).bfloat16()
    Wqf_fp8, Wqf_byte = requant_128x128_ue8m0(W_q_folded)
    Wqf_sfb = weight_scale_128x128_to_fused_sfb(Wqf_byte, q_lora_rank, K)
    # --- kv+rope slice: bf16 deep_gemm. Transform the FLOAT block scale into deep_gemm's packed layout
    # (mn-major, TMA-aligned, UE8M0). Feeding the raw [nb,kb] float to deep_gemm produces NaN. ---
    N_kv = N_full - q_lora_rank
    W_kvrope_fp8 = kv_a_weight_fp8[q_lora_rank:].contiguous()
    W_kvrope_scale = transform_sf_into_required_layout(
        kv_a_weight_scale[q_nb:].contiguous(), mn=N_kv, k=K, recipe=(1, 128, 128), is_sfa=False
    )
    return {
        "W_q_fp8": Wqf_fp8.contiguous(),
        "W_q_sfb": Wqf_sfb,
        "W_kvrope_fp8": W_kvrope_fp8,
        "W_kvrope_scale": W_kvrope_scale,
    }
