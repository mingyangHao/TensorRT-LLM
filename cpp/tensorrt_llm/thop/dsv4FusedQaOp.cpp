/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// torch.ops.trtllm.dsv4_fused_qa_fp8_out: a 2-CTA-cluster fp8-out block-scale GEMM used by the
// optional DeepSeek-V4 fused q_a path. Computes D = A @ B^T (fp8 e4m3 inputs, packed-UE8M0 scales)
// and requantizes the result per (token, 128-N-block) into the same packed UE8M0 layout deep_gemm
// consumes, so the fused qr activation can feed q_b_proj directly. SM100 family only (Blackwell
// B200 sm_100 + B300 sm_103).

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/dsv4FusedQa/fp8GemmQuant.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cstdint>
#include <limits>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> dsv4_fused_qa_fp8_out(at::Tensor const& a_fp8, at::Tensor const& a_sf,
    at::Tensor const& b_fp8, at::Tensor const& b_sf, bool use_single_cta)
{
    CHECK_TH_CUDA(a_fp8);
    CHECK_TH_CUDA(a_sf);
    CHECK_TH_CUDA(b_fp8);
    CHECK_TH_CUDA(b_sf);

    TORCH_CHECK(tensorrt_llm::common::isSM100Family(),
        "dsv4_fused_qa_fp8_out is only supported on the SM100 family (Blackwell B200 sm_100 / B300 sm_103).");

    TORCH_CHECK(a_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "A must be float8_e4m3fn.");
    TORCH_CHECK(b_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn, "B must be float8_e4m3fn.");
    TORCH_CHECK(a_sf.scalar_type() == at::ScalarType::Int, "A scale must be int32 (packed UE8M0).");
    TORCH_CHECK(b_sf.scalar_type() == at::ScalarType::Int, "B scale must be int32 (packed UE8M0).");
    TORCH_CHECK(a_fp8.dim() == 2, "A must be a matrix [M, K].");
    TORCH_CHECK(b_fp8.dim() == 2, "B must be a matrix [N, K].");
    TORCH_CHECK(a_sf.dim() == 2, "A scale must be 2-D (packed UE8M0).");
    TORCH_CHECK(b_sf.dim() == 2, "B scale must be 2-D (packed UE8M0).");

    // A/B are read densely, so reject (do not silently fix) non-contiguous inputs. The packed-UE8M0
    // scales are read via raw pointers with kernel-computed strides -- a_sf is intentionally a strided
    // packed view (not contiguous) -- so instead of a contiguity check their exact extents/strides are
    // validated below, once K/N/m_aligned are known. Require every operand on the same CUDA device.
    CHECK_CONTIGUOUS(a_fp8);
    CHECK_CONTIGUOUS(b_fp8);
    auto const device = a_fp8.device();
    TORCH_CHECK(b_fp8.device() == device && a_sf.device() == device && b_sf.device() == device,
        "A, B, and their scale tensors must all be on the same CUDA device.");

    int64_t const M = a_fp8.size(0);
    int64_t const K = a_fp8.size(1);
    int64_t const N = b_fp8.size(0);

    TORCH_CHECK(b_fp8.size(1) == K, "A/B inner dim mismatch: A is [M, ", K, "], B is [N, ", b_fp8.size(1), "].");
    TORCH_CHECK(M <= std::numeric_limits<int32_t>::max(), "M must fit in int32.");
    TORCH_CHECK(N <= std::numeric_limits<int32_t>::max(), "N must fit in int32.");
    TORCH_CHECK(K <= std::numeric_limits<int32_t>::max(), "K must fit in int32.");

    // The kernel does not mask K tails and groups scale factors in stages of 4 K-blocks, and the
    // 2-CTA epilogue stores full 128-wide N blocks; enforce the implied shape constraints.
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128, got ", K);
    TORCH_CHECK(K % 512 == 0, "K must be a multiple of 512 (scale-factor stage grouping), got ", K);
    TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128, got ", N);
    // M tiling: the single-CTA variant owns 128 M rows per CTA; the 2-CTA cluster owns 256. The kernel
    // does not handle a partial final tile, so the caller must pad M to the variant's granularity.
    int64_t const m_tile = use_single_cta ? 128 : 256;
    TORCH_CHECK(M % m_tile == 0, "M must be a multiple of ", m_tile, " for the ",
        (use_single_cta ? "single-CTA" : "2-CTA"), " variant, got ", M);

    auto const num_n_blocks = (N + 127) / 128;
    auto const num_packed_sf_k = (num_n_blocks + 3) / 4;
    auto const m_aligned = (M + 3) / 4 * 4; // activation/output scale leading-dim padding

    // Packed-UE8M0 scale-layout guards. The kernel reads the scales via raw pointers with fixed strides,
    // so a wrong layout would silently read garbage rather than error. b_sf is contiguous
    // [ceil(K/512), N]; a_sf is the strided packed view [M, ceil(K/512)] with leading dim m_aligned
    // (i.e. strides (1, m_aligned)) -- intentionally NOT contiguous, so we validate its strides instead.
    auto const sf_k = (K + 511) / 512; // packed K-block stages (1x128 scales, 4 blocks per int32)
    TORCH_CHECK(a_sf.size(0) == M && a_sf.size(1) == sf_k, "A scale must have shape [M, ceil(K/512)] = [", M, ", ",
        sf_k, "], got [", a_sf.size(0), ", ", a_sf.size(1), "].");
    TORCH_CHECK(a_sf.stride(0) == 1 && a_sf.stride(1) == m_aligned,
        "A scale must be the packed view with strides (1, m_aligned=", m_aligned, "), got (", a_sf.stride(0), ", ",
        a_sf.stride(1), ").");
    TORCH_CHECK(b_sf.is_contiguous() && b_sf.size(0) == sf_k && b_sf.size(1) == N,
        "B scale must be a contiguous [ceil(K/512), N] = [", sf_k, ", ", N, "] tensor, got [", b_sf.size(0), ", ",
        b_sf.size(1), "] (contiguous=", b_sf.is_contiguous(), ").");

    at::Tensor D = at::detail::empty_cuda({M, N}, at::ScalarType::Float8_e4m3fn, device, /* stride */ std::nullopt);
    // Packed output scale: physical [num_packed_sf_k, m_aligned] int32, deterministic in the padded
    // [M, m_aligned) tail. Zero-fill so padded rows / out-of-range slots are well defined.
    at::Tensor Dsf
        = at::detail::empty_cuda({num_packed_sf_k, m_aligned}, at::ScalarType::Int, device, /* stride */ std::nullopt);
    Dsf.zero_();

    tensorrt_llm::kernels::dsv4_fused_qa::GemmProblem prob;
    prob.M = static_cast<uint32_t>(M);
    prob.N = static_cast<uint32_t>(N);
    prob.K = static_cast<uint32_t>(K);
    prob.A = reinterpret_cast<__nv_fp8_e4m3*>(a_fp8.data_ptr());
    prob.B = reinterpret_cast<__nv_fp8_e4m3*>(b_fp8.data_ptr());
    prob.D = reinterpret_cast<__nv_fp8_e4m3*>(D.data_ptr());
    prob.sf = reinterpret_cast<uint8_t*>(Dsf.data_ptr());
    prob.sfa = reinterpret_cast<uint32_t*>(a_sf.data_ptr());
    prob.sfb = reinterpret_cast<uint32_t*>(b_sf.data_ptr());

    auto stream = at::cuda::getCurrentCUDAStream(a_fp8.get_device());
    if (use_single_cta)
    {
        tensorrt_llm::kernels::dsv4_fused_qa::launch_fp8_gemm_1cta(prob, stream);
    }
    else
    {
        tensorrt_llm::kernels::dsv4_fused_qa::launch_fp8_gemm_2cta(prob, stream);
    }

    return {D, Dsf};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv4_fused_qa_fp8_out(Tensor a_fp8, Tensor a_sf, Tensor b_fp8, Tensor b_sf, bool use_single_cta=False) -> "
        "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv4_fused_qa_fp8_out", &tensorrt_llm::torch_ext::dsv4_fused_qa_fp8_out);
}
