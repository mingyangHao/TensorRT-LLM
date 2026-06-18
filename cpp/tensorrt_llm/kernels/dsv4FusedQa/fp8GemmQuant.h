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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::dsv4_fused_qa
{

struct GemmProblem
{
    uint32_t M, N, K;
    __nv_fp8_e4m3* A;
    __nv_fp8_e4m3* B;
    __nv_fp8_e4m3* D; // FP8 output (quantized in the epilogue)
    uint8_t* sf;      // per-(token, 128-N-block) output UE8M0 scale byte, [M, N/128]
    uint32_t* sfa;
    uint32_t* sfb;
};

// 2-CTA-cluster fp8-out block-scale GEMM: D[M,N] = A[M,K] * B[N,K]^T, output requantized per
// (token, 128-N-block) into the packed UE8M0 layout deep_gemm consumes. SM100 family (sm_100/sm_103).
void launch_fp8_gemm_2cta(GemmProblem& prob, cudaStream_t stream = 0);

// Single-CTA (cta_group::1, no cluster) variant of the same fp8-out block-scale GEMM, optimized for
// small M (decode). Byte-identical output contract to launch_fp8_gemm_2cta. SM100 family (sm_100/sm_103).
void launch_fp8_gemm_1cta(GemmProblem& prob, cudaStream_t stream = 0);

} // namespace kernels::dsv4_fused_qa

TRTLLM_NAMESPACE_END
