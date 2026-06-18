#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#

# Standalone 2-CTA-cluster fp8-out block-scale GEMM (raw PTX, tcgen05) for the
# optional DeepSeek-V4 fused q_a path. Requires the SM100 family (tcgen05 +
# 2-CTA cluster): B200 (sm_100) and B300 (sm_103). set_cuda_architectures(...
# 100f) builds a 100f-real family cubin that runs across the family when the
# build enables SM100 (i.e. 100f is in CMAKE_CUDA_ARCHITECTURES_FAMILIES);
# otherwise it falls back to a PLACEHOLDER_KERNELS stub so non-family builds
# still link (the device PTX only assembles for the SM100 family). The op and
# device-side guards (isSM100Family() / __CUDA_ARCH__ >= 1000) admit both.
set(SRC_CU fp8GemmQuant.cu)

add_library(dsv4_fused_qa_src OBJECT ${SRC_CU})
set_property(TARGET dsv4_fused_qa_src PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET dsv4_fused_qa_src PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(dsv4_fused_qa_src
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
set_cuda_architectures(dsv4_fused_qa_src 100f)
