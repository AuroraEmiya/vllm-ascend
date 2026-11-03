// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "qwen_hc_tiling_data.h"

namespace vllm_ascend::hc {
void launchCombineNorm(uint32_t blockDim, void* stream, void* residual, void* block, void* injection, void* weight,
                       void* combined, void* normalized, HcTilingData tilingData);
void launchSilu(uint32_t blockDim, void* stream, void* input, void* output, HcTilingData tilingData);
void launchGateMix(uint32_t blockDim, void* stream, void* input, void* gate, void* output, HcTilingData tilingData);
}  // namespace vllm_ascend::hc
