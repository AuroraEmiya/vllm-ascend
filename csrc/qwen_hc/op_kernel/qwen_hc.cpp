// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#include "qwen_hc_combine_norm.h"
#include "qwen_hc_silu.h"
#include "qwen_hc_gate_mix.h"
#include "../qwen_hc_launch.h"
using namespace vllm_ascend::hc;

// Registered device entry names are stable for runtime/profile identity.
extern "C" __global__ __aicore__ void qwen_hc_combine_norm(GM_ADDR residual, GM_ADDR block,
    GM_ADDR injection, GM_ADDR weight, GM_ADDR combined, GM_ADDR normalized, HcTilingData tilingData) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tilingData.tilingKey == HcTilingKey::NormBatched) {
        BatchedCombineNorm<bfloat16_t> kernel;
        kernel.run(residual, block, injection, weight, combined, normalized, tilingData);
    } else {
        CombineNorm<bfloat16_t> kernel;
        kernel.run(residual, block, injection, weight, combined, normalized, tilingData);
    }
}
extern "C" __global__ __aicore__ void qwen_hc_silu(GM_ADDR input, GM_ADDR output, HcTilingData tilingData) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tilingData.tilingKey == HcTilingKey::SiluBatched) {
        BatchedSilu<bfloat16_t> kernel; kernel.run(input, output, tilingData);
    } else {
        HcSilu<bfloat16_t> kernel; kernel.run(input, output, tilingData);
    }
}
extern "C" __global__ __aicore__ void qwen_hc_gate_mix(GM_ADDR input, GM_ADDR gate,
    GM_ADDR output, HcTilingData tilingData) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tilingData.tilingKey == HcTilingKey::GateBatched) {
        BatchedGateMix<bfloat16_t> kernel; kernel.run(input, gate, output, tilingData);
    } else {
        GateMix<bfloat16_t> kernel; kernel.run(input, gate, output, tilingData);
    }
}
namespace vllm_ascend::hc {
void launchCombineNorm(uint32_t blockDim, void* stream, void* residual, void* block,
    void* injection, void* weight, void* combined, void* normalized, HcTilingData tilingData) {
    qwen_hc_combine_norm<<<blockDim, nullptr, stream>>>((GM_ADDR)residual, (GM_ADDR)block,
        (GM_ADDR)injection, (GM_ADDR)weight, (GM_ADDR)combined, (GM_ADDR)normalized, &tilingData);
}
void launchSilu(uint32_t blockDim, void* stream, void* input, void* output, HcTilingData tilingData) {
    qwen_hc_silu<<<blockDim, nullptr, stream>>>((GM_ADDR)input, (GM_ADDR)output, &tilingData);
}
void launchGateMix(uint32_t blockDim, void* stream, void* input, void* gate,
    void* output, HcTilingData tilingData) {
    qwen_hc_gate_mix<<<blockDim, nullptr, stream>>>((GM_ADDR)input, (GM_ADDR)gate,
        (GM_ADDR)output, &tilingData);
}
} // namespace vllm_ascend::hc
