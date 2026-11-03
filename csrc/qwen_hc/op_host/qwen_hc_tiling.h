// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "../qwen_hc_tiling_data.h"

namespace vllm_ascend::hc {
struct DeviceResources {
    uint32_t vectorCoreCount;
    uint64_t ubSizeBytes;
};
struct HcTilingPlan {
    HcTilingData data;
    uint64_t ubBytes;
};
DeviceResources queryDeviceResources(int32_t deviceIndex);
HcTilingPlan makeCombineNormTiling(HcTilingData data, DeviceResources resources);
HcTilingPlan makeSiluTiling(HcTilingData data, DeviceResources resources);
HcTilingPlan makeGateMixTiling(HcTilingData data, DeviceResources resources);
uint64_t requiredUbBytes(const HcTilingData& data);
} // namespace vllm_ascend::hc
