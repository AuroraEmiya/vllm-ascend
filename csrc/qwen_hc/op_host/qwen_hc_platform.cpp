// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#if defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS) || defined(VLLM_ASCEND_HC_STANDALONE)
#include "qwen_hc_tiling.h"
#include <acl/acl.h>
#include <platform/platform_info.h>
#include <tiling/platform/platform_ascendc.h>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vllm_ascend::hc {
DeviceResources queryDeviceResources(int32_t deviceIndex) {
    // Callers install NPUGuard first. Cache immutable resources per device,
    // not one global first-device result; never cache a shape or tensor address.
    static std::mutex resourceMutex;
    static std::map<int32_t, DeviceResources> deviceResources;
    std::lock_guard<std::mutex> resourceLock(resourceMutex);
    auto found = deviceResources.find(deviceIndex);
    if (found != deviceResources.end()) return found->second;
    int32_t currentDevice = -1;
    if (aclrtGetDevice(&currentDevice) != ACL_SUCCESS || currentDevice != deviceIndex) {
        throw std::runtime_error("HC device resources must be queried under the input device guard");
    }
    int64_t vectorCoreCount = 0;
    if (aclrtGetDeviceInfo(deviceIndex, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreCount) != ACL_SUCCESS
        || vectorCoreCount <= 0 || uint64_t(vectorCoreCount) > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("HC failed to query the current device Vector Core count");
    }
    // Some A3 runtimes return success with zero for the UB runtime attribute.
    // Resolve UB through the public platform API using this device's exact SoC.
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) throw std::runtime_error("HC failed to query current device SoC");
    auto& platformManager = fe::PlatformInfoManager::Instance();
    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    if (platformManager.InitializePlatformInfo() != 0
        || platformManager.GetPlatformInfos(socName, platformInfos, optionalInfos) != 0) {
        throw std::runtime_error("HC failed to resolve current-SoC platform resources");
    }
    platform_ascendc::PlatformAscendC platform(&platformInfos);
    uint64_t ubSizeBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytes);
    if (!ubSizeBytes) throw std::runtime_error("HC platform returned zero UB capacity");
    DeviceResources resources{static_cast<uint32_t>(vectorCoreCount), ubSizeBytes};
    deviceResources.emplace(deviceIndex, resources);
    return resources;
}
} // namespace vllm_ascend::hc
#endif
