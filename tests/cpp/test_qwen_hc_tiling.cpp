// SPDX-License-Identifier: Apache-2.0
#include "../../csrc/qwen_hc/op_host/qwen_hc_tiling.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace vllm_ascend::hc;
namespace {
void checkPlan(const HcTilingPlan& plan, DeviceResources resources) {
  const auto& data = plan.data;
  assert(plan.ubBytes <= resources.ubSizeBytes);
  assert(data.blockDim <= resources.vectorCoreCount);
  assert(data.blockDim == std::min(data.taskCount, resources.vectorCoreCount));
  assert((data.rows == 0) == (data.blockDim == 0));
  if (data.taskCount < 100000) {
    std::vector<uint32_t> visits(data.taskCount, 0);
    for (uint32_t coreIndex = 0; coreIndex < data.blockDim; ++coreIndex) {
      for (uint32_t taskIndex = coreIndex; taskIndex < data.taskCount; taskIndex += data.blockDim) {
        ++visits[taskIndex];
      }
    }
    for (auto count : visits) assert(count == 1);
  }
  if (data.tilingKey == HcTilingKey::NormBatched) {
    assert(data.groupCount % data.rowBatch == 0);
    assert(data.hiddenSize % kFloatVectorElements == 0);
    assert(data.rowBatch <= kMaxNormGroupBatch);
  }
  if (data.tilingKey == HcTilingKey::SiluBatched) {
    assert(data.hiddenSize % kBf16BlockElements == 0);
    assert(data.rowBatch <= kMaxDmaBlockCount);
  }
  if (data.tilingKey == HcTilingKey::GateBatched || data.tilingKey == HcTilingKey::GateSequential) {
    assert(data.tileSize && data.tileSize % kBf16BlockElements == 0);
    uint32_t covered = 0;
    for (uint32_t column = 0; column < data.hiddenSize; column += data.tileSize) {
      covered += std::min(data.tileSize, data.hiddenSize - column);
    }
    assert(covered == data.hiddenSize);
  }
}
}  // namespace
int main() {
  size_t accepted = 0, rejected = 0;
  for (uint32_t coreCount : {1u, 8u, 20u, 32u, 40u, 48u, 64u}) {
    for (uint64_t ubKiB : {64u, 128u, 192u, 248u}) {
      DeviceResources resources{coreCount, ubKiB * 1024};
      for (uint32_t rows : {0u, 1u, 16u, 32u, 64u, 256u}) {
        for (uint32_t hiddenSize : {1u, 15u, 16u, 63u, 64u, 65u, 320u, 2560u, 8192u}) {
          for (uint32_t groups : {1u, 2u, 3u, 4u, 6u, 8u, 9u}) {
            HcTilingData data{};
            data.rows = rows;
            data.hiddenSize = hiddenSize;
            data.groupCount = groups;
            data.residualStride = groups * hiddenSize;
            for (auto makePlan : {makeCombineNormTiling, makeSiluTiling, makeGateMixTiling}) {
              try {
                checkPlan(makePlan(data, resources), resources);
                ++accepted;
              } catch (const std::invalid_argument&) {
                ++rejected;
              }
            }
          }
        }
      }
    }
  }
  HcTilingData data{};
  data.rows = 64;
  data.hiddenSize = 2560;
  data.groupCount = 4;
  data.residualStride = 10240;
  auto smallerDevice = makeCombineNormTiling(data, {20, 192 * 1024});
  auto largerDevice = makeCombineNormTiling(data, {64, 192 * 1024});
  assert(smallerDevice.data.blockDim == 20 && largerDevice.data.blockDim == 64);
  auto smallUb = makeCombineNormTiling(data, {40, 64 * 1024});
  assert(smallUb.data.rowBatch == 1 && smallerDevice.data.rowBatch == 4);
  // Missing resource evidence must fail, rather than inventing a default.
  for (DeviceResources resources : {DeviceResources{0, 192 * 1024}, DeviceResources{40, 0}}) {
    bool failed = false;
    try {
      makeCombineNormTiling(data, resources);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    assert(failed);
  }
  std::cout << "accepted=" << accepted << " capacity_rejected=" << rejected << std::endl;
}
