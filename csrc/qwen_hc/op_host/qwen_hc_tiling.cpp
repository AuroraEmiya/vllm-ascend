// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#include "qwen_hc_tiling.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vllm_ascend::hc {
namespace {
uint64_t ceilDivide(uint64_t value, uint64_t divisor) { return value / divisor + (value % divisor != 0); }
uint64_t alignUp(uint64_t value, uint64_t alignment) { return ceilDivide(value, alignment) * alignment; }
void checkInputs(const HcTilingData& data, DeviceResources resources) {
  if (!resources.vectorCoreCount || !resources.ubSizeBytes) {
    throw std::invalid_argument("HC requires positive queried device resources");
  }
  if (!data.hiddenSize || data.hiddenSize > kMaxSupportedHiddenSize || !data.groupCount) {
    throw std::invalid_argument("HC shape is outside the validated hidden-size/group range");
  }
}
HcTilingPlan finishPlan(HcTilingData data, DeviceResources resources, uint64_t tasks) {
  // Grid-stride task counters must not wrap on their last increment.
  if (tasks > std::numeric_limits<uint32_t>::max() - uint64_t(resources.vectorCoreCount)) {
    throw std::invalid_argument("HC task count exceeds safe uint32 indexing");
  }
  data.taskCount = static_cast<uint32_t>(tasks);
  data.blockDim = static_cast<uint32_t>(std::min(tasks, uint64_t(resources.vectorCoreCount)));
  const auto ubBytes = requiredUbBytes(data);
  if (ubBytes > resources.ubSizeBytes) {
    throw std::invalid_argument("HC full-row branch exceeds the queried device UB capacity");
  }
  return {data, ubBytes};
}
}  // namespace

uint64_t requiredUbBytes(const HcTilingData& data) {
  const uint64_t hiddenSize = data.hiddenSize;
  const uint64_t flatElements = hiddenSize * data.rowBatch;
  switch (data.tilingKey) {
    case HcTilingKey::NormSingle:
      // Four BF16 rows, three FP32 rows, injection, scalars and broadcast.
      return (4 * kBf16Bytes + 3 * kFloatBytes) * alignUp(hiddenSize, kFloatVectorElements) +
             kBf16BlockElements * kBf16Bytes + (kNormScalarElements + kNormBroadcastElements) * kFloatBytes;
    case HcTilingKey::NormBatched:
      // rt/wt/nt and xf/tmp are batched; block/work remain one row.
      return (3 * kBf16Bytes + 2 * kFloatBytes) * flatElements + (kBf16Bytes + kFloatBytes) * hiddenSize +
             kBf16BlockElements * kBf16Bytes + (kNormScalarElements + kNormBroadcastElements) * kFloatBytes +
             8 * kTransferGuardBytes;
    case HcTilingKey::SiluSingle:
      return (2 * kBf16Bytes + 2 * kFloatBytes) * alignUp(hiddenSize, kBf16BlockElements);
    case HcTilingKey::SiluBatched:
      return (2 * kBf16Bytes + 2 * kFloatBytes) * flatElements + 4 * kTransferGuardBytes;
    case HcTilingKey::GateSequential:
      return (3 * kBf16Bytes + 3 * kFloatBytes) * uint64_t(data.tileSize);
    case HcTilingKey::GateBatched:
      return ((2 * kBf16Bytes + 2 * kFloatBytes) * uint64_t(data.groupCount) + kBf16Bytes + kFloatBytes) *
                 data.tileSize +
             6 * kTransferGuardBytes;
  }
  throw std::invalid_argument("Unknown HC tiling key");
}

HcTilingPlan makeCombineNormTiling(HcTilingData data, DeviceResources resources) {
  checkInputs(data, resources);
  data.rowBatch = 1;
  data.tilingKey = HcTilingKey::NormSingle;
  const uint64_t groupTasks = uint64_t(data.rows) * data.groupCount;
  // Batch only work that would revisit a core. Each norm owns a full H row.
  const uint64_t groupsPerCore = std::max(uint64_t(1), ceilDivide(groupTasks, resources.vectorCoreCount));
  if (data.hiddenSize % kFloatVectorElements == 0) {
    const auto maxBatch = std::min(uint64_t(kMaxNormGroupBatch), groupsPerCore);
    for (uint32_t batch = static_cast<uint32_t>(maxBatch); batch > 1; --batch) {
      if (data.groupCount % batch != 0) continue;
      data.rowBatch = batch;
      data.tilingKey = HcTilingKey::NormBatched;
      if (requiredUbBytes(data) <= resources.ubSizeBytes) break;
      data.rowBatch = 1;
      data.tilingKey = HcTilingKey::NormSingle;
    }
  }
  return finishPlan(data, resources, groupTasks / data.rowBatch);
}

HcTilingPlan makeSiluTiling(HcTilingData data, DeviceResources resources) {
  checkInputs(data, resources);
  data.rowBatch = 1;
  data.tilingKey = HcTilingKey::SiluSingle;
  if (data.hiddenSize % kBf16BlockElements == 0 && data.residualStride >= data.hiddenSize) {
    const auto rowWaves = std::max(uint64_t(1), ceilDivide(data.rows, resources.vectorCoreCount));
    constexpr uint64_t guardBytes = 4 * kTransferGuardBytes;
    constexpr uint64_t bytesPerElement = 2 * kBf16Bytes + 2 * kFloatBytes;
    const uint64_t ubRows = resources.ubSizeBytes > guardBytes
                                ? (resources.ubSizeBytes - guardBytes) / (bytesPerElement * data.hiddenSize)
                                : 0;
    const auto batch = std::min({rowWaves, ubRows, uint64_t(kMaxDmaBlockCount)});
    if (batch > 1) {
      data.rowBatch = static_cast<uint32_t>(batch);
      data.tilingKey = HcTilingKey::SiluBatched;
    }
  }
  return finishPlan(data, resources, ceilDivide(data.rows, data.rowBatch));
}

HcTilingPlan makeGateMixTiling(HcTilingData data, DeviceResources resources) {
  checkInputs(data, resources);
  data.tilingKey = data.groupCount <= kMaxBatchedGateGroups ? HcTilingKey::GateBatched : HcTilingKey::GateSequential;
  data.tileSize = kBf16BlockElements;
  const uint64_t guardBytes = data.tilingKey == HcTilingKey::GateBatched ? 6 * kTransferGuardBytes : 0;
  const uint64_t bytesPerColumn =
      data.tilingKey == HcTilingKey::GateBatched
          ? (2 * kBf16Bytes + 2 * kFloatBytes) * uint64_t(data.groupCount) + kBf16Bytes + kFloatBytes
          : 3 * kBf16Bytes + 3 * kFloatBytes;
  const uint64_t ubColumns =
      resources.ubSizeBytes > guardBytes
          ? (resources.ubSizeBytes - guardBytes) / bytesPerColumn / kBf16BlockElements * kBf16BlockElements
          : 0;
  if (!ubColumns) throw std::invalid_argument("HC device UB cannot hold one aligned gate column tile");
  const uint64_t tilesPerRow =
      std::max(uint64_t(1), ceilDivide(resources.vectorCoreCount, std::max(uint64_t(1), uint64_t(data.rows))));
  const uint64_t balancedColumns = alignUp(ceilDivide(data.hiddenSize, tilesPerRow), kBf16BlockElements);
  data.tileSize = static_cast<uint32_t>(std::min(ubColumns, balancedColumns));
  return finishPlan(data, resources, uint64_t(data.rows) * ceilDivide(data.hiddenSize, data.tileSize));
}
}  // namespace vllm_ascend::hc
