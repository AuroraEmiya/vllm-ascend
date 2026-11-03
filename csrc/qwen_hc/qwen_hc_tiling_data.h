// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include <cstdint>

namespace vllm_ascend::hc {
// DAV2201 DMA blocks and vector repeat widths; these are instruction units,
// not device capacities. Core count and UB capacity are queried at runtime.
constexpr uint32_t kDmaBlockBytes = 32;
constexpr uint32_t kVectorRepeatBytes = 256;
constexpr uint32_t kBf16Bytes = sizeof(uint16_t);
constexpr uint32_t kFloatBytes = sizeof(float);
constexpr uint32_t kBf16BlockElements = kDmaBlockBytes / kBf16Bytes;
constexpr uint32_t kFloatBlockElements = kDmaBlockBytes / kFloatBytes;
constexpr uint32_t kFloatVectorElements = kVectorRepeatBytes / kFloatBytes;
constexpr uint32_t kTransferGuardBytes = kVectorRepeatBytes;
constexpr uint32_t kMaxNormGroupBatch = 4;
constexpr uint32_t kNormScalarElements = kMaxNormGroupBatch * kFloatBlockElements;
constexpr uint32_t kNormBroadcastElements = kFloatVectorElements;
constexpr uint32_t kMaxBatchedGateGroups = 8;
// Measured A3 launch policy: amortize scheduling with at least eight FP32 repeats.
constexpr uint32_t kMinGateVectorRepeats = 8;
constexpr uint32_t kMinGateTileElements = kMinGateVectorRepeats * kFloatVectorElements;
// Candidate policy: amortize short-row launches with at least 4 KiB input per core.
constexpr uint32_t kMinSiluInputBytesPerCore = 4 * 1024;
constexpr uint32_t kMaxDmaBlockCount = 4095;
constexpr uint32_t kMaxSupportedHiddenSize = 8192;

} // namespace vllm_ascend::hc

// Ascend C launch generation forward-declares parameter types globally.
enum HcTilingKey : uint32_t { NormSingle, NormBatched, SiluSingle, SiluBatched, GateSequential, GateBatched };

// Host computes the complete launch plan. Device code does no capacity guessing.
struct HcTilingData {
  uint32_t rows = 0;
  uint32_t hiddenSize = 0;
  uint32_t groupCount = 0;
  uint32_t residualStride = 0;
  uint32_t blockStride = 0;
  uint32_t injectionStride = 0;
  uint32_t sharedWeight = 0;
  uint32_t hasInjection = 0;
  uint32_t tileSize = 0;
  uint32_t rowBatch = 1;
  uint32_t taskCount = 0;
  uint32_t blockDim = 0;
  uint32_t tilingKey = 0;  // HcTilingKey; generated launch ABI copies primitive fields only.
  float eps = 0;
};
