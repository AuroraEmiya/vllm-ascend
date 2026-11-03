// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "qwen_hc_common.h"
namespace vllm_ascend::hc {
template <typename T>
class CombineNorm {
 public:
  __aicore__ inline void run(GM_ADDR residual, GM_ADDR block, GM_ADDR injection, GM_ADDR weight, GM_ADDR combined,
                             GM_ADDR normalized, HcTilingData tilingData) {
    GlobalTensor<T> residualGm, blockGm, injectionGm, weightGm, combinedGm, normalizedGm;
    residualGm.SetGlobalBuffer((__gm__ T*)residual);
    blockGm.SetGlobalBuffer((__gm__ T*)block);
    injectionGm.SetGlobalBuffer((__gm__ T*)injection);
    weightGm.SetGlobalBuffer((__gm__ T*)weight);
    combinedGm.SetGlobalBuffer((__gm__ T*)combined);
    normalizedGm.SetGlobalBuffer((__gm__ T*)normalized);
    const uint32_t alignedHiddenSize =
        (tilingData.hiddenSize + (kFloatVectorElements - 1)) / kFloatVectorElements * kFloatVectorElements;
    TPipe pipe;
    TBuf<TPosition::VECCALC> residualBuffer, blockBuffer, weightBuffer, normalizedBuffer, injectionBuffer, valueBuffer,
        squareBuffer, workBuffer, scalarBuffer, broadcastBuffer;
    pipe.InitBuffer(residualBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(blockBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(weightBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(normalizedBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(injectionBuffer, kBf16BlockElements * sizeof(T));
    pipe.InitBuffer(valueBuffer, alignedHiddenSize * sizeof(float));
    pipe.InitBuffer(squareBuffer, alignedHiddenSize * sizeof(float));
    pipe.InitBuffer(workBuffer, alignedHiddenSize * sizeof(float));
    pipe.InitBuffer(scalarBuffer, kNormScalarElements * sizeof(float));
    pipe.InitBuffer(broadcastBuffer, kFloatVectorElements * sizeof(float));
    auto residualLocal = residualBuffer.Get<T>();
    auto blockLocal = blockBuffer.Get<T>();
    auto weightLocal = weightBuffer.Get<T>();
    auto normalizedLocal = normalizedBuffer.Get<T>();
    auto injectionLocal = injectionBuffer.Get<T>();
    auto values = valueBuffer.Get<float>();
    auto temporary = squareBuffer.Get<float>();
    auto workspace = workBuffer.Get<float>();
    auto scalars = scalarBuffer.Get<float>();
    auto broadcast = broadcastBuffer.Get<float>();
    for (uint32_t taskIndex = GetBlockIdx(); taskIndex < tilingData.rows * tilingData.groupCount;
         taskIndex += GetBlockNum()) {
      const uint32_t row = taskIndex / tilingData.groupCount, groupIndex = taskIndex % tilingData.groupCount;
      // Reuse is safe only after the previous MTE3 output finishes.
      syncPipeline<HardEvent::MTE3_MTE2>();
      loadTile(residualLocal, residualGm, row * tilingData.residualStride + groupIndex * tilingData.hiddenSize,
               tilingData.hiddenSize);
      loadTile(blockLocal, blockGm, row * tilingData.blockStride, tilingData.hiddenSize);
      loadTile(weightLocal, weightGm, tilingData.sharedWeight ? 0 : groupIndex * tilingData.hiddenSize,
               tilingData.hiddenSize);
      if (tilingData.hasInjection)
        loadTile(injectionLocal, injectionGm, row * tilingData.injectionStride + groupIndex, 1);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(values, residualLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      Cast(temporary, blockLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      if (tilingData.hasInjection) {
        Cast(scalars, injectionLocal, RoundMode::CAST_NONE, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Muls(scalars, scalars, -1.0f / tilingData.groupCount, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Exp(scalars, scalars, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Adds(scalars, scalars, 1.0f, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Reciprocal(scalars, scalars, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Muls(scalars, scalars, 2.0f, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        // Brcb replicates the first FP32 value into one 32B block.
        // Reuse that block in every lane/repeat, keeping the reduction
        // result in V rather than stalling the scalar pipeline.
        Brcb(broadcast, scalars, 1, {1, kFloatBlockElements});
        PipeBarrier<PIPE_V>();
        MulAddDst(values, temporary, broadcast, uint64_t(kFloatVectorElements),
                  (tilingData.hiddenSize + (kFloatVectorElements - 1)) / kFloatVectorElements,
                  {1, 1, 0, kFloatBlockElements, kFloatBlockElements, 0});
        PipeBarrier<PIPE_V>();
      }
      if (!tilingData.hasInjection) {
        Add(values, values, temporary, tilingData.hiddenSize);
        PipeBarrier<PIPE_V>();
      }
      // Preserve the original low-precision combine materialization before RMS.
      Cast(residualLocal, values, RoundMode::CAST_RINT, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Cast(values, residualLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Mul(temporary, values, values, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      ReduceSum(scalars, temporary, workspace, static_cast<int32_t>(tilingData.hiddenSize));
      PipeBarrier<PIPE_V>();
      Muls(scalars, scalars, 1.0f / tilingData.hiddenSize, kFloatBlockElements);
      PipeBarrier<PIPE_V>();
      Adds(scalars, scalars, tilingData.eps, kFloatBlockElements);
      PipeBarrier<PIPE_V>();
      Rsqrt(scalars, scalars, kFloatBlockElements);
      PipeBarrier<PIPE_V>();
      Brcb(broadcast, scalars, 1, {1, kFloatBlockElements});
      PipeBarrier<PIPE_V>();
      Mul(values, values, broadcast, uint64_t(kFloatVectorElements),
          (tilingData.hiddenSize + (kFloatVectorElements - 1)) / kFloatVectorElements,
          {1, 1, 0, kFloatBlockElements, kFloatBlockElements, 0});
      Cast(temporary, weightLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      MulAddDst(values, values, temporary, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Cast(normalizedLocal, values, RoundMode::CAST_RINT, tilingData.hiddenSize);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(combinedGm, taskIndex * tilingData.hiddenSize, residualLocal, tilingData.hiddenSize);
      storeTile(normalizedGm, taskIndex * tilingData.hiddenSize, normalizedLocal, tilingData.hiddenSize);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};

template <typename T, bool cacheWeight = false>
class BatchedCombineNorm {
 public:
  __aicore__ inline void run(GM_ADDR residual, GM_ADDR block, GM_ADDR injection, GM_ADDR weight, GM_ADDR combined,
                             GM_ADDR normalized, HcTilingData tilingData) {
    GlobalTensor<T> residualGm, blockGm, injectionGm, weightGm, combinedGm, normalizedGm;
    residualGm.SetGlobalBuffer((__gm__ T*)residual);
    blockGm.SetGlobalBuffer((__gm__ T*)block);
    injectionGm.SetGlobalBuffer((__gm__ T*)injection);
    weightGm.SetGlobalBuffer((__gm__ T*)weight);
    combinedGm.SetGlobalBuffer((__gm__ T*)combined);
    normalizedGm.SetGlobalBuffer((__gm__ T*)normalized);
    const uint32_t flatElements = tilingData.rowBatch * tilingData.hiddenSize;
    TPipe pipe;
    TBuf<TPosition::VECCALC> residualBuffer, blockBuffer, weightBuffer, normalizedBuffer, injectionBuffer, valueBuffer,
        squareBuffer, workBuffer, scalarBuffer, broadcastBuffer, cachedWeightBuffer;
    pipe.InitBuffer(residualBuffer, flatElements * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(blockBuffer, tilingData.hiddenSize * sizeof(T) + kTransferGuardBytes);
    if constexpr (cacheWeight) {
      pipe.InitBuffer(cachedWeightBuffer, flatElements * sizeof(float) + kTransferGuardBytes);
    } else {
      pipe.InitBuffer(weightBuffer, flatElements * sizeof(T) + kTransferGuardBytes);
    }
    pipe.InitBuffer(normalizedBuffer, flatElements * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(injectionBuffer, kBf16BlockElements * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(valueBuffer, flatElements * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(squareBuffer, flatElements * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(workBuffer, tilingData.hiddenSize * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(scalarBuffer, kNormScalarElements * sizeof(float));
    pipe.InitBuffer(broadcastBuffer, kFloatVectorElements * sizeof(float));
    auto residualLocal = residualBuffer.Get<T>();
    auto blockLocal = blockBuffer.Get<T>();
    auto normalizedLocal = normalizedBuffer.Get<T>();
    auto weightLocal = normalizedLocal;
    if constexpr (!cacheWeight) weightLocal = weightBuffer.Get<T>();
    LocalTensor<float> cachedWeight;
    if constexpr (cacheWeight) cachedWeight = cachedWeightBuffer.Get<float>();
    auto injectionLocal = injectionBuffer.Get<T>();
    auto values = valueBuffer.Get<float>();
    auto temporary = squareBuffer.Get<float>();
    auto workspace = workBuffer.Get<float>();
    auto scalars = scalarBuffer.Get<float>();
    auto broadcast = broadcastBuffer.Get<float>();
    const uint32_t groupsPerRow = tilingData.groupCount / tilingData.rowBatch;
    if constexpr (cacheWeight) {
      // Host guarantees the group chunk is invariant across this core's grid-stride loop.
      const uint32_t firstGroup = (GetBlockIdx() % groupsPerRow) * tilingData.rowBatch;
      const uint32_t weightElements = tilingData.sharedWeight ? tilingData.hiddenSize : flatElements;
      loadTile(weightLocal, weightGm, tilingData.sharedWeight ? 0 : firstGroup * tilingData.hiddenSize,
               weightElements);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(cachedWeight, weightLocal, RoundMode::CAST_NONE, weightElements);
      PipeBarrier<PIPE_V>();
    }
    for (uint32_t taskIndex = GetBlockIdx(); taskIndex < tilingData.rows * groupsPerRow; taskIndex += GetBlockNum()) {
      const uint32_t row = taskIndex / groupsPerRow, firstGroup = (taskIndex % groupsPerRow) * tilingData.rowBatch;
      syncPipeline<HardEvent::MTE3_MTE2>();
      loadTile(residualLocal, residualGm, row * tilingData.residualStride + firstGroup * tilingData.hiddenSize,
               flatElements);
      loadTile(blockLocal, blockGm, row * tilingData.blockStride, tilingData.hiddenSize);
      if constexpr (!cacheWeight) {
        loadTile(weightLocal, weightGm, tilingData.sharedWeight ? 0 : firstGroup * tilingData.hiddenSize,
                 tilingData.sharedWeight ? tilingData.hiddenSize : flatElements);
      }
      if (tilingData.hasInjection)
        loadTile(injectionLocal, injectionGm, row * tilingData.injectionStride + firstGroup, tilingData.rowBatch);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(values, residualLocal, RoundMode::CAST_NONE, flatElements);
      Cast(workspace, blockLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      if (tilingData.hasInjection) {
        Cast(scalars, injectionLocal, RoundMode::CAST_NONE, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Muls(scalars, scalars, -1.0f / tilingData.groupCount, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Exp(scalars, scalars, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Adds(scalars, scalars, 1.0f, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Reciprocal(scalars, scalars, kBf16BlockElements);
        PipeBarrier<PIPE_V>();
        Muls(scalars, scalars, 2.0f, kBf16BlockElements);
      } else {
        Duplicate(scalars, 1.0f, kBf16BlockElements);
      }
      PipeBarrier<PIPE_V>();
      Brcb(broadcast, scalars, 1, {1, kFloatBlockElements});
      PipeBarrier<PIPE_V>();
      for (uint32_t batchIndex = 0; batchIndex < tilingData.rowBatch; batchIndex++) {
        MulAddDst(values[batchIndex * tilingData.hiddenSize], workspace, broadcast[batchIndex * kFloatBlockElements],
                  uint64_t(kFloatVectorElements), tilingData.hiddenSize / kFloatVectorElements,
                  {1, 1, 0, kFloatBlockElements, kFloatBlockElements, 0});
      }
      PipeBarrier<PIPE_V>();
      Cast(residualLocal, values, RoundMode::CAST_RINT, flatElements);
      PipeBarrier<PIPE_V>();
      Cast(values, residualLocal, RoundMode::CAST_NONE, flatElements);
      PipeBarrier<PIPE_V>();
      Mul(temporary, values, values, flatElements);
      PipeBarrier<PIPE_V>();
      for (uint32_t batchIndex = 0; batchIndex < tilingData.rowBatch; batchIndex++) {
        ReduceSum(scalars[batchIndex * kFloatBlockElements], temporary[batchIndex * tilingData.hiddenSize], workspace,
                  static_cast<int32_t>(tilingData.hiddenSize));
        PipeBarrier<PIPE_V>();
      }
      const uint32_t scalarCount = tilingData.rowBatch * kFloatBlockElements;
      Muls(scalars, scalars, 1.0f / tilingData.hiddenSize, scalarCount);
      PipeBarrier<PIPE_V>();
      Adds(scalars, scalars, tilingData.eps, scalarCount);
      PipeBarrier<PIPE_V>();
      Rsqrt(scalars, scalars, scalarCount);
      PipeBarrier<PIPE_V>();
      for (uint32_t batchIndex = 0; batchIndex < tilingData.rowBatch; batchIndex++) {
        Brcb(broadcast, scalars[batchIndex * kFloatBlockElements], 1, {1, kFloatBlockElements});
        PipeBarrier<PIPE_V>();
        Mul(values[batchIndex * tilingData.hiddenSize], values[batchIndex * tilingData.hiddenSize], broadcast,
            uint64_t(kFloatVectorElements), tilingData.hiddenSize / kFloatVectorElements,
            {1, 1, 0, kFloatBlockElements, kFloatBlockElements, 0});
        PipeBarrier<PIPE_V>();
      }
      if (tilingData.sharedWeight) {
        auto sharedWeightValues = workspace;
        if constexpr (cacheWeight) {
          sharedWeightValues = cachedWeight;
        } else {
          Cast(workspace, weightLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
          PipeBarrier<PIPE_V>();
        }
        for (uint32_t batchIndex = 0; batchIndex < tilingData.rowBatch; batchIndex++) {
          MulAddDst(values[batchIndex * tilingData.hiddenSize], values[batchIndex * tilingData.hiddenSize], sharedWeightValues,
                    tilingData.hiddenSize);
        }
        PipeBarrier<PIPE_V>();
      } else {
        auto groupWeightValues = temporary;
        if constexpr (cacheWeight) {
          groupWeightValues = cachedWeight;
        } else {
          Cast(temporary, weightLocal, RoundMode::CAST_NONE, flatElements);
          PipeBarrier<PIPE_V>();
        }
        MulAddDst(values, values, groupWeightValues, flatElements);
        PipeBarrier<PIPE_V>();
      }
      Cast(normalizedLocal, values, RoundMode::CAST_RINT, flatElements);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(combinedGm, row * tilingData.groupCount * tilingData.hiddenSize + firstGroup * tilingData.hiddenSize,
                residualLocal, flatElements);
      storeTile(normalizedGm, row * tilingData.groupCount * tilingData.hiddenSize + firstGroup * tilingData.hiddenSize,
                normalizedLocal, flatElements);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};

}  // namespace vllm_ascend::hc
