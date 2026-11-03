// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "qwen_hc_common.h"
namespace vllm_ascend::hc {
template <typename T>
class GateMix {
 public:
  __aicore__ inline void run(GM_ADDR values, GM_ADDR gates, GM_ADDR output, HcTilingData tilingData) {
    GlobalTensor<T> inputGm, gateGm, outputGm;
    inputGm.SetGlobalBuffer((__gm__ T*)values);
    gateGm.SetGlobalBuffer((__gm__ T*)gates);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, gateBuffer, outputBuffer, valueBuffer, activationBuffer, accumulatorBuffer;
    pipe.InitBuffer(inputBuffer, tilingData.tileSize * sizeof(T));
    pipe.InitBuffer(gateBuffer, tilingData.tileSize * sizeof(T));
    pipe.InitBuffer(outputBuffer, tilingData.tileSize * sizeof(T));
    pipe.InitBuffer(valueBuffer, tilingData.tileSize * sizeof(float));
    pipe.InitBuffer(activationBuffer, tilingData.tileSize * sizeof(float));
    pipe.InitBuffer(accumulatorBuffer, tilingData.tileSize * sizeof(float));
    auto inputLocal = inputBuffer.Get<T>();
    auto gateLocal = gateBuffer.Get<T>();
    auto outputLocal = outputBuffer.Get<T>();
    auto valuesLocal = valueBuffer.Get<float>();
    auto activation = activationBuffer.Get<float>();
    auto accumulator = accumulatorBuffer.Get<float>();
    const uint32_t tiles = (tilingData.hiddenSize + tilingData.tileSize - 1) / tilingData.tileSize;
    for (uint32_t taskIndex = GetBlockIdx(); taskIndex < tilingData.rows * tiles; taskIndex += GetBlockNum()) {
      const uint32_t row = taskIndex / tiles, offset = (taskIndex % tiles) * tilingData.tileSize;
      const uint32_t count =
          (tilingData.hiddenSize - offset < tilingData.tileSize) ? tilingData.hiddenSize - offset : tilingData.tileSize;
      syncPipeline<HardEvent::MTE3_V>();
      Duplicate(accumulator, 0.0f, count);
      PipeBarrier<PIPE_V>();
      for (uint32_t groupIndex = 0; groupIndex < tilingData.groupCount; groupIndex++) {
        syncPipeline<HardEvent::V_MTE2>();
        loadTile(inputLocal, inputGm, row * tilingData.residualStride + groupIndex * tilingData.hiddenSize + offset,
                 count);
        loadTile(gateLocal, gateGm, row * tilingData.blockStride + groupIndex * tilingData.hiddenSize + offset, count);
        syncPipeline<HardEvent::MTE2_V>();
        Cast(valuesLocal, inputLocal, RoundMode::CAST_NONE, count);
        Cast(activation, gateLocal, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(activation, activation, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(activation, activation, count);
        PipeBarrier<PIPE_V>();
        Adds(activation, activation, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Reciprocal(activation, activation, count);
        PipeBarrier<PIPE_V>();
        Mul(valuesLocal, valuesLocal, activation, count);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, valuesLocal, count);
        PipeBarrier<PIPE_V>();
      }
      Muls(accumulator, accumulator, 1.0f / tilingData.groupCount, count);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, accumulator, RoundMode::CAST_RINT, count);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(outputGm, row * tilingData.hiddenSize + offset, outputLocal, count);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};
template <typename T>
class BatchedGateMix {
 public:
  __aicore__ inline void run(GM_ADDR values, GM_ADDR gates, GM_ADDR output, HcTilingData tilingData) {
    GlobalTensor<T> inputGm, gateGm, outputGm;
    inputGm.SetGlobalBuffer((__gm__ T*)values);
    gateGm.SetGlobalBuffer((__gm__ T*)gates);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, gateBuffer, outputBuffer, valueBuffer, activationBuffer, accumulatorBuffer;
    const uint32_t capacity = tilingData.groupCount * tilingData.tileSize;
    pipe.InitBuffer(inputBuffer, capacity * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(gateBuffer, capacity * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(outputBuffer, tilingData.tileSize * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(valueBuffer, capacity * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(activationBuffer, capacity * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(accumulatorBuffer, tilingData.tileSize * sizeof(float) + kTransferGuardBytes);
    auto inputLocal = inputBuffer.Get<T>();
    auto gateLocal = gateBuffer.Get<T>();
    auto outputLocal = outputBuffer.Get<T>();
    auto valuesLocal = valueBuffer.Get<float>();
    auto activation = activationBuffer.Get<float>();
    auto accumulator = accumulatorBuffer.Get<float>();
    const uint32_t tiles = (tilingData.hiddenSize + tilingData.tileSize - 1) / tilingData.tileSize;
    for (uint32_t taskIndex = GetBlockIdx(); taskIndex < tilingData.rows * tiles; taskIndex += GetBlockNum()) {
      const uint32_t row = taskIndex / tiles, offset = (taskIndex % tiles) * tilingData.tileSize;
      const uint32_t count =
          (tilingData.hiddenSize - offset < tilingData.tileSize) ? tilingData.hiddenSize - offset : tilingData.tileSize;
      const uint32_t alignedCount = alignElements(count), flatElements = tilingData.groupCount * alignedCount;
      syncPipeline<HardEvent::MTE3_MTE2>();
      DataCopyExtParams copyParams{static_cast<uint16_t>(tilingData.groupCount), count * uint32_t(sizeof(T)),
                                   (tilingData.hiddenSize - count) * uint32_t(sizeof(T)), 0, 0};
      DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(alignedCount - count), 0};
      DataCopyPad(inputLocal, inputGm[row * tilingData.residualStride + offset], copyParams, padParams);
      DataCopyPad(gateLocal, gateGm[row * tilingData.blockStride + offset], copyParams, padParams);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(valuesLocal, inputLocal, RoundMode::CAST_NONE, flatElements);
      Cast(activation, gateLocal, RoundMode::CAST_NONE, flatElements);
      PipeBarrier<PIPE_V>();
      Muls(activation, activation, -1.0f, flatElements);
      PipeBarrier<PIPE_V>();
      Exp(activation, activation, flatElements);
      PipeBarrier<PIPE_V>();
      Adds(activation, activation, 1.0f, flatElements);
      PipeBarrier<PIPE_V>();
      Reciprocal(activation, activation, flatElements);
      PipeBarrier<PIPE_V>();
      Mul(valuesLocal, valuesLocal, activation, flatElements);
      PipeBarrier<PIPE_V>();
      Duplicate(accumulator, 0.0f, count);
      PipeBarrier<PIPE_V>();
      if (tilingData.groupCount == 4) {
        // Two independent pair sums, then one final K reduction.
        Add(accumulator, valuesLocal, valuesLocal[alignedCount], count);
        Add(activation, valuesLocal[2 * alignedCount], valuesLocal[3 * alignedCount], count);
        PipeBarrier<PIPE_V>();
        Add(accumulator, accumulator, activation, count);
        PipeBarrier<PIPE_V>();
      } else {
        for (uint32_t groupIndex = 0; groupIndex < tilingData.groupCount; groupIndex++) {
          Add(accumulator, accumulator, valuesLocal[groupIndex * alignedCount], count);
          PipeBarrier<PIPE_V>();
        }
      }
      Muls(accumulator, accumulator, 1.0f / tilingData.groupCount, count);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, accumulator, RoundMode::CAST_RINT, count);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(outputGm, row * tilingData.hiddenSize + offset, outputLocal, count);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};
}  // namespace vllm_ascend::hc
