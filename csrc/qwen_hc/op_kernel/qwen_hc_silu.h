// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "qwen_hc_common.h"
namespace vllm_ascend::hc {
template <typename T>
class HcSilu {
 public:
  __aicore__ inline void run(GM_ADDR input, GM_ADDR output, HcTilingData tilingData) {
    GlobalTensor<T> inputGm, outputGm;
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, outputBuffer, valueBuffer, activationBuffer;
    const uint32_t alignedHiddenSize = alignElements(tilingData.hiddenSize);
    pipe.InitBuffer(inputBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(outputBuffer, alignedHiddenSize * sizeof(T));
    pipe.InitBuffer(valueBuffer, alignedHiddenSize * sizeof(float));
    pipe.InitBuffer(activationBuffer, alignedHiddenSize * sizeof(float));
    auto inputLocal = inputBuffer.Get<T>();
    auto outputLocal = outputBuffer.Get<T>();
    auto values = valueBuffer.Get<float>();
    auto activation = activationBuffer.Get<float>();
    for (uint32_t row = GetBlockIdx(); row < tilingData.rows; row += GetBlockNum()) {
      syncPipeline<HardEvent::MTE3_MTE2>();
      loadTile(inputLocal, inputGm, row * tilingData.residualStride, tilingData.hiddenSize);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(values, inputLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Muls(values, values, 1.0f / tilingData.groupCount, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Muls(activation, values, -1.0f, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Exp(activation, activation, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Adds(activation, activation, 1.0f, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Reciprocal(activation, activation, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Mul(values, values, activation, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, values, RoundMode::CAST_RINT, tilingData.hiddenSize);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(outputGm, row * tilingData.hiddenSize, outputLocal, tilingData.hiddenSize);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};

template <typename T>
class BatchedSilu {
 public:
  __aicore__ inline void run(GM_ADDR input, GM_ADDR output, HcTilingData tilingData) {
    GlobalTensor<T> inputGm, outputGm;
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    const uint32_t capacity = tilingData.rowBatch * tilingData.hiddenSize;
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, outputBuffer, valueBuffer, activationBuffer;
    pipe.InitBuffer(inputBuffer, capacity * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(outputBuffer, capacity * sizeof(T) + kTransferGuardBytes);
    pipe.InitBuffer(valueBuffer, capacity * sizeof(float) + kTransferGuardBytes);
    pipe.InitBuffer(activationBuffer, capacity * sizeof(float) + kTransferGuardBytes);
    auto inputLocal = inputBuffer.Get<T>();
    auto outputLocal = outputBuffer.Get<T>();
    auto values = valueBuffer.Get<float>();
    auto activation = activationBuffer.Get<float>();
    const uint32_t taskCount = (tilingData.rows + tilingData.rowBatch - 1) / tilingData.rowBatch;
    for (uint32_t taskIndex = GetBlockIdx(); taskIndex < taskCount; taskIndex += GetBlockNum()) {
      const uint32_t row = taskIndex * tilingData.rowBatch;
      const uint32_t rowCount =
          (tilingData.rows - row < tilingData.rowBatch) ? tilingData.rows - row : tilingData.rowBatch;
      const uint32_t count = rowCount * tilingData.hiddenSize;
      syncPipeline<HardEvent::MTE3_MTE2>();
      DataCopyExtParams copyParams{static_cast<uint16_t>(rowCount), tilingData.hiddenSize * uint32_t(sizeof(T)),
                                   (tilingData.residualStride - tilingData.hiddenSize) * uint32_t(sizeof(T)), 0, 0};
      DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
      DataCopyPad(inputLocal, inputGm[row * tilingData.residualStride], copyParams, padParams);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(values, inputLocal, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Muls(values, values, 1.0f / tilingData.groupCount, count);
      PipeBarrier<PIPE_V>();
      Muls(activation, values, -1.0f, count);
      PipeBarrier<PIPE_V>();
      Exp(activation, activation, count);
      PipeBarrier<PIPE_V>();
      Adds(activation, activation, 1.0f, count);
      PipeBarrier<PIPE_V>();
      Reciprocal(activation, activation, count);
      PipeBarrier<PIPE_V>();
      Mul(values, values, activation, count);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, values, RoundMode::CAST_RINT, count);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(outputGm, row * tilingData.hiddenSize, outputLocal, count);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};

// HC count is small inputGm the model. Batch [K,tileSize] DMA and activation instead of
// issuing inputLocal separate copy/activation/synchronization chain for every branch.
}  // namespace vllm_ascend::hc
