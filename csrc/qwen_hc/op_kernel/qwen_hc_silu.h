// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "qwen_hc_common.h"
#include "adv_api/activation/silu.h"
namespace vllm_ascend::hc {
template <typename T>
class HcSilu {
 public:
  __aicore__ inline void run(GM_ADDR input, GM_ADDR output, HcTilingData tilingData) {
    GlobalTensor<T> inputGm, outputGm;
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    TPipe pipe;
    const uint32_t alignedHiddenSize = alignElements(tilingData.hiddenSize);
    const uint32_t inputBytes = alignedHiddenSize * sizeof(T);
    const uint32_t floatBytes = alignedHiddenSize * sizeof(float);
    const uint32_t outputOffsetBytes = inputBytes;
    const uint32_t valueOffsetBytes = outputOffsetBytes + inputBytes;
    const uint32_t activationOffsetBytes = valueOffsetBytes + floatBytes;
    TBuf<TPosition::VECCALC> storage;
    // Disjoint DMA-aligned regions, with the same total live UB as four TBufs.
    pipe.InitBuffer(storage, activationOffsetBytes + floatBytes);
    auto inputLocal = storage.Get<T>();
    auto outputLocal = storage.Get<T>()[outputOffsetBytes / sizeof(T)];
    auto values = storage.Get<float>()[valueOffsetBytes / sizeof(float)];
    auto activation = storage.Get<float>()[activationOffsetBytes / sizeof(float)];
    for (uint32_t row = GetBlockIdx(); row < tilingData.rows; row += GetBlockNum()) {
      syncPipeline<HardEvent::MTE3_MTE2>();
      loadTile(inputLocal, inputGm, row * tilingData.residualStride, tilingData.hiddenSize);
      syncPipeline<HardEvent::MTE2_V>();
      Cast(values, inputLocal, RoundMode::CAST_NONE, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Muls(values, values, 1.0f / tilingData.groupCount, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      // Public Silu uses Div on A3; keep source separate from destination.
      Silu<float>(activation, values, tilingData.hiddenSize);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, activation, RoundMode::CAST_RINT, tilingData.hiddenSize);
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
    const uint32_t inputBytes = capacity * sizeof(T) + kTransferGuardBytes;
    const uint32_t floatBytes = capacity * sizeof(float) + kTransferGuardBytes;
    const uint32_t outputOffsetBytes = inputBytes;
    const uint32_t valueOffsetBytes = outputOffsetBytes + inputBytes;
    const uint32_t activationOffsetBytes = valueOffsetBytes + floatBytes;
    TBuf<TPosition::VECCALC> storage;
    // Retain a separate transfer guard for every logical region.
    pipe.InitBuffer(storage, activationOffsetBytes + floatBytes);
    auto inputLocal = storage.Get<T>();
    auto outputLocal = storage.Get<T>()[outputOffsetBytes / sizeof(T)];
    auto values = storage.Get<float>()[valueOffsetBytes / sizeof(float)];
    auto activation = storage.Get<float>()[activationOffsetBytes / sizeof(float)];
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
      // Public Silu uses Div on A3; keep source separate from destination.
      Silu<float>(activation, values, count);
      PipeBarrier<PIPE_V>();
      Cast(outputLocal, activation, RoundMode::CAST_RINT, count);
      syncPipeline<HardEvent::V_MTE3>();
      storeTile(outputGm, row * tilingData.hiddenSize, outputLocal, count);
    }
    syncPipeline<HardEvent::MTE3_S>();
  }
};

}  // namespace vllm_ascend::hc
