// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_tgr_mix_tiling.h"
using namespace AscendC;
namespace {
class XTgrMixKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate, GM_ADDR y, GM_ADDR inv,
                              GM_ADDR inject, const XTgrMixTilingData& data) {
    t = data;
    xG.SetGlobalBuffer((__gm__ bfloat16_t*)x);
    gG.SetGlobalBuffer((__gm__ float*)gamma);
    uG.SetGlobalBuffer((__gm__ bfloat16_t*)aux);
    gateG.SetGlobalBuffer((__gm__ bfloat16_t*)gate);
    yG.SetGlobalBuffer((__gm__ bfloat16_t*)y);
    invG.SetGlobalBuffer((__gm__ float*)inv);
    injectG.SetGlobalBuffer((__gm__ bfloat16_t*)inject);
    const uint32_t count = t.workElements;
    buffered = !t.hasGate && t.units >= 2 * GetBlockNum();
    if (buffered) {
      pipe.InitBuffer(mixInput, 2, count * 4);
      pipe.InitBuffer(mixOutput, 2, t.tile * 2);
    } else {
      pipe.InitBuffer(b0, count * 2);
      pipe.InitBuffer(b1, count * 2);
    }
    pipe.InitBuffer(f0, count * 4);
    pipe.InitBuffer(f1, count * 4);
    pipe.InitBuffer(f2, t.tile * 4);
  }
  __aicore__ inline void Process() {
    if (buffered) {
      Prefetch(GetBlockIdx());
      for (uint32_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum()) {
        auto input = mixInput.DeQue<bfloat16_t>();
        if (unit + GetBlockNum() < t.units) Prefetch(unit + GetBlockNum());
        auto output = mixOutput.AllocTensor<bfloat16_t>();
        ComputeMix(input, input[t.workElements], output, t.tile);
        mixInput.FreeTensor(input);
        mixOutput.EnQue(output);
        auto ready = mixOutput.DeQue<bfloat16_t>();
        const uint32_t h = t.width / t.channels, tiles = h / t.tile;
        DataCopy(yG[(unit / tiles) * h + (unit % tiles) * t.tile], ready, t.tile);
        mixOutput.FreeTensor(ready);
      }
      Fence<HardEvent::MTE3_MTE2>();
      return;
    }
    for (uint32_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum()) {
      Mix(unit);
    }
  }

 private:
  __aicore__ inline void Barrier() { PipeBarrier<PIPE_V>(); }

  template <HardEvent event>
  __aicore__ inline void Fence() {
    const auto id = static_cast<event_t>(pipe.FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
  }

  __aicore__ inline void Sigmoid(LocalTensor<float> v, uint32_t n) {
    Muls(v, v, -1.0f, n);
    Barrier();
    Exp(v, v, n);
    Barrier();
    Adds(v, v, 1.0f, n);
    Barrier();
    // This path ends at BF16 outputs. The precise FP32 inverse-RMS
    // auxiliary remains on Sqrt+Div in Norm, with its stricter checks.
    Reciprocal(v, v, n);
    Barrier();
  }

  __aicore__ inline void Prefetch(uint32_t unit) {
    const uint32_t h = t.width / t.channels, tiles = h / t.tile;
    const uint32_t off = (unit / tiles) * t.width + (unit % tiles) * t.tile;
    auto input = mixInput.AllocTensor<bfloat16_t>();
    const DataCopyParams cp{static_cast<uint16_t>(t.channels), static_cast<uint16_t>(t.tile / 16),
                            static_cast<uint16_t>((h - t.tile) / 16), 0};
    DataCopy(input, xG[off], cp);
    DataCopy(input[t.workElements], uG[off], cp);
    mixInput.EnQue(input);
  }
  __aicore__ inline void ComputeMix(LocalTensor<bfloat16_t> xb, LocalTensor<bfloat16_t> ub,
                                    LocalTensor<bfloat16_t> output, uint32_t n) {
    const uint32_t all = n * t.channels;
    auto z = f0.Get<float>(), q = f1.Get<float>(), sum = f2.Get<float>();
    Cast(z, xb, RoundMode::CAST_NONE, all);
    Cast(q, ub, RoundMode::CAST_NONE, all);
    Barrier();
    Sigmoid(q, all);
    Mul(z, z, q, all);
    Barrier();
    Add(sum, z, z[n], n);
    Barrier();
    Add(sum, sum, z[2 * n], n);
    Barrier();
    Add(sum, sum, z[3 * n], n);
    Barrier();
    Muls(sum, sum, 1.0f / t.channels, n);
    Barrier();
    Cast(output, sum, RoundMode::CAST_RINT, n);
  }
  __aicore__ inline void Mix(uint32_t unit) {
    uint32_t h = t.width / t.channels, tiles = (h + t.tile - 1) / t.tile;
    uint32_t row = unit / tiles, col = (unit % tiles) * t.tile, n = h - col;
    n = n < t.tile ? n : t.tile;
    auto xb = b0.Get<bfloat16_t>(), ub = b1.Get<bfloat16_t>();
    auto q = f1.Get<float>();
    // DataCopy strides and lengths are in 32-byte blocks. All supported
    // tiles divide the hidden width and are multiples of 16 BF16 values.
    const DataCopyParams cp{static_cast<uint16_t>(t.channels), static_cast<uint16_t>(n / 16),
                            static_cast<uint16_t>((h - n) / 16), 0};
    const uint32_t off = row * t.width + col;
    DataCopy(xb, xG[off], cp);
    DataCopy(ub, uG[off], cp);
    Fence<HardEvent::MTE2_V>();
    ComputeMix(xb, ub, xb, n);
    Fence<HardEvent::V_MTE3>();
    DataCopy(yG[row * h + col], xb, n);
    if (t.hasGate && col == 0) {
      Fence<HardEvent::MTE3_MTE2>();
      DataCopyExtParams cp{1, t.channels * 2, 0, 0, 0};
      DataCopyPadExtParams<bfloat16_t> pad{true, 0, 0, (bfloat16_t)0};
      DataCopyPad(xb, gateG[row * t.channels], cp, pad);
      Fence<HardEvent::MTE2_V>();
      Cast(q, xb, RoundMode::CAST_NONE, t.channels);
      Barrier();
      Muls(q, q, 1.0f / t.channels, t.channels);
      Barrier();
      Sigmoid(q, t.channels);
      Muls(q, q, 2.0f, t.channels);
      Barrier();
      Cast(xb, q, RoundMode::CAST_RINT, t.channels);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(injectG[row * t.channels], xb, cp);
    }
    Fence<HardEvent::MTE3_MTE2>();
  }
  TPipe pipe;
  bool buffered = false;
  TQue<TPosition::VECIN, 2> mixInput;
  TQue<TPosition::VECOUT, 2> mixOutput;
  TBuf<TPosition::VECCALC> b0, b1, f0, f1, f2, f3, scalar;
  GlobalTensor<bfloat16_t> xG, uG, gateG, yG, injectG;
  GlobalTensor<float> gG, invG;
  XTgrMixTilingData t;
};
}  // namespace
extern "C" __global__ __aicore__ void x_tgr_mix(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate, GM_ADDR y,
                                                GM_ADDR inv, GM_ADDR inject, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XTgrMixTilingData);
  GET_TILING_DATA_WITH_STRUCT(XTgrMixTilingData, tilingData, tiling);
  XTgrMixKernel op;
  op.Init(x, gamma, aux, gate, y, inv, inject, tilingData);
  op.Process();
}
