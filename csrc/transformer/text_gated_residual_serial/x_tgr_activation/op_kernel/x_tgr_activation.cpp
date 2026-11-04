// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_tgr_activation_tiling.h"
using namespace AscendC;
namespace {
class XTgrActivationKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate, GM_ADDR y, GM_ADDR inv,
                              GM_ADDR inject, const XTgrActivationTilingData& data) {
    t = data;
    xG.SetGlobalBuffer((__gm__ bfloat16_t*)x);
    gG.SetGlobalBuffer((__gm__ float*)gamma);
    uG.SetGlobalBuffer((__gm__ bfloat16_t*)aux);
    gateG.SetGlobalBuffer((__gm__ bfloat16_t*)gate);
    yG.SetGlobalBuffer((__gm__ bfloat16_t*)y);
    invG.SetGlobalBuffer((__gm__ float*)inv);
    injectG.SetGlobalBuffer((__gm__ bfloat16_t*)inject);
    const uint32_t count = t.workElements;
    pipe.InitBuffer(b0, count * 2);
    pipe.InitBuffer(b1, count * 2);
    pipe.InitBuffer(f0, count * 4);
    pipe.InitBuffer(f1, count * 4);
  }
  __aicore__ inline void Process() {
    for (uint32_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum()) {
      if (t.mode == 1)
        Activation(unit);
      else if (t.mode == 5 && t.tile > 1)
        BatchedPackedActivation(unit);
      else
        PackedActivation(unit);
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

  __aicore__ inline void Activation(uint32_t unit) {
    uint32_t off = unit * t.tile, n = t.rows * t.width - off;
    n = n < t.tile ? n : t.tile;
    auto xb = b0.Get<bfloat16_t>(), yb = b1.Get<bfloat16_t>();
    auto z = f0.Get<float>(), q = f1.Get<float>();
    DataCopy(xb, xG[off], n);
    Fence<HardEvent::MTE2_V>();
    Cast(z, xb, RoundMode::CAST_NONE, n);
    Barrier();
    Muls(z, z, 1.0f / t.channels, n);
    Barrier();
    Muls(q, z, 1.0f, n);
    Barrier();
    Sigmoid(q, n);
    Mul(z, z, q, n);
    Barrier();
    Cast(yb, z, RoundMode::CAST_RINT, n);
    Fence<HardEvent::V_MTE3>();
    DataCopy(yG[off], yb, n);
    Fence<HardEvent::MTE3_MTE2>();
  }

  __aicore__ inline void PackedActivation(uint32_t row) {
    // One aligned projection contains [down320, gate4, zero padding12].
    // The complete reduction remains in public mm. This V stage owns the
    // activation and gate outputs for its row and reads only live columns.
    auto xb = b0.Get<bfloat16_t>(), yb = b1.Get<bfloat16_t>();
    auto z = f0.Get<float>(), q = f1.Get<float>();
    DataCopy(xb, xG[row * t.width], 320);
    DataCopyExtParams cp{1, 8, 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> pad{true, 0, 0, (bfloat16_t)0};
    DataCopyPad(xb[320], xG[row * t.width + 320], cp, pad);
    Fence<HardEvent::MTE2_V>();
    Cast(z, xb, RoundMode::CAST_NONE, 320);
    Barrier();
    Muls(z, z, 1.0f / t.channels, 320);
    Barrier();
    Muls(q, z, 1.0f, 320);
    Barrier();
    Sigmoid(q, 320);
    Mul(z, z, q, 320);
    Barrier();
    Cast(yb, z, RoundMode::CAST_RINT, 320);
    Fence<HardEvent::V_MTE3>();
    DataCopy(yG[row * 320], yb, 320);
    Cast(q, xb[320], RoundMode::CAST_NONE, 4);
    Barrier();
    Muls(q, q, 1.0f / t.channels, 4);
    Barrier();
    Sigmoid(q, 4);
    Muls(q, q, 2.0f, 4);
    Barrier();
    Cast(xb, q, RoundMode::CAST_RINT, 4);
    // A padded internal row keeps independent cores' small gate stores
    // in separate 32-byte blocks. The Python API returns the live view.
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(injectG[row * (t.mode == 5 ? 16 : 4)], xb, cp);
    Fence<HardEvent::MTE3_MTE2>();
  }

  __aicore__ inline void BatchedPackedActivation(uint32_t unit) {
    const uint32_t row = unit * t.tile;
    const uint32_t rows = (t.rows - row) < t.tile ? (t.rows - row) : t.tile;
    const uint32_t all = rows * 336;
    auto xb = b0.Get<bfloat16_t>(), yb = b1.Get<bfloat16_t>();
    auto z = f0.Get<float>(), q = f1.Get<float>();
    // Include the16 padding/gate columns in the one vector sequence,
    // then store only the live320 down columns and padded gate16.
    const DataCopyParams input{static_cast<uint16_t>(rows), 21, static_cast<uint16_t>((t.width - 336) / 16), 0};
    DataCopy(xb, xG[row * t.width], input);
    Fence<HardEvent::MTE2_V>();
    Cast(z, xb, RoundMode::CAST_NONE, all);
    Barrier();
    Muls(z, z, 1.0f / t.channels, all);
    Barrier();
    Muls(q, z, 1.0f, all);
    Barrier();
    Sigmoid(q, all);
    Mul(z, z, q, all);
    Barrier();
    Cast(yb, z, RoundMode::CAST_RINT, all);
    Fence<HardEvent::V_MTE3>();
    const DataCopyParams down{static_cast<uint16_t>(rows), 20, 1, 0};
    DataCopy(yG[row * 320], yb, down);
    Muls(q, q, 2.0f, all);
    Barrier();
    Cast(xb, q, RoundMode::CAST_RINT, all);
    Fence<HardEvent::V_MTE3>();
    const DataCopyParams gate{static_cast<uint16_t>(rows), 1, 20, 0};
    DataCopy(injectG[row * 16], xb[320], gate);
    Fence<HardEvent::MTE3_MTE2>();
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, f0, f1, f2, f3, scalar;
  GlobalTensor<bfloat16_t> xG, uG, gateG, yG, injectG;
  GlobalTensor<float> gG, invG;
  XTgrActivationTilingData t;
};
}  // namespace
extern "C" __global__ __aicore__ void x_tgr_activation(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate, GM_ADDR y,
                                                       GM_ADDR inv, GM_ADDR inject, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XTgrActivationTilingData);
  GET_TILING_DATA_WITH_STRUCT(XTgrActivationTilingData, tilingData, tiling);
  XTgrActivationKernel op;
  op.Init(x, gamma, aux, gate, y, inv, inject, tilingData);
  op.Process();
}
