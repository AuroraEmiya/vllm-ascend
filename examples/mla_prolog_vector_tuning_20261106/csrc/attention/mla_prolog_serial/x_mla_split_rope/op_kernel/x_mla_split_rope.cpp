// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_split_rope_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t NOPE_DIM = 128;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t UP_HEAD_WIDTH = NOPE_DIM + ROPE_DIM;

template <HardEvent EVENT>
__aicore__ inline void Sync() {
  SetFlag<EVENT>(EVENT_ID0);
  WaitFlag<EVENT>(EVENT_ID0);
}

class XMlaSplitRopeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR sine, GM_ADDR cosine, GM_ADDR headQ, GM_ADDR queryRope,
                              const XMlaSplitRopeTilingData& tiling) {
    tilingData = tiling;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sine));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cosine));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(headQ));
    qrGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(queryRope));
    constexpr uint32_t copies = 2;
    pipe.InitBuffer(headInputBf16Buffer, tilingData.headTile * tilingData.inputHeadWidth * 2 * copies);
    pipe.InitBuffer(sinBf16Buffer, 64 * 2 * copies);
    pipe.InitBuffer(cosBf16Buffer, 64 * 2 * copies);
    pipe.InitBuffer(ropeOutputBf16Buffer, tilingData.workElements * 2 * copies);
    pipe.InitBuffer(valuesOrSinProductsBuffer, tilingData.floatElements * 4);
    pipe.InitBuffer(sinFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(cosFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(evenOddFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(ropeResultFp32Buffer, tilingData.floatElements * 4);
    headInputBf16 = headInputBf16Buffer.Get<bfloat16_t>();
    sinBf16 = sinBf16Buffer.Get<bfloat16_t>();
    cosBf16 = cosBf16Buffer.Get<bfloat16_t>();
    ropeOutputBf16 = ropeOutputBf16Buffer.Get<bfloat16_t>();
    valuesOrSinProducts = valuesOrSinProductsBuffer.Get<float>();
    sinFp32 = sinFp32Buffer.Get<float>();
    cosFp32 = cosFp32Buffer.Get<float>();
    evenOddFp32 = evenOddFp32Buffer.Get<float>();
    ropeResultFp32 = ropeResultFp32Buffer.Get<float>();
  }
  __aicore__ inline void Process() {
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t iteration = 0;
    for (uint32_t unit = GetBlockIdx(); unit < tilingData.units; unit += GetBlockNum(), ++iteration) {
      const uint32_t slot = iteration & 1u;
      const int32_t eventId = slot ? EVENT_ID1 : EVENT_ID0;
      WaitFlag<HardEvent::MTE3_MTE2>(eventId);
      headInputBf16 = headInputBf16Buffer.Get<bfloat16_t>()[slot * tilingData.headTile * tilingData.inputHeadWidth];
      sinBf16 = sinBf16Buffer.Get<bfloat16_t>()[slot * 64];
      cosBf16 = cosBf16Buffer.Get<bfloat16_t>()[slot * 64];
      ropeOutputBf16 = ropeOutputBf16Buffer.Get<bfloat16_t>()[slot * tilingData.workElements];
      SplitAndRope(unit);
      // Covers both the NoPE DMA and the later QR DMA from this slot.
      SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  }

 private:
  __aicore__ inline void SplitAndRope(uint32_t unit) {
    const uint32_t tileSize = tilingData.headTile;
    const uint32_t tiles = (tilingData.heads + tileSize - 1) / tileSize;
    const uint32_t row = unit / tiles;
    const uint32_t head = (unit % tiles) * tileSize;
    const uint32_t headCount = tilingData.heads - head < tileSize ? tilingData.heads - head : tileSize;
    const bool contiguousHeadInput = tilingData.inputHeadWidth == UP_HEAD_WIDTH;
    const uint64_t source = static_cast<uint64_t>(row * tilingData.heads + head) * UP_HEAD_WIDTH;
    DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, 0};
    DataCopyExtParams copyQ{static_cast<uint16_t>(headCount), NOPE_DIM * sizeof(bfloat16_t),
                            ROPE_DIM * sizeof(bfloat16_t), 0, 0};
    if (contiguousHeadInput)
      DataCopy(headInputBf16, xGm[source], headCount * UP_HEAD_WIDTH);
    else
      DataCopyPad(headInputBf16, xGm[source], copyQ, noPad);
    Sync<HardEvent::MTE2_MTE3>();
    // UB source stride is in 32-byte blocks; GM destination stride is bytes.
    DataCopyExtParams writeQ{static_cast<uint16_t>(headCount), NOPE_DIM * sizeof(bfloat16_t),
                             static_cast<uint32_t>(contiguousHeadInput ? ROPE_DIM * sizeof(bfloat16_t) / 32 : 0),
                             static_cast<uint32_t>((tilingData.rows - 1) * NOPE_DIM * sizeof(bfloat16_t)), 0};
    DataCopyPad(yGm[(head * tilingData.rows + row) * NOPE_DIM], headInputBf16, writeQ);

    DataCopyExtParams copyRope{static_cast<uint16_t>(headCount), ROPE_DIM * sizeof(bfloat16_t),
                               NOPE_DIM * sizeof(bfloat16_t), 0, 0};
    // Full-head input already includes RoPE; the small-input path loads it separately.
    if (!contiguousHeadInput) DataCopyPad(ropeOutputBf16, xGm[source + NOPE_DIM], copyRope, noPad);
    DataCopy(sinBf16, sinGm[row * ROPE_DIM], ROPE_DIM);
    DataCopy(cosBf16, cosGm[row * ROPE_DIM], ROPE_DIM);
    Sync<HardEvent::MTE2_V>();
    const uint32_t count = headCount * ROPE_DIM;
    const uint32_t halfCount = count / 2;
    if (contiguousHeadInput)
      Cast(valuesOrSinProducts, headInputBf16[NOPE_DIM], RoundMode::CAST_NONE, static_cast<uint64_t>(ROPE_DIM),
           static_cast<uint8_t>(headCount), UnaryRepeatParams(1, 1, 8, 12));
    else
      Cast(valuesOrSinProducts, ropeOutputBf16, RoundMode::CAST_NONE, count);
    Cast(sinFp32, sinBf16, RoundMode::CAST_NONE, ROPE_DIM);
    Cast(cosFp32, cosBf16, RoundMode::CAST_NONE, ROPE_DIM);
    PipeBarrier<PIPE_V>();
    uint64_t reserved = 0;
    GatherMask(evenOddFp32, valuesOrSinProducts, static_cast<uint8_t>(1), true, count, GatherMaskParams(1, 1, 8, 0), reserved);
    GatherMask(evenOddFp32[halfCount], valuesOrSinProducts, static_cast<uint8_t>(2), true, count, GatherMaskParams(1, 1, 8, 0), reserved);
    PipeBarrier<PIPE_V>();

    // Gather patterns1/2 select even/odd source elements, not consecutive
    // halves. Retain that required permutation; directly form head-major
    // [even-result32, odd-result32] output with strided V writes.
    const uint8_t repeats = static_cast<uint8_t>(headCount);
    BinaryRepeatParams coefficients(1, 1, 1, 8, 4, 0);
    BinaryRepeatParams halves(1, 1, 1, 8, 8, 8);
    Mul(ropeResultFp32, evenOddFp32, cosFp32, static_cast<uint64_t>(ROPE_DIM / 2), repeats, coefficients);
    Mul(ropeResultFp32[ROPE_DIM / 2], evenOddFp32[halfCount], cosFp32[ROPE_DIM / 2], static_cast<uint64_t>(ROPE_DIM / 2), repeats,
        coefficients);
    // Gather has finished consuming valuesOrSinProducts, so it now holds sine products.
    Mul(valuesOrSinProducts, evenOddFp32[halfCount], sinFp32, static_cast<uint64_t>(ROPE_DIM / 2), repeats, coefficients);
    Mul(valuesOrSinProducts[ROPE_DIM / 2], evenOddFp32, sinFp32[ROPE_DIM / 2], static_cast<uint64_t>(ROPE_DIM / 2), repeats, coefficients);
    PipeBarrier<PIPE_V>();
    Sub(ropeResultFp32, ropeResultFp32, valuesOrSinProducts, static_cast<uint64_t>(ROPE_DIM / 2), repeats, halves);
    Add(ropeResultFp32[ROPE_DIM / 2], ropeResultFp32[ROPE_DIM / 2], valuesOrSinProducts[ROPE_DIM / 2], static_cast<uint64_t>(ROPE_DIM / 2), repeats, halves);
    PipeBarrier<PIPE_V>();
    Cast(ropeOutputBf16, ropeResultFp32, RoundMode::CAST_RINT, count);
    Sync<HardEvent::V_MTE3>();
    DataCopy(qrGm[static_cast<uint64_t>(row * tilingData.heads + head) * ROPE_DIM], ropeOutputBf16, count);
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> headInputBf16Buffer, sinBf16Buffer, cosBf16Buffer, ropeOutputBf16Buffer;
  TBuf<TPosition::VECCALC> valuesOrSinProductsBuffer, sinFp32Buffer, cosFp32Buffer;
  TBuf<TPosition::VECCALC> evenOddFp32Buffer, ropeResultFp32Buffer;
  LocalTensor<bfloat16_t> headInputBf16, sinBf16, cosBf16, ropeOutputBf16;
  LocalTensor<float> valuesOrSinProducts, sinFp32, cosFp32, evenOddFp32, ropeResultFp32;
  GlobalTensor<bfloat16_t> xGm, sinGm, cosGm, yGm, qrGm;
  XMlaSplitRopeTilingData tilingData;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_split_rope(
    GM_ADDR x, GM_ADDR sin, GM_ADDR cos, GM_ADDR head_q, GM_ADDR query_rope, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaSplitRopeTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaSplitRopeTilingData, tilingData, tiling);
  XMlaSplitRopeKernel op;
  op.Init(x, sin, cos, head_q, query_rope, tilingData);
  op.Process();
}
