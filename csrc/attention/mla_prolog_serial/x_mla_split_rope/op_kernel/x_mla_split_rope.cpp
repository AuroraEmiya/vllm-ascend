// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_split_rope_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t QUERY_RANK = 1536;
constexpr uint32_t KV_RANK = 512;
constexpr uint32_t NOPE_DIM = 128;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t UP_HEAD_WIDTH = NOPE_DIM + ROPE_DIM;
constexpr uint32_t HEAD_TILE = 32;
constexpr uint32_t SMALL_HEAD_TILE = 16;
constexpr uint32_t WORK_WIDTH = HEAD_TILE * ROPE_DIM;
constexpr uint32_t GAMMA_OFFSET = ROPE_DIM;
static_assert(GAMMA_OFFSET + QUERY_RANK <= WORK_WIDTH, "cached gamma exceeds f1 buffer");

template <HardEvent EVENT>
__aicore__ inline void Sync() {
  SetFlag<EVENT>(EVENT_ID0);
  WaitFlag<EVENT>(EVENT_ID0);
}

class XMlaSplitRopeKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR aux, GM_ADDR gammaQ, GM_ADDR gammaKv, GM_ADDR sine, GM_ADDR cosine,
                              GM_ADDR index, GM_ADDR y, GM_ADDR queryRope, GM_ADDR cacheKv, GM_ADDR cacheKr,
                              const XMlaSplitRopeTilingData& tiling) {
    t = tiling;
    cachedGammaWidth = 0;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    auxGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(aux));
    gammaQGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gammaQ));
    gammaKvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gammaKv));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sine));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cosine));
    indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    qrGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(queryRope));
    kvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cacheKv));
    krGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cacheKr));
    pipe.InitBuffer(b0Buf, (t.headTile * 128) * 2);
    pipe.InitBuffer(b1Buf, (64) * 2);
    pipe.InitBuffer(b2Buf, 64 * 2);
    pipe.InitBuffer(b3Buf, (t.workElements) * 2);
    pipe.InitBuffer(f0Buf, t.floatElements * 4);
    pipe.InitBuffer(f1Buf, t.floatElements * 4);
    pipe.InitBuffer(f2Buf, t.floatElements * 4);
    pipe.InitBuffer(f3Buf, t.floatElements * 4);
    pipe.InitBuffer(f4Buf, t.floatElements * 4);
    pipe.InitBuffer(indexBuf, 32);
    b0 = b0Buf.Get<bfloat16_t>();
    b1 = b1Buf.Get<bfloat16_t>();
    b2 = b2Buf.Get<bfloat16_t>();
    b3 = b3Buf.Get<bfloat16_t>();
    f0 = f0Buf.Get<float>();
    f1 = f1Buf.Get<float>();
    f2 = f2Buf.Get<float>();
    f3 = f3Buf.Get<float>();
    f4 = f4Buf.Get<float>();
    idx = indexBuf.Get<int64_t>();
  }
  __aicore__ inline void Process() {
    for (uint32_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum()) {
      Sync<HardEvent::V_MTE2>();
      Sync<HardEvent::MTE3_MTE2>();
      SplitAndRope(unit);
    }
    Sync<HardEvent::MTE3_S>();
  }

 private:
  __aicore__ inline void BroadcastHalf(const LocalTensor<float>& source) {
    // One repeat contains two heads' 32-element halves. Subsequent vector
    // repeats reuse these 64 coefficients through src1RepStride=0.
    Adds(f0, source, 0.0f, ROPE_DIM / 2);
    Adds(f0[ROPE_DIM / 2], source, 0.0f, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void PackHalf(uint32_t offset, const LocalTensor<float>& source) {
    // The four coefficient pairs occupy disjoint f0 windows. Their
    // producer/consumer barrier is issued once after all eight Adds.
    Adds(f0[offset], source, 0.0f, ROPE_DIM / 2);
    Adds(f0[offset + ROPE_DIM / 2], source, 0.0f, ROPE_DIM / 2);
  }

  __aicore__ inline void RopePackedCoefficients(uint32_t halfCount, uint64_t mask, uint8_t repeats,
                                                const BinaryRepeatParams& broadcast) {
    // GatherMask has consumed the old f0 before this helper is called.
    // [0,64)/[64,128) hold cos halves; [128,192)/[192,256) sine halves.
    PackHalf(0, f2);
    PackHalf(ROPE_DIM, f2[ROPE_DIM / 2]);
    PackHalf(2 * ROPE_DIM, f1);
    PackHalf(3 * ROPE_DIM, f1[ROPE_DIM / 2]);
    PipeBarrier<PIPE_V>();
    Mul(f4, f3, f0, mask, repeats, broadcast);
    Mul(f4[halfCount], f3[halfCount], f0[ROPE_DIM], mask, repeats, broadcast);
    // Both cos operations must finish reading f3 before sine products
    // overwrite its two halves. Those writes have disjoint destinations.
    PipeBarrier<PIPE_V>();
    Mul(f3[halfCount], f3[halfCount], f0[2 * ROPE_DIM], mask, repeats, broadcast);
    Mul(f3, f3, f0[3 * ROPE_DIM], mask, repeats, broadcast);
    PipeBarrier<PIPE_V>();
    Sub(f4, f4, f3[halfCount], halfCount);
    Add(f4[halfCount], f4[halfCount], f3, halfCount);
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void SplitAndRope(uint32_t unit) {
    const uint32_t tileSize = t.headTile;
    const uint32_t tiles = (t.heads + tileSize - 1) / tileSize;
    const uint32_t row = unit / tiles;
    const uint32_t head = (unit % tiles) * tileSize;
    const uint32_t headCount = t.heads - head < tileSize ? t.heads - head : tileSize;
    // Preserve round6's retained DMA selection in the mode8 path.
    const bool overlapInput = t.mode == 6 || ((t.mode == 8 || t.mode == 10) && t.rows >= 32 && t.rows <= 64);
    const uint64_t source = static_cast<uint64_t>(row * t.heads + head) * UP_HEAD_WIDTH;
    DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, 0};
    DataCopyExtParams copyQ{static_cast<uint16_t>(headCount), NOPE_DIM * sizeof(bfloat16_t),
                            ROPE_DIM * sizeof(bfloat16_t), 0, 0};
    DataCopyPad(b0, xGm[source], copyQ, noPad);
    Sync<HardEvent::MTE2_MTE3>();
    DataCopyExtParams writeQ{static_cast<uint16_t>(headCount), NOPE_DIM * sizeof(bfloat16_t), 0,
                             static_cast<uint32_t>((t.rows - 1) * NOPE_DIM * sizeof(bfloat16_t)), 0};
    DataCopyPad(yGm[(head * t.rows + row) * NOPE_DIM], b0, writeQ);
    // Mode1 retains the original shared-b0 wait. Mode6 never overwrites
    // b0 here: Q can keep reading it while RoPE is loaded into b3.
    if (!overlapInput) Sync<HardEvent::MTE3_MTE2>();

    DataCopyExtParams copyRope{static_cast<uint16_t>(headCount), ROPE_DIM * sizeof(bfloat16_t),
                               NOPE_DIM * sizeof(bfloat16_t), 0, 0};
    if (overlapInput)
      DataCopyPad(b3, xGm[source + NOPE_DIM], copyRope, noPad);
    else
      DataCopyPad(b0, xGm[source + NOPE_DIM], copyRope, noPad);
    DataCopy(b1, sinGm[row * ROPE_DIM], ROPE_DIM);
    DataCopy(b2, cosGm[row * ROPE_DIM], ROPE_DIM);
    Sync<HardEvent::MTE2_V>();
    const uint32_t count = headCount * ROPE_DIM;
    const uint32_t halfCount = count / 2;
    if (overlapInput)
      Cast(f0, b3, RoundMode::CAST_NONE, count);
    else
      Cast(f0, b0, RoundMode::CAST_NONE, count);
    Cast(f1, b1, RoundMode::CAST_NONE, ROPE_DIM);
    Cast(f2, b2, RoundMode::CAST_NONE, ROPE_DIM);
    PipeBarrier<PIPE_V>();
    if (t.mode == 10) {
      // Keep each head's two halves in their original contiguous layout.
      // Repeat strides broadcast one coefficient row across all live heads.
      BinaryRepeatParams coefficients(1, 1, 1, 8, 8, 0);
      BinaryRepeatParams halves(1, 1, 1, 8, 8, 8);
      Mul(f4, f0, f2, static_cast<uint64_t>(ROPE_DIM), headCount, coefficients);
      Mul(f3, f0[ROPE_DIM / 2], f1, static_cast<uint64_t>(ROPE_DIM / 2), headCount, coefficients);
      Mul(f3[ROPE_DIM / 2], f0, f1[ROPE_DIM / 2], static_cast<uint64_t>(ROPE_DIM / 2), headCount, coefficients);
      PipeBarrier<PIPE_V>();
      Sub(f4, f4, f3, static_cast<uint64_t>(ROPE_DIM / 2), headCount, halves);
      Add(f4[ROPE_DIM / 2], f4[ROPE_DIM / 2], f3[ROPE_DIM / 2], static_cast<uint64_t>(ROPE_DIM / 2), headCount, halves);
      PipeBarrier<PIPE_V>();
      Cast(b3, f4, RoundMode::CAST_RINT, count);
      Sync<HardEvent::V_MTE3>();
      DataCopy(qrGm[static_cast<uint64_t>(row * t.heads + head) * ROPE_DIM], b3, count);
      return;
    }
    uint64_t reserved = 0;
    GatherMask(f3, f0, static_cast<uint8_t>(1), true, count, GatherMaskParams(1, 1, 8, 0), reserved);
    GatherMask(f3[halfCount], f0, static_cast<uint8_t>(2), true, count, GatherMaskParams(1, 1, 8, 0), reserved);
    PipeBarrier<PIPE_V>();

    const uint64_t mask = headCount == 1 ? ROPE_DIM / 2 : ROPE_DIM;
    const uint8_t repeats = headCount <= 2 ? 1 : headCount / 2;
    BinaryRepeatParams broadcast(1, 1, 1, 8, 8, 0);
    if (t.mode == 8 || t.mode == 10) {
      RopePackedCoefficients(halfCount, mask, repeats, broadcast);
    } else {
      BroadcastHalf(f2);
      Mul(f4, f3, f0, mask, repeats, broadcast);
      PipeBarrier<PIPE_V>();
      BroadcastHalf(f2[ROPE_DIM / 2]);
      Mul(f4[halfCount], f3[halfCount], f0, mask, repeats, broadcast);
      PipeBarrier<PIPE_V>();
      BroadcastHalf(f1);
      Mul(f3[halfCount], f3[halfCount], f0, mask, repeats, broadcast);
      PipeBarrier<PIPE_V>();
      Sub(f4, f4, f3[halfCount], halfCount);
      PipeBarrier<PIPE_V>();
      BroadcastHalf(f1[ROPE_DIM / 2]);
      Mul(f3, f3, f0, mask, repeats, broadcast);
      PipeBarrier<PIPE_V>();
      Add(f4[halfCount], f4[halfCount], f3, halfCount);
      PipeBarrier<PIPE_V>();
    }
    Cast(b3, f4, RoundMode::CAST_RINT, count);
    Sync<HardEvent::V_MTE3>();
    DataCopyExtParams writeRope{static_cast<uint16_t>(headCount), (ROPE_DIM / 2) * sizeof(bfloat16_t), 0,
                                (ROPE_DIM / 2) * sizeof(bfloat16_t), 0};
    const uint64_t ropeOffset = static_cast<uint64_t>(row * t.heads + head) * ROPE_DIM;
    DataCopyPad(qrGm[ropeOffset], b3, writeRope);
    DataCopyPad(qrGm[ropeOffset + ROPE_DIM / 2], b3[halfCount], writeRope);
  }
  uint32_t cachedGammaWidth;
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0Buf, b1Buf, b2Buf, b3Buf, f0Buf, f1Buf, f2Buf, f3Buf, f4Buf, indexBuf;
  LocalTensor<bfloat16_t> b0, b1, b2, b3;
  LocalTensor<float> f0, f1, f2, f3, f4;
  LocalTensor<int64_t> idx;
  GlobalTensor<bfloat16_t> xGm, auxGm, gammaQGm, gammaKvGm, sinGm, cosGm, yGm, qrGm, kvGm, krGm;
  GlobalTensor<int64_t> indexGm;
  XMlaSplitRopeTilingData t;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_split_rope(GM_ADDR x, GM_ADDR aux, GM_ADDR gamma_q, GM_ADDR gamma_kv,
                                                       GM_ADDR sin, GM_ADDR cos, GM_ADDR index, GM_ADDR y,
                                                       GM_ADDR query_rope, GM_ADDR kv_cache, GM_ADDR kr_cache,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaSplitRopeTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaSplitRopeTilingData, tilingData, tiling);
  XMlaSplitRopeKernel op;
  op.Init(x, aux, gamma_q, gamma_kv, sin, cos, index, y, query_rope, kv_cache, kr_cache, tilingData);
  op.Process();
}
