// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_norm_cache_tiling.h"
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

class XMlaNormCacheKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR aux, GM_ADDR gammaQ, GM_ADDR gammaKv, GM_ADDR sine, GM_ADDR cosine,
                              GM_ADDR index, GM_ADDR y, GM_ADDR queryRope, GM_ADDR cacheKv, GM_ADDR cacheKr,
                              const XMlaNormCacheTilingData& tiling) {
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
    pipe.InitBuffer(b0Buf, (1536) * 2);
    pipe.InitBuffer(b1Buf, (1536) * 2);
    pipe.InitBuffer(b2Buf, 64 * 2);
    pipe.InitBuffer(b3Buf, (1536) * 2);
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
      if (t.mode != 7) Sync<HardEvent::MTE3_MTE2>();
      NormAndCache(unit);
    }
    Sync<HardEvent::MTE3_S>();
  }

 private:
  __aicore__ inline void Normalize(GlobalTensor<bfloat16_t> input, uint64_t offset, GlobalTensor<bfloat16_t> gamma,
                                   uint32_t width, float eps, GlobalTensor<bfloat16_t> output, uint64_t outputOffset) {
    DataCopy(b0, input[offset], width);
    if (t.mode == 5 || t.mode == 7) {
      // The width tag also handles a platform with odd blockDim where
      // a core alternates query/KV units. Init invalidates it per launch.
      const bool loadGamma = cachedGammaWidth != width;
      if (loadGamma) DataCopy(b1, gamma, width);
      Sync<HardEvent::MTE2_V>();
      Cast(f0, b0, RoundMode::CAST_NONE, width);
      if (loadGamma) Cast(f1[GAMMA_OFFSET], b1, RoundMode::CAST_NONE, width);
      cachedGammaWidth = width;
    } else {
      // Preserve the mode4 control's load/cast algorithm.
      DataCopy(b1, gamma, width);
      Sync<HardEvent::MTE2_V>();
      Cast(f0, b0, RoundMode::CAST_NONE, width);
      Cast(f1, b1, RoundMode::CAST_NONE, width);
    }
    PipeBarrier<PIPE_V>();
    Mul(f4, f0, f0, width);
    Duplicate(f2, 0.0f, 8);
    PipeBarrier<PIPE_V>();
    ReduceSum(f2, f4, f3, width);
    PipeBarrier<PIPE_V>();
    const float reciprocalWidth = width == QUERY_RANK ? (1.0f / 1536.0f) : (1.0f / 512.0f);
    Muls(f2, f2, reciprocalWidth, 8);
    PipeBarrier<PIPE_V>();
    Adds(f2, f2, eps, 8);
    PipeBarrier<PIPE_V>();
    Rsqrt(f3, f2, 8);
    if (t.mode >= 4) {
      // Broadcast only the exact FP32 inverse in lane0. The first Brcb
      // output block repeats it eight times; Mul reuses this 32-byte
      // block for every block/repeat without a vector-to-scalar read.
      PipeBarrier<PIPE_V>();
      Brcb(f4, f3, 1, {1, 8});
      PipeBarrier<PIPE_V>();
      const BinaryRepeatParams inverseBroadcast(1, 1, 0, 8, 8, 0);
      Mul(f0, f0, f4, static_cast<uint64_t>(64), static_cast<uint8_t>(width / 64), inverseBroadcast);
    } else {
      // Mode3 keeps the original control algorithm in this same binary.
      Sync<HardEvent::V_S>();
      const float inverse = f3.GetValue(0);
      Muls(f0, f0, inverse, width);
    }
    PipeBarrier<PIPE_V>();
    if (t.mode == 5 || t.mode == 7)
      Mul(f0, f0, f1[GAMMA_OFFSET], width);
    else
      Mul(f0, f0, f1, width);
    PipeBarrier<PIPE_V>();
    // Mode7 may still have a previous norm/RoPE DMA reading b3.
    // Input b0 and FP32 arithmetic can proceed before this exact reuse.
    if (t.mode == 7) Sync<HardEvent::MTE3_V>();
    Cast(b3, f0, RoundMode::CAST_RINT, width);
    Sync<HardEvent::V_MTE3>();
    DataCopy(output[outputOffset], b3, width);
  }

  __aicore__ inline void Rope(GlobalTensor<bfloat16_t> input, uint64_t offset, uint32_t row,
                              GlobalTensor<bfloat16_t> output, uint64_t outputOffset) {
    Sync<HardEvent::V_MTE2>();
    if (t.mode != 7) Sync<HardEvent::MTE3_MTE2>();
    DataCopy(b0, input[offset], ROPE_DIM);
    DataCopy(b1, sinGm[row * ROPE_DIM], ROPE_DIM);
    DataCopy(b2, cosGm[row * ROPE_DIM], ROPE_DIM);
    Sync<HardEvent::MTE2_V>();
    Cast(f0, b0, RoundMode::CAST_NONE, ROPE_DIM);
    // Cached gamma starts at GAMMA_OFFSET; RoPE only overwrites [0,64).
    Cast(f1, b1, RoundMode::CAST_NONE, ROPE_DIM);
    Cast(f2, b2, RoundMode::CAST_NONE, ROPE_DIM);
    PipeBarrier<PIPE_V>();
    // Public RoPE convention verified by a structured black-box ABI probe:
    // even elements, then odd elements; rotate the two resulting halves.
    uint64_t reserved = 0;
    GatherMask(f3, f0, static_cast<uint8_t>(1), true, ROPE_DIM, GatherMaskParams(1, 1, 8, 0), reserved);
    GatherMask(f3[ROPE_DIM / 2], f0, static_cast<uint8_t>(2), true, ROPE_DIM, GatherMaskParams(1, 1, 8, 0), reserved);
    PipeBarrier<PIPE_V>();
    Mul(f4, f3, f2, ROPE_DIM);
    Muls(f0, f3[ROPE_DIM / 2], -1.0f, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Mul(f0, f0, f1, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Add(f4, f4, f0, ROPE_DIM / 2);
    Mul(f0, f3, f1[ROPE_DIM / 2], ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Add(f4[ROPE_DIM / 2], f4[ROPE_DIM / 2], f0, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    // Protect the KV output still being read from this same b3.
    if (t.mode == 7) Sync<HardEvent::MTE3_V>();
    Cast(b3, f4, RoundMode::CAST_RINT, ROPE_DIM);
    Sync<HardEvent::V_MTE3>();
    DataCopy(output[outputOffset], b3, ROPE_DIM);
  }

  __aicore__ inline void NormAndCache(uint32_t unit) {
    const uint32_t row = unit / 2;
    if ((unit & 1u) == 0) {
      const uint32_t inputWidth = t.mode >= 3 ? QUERY_RANK + KV_RANK + ROPE_DIM : QUERY_RANK;
      Normalize(xGm, row * inputWidth, gammaQGm, QUERY_RANK, t.epsQ, yGm, row * QUERY_RANK);
      return;
    }
    DataCopyExtParams params{1, sizeof(int64_t), 0, 0, 0};
    DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
    DataCopyPad(idx, indexGm[row], params, pad);
    Sync<HardEvent::MTE2_S>();
    const int64_t slot = idx.GetValue(0);
    // Invalid slots never produce out-of-bounds device accesses. The public
    // wrapper's setup checker rejects invalid/duplicate host metadata.
    if (slot < 0 || slot >= t.slots) return;
    if (t.mode >= 3) {
      const uint64_t offset = row * (QUERY_RANK + KV_RANK + ROPE_DIM) + QUERY_RANK;
      Normalize(xGm, offset, gammaKvGm, KV_RANK, t.epsKv, kvGm, static_cast<uint64_t>(slot) * KV_RANK);
      Rope(xGm, offset + KV_RANK, row, krGm, static_cast<uint64_t>(slot) * ROPE_DIM);
    } else {
      Normalize(auxGm, row * (KV_RANK + ROPE_DIM), gammaKvGm, KV_RANK, t.epsKv, kvGm,
                static_cast<uint64_t>(slot) * KV_RANK);
      Rope(auxGm, row * (KV_RANK + ROPE_DIM) + KV_RANK, row, krGm, static_cast<uint64_t>(slot) * ROPE_DIM);
    }
  }
  uint32_t cachedGammaWidth;
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0Buf, b1Buf, b2Buf, b3Buf, f0Buf, f1Buf, f2Buf, f3Buf, f4Buf, indexBuf;
  LocalTensor<bfloat16_t> b0, b1, b2, b3;
  LocalTensor<float> f0, f1, f2, f3, f4;
  LocalTensor<int64_t> idx;
  GlobalTensor<bfloat16_t> xGm, auxGm, gammaQGm, gammaKvGm, sinGm, cosGm, yGm, qrGm, kvGm, krGm;
  GlobalTensor<int64_t> indexGm;
  XMlaNormCacheTilingData t;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_norm_cache(GM_ADDR x, GM_ADDR aux, GM_ADDR gamma_q, GM_ADDR gamma_kv,
                                                       GM_ADDR sin, GM_ADDR cos, GM_ADDR index, GM_ADDR y,
                                                       GM_ADDR query_rope, GM_ADDR kv_cache, GM_ADDR kr_cache,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaNormCacheTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaNormCacheTilingData, tilingData, tiling);
  XMlaNormCacheKernel op;
  op.Init(x, aux, gamma_q, gamma_kv, sin, cos, index, y, query_rope, kv_cache, kr_cache, tilingData);
  op.Process();
}
