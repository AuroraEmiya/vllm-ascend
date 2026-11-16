// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_norm_cache_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t QUERY_RANK = 1536;
constexpr uint32_t KV_RANK = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t GAMMA_OFFSET = ROPE_DIM;

template <HardEvent EVENT>
__aicore__ inline void Sync() {
  SetFlag<EVENT>(EVENT_ID0);
  WaitFlag<EVENT>(EVENT_ID0);
}

class XMlaNormCacheKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gammaQ, GM_ADDR gammaKv, GM_ADDR sine, GM_ADDR cosine,
                              GM_ADDR index, GM_ADDR y, GM_ADDR cacheKv, GM_ADDR cacheKr,
                              const XMlaNormCacheTilingData& tiling) {
    tilingData = tiling;
    cachedGammaWidth = 0;
    cachedGammaKinds = 0;
    outputSlot = 0;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    gammaQGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gammaQ));
    gammaKvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gammaKv));
    sinGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(sine));
    cosGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cosine));
    indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(index));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    kvGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cacheKv));
    krGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cacheKr));
    pipe.InitBuffer(inputBf16Buffer, (1536) * 2);
    pipe.InitBuffer(gammaOrSinBf16Buffer, (1536) * 2);
    pipe.InitBuffer(cosBf16Buffer, 64 * 2);
    pipe.InitBuffer(outputBf16Buffer, QUERY_RANK * sizeof(bfloat16_t) * 2);
    pipe.InitBuffer(valuesFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(gammaOrSinFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(reduceOrCosFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(reduceOrRotatedFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(productsFp32Buffer, tilingData.floatElements * 4);
    pipe.InitBuffer(indexBuf, 32);
    inputBf16 = inputBf16Buffer.Get<bfloat16_t>();
    gammaOrSinBf16 = gammaOrSinBf16Buffer.Get<bfloat16_t>();
    cosBf16 = cosBf16Buffer.Get<bfloat16_t>();
    outputBf16 = outputBf16Buffer.Get<bfloat16_t>();
    valuesFp32 = valuesFp32Buffer.Get<float>();
    gammaOrSinFp32 = gammaOrSinFp32Buffer.Get<float>();
    reduceOrCosFp32 = reduceOrCosFp32Buffer.Get<float>();
    reduceOrRotatedFp32 = reduceOrRotatedFp32Buffer.Get<float>();
    productsFp32 = productsFp32Buffer.Get<float>();
    idx = indexBuf.Get<int64_t>();
  }
  __aicore__ inline void Process() {
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    for (uint32_t unit = GetBlockIdx(); unit < tilingData.units; unit += GetBlockNum()) {
      // One or two adjacent Q/KV branches per work unit; the math is shared.
      for (uint32_t branch = 0; branch < tilingData.branchesPerUnit; ++branch) {
        Sync<HardEvent::V_MTE2>();
        NormAndCache(unit * tilingData.branchesPerUnit + branch);
      }
    }
    // Drain both output slots before returning to the parent SuperKernel.
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }

 private:
  __aicore__ inline void AcquireOutput() {
    const int32_t eventId = outputSlot ? EVENT_ID1 : EVENT_ID0;
    WaitFlag<HardEvent::MTE3_V>(eventId);
    outputBf16 = outputBf16Buffer.Get<bfloat16_t>()[outputSlot * QUERY_RANK];
  }

  __aicore__ inline void ReleaseOutput() {
    const int32_t eventId = outputSlot ? EVENT_ID1 : EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(eventId);
    outputSlot ^= 1u;
  }

  __aicore__ inline void Normalize(GlobalTensor<bfloat16_t> input, uint64_t offset, GlobalTensor<bfloat16_t> gamma,
                                   uint32_t width, float eps, GlobalTensor<bfloat16_t> output, uint64_t outputOffset) {
    const uint32_t gammaOffset = GAMMA_OFFSET + (tilingData.branchesPerUnit == 2 && width == KV_RANK ? QUERY_RANK : 0);
    const uint32_t gammaKind = width == QUERY_RANK ? 1u : 2u;
    DataCopy(inputBf16, input[offset], width);
    // Small inputs reuse one gamma window; paired-token work keeps both.
    const bool loadGamma = tilingData.branchesPerUnit == 2
                               ? (cachedGammaKinds & gammaKind) == 0
                               : cachedGammaWidth != width;
    if (loadGamma) DataCopy(gammaOrSinBf16, gamma, width);
    Sync<HardEvent::MTE2_V>();
    Cast(valuesFp32, inputBf16, RoundMode::CAST_NONE, width);
    if (loadGamma) Cast(gammaOrSinFp32[gammaOffset], gammaOrSinBf16, RoundMode::CAST_NONE, width);
    cachedGammaWidth = width;
    cachedGammaKinds |= gammaKind;
    PipeBarrier<PIPE_V>();
    Mul(productsFp32, valuesFp32, valuesFp32, width);
    Duplicate(reduceOrCosFp32, 0.0f, 8);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceOrCosFp32, productsFp32, reduceOrRotatedFp32, width);
    PipeBarrier<PIPE_V>();
    const float reciprocalWidth = width == QUERY_RANK ? (1.0f / 1536.0f) : (1.0f / 512.0f);
    Muls(reduceOrCosFp32, reduceOrCosFp32, reciprocalWidth, 8);
    PipeBarrier<PIPE_V>();
    Adds(reduceOrCosFp32, reduceOrCosFp32, eps, 8);
    PipeBarrier<PIPE_V>();
    Rsqrt(reduceOrRotatedFp32, reduceOrCosFp32, 8);
    // Broadcast the exact inverse from lane 0 without a vector-to-scalar read.
    PipeBarrier<PIPE_V>();
    Brcb(productsFp32, reduceOrRotatedFp32, 1, {1, 8});
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams inverseBroadcast(1, 1, 0, 8, 8, 0);
    Mul(valuesFp32, valuesFp32, productsFp32, static_cast<uint64_t>(64), static_cast<uint8_t>(width / 64), inverseBroadcast);
    PipeBarrier<PIPE_V>();
    Mul(valuesFp32, valuesFp32, gammaOrSinFp32[gammaOffset], width);
    PipeBarrier<PIPE_V>();
    // Wait only when reusing the BF16 output slot consumed by MTE3.
    AcquireOutput();
    Cast(outputBf16, valuesFp32, RoundMode::CAST_RINT, width);
    Sync<HardEvent::V_MTE3>();
    DataCopy(output[outputOffset], outputBf16, width);
    ReleaseOutput();
  }

  __aicore__ inline void Rope(GlobalTensor<bfloat16_t> input, uint64_t offset, uint32_t row,
                              GlobalTensor<bfloat16_t> output, uint64_t outputOffset) {
    Sync<HardEvent::V_MTE2>();
    DataCopy(inputBf16, input[offset], ROPE_DIM);
    DataCopy(gammaOrSinBf16, sinGm[row * ROPE_DIM], ROPE_DIM);
    DataCopy(cosBf16, cosGm[row * ROPE_DIM], ROPE_DIM);
    Sync<HardEvent::MTE2_V>();
    Cast(valuesFp32, inputBf16, RoundMode::CAST_NONE, ROPE_DIM);
    // Cached gamma starts at GAMMA_OFFSET; RoPE only overwrites [0,64).
    Cast(gammaOrSinFp32, gammaOrSinBf16, RoundMode::CAST_NONE, ROPE_DIM);
    Cast(reduceOrCosFp32, cosBf16, RoundMode::CAST_NONE, ROPE_DIM);
    PipeBarrier<PIPE_V>();
    // Public RoPE convention verified by a structured black-box ABI probe:
    // even elements, then odd elements; rotate the two resulting halves.
    uint64_t reserved = 0;
    GatherMask(reduceOrRotatedFp32, valuesFp32, static_cast<uint8_t>(1), true, ROPE_DIM, GatherMaskParams(1, 1, 8, 0), reserved);
    GatherMask(reduceOrRotatedFp32[ROPE_DIM / 2], valuesFp32, static_cast<uint8_t>(2), true, ROPE_DIM, GatherMaskParams(1, 1, 8, 0), reserved);
    PipeBarrier<PIPE_V>();
    Mul(productsFp32, reduceOrRotatedFp32, reduceOrCosFp32, ROPE_DIM);
    Muls(valuesFp32, reduceOrRotatedFp32[ROPE_DIM / 2], -1.0f, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Mul(valuesFp32, valuesFp32, gammaOrSinFp32, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Add(productsFp32, productsFp32, valuesFp32, ROPE_DIM / 2);
    Mul(valuesFp32, reduceOrRotatedFp32, gammaOrSinFp32[ROPE_DIM / 2], ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    Add(productsFp32[ROPE_DIM / 2], productsFp32[ROPE_DIM / 2], valuesFp32, ROPE_DIM / 2);
    PipeBarrier<PIPE_V>();
    AcquireOutput();
    Cast(outputBf16, productsFp32, RoundMode::CAST_RINT, ROPE_DIM);
    Sync<HardEvent::V_MTE3>();
    DataCopy(output[outputOffset], outputBf16, ROPE_DIM);
    ReleaseOutput();
  }

  __aicore__ inline void NormAndCache(uint32_t unit) {
    const uint32_t row = unit / 2;
    if ((unit & 1u) == 0) {
      const uint32_t inputWidth = QUERY_RANK + KV_RANK + ROPE_DIM;
      Normalize(xGm, row * inputWidth, gammaQGm, QUERY_RANK, tilingData.epsQ, yGm, row * QUERY_RANK);
      return;
    }
    DataCopyExtParams params{1, sizeof(int64_t), 0, 0, 0};
    DataCopyPadExtParams<int64_t> pad{false, 0, 0, 0};
    DataCopyPad(idx, indexGm[row], params, pad);
    Sync<HardEvent::MTE2_S>();
    const int64_t slot = idx.GetValue(0);
    // Invalid slots never produce out-of-bounds device accesses. The public
    // wrapper's setup checker rejects invalid/duplicate host metadata.
    if (slot < 0 || slot >= tilingData.slots) return;
    const uint64_t offset = row * (QUERY_RANK + KV_RANK + ROPE_DIM) + QUERY_RANK;
    Normalize(xGm, offset, gammaKvGm, KV_RANK, tilingData.epsKv, kvGm, static_cast<uint64_t>(slot) * KV_RANK);
    Rope(xGm, offset + KV_RANK, row, krGm, static_cast<uint64_t>(slot) * ROPE_DIM);
  }
  uint32_t cachedGammaWidth, cachedGammaKinds, outputSlot;
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBf16Buffer, gammaOrSinBf16Buffer, cosBf16Buffer, outputBf16Buffer, valuesFp32Buffer, gammaOrSinFp32Buffer, reduceOrCosFp32Buffer, reduceOrRotatedFp32Buffer, productsFp32Buffer, indexBuf;
  LocalTensor<bfloat16_t> inputBf16, gammaOrSinBf16, cosBf16, outputBf16;
  LocalTensor<float> valuesFp32, gammaOrSinFp32, reduceOrCosFp32, reduceOrRotatedFp32, productsFp32;
  LocalTensor<int64_t> idx;
  GlobalTensor<bfloat16_t> xGm, gammaQGm, gammaKvGm, sinGm, cosGm, yGm, kvGm, krGm;
  GlobalTensor<int64_t> indexGm;
  XMlaNormCacheTilingData tilingData;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_norm_cache(
    GM_ADDR x, GM_ADDR gamma_q, GM_ADDR gamma_kv, GM_ADDR sin, GM_ADDR cos, GM_ADDR index,
    GM_ADDR normalized_q, GM_ADDR kv_cache, GM_ADDR kr_cache, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaNormCacheTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaNormCacheTilingData, tilingData, tiling);
  XMlaNormCacheKernel op;
  op.Init(x, gamma_q, gamma_kv, sin, cos, index, normalized_q, kv_cache, kr_cache, tilingData);
  op.Process();
}
