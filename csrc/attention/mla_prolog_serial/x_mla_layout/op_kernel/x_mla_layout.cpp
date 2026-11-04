// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_layout_tiling.h"
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

class XMlaLayoutKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR aux, GM_ADDR gammaQ, GM_ADDR gammaKv, GM_ADDR sine, GM_ADDR cosine,
                              GM_ADDR index, GM_ADDR y, GM_ADDR queryRope, GM_ADDR cacheKv, GM_ADDR cacheKr,
                              const XMlaLayoutTilingData& tiling) {
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
    pipe.InitBuffer(b0Buf, t.workElements * 2);
    b0 = b0Buf.Get<bfloat16_t>();
  }
  __aicore__ inline void Process() {
    for (uint32_t unit = GetBlockIdx(); unit < t.units; unit += GetBlockNum()) {
      Sync<HardEvent::V_MTE2>();
      Sync<HardEvent::MTE3_MTE2>();
      RestoreLayout(unit);
    }
    Sync<HardEvent::MTE3_S>();
  }

 private:
  __aicore__ inline void RestoreLayout(uint32_t unit) {
    // Read all heads of one row with a strided GM DMA, then write one
    // contiguous batch-major row. The UB-only mode needs no vector buffers.
    const uint32_t tileHeads = t.headTile;
    const uint32_t tiles = (t.heads + tileHeads - 1) / tileHeads;
    const uint32_t row = unit / tiles, head = (unit % tiles) * tileHeads;
    const uint32_t liveHeads = t.heads - head < tileHeads ? t.heads - head : tileHeads;
    DataCopyExtParams copy{static_cast<uint16_t>(liveHeads), KV_RANK * sizeof(bfloat16_t),
                           static_cast<uint32_t>((t.rows - 1) * KV_RANK * sizeof(bfloat16_t)), 0, 0};
    DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, 0};
    DataCopyPad(b0, xGm[(head * t.rows + row) * KV_RANK], copy, noPad);
    Sync<HardEvent::MTE2_MTE3>();
    DataCopy(yGm[(static_cast<uint64_t>(row) * t.heads + head) * KV_RANK], b0, liveHeads * KV_RANK);
  }
  uint32_t cachedGammaWidth;
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0Buf, b1Buf, b2Buf, b3Buf, f0Buf, f1Buf, f2Buf, f3Buf, f4Buf, indexBuf;
  LocalTensor<bfloat16_t> b0, b1, b2, b3;
  LocalTensor<float> f0, f1, f2, f3, f4;
  LocalTensor<int64_t> idx;
  GlobalTensor<bfloat16_t> xGm, auxGm, gammaQGm, gammaKvGm, sinGm, cosGm, yGm, qrGm, kvGm, krGm;
  GlobalTensor<int64_t> indexGm;
  XMlaLayoutTilingData t;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_layout(GM_ADDR x, GM_ADDR aux, GM_ADDR gamma_q, GM_ADDR gamma_kv,
                                                   GM_ADDR sin, GM_ADDR cos, GM_ADDR index, GM_ADDR y,
                                                   GM_ADDR query_rope, GM_ADDR kv_cache, GM_ADDR kr_cache,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaLayoutTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaLayoutTilingData, tilingData, tiling);
  XMlaLayoutKernel op;
  op.Init(x, aux, gamma_q, gamma_kv, sin, cos, index, y, query_rope, kv_cache, kr_cache, tilingData);
  op.Process();
}
