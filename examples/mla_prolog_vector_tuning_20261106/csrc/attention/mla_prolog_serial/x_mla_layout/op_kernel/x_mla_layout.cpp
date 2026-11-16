// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include "x_mla_layout_tiling.h"
using namespace AscendC;
namespace {
constexpr uint32_t KV_RANK = 512;

template <HardEvent EVENT>
__aicore__ inline void Sync() {
  SetFlag<EVENT>(EVENT_ID0);
  WaitFlag<EVENT>(EVENT_ID0);
}

class XMlaLayoutKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR query, const XMlaLayoutTilingData& tiling) {
    tilingData = tiling;
    headMajorGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
    queryGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query));
    // Two disjoint BF16 tiles, each protected until its output DMA completes.
    pipe.InitBuffer(tileBuffer, tilingData.workElements * sizeof(bfloat16_t) * 2);
    tileBf16 = tileBuffer.Get<bfloat16_t>();
  }
  __aicore__ inline void Process() {
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint32_t iteration = 0;
    for (uint32_t unit = GetBlockIdx(); unit < tilingData.units; unit += GetBlockNum(), ++iteration) {
      const int32_t eventId = (iteration & 1u) ? EVENT_ID1 : EVENT_ID0;
      WaitFlag<HardEvent::MTE3_MTE2>(eventId);
      RestoreLayout(unit, (iteration & 1u) * tilingData.workElements);
      SetFlag<HardEvent::MTE3_MTE2>(eventId);
    }
    // Consume both tokens before returning to a parent SuperKernel.
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  }

 private:
  __aicore__ inline void RestoreLayout(uint32_t unit, uint32_t bufferOffset = 0) {
    // Read a head/row tile, then write batch-major rows using DMA strides.
    const uint32_t tileHeads = tilingData.headTile;
    const uint32_t tiles = (tilingData.heads + tileHeads - 1) / tileHeads;
    const uint32_t rowTile = tilingData.rowTile;
    const uint32_t row = (unit / tiles) * rowTile, head = (unit % tiles) * tileHeads;
    const uint32_t liveHeads = tilingData.heads - head < tileHeads ? tilingData.heads - head : tileHeads;
    const uint32_t liveRows = tilingData.rows - row < rowTile ? tilingData.rows - row : rowTile;
    DataCopyExtParams copy{static_cast<uint16_t>(liveHeads), static_cast<uint32_t>(liveRows * KV_RANK * sizeof(bfloat16_t)),
                           static_cast<uint32_t>((tilingData.rows - liveRows) * KV_RANK * sizeof(bfloat16_t)),
                           static_cast<uint32_t>((rowTile - liveRows) * KV_RANK * sizeof(bfloat16_t) / 32), 0};
    DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, 0};
    auto buffer = tileBf16[bufferOffset];
    DataCopyPad(buffer, headMajorGm[(head * tilingData.rows + row) * KV_RANK], copy, noPad);
    Sync<HardEvent::MTE2_MTE3>();
    // DMA strides perform the [head,row,512] -> [row,head,512] mapping.
    // UB strides are in 32-byte blocks; GM strides above are bytes.
    DataCopyExtParams write{static_cast<uint16_t>(liveHeads), KV_RANK * sizeof(bfloat16_t),
                           static_cast<uint32_t>((rowTile - 1) * KV_RANK * sizeof(bfloat16_t) / 32), 0, 0};
    for (uint32_t rowOffset = 0; rowOffset < liveRows; ++rowOffset) {
      DataCopyPad(queryGm[(static_cast<uint64_t>(row + rowOffset) * tilingData.heads + head) * KV_RANK],
                  buffer[rowOffset * KV_RANK], write);
    }
  }
  TPipe pipe;
  TBuf<TPosition::VECCALC> tileBuffer;
  LocalTensor<bfloat16_t> tileBf16;
  GlobalTensor<bfloat16_t> headMajorGm, queryGm;
  XMlaLayoutTilingData tilingData;
};
}  // namespace
extern "C" __global__ __aicore__ void x_mla_layout(
    GM_ADDR x, GM_ADDR query, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(XMlaLayoutTilingData);
  GET_TILING_DATA_WITH_STRUCT(XMlaLayoutTilingData, tilingData, tiling);
  XMlaLayoutKernel op;
  op.Init(x, query, tilingData);
  op.Process();
}
