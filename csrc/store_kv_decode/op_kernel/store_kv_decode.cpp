/**
 * @file store_kv_decode.cpp
 * @brief StoreKVDecode AICore kernel — single-token KV cache store (D stage)
 *
 * Copies exactly one row from keyIn into keyCacheIn at the slot position.
 * Single core, no grouping.  keyCacheIn is mutated in-place.
 *
 * Data flow:
 *   src = keyIn[0]          (one row)
 *   dst = keyCacheIn[slot]  (one row at slot position)
 *   Copy src → dst via UB bounce buffer.
 */

#include "kernel_utils.h"

constexpr int32_t ALIGN = 32;
using namespace AscendC;

class StoreKVDecode {
public:
    __aicore__ inline StoreKVDecode(StoreKVDecodeTilingData tiling)
        : rowBytes_(tiling.rowBytes)
    {}

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR keyCacheIn,
                                GM_ADDR slotPos, GM_ADDR /*keyCacheOut*/)
    {
        AscendC::TPipe pipe;
        pipe.InitBuffer(ubBuf_, RoundUp(rowBytes_, ALIGN));
        ubTensor_ = ubBuf_.Get<uint8_t>();

        srcGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIn);
        cacheGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheIn);
        slotGm_.SetGlobalBuffer((__gm__ int32_t *)slotPos);
    }

    __aicore__ inline void Process()
    {
        int32_t slot = slotGm_.GetValue(0);
        if (slot < 0) {
            return;
        }

        uint32_t paddedBytes = RoundUp(rowBytes_, ALIGN);
        uint16_t blockCount = static_cast<uint16_t>(paddedBytes / ALIGN);

        AscendC::DataCopyParams copyParams = {1, blockCount, 0, 0};
        int64_t srcByteOffset = 0;  // keyIn[0] always
        int64_t dstByteOffset = static_cast<int64_t>(slot) * rowBytes_;

        DataCopy(ubTensor_, srcGm_[srcByteOffset], copyParams);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(cacheGm_[dstByteOffset], ubTensor_, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

private:
    GlobalTensor<uint8_t> srcGm_;
    GlobalTensor<uint8_t> cacheGm_;
    GlobalTensor<int32_t> slotGm_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> ubTensor_;

    uint32_t rowBytes_{0};
};

extern "C" __global__ __aicore__ void store_kv_decode(
    GM_ADDR keyIn, GM_ADDR keyCacheIn,
    GM_ADDR slotPos, GM_ADDR keyCacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    StoreKVDecode op(tilingData);
    op.Init(keyIn, keyCacheIn, slotPos, keyCacheOut);
    op.Process();
}
