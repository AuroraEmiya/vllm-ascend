/**
 * @file store_kv_block.cpp
 * @brief StoreKVBlock AICore kernel — grouped KV cache store (in-place)
 *
 * Copies row-contiguous segments from keyIn into keyCacheIn using precomputed
 * group metadata.  keyCacheIn is mutated in-place; rows not covered by any
 * group retain their original values.
 *
 * Data flow (per group g):
 *   src = keyIn        [ groupSrcIdx[g] : groupSrcIdx[g] + groupLen[g] ]
 *   dst = keyCacheIn   [ groupDstIdx[g] : groupDstIdx[g] + groupLen[g] ]
 *   Copy src rows → dst rows via UB bounce buffer.
 *
 * Multi-core: groups are distributed round-robin across active cores.
 */

#include "kernel_utils.h"

constexpr int32_t ALIGN = 32;
using namespace AscendC;

class StoreKVBlock {
public:
    __aicore__ inline StoreKVBlock(StoreKVBlockTilingData tiling)
        : rowBytes_(tiling.rowBytes),
          maxGroupLen_(tiling.maxGroupLen),
          groupCount_(tiling.groupCount),
          coreCount_(tiling.coreCount)
    {}

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR keyCacheIn,
                                GM_ADDR groupLen, GM_ADDR groupSrcIdx,
                                GM_ADDR groupDstIdx, GM_ADDR /*keyCacheOut*/)
    {
        AscendC::TPipe pipe;
        pipe.InitBuffer(ubBuf_, RoundUp(maxGroupLen_ * rowBytes_, ALIGN));
        ubTensor_ = ubBuf_.Get<uint8_t>();

        srcGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIn);
        cacheGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheIn);
        groupLenGm_.SetGlobalBuffer((__gm__ int32_t *)groupLen);
        groupSrcGm_.SetGlobalBuffer((__gm__ int32_t *)groupSrcIdx);
        groupDstGm_.SetGlobalBuffer((__gm__ int32_t *)groupDstIdx);
    }

    __aicore__ inline void Process()
    {
        uint32_t activeCores = Min(groupCount_, coreCount_);
        uint32_t basePerCore = groupCount_ / activeCores;
        uint32_t remainder = groupCount_ - basePerCore * activeCores;

        uint32_t coreId = GetBlockIdx();
        if (coreId >= activeCores) {
            return;
        }

        uint32_t myGroupCount = basePerCore + (coreId < remainder ? 1 : 0);
        uint32_t myStartGroup = basePerCore * coreId + Min(coreId, remainder);

        for (uint32_t g = 0; g < myGroupCount; g++) {
            uint32_t groupId = myStartGroup + g;
            int32_t groupLen = groupLenGm_.GetValue(groupId);
            if (groupLen <= 0) {
                continue;
            }
            int32_t srcRow = groupSrcGm_.GetValue(groupId);
            int32_t dstRow = groupDstGm_.GetValue(groupId);

            uint32_t dataBytes = static_cast<uint32_t>(groupLen) * rowBytes_;
            uint16_t byteCount = static_cast<uint16_t>(dataBytes);

            AscendC::DataCopyParams copyParams = {1, byteCount, 0, 0};
            int64_t srcByteOffset = static_cast<int64_t>(srcRow) * rowBytes_;
            int64_t dstByteOffset = static_cast<int64_t>(dstRow) * rowBytes_;

            DataCopy(ubTensor_, srcGm_[srcByteOffset], copyParams);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(cacheGm_[dstByteOffset], ubTensor_, copyParams);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    GlobalTensor<uint8_t> srcGm_;
    GlobalTensor<uint8_t> cacheGm_;
    GlobalTensor<int32_t> groupLenGm_;
    GlobalTensor<int32_t> groupSrcGm_;
    GlobalTensor<int32_t> groupDstGm_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> ubTensor_;

    uint32_t rowBytes_{0};
    uint32_t maxGroupLen_{0};
    uint32_t groupCount_{0};
    uint32_t coreCount_{0};
};

extern "C" __global__ __aicore__ void store_kv_block(
    GM_ADDR keyIn, GM_ADDR keyCacheIn,
    GM_ADDR groupLen, GM_ADDR groupSrcIdx,
    GM_ADDR groupDstIdx, GM_ADDR keyCacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    StoreKVBlock op(tilingData);
    op.Init(keyIn, keyCacheIn, groupLen, groupSrcIdx, groupDstIdx, keyCacheOut);
    op.Process();
}
