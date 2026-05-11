/**
 * @file store_kv_block.cpp
 * @brief StoreKVBlock AICore kernel - grouped KV cache store
 *
 * Uses precomputed group metadata to copy KV data from keyIn into keyCache.
 * Each group g defines a contiguous copy:
 *   dst = keyCacheOut[groupKeyCacheIdx[g] : groupKeyCacheIdx[g] + groupLen[g]]
 *   src = keyIn[groupKeyIdx[g] : groupKeyIdx[g] + groupLen[g]]
 *
 * Multi-core: groups are distributed across cores by group count.
 */

#include "kernel_utils.h"

constexpr int32_t ALIGN = 32;
using namespace AscendC;

class StoreKVBlock {
public:
    __aicore__ inline StoreKVBlock(StoreKVBlockTilingData tilingData)
        : numTokens_(tilingData.numTokens),
          headDim_(tilingData.headDim),
          numGroups_(tilingData.numGroups),
          blockSize_(tilingData.blockSize),
          coreNum_(tilingData.numCore)
        {}

    __aicore__ inline void Init(GM_ADDR keyIn, GM_ADDR keyCacheIn,
                                GM_ADDR groupLen, GM_ADDR groupKeyIdx,
                                GM_ADDR groupKeyCacheIdx, GM_ADDR keyCacheOut)
    {
        AscendC::TPipe pipe;
        pipe.InitBuffer(ubBuf_, RoundUp(blockSize_ * headDim_, ALIGN));
        tmpTensor_ = ubBuf_.Get<uint8_t>();
        keyInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyIn);
        keyCacheInGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheIn);
        keyCacheOutGm_.SetGlobalBuffer((__gm__ uint8_t *)keyCacheOut);
        groupLenGm_.SetGlobalBuffer((__gm__ int32_t *)groupLen);
        groupKeyIdxGm_.SetGlobalBuffer((__gm__ int32_t *)groupKeyIdx);
        groupKeyCacheIdxGm_.SetGlobalBuffer((__gm__ int32_t *)groupKeyCacheIdx);
    }

    __aicore__ inline void Process()
    {
        uint32_t actualCoreNum = numGroups_ <= coreNum_ ? numGroups_ : coreNum_;
        uint32_t groupsPerCore = numGroups_ / actualCoreNum;
        uint32_t leftGroups = numGroups_ - groupsPerCore * actualCoreNum;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t myNumGroups = blockIdx < leftGroups ? groupsPerCore + 1 : groupsPerCore;
        uint32_t myStartGroup = blockIdx < leftGroups
            ? (groupsPerCore * blockIdx + blockIdx)
            : (groupsPerCore * blockIdx + leftGroups);

        if (blockIdx >= actualCoreNum) {
            return;
        }

        for (uint32_t g = 0; g < myNumGroups; g++) {
            uint32_t groupId = myStartGroup + g;
            int32_t length = groupLenGm_.GetValue(groupId);
            int32_t srcIdx = groupKeyIdxGm_.GetValue(groupId);
            int32_t dstIdx = groupKeyCacheIdxGm_.GetValue(groupId);

            if (length <= 0) {
                continue;
            }

            uint32_t copyBytes = static_cast<uint32_t>(length) * headDim_;
            uint32_t alignedBytes = RoundUp(copyBytes, ALIGN);
            uint16_t numBlocks = static_cast<uint16_t>(alignedBytes / ALIGN);

            AscendC::DataCopyParams copyParams = {1, numBlocks, 0, 0};
            int64_t srcByteOffset = static_cast<int64_t>(srcIdx) * headDim_;
            int64_t dstByteOffset = static_cast<int64_t>(dstIdx) * headDim_;

            DataCopy(tmpTensor_, keyInGm_[srcByteOffset], copyParams);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopy(keyCacheOutGm_[dstByteOffset], tmpTensor_, copyParams);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    GlobalTensor<uint8_t> keyInGm_;
    GlobalTensor<uint8_t> keyCacheInGm_;
    GlobalTensor<int32_t> groupLenGm_;
    GlobalTensor<int32_t> groupKeyIdxGm_;
    GlobalTensor<int32_t> groupKeyCacheIdxGm_;
    GlobalTensor<uint8_t> keyCacheOutGm_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<uint8_t> tmpTensor_;

    uint32_t numTokens_{0};
    uint32_t headDim_{0};
    uint32_t numGroups_{0};
    uint32_t blockSize_{0};
    uint32_t coreNum_{0};
};

extern "C" __global__ __aicore__ void store_kv_block(
    GM_ADDR keyIn, GM_ADDR keyCacheIn,
    GM_ADDR groupLen, GM_ADDR groupKeyIdx,
    GM_ADDR groupKeyCacheIdx, GM_ADDR keyCacheOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    StoreKVBlock op(tilingData);
    op.Init(keyIn, keyCacheIn, groupLen, groupKeyIdx, groupKeyCacheIdx, keyCacheOut);
    op.Process();
}
