/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file decode_index_score_service_vector.h
 * \brief Vector service (AIV). Writes the init/local/-inf sentinel runs,
 *        waits for the cube service's mmad results in workspace GM, folds the
 *        per-block max over the 128 positions (or the partial block's valid
 *        prefix) and stores the block scores into the score buffer.
 *
 * The sentinel runs are written before waiting for the cube so the two cores
 * overlap. The max fold mirrors lightning_indexer's binary DoReduce but uses
 * Max instead of Add (associative+commutative).
 */

#ifndef DECODE_INDEX_SCORE_SERVICE_VECTOR_H_
#define DECODE_INDEX_SCORE_SERVICE_VECTOR_H_

#include "kernel_operator.h"
#include "../decode_index_score_common.h"

namespace DecodeIndexScoreKernel {
using namespace AscendC;

template <typename Q_T>
class DecodeIndexScoreServiceVector {
public:
    static constexpr uint32_t M_PAD = 16;      // align16(H) upper bound (H <= 16)
    static constexpr uint32_t BLOCK_POS = 128; // positions per sparse block
    static constexpr uint32_t MAX_SENT_CHUNK = 4096; // cap one sentinel run copy to 16KB

    __aicore__ inline DecodeIndexScoreServiceVector(){};
    __aicore__ inline void Init(const ConstInfo &constInfo, const GlobalTensor<float> &scoreGm,
                                const GlobalTensor<float> &wsGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void Run(const RangeInfo &range, uint32_t queryId);

protected:
    __aicore__ inline void SignalVecReady();
    __aicore__ inline void WaitCubeReady();
    __aicore__ inline uint64_t NearestPower2(uint64_t value);
    __aicore__ inline void ReduceRowMax(LocalTensor<float> &row, uint32_t validLen);
    __aicore__ inline void WriteSentinelRun(uint32_t head, uint32_t start, uint32_t end, float value);
    __aicore__ inline void ProcessFullBlock(uint32_t blockId);
    __aicore__ inline void ProcessPartialBlock(uint32_t partialBlockId, uint32_t validPos);

    TPipe *pipe_ = nullptr;
    ConstInfo constInfo_{};
    RangeInfo range_{};
    uint32_t queryId_ = 0;

    GlobalTensor<float> scoreGm_;
    GlobalTensor<float> wsGm_;

    TBuf<TPosition::VECCALC> bufIn_;
    LocalTensor<float> inUb_;  // [align16(H), 128] workspace read buffer
    TBuf<TPosition::VECCALC> bufSent_;
    LocalTensor<float> sentUb_; // sentinel / single-value run buffer
};

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::Init(const ConstInfo &constInfo,
                                                                const GlobalTensor<float> &scoreGm,
                                                                const GlobalTensor<float> &wsGm)
{
    constInfo_ = constInfo;
    scoreGm_ = scoreGm;
    wsGm_ = wsGm;
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::InitBuffers(TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(bufIn_, M_PAD * BLOCK_POS * sizeof(float));
    inUb_ = bufIn_.Get<float>();
    uint32_t sentChunk = constInfo_.wsBlocksPerChunk < MAX_SENT_CHUNK ? constInfo_.wsBlocksPerChunk : MAX_SENT_CHUNK;
    pipe_->InitBuffer(bufSent_, sentChunk * sizeof(float));
    sentUb_ = bufSent_.Get<float>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::SignalVecReady()
{
    CrossCoreSetFlag<DI_SYNC_MODE, PIPE_MTE2>(CROSS_VC_EVENT);
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::WaitCubeReady()
{
    CrossCoreWaitFlag(CROSS_CV_EVENT);
}

template <typename Q_T>
__aicore__ inline uint64_t DecodeIndexScoreServiceVector<Q_T>::NearestPower2(uint64_t value)
{
    if (value <= 2) {
        return value;
    }
    const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(value);
    return (1ull << pow);
}

// Fold the elementwise max of the first `validLen` elements of `row` into
// row[0]. Binary fold; the tail (validLen - nearest_pow2) is folded first so
// arbitrary validLen is supported (used by the partial block).
template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::ReduceRowMax(LocalTensor<float> &row, uint32_t validLen)
{
    if (validLen <= 1) {
        return;
    }
    uint32_t pow = (uint32_t)NearestPower2(validLen);
    if (validLen > pow) {
        AscendC::Max(row, row, row[pow], validLen - pow);
        PipeBarrier<PIPE_V>();
    }
    uint32_t nowRows = pow;
    while (nowRows > 1) {
        nowRows /= 2;
        AscendC::Max(row, row, row[nowRows], nowRows);
        PipeBarrier<PIPE_V>();
    }
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::WriteSentinelRun(uint32_t head, uint32_t start, uint32_t end,
                                                                            float value)
{
    uint32_t len = end - start;
    if (len == 0) {
        return;
    }
    uint64_t gmBase = (uint64_t)head * constInfo_.totalQ * constInfo_.scoreBlockStride +
                      queryId_ * constInfo_.scoreBlockStride + start;
    // chunk the run so a single DataCopyPad stays within 16KB (uint16 blockLen)
    uint32_t chunk = len;
    uint32_t offset = 0;
    while (chunk > MAX_SENT_CHUNK) {
        AscendC::Duplicate(sentUb_, value, MAX_SENT_CHUNK);
        PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(scoreGm_[gmBase + offset], sentUb_,
                             {1, (uint16_t)(MAX_SENT_CHUNK * sizeof(float)), 0, 0});
        PipeBarrier<PIPE_ALL>();
        chunk -= MAX_SENT_CHUNK;
        offset += MAX_SENT_CHUNK;
    }
    if (chunk > 0) {
        AscendC::Duplicate(sentUb_, value, chunk);
        PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(scoreGm_[gmBase + offset], sentUb_, {1, (uint16_t)(chunk * sizeof(float)), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::ProcessFullBlock(uint32_t blockId)
{
    uint64_t stripOffset = (uint64_t)(blockId - range_.chunkStart) * constInfo_.wsStripSize;
    // read the H rows of the strip: [H, 128] fp32 (contiguous at the strip head)
    AscendC::DataCopy(inUb_, wsGm_[stripOffset],
                      {1, (uint16_t)(constInfo_.numIdxHeads * BLOCK_POS * sizeof(float)), 0, 0});
    PipeBarrier<PIPE_ALL>();
    for (uint32_t h = 0; h < constInfo_.numIdxHeads; h++) {
        LocalTensor<float> row = inUb_[h * BLOCK_POS];
        ReduceRowMax(row, BLOCK_POS);
        uint64_t gmOffset = (uint64_t)h * constInfo_.totalQ * constInfo_.scoreBlockStride +
                            queryId_ * constInfo_.scoreBlockStride + blockId;
        AscendC::DataCopyPad(scoreGm_[gmOffset], row, {1, (uint16_t)sizeof(float), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::ProcessPartialBlock(uint32_t partialBlockId,
                                                                               uint32_t validPos)
{
    uint64_t stripOffset = (uint64_t)(partialBlockId - range_.chunkStart) * constInfo_.wsStripSize;
    AscendC::DataCopy(inUb_, wsGm_[stripOffset],
                      {1, (uint16_t)(constInfo_.numIdxHeads * BLOCK_POS * sizeof(float)), 0, 0});
    PipeBarrier<PIPE_ALL>();
    for (uint32_t h = 0; h < constInfo_.numIdxHeads; h++) {
        LocalTensor<float> row = inUb_[h * BLOCK_POS];
        ReduceRowMax(row, validPos);
        uint64_t gmOffset = (uint64_t)h * constInfo_.totalQ * constInfo_.scoreBlockStride +
                            queryId_ * constInfo_.scoreBlockStride + partialBlockId;
        AscendC::DataCopyPad(scoreGm_[gmOffset], row, {1, (uint16_t)sizeof(float), 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceVector<Q_T>::Run(const RangeInfo &range, uint32_t queryId)
{
    range_ = range;
    queryId_ = queryId;

    SignalVecReady();

    // sentinel ranges (independent of the cube; overlap while the AIC computes)
    for (uint32_t h = 0; h < constInfo_.numIdxHeads; h++) {
        WriteSentinelRun(h, range.chunkStart, range.initEnd, SCORE_INIT);
        WriteSentinelRun(h, range.localBegin, range.validEnd, SCORE_LOCAL);
        WriteSentinelRun(h, range.tailBegin, range.chunkEnd, SCORE_NEG_INF);
    }

    WaitCubeReady();

    for (uint32_t blockId = range.normalStart; blockId < range.normalFullEnd; blockId++) {
        ProcessFullBlock(blockId);
    }
    if (range.hasPartial) {
        ProcessPartialBlock(range.fullBlockCount, range.partialValidPos);
    }
}

} // namespace DecodeIndexScoreKernel
#endif // DECODE_INDEX_SCORE_SERVICE_VECTOR_H_
