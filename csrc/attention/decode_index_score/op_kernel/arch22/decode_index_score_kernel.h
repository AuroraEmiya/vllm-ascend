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
 * \file decode_index_score_kernel.h
 * \brief arch22 kernel main class. One program handles one (query, chunk)
 *        pair, mirroring the Triton grid. The cube core computes per-block
 *        mmads, the vector core reduces and writes scores.
 *
 * 910B is a 1:2 (cube:vector) machine, so the kernel runs with
 * KERNEL_TYPE_MIX_AIC_1_2; only sub-AIV 0 performs the vector work and
 * sub-AIV 1 idles (the per-program vector workload is too small to split).
 */

#ifndef DECODE_INDEX_SCORE_KERNEL_H_
#define DECODE_INDEX_SCORE_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../../op_host/decode_index_score_tiling.h"
#include "../decode_index_score_common.h"
#include "decode_index_score_service_cube.h"
#include "decode_index_score_service_vector.h"

namespace DecodeIndexScoreKernel {
using namespace AscendC;

template <typename Q_T>
class DecodeIndexScoreKernelT {
public:
    using K_T = Q_T;

    __aicore__ inline DecodeIndexScoreKernelT(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *indexKvCache, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *seqLens, __gm__ uint8_t *globalSeqLens, __gm__ uint8_t *score,
                                __gm__ uint8_t *workspace, const DecodeIndexScoreTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitTilingData(const DecodeIndexScoreTilingData *__restrict tilingData);

    TPipe *pipe = nullptr;
    ConstInfo constInfo_{};
    RangeInfo range_{};
    uint32_t coreIdx_ = 0;
    uint32_t aivSubIdx_ = 0;
    uint32_t programCount_ = 0;
    uint32_t queryId_ = 0;
    uint32_t chunkId_ = 0;

    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<int32_t> globalSeqLensGm_;
    GlobalTensor<float> scoreGm_;
    GlobalTensor<float> wsGm_;

    DecodeIndexScoreServiceCube<Q_T> cubeService_;
    DecodeIndexScoreServiceVector<Q_T> vectorService_;
};

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreKernelT<Q_T>::InitTilingData(const DecodeIndexScoreTilingData *__restrict tiling)
{
    constInfo_.totalQ = tiling->totalQ;
    constInfo_.numIdxHeads = tiling->numIdxHeads;
    constInfo_.headDim = tiling->headDim;
    constInfo_.decodeQueryLen = tiling->decodeQueryLen;
    constInfo_.blockOffset = tiling->blockOffset;
    constInfo_.initBlocks = tiling->initBlocks;
    constInfo_.localBlocks = tiling->localBlocks;
    constInfo_.scoreBlockStride = tiling->scoreBlockStride;
    constInfo_.numChunks = tiling->numChunks;
    constInfo_.localMaxBlocks = tiling->localMaxBlocks;
    constInfo_.wsBlocksPerChunk = tiling->wsBlocksPerChunk;
    constInfo_.wsStripSize = tiling->wsStripSize;
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreKernelT<Q_T>::Init(__gm__ uint8_t *query, __gm__ uint8_t *indexKvCache,
                                                          __gm__ uint8_t *blockTable, __gm__ uint8_t *seqLens,
                                                          __gm__ uint8_t *globalSeqLens, __gm__ uint8_t *score,
                                                          __gm__ uint8_t *workspace,
                                                          const DecodeIndexScoreTilingData *__restrict tiling,
                                                          TPipe *tPipe)
{
    pipe = tPipe;
    InitTilingData(tiling);

    if ASCEND_IS_AIV {
        // 910B: one block spawns one AIC and two AIVs (1:2).
        coreIdx_ = GetBlockIdx() / 2;
        aivSubIdx_ = GetBlockIdx() % 2;
    } else {
        coreIdx_ = GetBlockIdx();
        aivSubIdx_ = 0;
    }
    programCount_ = constInfo_.totalQ * constInfo_.numChunks;
    queryId_ = coreIdx_ % constInfo_.totalQ;
    chunkId_ = coreIdx_ / constInfo_.totalQ;

    queryGm_.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm_.SetGlobalBuffer((__gm__ K_T *)indexKvCache);
    blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    seqLensGm_.SetGlobalBuffer((__gm__ int32_t *)seqLens);
    globalSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)globalSeqLens);
    scoreGm_.SetGlobalBuffer((__gm__ float *)score);

    // per-program workspace slice: cube -> vector mmad hand-off
    uint64_t perProgramBytes =
        (uint64_t)constInfo_.wsBlocksPerChunk * constInfo_.wsStripSize * sizeof(float);
    wsGm_.SetGlobalBuffer((__gm__ float *)(workspace + coreIdx_ * perProgramBytes));

    uint32_t requestId = queryId_ / constInfo_.decodeQueryLen;
    int32_t localSeqLen = seqLensGm_.GetValue(requestId);
    int32_t globalSeqLen = globalSeqLensGm_.GetValue(requestId);
    range_ = ComputeRanges(queryId_, chunkId_, constInfo_, localSeqLen, globalSeqLen);

    cubeService_.Init(constInfo_, queryGm_, keyGm_, blockTableGm_, wsGm_);
    vectorService_.Init(constInfo_, scoreGm_, wsGm_);
    if ASCEND_IS_AIV {
        vectorService_.InitBuffers(pipe);
    } else {
        cubeService_.InitBuffers(pipe);
    }
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreKernelT<Q_T>::Process()
{
    if (coreIdx_ >= programCount_) {
        return;
    }
    if ASCEND_IS_AIC {
        cubeService_.Run(range_, queryId_);
    } else if (aivSubIdx_ == 0) {
        vectorService_.Run(range_, queryId_);
    }
}

} // namespace DecodeIndexScoreKernel
#endif // DECODE_INDEX_SCORE_KERNEL_H_
