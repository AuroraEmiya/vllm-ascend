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
 * \file decode_index_score_service_cube.h
 * \brief Cube service (AIC). For every normal candidate block of this
 *        (query, chunk) program it gathers the physical page's index-K tile,
 *        runs a single-block mmad against the index query and fixes the
 *        [align16(H), 128] fp32 result out to the workspace GM strip.
 *
 * This is a deliberately simplified arch22 implementation (single buffering,
 * one block per mmad, PipeBarrier<PIPE_ALL> between stages). It mirrors the
 * Nd2Nz / LoadData / Mmad / Fixp mechanics of the lightning_indexer arch22
 * cube service but drops the multi-buffer pipeline. The per-block batch
 * optimization (BLOCKS_PER_MMAD) is a follow-up.
 */

#ifndef DECODE_INDEX_SCORE_SERVICE_CUBE_H_
#define DECODE_INDEX_SCORE_SERVICE_CUBE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../decode_index_score_common.h"

namespace DecodeIndexScoreKernel {
using namespace AscendC;

template <typename Q_T>
class DecodeIndexScoreServiceCube {
public:
    using K_T = Q_T;

    static constexpr uint32_t M_PAD = 16;          // align16(H) upper bound (H <= 16)
    static constexpr uint32_t BLOCK_POS = 128;     // positions per sparse block
    static constexpr uint32_t L0AB_ELEMS = M_PAD * 128;    // L0A/L0B tile elements
    static constexpr uint32_t L1_Q_ELEMS = M_PAD * 128;    // L1 query tile (NZ)
    static constexpr uint32_t L1_K_ELEMS = 128 * 128;      // L1 key tile (NZ)
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix, isSetPadding

    __aicore__ inline DecodeIndexScoreServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void Init(const ConstInfo &constInfo, const GlobalTensor<Q_T> &queryGm,
                                const GlobalTensor<K_T> &keyGm, const GlobalTensor<int32_t> &blockTableGm,
                                const GlobalTensor<float> &wsGm);
    __aicore__ inline void Run(const RangeInfo &range, uint32_t queryId);
    __aicore__ inline void SetReadyEvent();
    __aicore__ inline void WaitVecReadyEvent();

protected:
    __aicore__ inline void LoadQueryToL0a();
    __aicore__ inline void LoadKeyToL0b(uint64_t pageId);
    __aicore__ inline void ComputeL0c();
    __aicore__ inline void FixpToWorkspace(uint32_t blockLocalIdx);
    __aicore__ inline void DoOneBlock(uint32_t blockId, uint32_t blockLocalIdx);

    TPipe *pipe_ = nullptr;
    ConstInfo constInfo_{};
    RangeInfo range_{};
    uint32_t queryId_ = 0;

    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<float> wsGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;
    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;
};

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::InitBuffers(TPipe *pipe)
{
    pipe_ = pipe;
    pipe_->InitBuffer(bufQL1_, L1_Q_ELEMS * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe_->InitBuffer(bufKeyL1_, L1_K_ELEMS * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe_->InitBuffer(bufQL0_, L0AB_ELEMS * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe_->InitBuffer(bufKeyL0_, L0AB_ELEMS * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe_->InitBuffer(bufL0C_, M_PAD * BLOCK_POS * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::Init(const ConstInfo &constInfo,
                                                              const GlobalTensor<Q_T> &queryGm,
                                                              const GlobalTensor<K_T> &keyGm,
                                                              const GlobalTensor<int32_t> &blockTableGm,
                                                              const GlobalTensor<float> &wsGm)
{
    constInfo_ = constInfo;
    queryGm_ = queryGm;
    keyGm_ = keyGm;
    blockTableGm_ = blockTableGm;
    wsGm_ = wsGm;
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::SetReadyEvent()
{
    CrossCoreSetFlag<DI_SYNC_MODE, PIPE_FIX>(CROSS_CV_EVENT);
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::WaitVecReadyEvent()
{
    CrossCoreWaitFlag(CROSS_VC_EVENT);
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::LoadQueryToL0a()
{
    // GM -> L1 (ND -> NZ) for q [H, head_dim]
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = constInfo_.numIdxHeads; // rows
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(constInfo_.numIdxHeads, (uint64_t)BLOCK_CUBE_SIZE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_, queryGm_[queryId_ * constInfo_.numIdxHeads * constInfo_.headDim], nd2nzPara);
    PipeBarrier<PIPE_ALL>();

    // L1 -> L0A
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(constInfo_.numIdxHeads, BLOCK_CUBE_SIZE); // Hin
    loadData3DParams.l1W = BLOCK_CUBE_SIZE;                                 // Win = M0
    loadData3DParams.channelSize = constInfo_.headDim;                       // Cin = K
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;
    loadData3DParams.mExtension = CeilAlign(constInfo_.numIdxHeads, BLOCK_CUBE_SIZE);
    loadData3DParams.kExtension = constInfo_.headDim;
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    LoadData<Q_T, LOAD3DV2_CONFIG>(queryL0_, queryL1_, loadData3DParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::LoadKeyToL0b(uint64_t pageId)
{
    // GM -> L1 (ND -> NZ) for the K tile [128, head_dim] of the given page
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = BLOCK_POS; // 128 rows
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(BLOCK_POS, (uint64_t)BLOCK_CUBE_SIZE); // 128
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // cache layout: [num_blocks, 128, head_dim] contiguous
    DataCopy(keyL1_, keyGm_[pageId * BLOCK_POS * constInfo_.headDim], nd2nzPara);
    PipeBarrier<PIPE_ALL>();

    // L1 -> L0B
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(BLOCK_POS, BLOCK_CUBE_SIZE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE_SIZE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_, keyL1_, loadData2DParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::ComputeL0c()
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(constInfo_.numIdxHeads, BLOCK_CUBE_SIZE);
    mmadParams.n = BLOCK_POS;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_, queryL0_, keyL0_, mmadParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::FixpToWorkspace(uint32_t blockLocalIdx)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = CeilAlign(constInfo_.numIdxHeads, BLOCK_CUBE_SIZE);
    intriParams.nSize = BLOCK_POS;
    intriParams.dstStride = BLOCK_POS;
    intriParams.srcStride = CeilAlign(constInfo_.numIdxHeads, BLOCK_CUBE_SIZE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11;
    intriParams.reluPre = 1;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    uint64_t stripOffset = (uint64_t)blockLocalIdx * constInfo_.wsStripSize;
    AscendC::DataCopy(wsGm_[stripOffset], cL0_, intriParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::DoOneBlock(uint32_t blockId, uint32_t blockLocalIdx)
{
    int32_t pageId = blockTableGm_.GetValue(range_.requestId * constInfo_.localMaxBlocks + blockId);
    pageId = MaxT(pageId, 0); // safe page id (block_table values are >= 0)
    LoadKeyToL0b((uint64_t)pageId);
    ComputeL0c();
    FixpToWorkspace(blockLocalIdx);
}

template <typename Q_T>
__aicore__ inline void DecodeIndexScoreServiceCube<Q_T>::Run(const RangeInfo &range, uint32_t queryId)
{
    range_ = range;
    queryId_ = queryId;

    WaitVecReadyEvent();

    // The C->V event must be set even when this chunk has no cube work,
    // because the vector side always waits on it after its sentinel writes.
    if (range.normalStart < range.normalEnd || range.hasPartial) {
        LoadQueryToL0a();

        // normal full blocks
        for (uint32_t blockId = range.normalStart; blockId < range.normalFullEnd; blockId++) {
            DoOneBlock(blockId, blockId - range.chunkStart);
        }
        // partial block (at most one) — same mmad; the vector side folds only the
        // valid prefix of its 128 positions
        if (range.hasPartial) {
            DoOneBlock(range.fullBlockCount, range.fullBlockCount - range.chunkStart);
        }
    }

    SetReadyEvent();
}

} // namespace DecodeIndexScoreKernel
#endif // DECODE_INDEX_SCORE_SERVICE_CUBE_H_
