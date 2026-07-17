/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file sparse_kv_gather_kernel.h
 * \brief Vector-core kernel: sparse KV gather into two contiguous 3D outputs.
 *
 *   out_ctkv [num_actual, topk_n, 512]
 *   out_kpe  [num_actual, topk_n,  64]
 *
 * Invalid positions (index == -1, or index == cur_pos[q] when provided)
 * produce zero rows in the output.
 */

#ifndef SPARSE_KV_GATHER_KERNEL_H
#define SPARSE_KV_GATHER_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

namespace BaseApi {

constexpr uint32_t D_NOPE     = 512;
constexpr uint32_t D_ROPE     = 64;
constexpr uint32_t D_COMBINED = 576;
constexpr uint32_t MAX_ROWS_PER_LOOP = 16;
constexpr uint32_t BYTES_PER_BLOCK   = 32;

enum class SKGLayoutKernel : uint32_t {
    BSND    = 0,
    TND     = 1,
    PA_BSND = 2,
};

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
class SparseKvGatherKernel {
public:
    __aicore__ inline SparseKvGatherKernel() {}

    __aicore__ inline void Init(__gm__ uint8_t *sparseIndices,
                                __gm__ uint8_t *keyNope,
                                __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actSeqLenQ,
                                __gm__ uint8_t *actSeqLenKV,
                                __gm__ uint8_t *curPos,
                                __gm__ uint8_t *outCtkv,
                                __gm__ uint8_t *outKpe,
                                const SparseKvGatherTilingData *__restrict tilingData,
                                TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t GetKeyOffset(int64_t tokenIdx, uint32_t bIdx);
    __aicore__ inline void    GetTokenPair(int64_t &t0, int64_t &t1,
                                           uint64_t topkBase, uint64_t kIdx);
    __aicore__ inline void    CopyInSingleKv(LocalTensor<KV_T> &kvUb,
                                             uint32_t startRow, int64_t keyOffset);
    __aicore__ inline uint32_t CopyInKvPair(LocalTensor<KV_T> &kvUb,
                                            uint32_t startRow,
                                            int64_t t0Idx, int64_t t1Idx,
                                            uint32_t bIdx);
    __aicore__ inline void    ProcessOneGroup(uint32_t qIdx);

    // --- Members ---
    const SparseKvGatherTilingData *__restrict tilingData_ = nullptr;
    TPipe *pipe_ = nullptr;

    GlobalTensor<int32_t> sparseIndicesGm_;
    GlobalTensor<KV_T>    keyNopeGm_;
    GlobalTensor<KV_T>    keyRopeGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> actSeqLenQGm_;
    GlobalTensor<int32_t> actSeqLenKVGm_;
    GlobalTensor<int32_t> curPosGm_;
    GlobalTensor<KV_T>    outCtkvGm_;
    GlobalTensor<KV_T>    outKpeGm_;

    uint32_t batchSize_        = 0;
    uint32_t s1Size_           = 0;
    uint32_t s2Size_           = 0;
    uint32_t numActual_        = 0;
    uint32_t topkN_            = 0;
    int64_t  sparseBlockSize_  = 1;
    uint32_t groupsPerCore_    = 0;
    int64_t  paBlockSize_      = 0;
    uint32_t paMaxBlocks_      = 0;

    uint32_t coreIdx_   = 0;
    uint32_t numCores_  = 0;
    bool     hasActSeqLenQ_  = false;
    bool     hasActSeqLenKV_ = false;
    bool     hasCurPos_       = false;

    TBuf<> stageBuf_;
    TEventID mte2ToMte3_[2];
};

// ==================== Init ====================

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline void
SparseKvGatherKernel<KV_T, IS_PA, LAYOUT_Q>::Init(
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *keyNope,
    __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actSeqLenQ,
    __gm__ uint8_t *actSeqLenKV,
    __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv,
    __gm__ uint8_t *outKpe,
    const SparseKvGatherTilingData *__restrict tilingData,
    TPipe *pipe)
{
    tilingData_ = tilingData;
    pipe_       = pipe;

    const auto &bp = tilingData_->baseParams;
    batchSize_        = bp.batchSize;
    s1Size_           = bp.s1Size;
    s2Size_           = bp.s2Size;
    numActual_        = bp.numActual;
    topkN_            = bp.topkN;
    sparseBlockSize_  = bp.sparseBlockSize;
    groupsPerCore_    = bp.groupsPerCore;
    paBlockSize_      = bp.blockSize;
    paMaxBlocks_      = bp.maxBlockNumPerBatch;
    hasActSeqLenQ_    = !bp.isActualLenDimsNull;
    hasActSeqLenKV_   = !bp.isActualLenDimsKVNull;
    hasCurPos_         = bp.hasCurPos;

    coreIdx_  = GetBlockIdx();
    numCores_ = bp.usedCoreNum;

    sparseIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sparseIndices));
    keyNopeGm_     .SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(keyNope));
    keyRopeGm_     .SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(keyRope));
    outCtkvGm_     .SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(outCtkv));
    outKpeGm_      .SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(outKpe));

    if constexpr (IS_PA) {
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    }
    if (actSeqLenQ != nullptr) {
        actSeqLenQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actSeqLenQ));
    }
    if (actSeqLenKV != nullptr) {
        actSeqLenKVGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actSeqLenKV));
    }
    if (curPos != nullptr) {
        curPosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(curPos));
    }

    pipe_->InitBuffer(stageBuf_, D_COMBINED * MAX_ROWS_PER_LOOP * sizeof(KV_T));

    mte2ToMte3_[0] = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3_[1] = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[0]);
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[1]);
}

// ==================== Index Translation ====================

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline int64_t
SparseKvGatherKernel<KV_T, IS_PA, LAYOUT_Q>::GetKeyOffset(
    int64_t tokenIdx, uint32_t bIdx)
{
    if (tokenIdx < 0) return -1;

    if constexpr (IS_PA) {
        int64_t blkIdx = tokenIdx / paBlockSize_;
        int64_t blkOff  = tokenIdx % paBlockSize_;
        int64_t physBlk = blockTableGm_.GetValue(
            static_cast<int64_t>(bIdx) * paMaxBlocks_ + blkIdx);
        return physBlk * paBlockSize_ + blkOff;
    } else {
        if constexpr (LAYOUT_Q == SKGLayoutKernel::TND) {
            int64_t prefix = (bIdx == 0) ? 0
                : actSeqLenKVGm_.GetValue(bIdx - 1);
            return prefix + tokenIdx;
        } else {
            return static_cast<int64_t>(bIdx) * s2Size_ + tokenIdx;
        }
    }
}

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline void
SparseKvGatherKernel<KV_T, IS_PA, SKGLayoutKernel LAYOUT_Q>::GetTokenPair(
    int64_t &t0, int64_t &t1, uint64_t topkBase, uint64_t kIdx)
{
    if (kIdx >= topkN_) { t0 = -1; }
    else { int32_t r = sparseIndicesGm_.GetValue(topkBase + kIdx);
           t0 = (r == -1) ? -1 : static_cast<int64_t>(r); }

    if (kIdx + 1 >= topkN_) { t1 = -1; }
    else { int32_t r = sparseIndicesGm_.GetValue(topkBase + kIdx + 1);
           t1 = (r == -1) ? -1 : static_cast<int64_t>(r); }
}

// ==================== DMA Helpers ====================

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline void
SparseKvGatherKernel<KV_T, IS_PA, SKGLayoutKernel LAYOUT_Q>::CopyInSingleKv(
    LocalTensor<KV_T> &kvUb, uint32_t startRow, int64_t keyOffset)
{
    if (keyOffset < 0) return;

    constexpr uint32_t nopeAlign = (D_NOPE * sizeof(KV_T) + BYTES_PER_BLOCK - 1)
                                   / BYTES_PER_BLOCK * BYTES_PER_BLOCK / sizeof(KV_T);
    constexpr uint32_t ropeAlign = (D_ROPE * sizeof(KV_T) + BYTES_PER_BLOCK - 1)
                                   / BYTES_PER_BLOCK * BYTES_PER_BLOCK / sizeof(KV_T);
    constexpr uint32_t padNope = nopeAlign - D_NOPE;
    constexpr uint32_t padRope = ropeAlign - D_ROPE;

    DataCopyExtParams np;
    np.blockCount = 1; np.blockLen = D_NOPE * sizeof(KV_T);
    np.srcStride = 0; np.dstStride = 0;
    DataCopyPadExtParams<KV_T> nPad;
    nPad.isPad = true; nPad.leftPadding = 0;
    nPad.rightPadding = padNope; nPad.paddingValue = 0;
    DataCopyPad(kvUb[startRow * D_COMBINED],
                keyNopeGm_[keyOffset * D_NOPE], np, nPad);

    DataCopyExtParams rp;
    rp.blockCount = 1; rp.blockLen = D_ROPE * sizeof(KV_T);
    rp.srcStride = 0; rp.dstStride = 0;
    DataCopyPadExtParams<KV_T> rPad;
    rPad.isPad = true; rPad.leftPadding = 0;
    rPad.rightPadding = padRope; rPad.paddingValue = 0;
    DataCopyPad(kvUb[startRow * D_COMBINED + D_NOPE],
                keyRopeGm_[keyOffset * D_ROPE], rp, rPad);
}

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline uint32_t
SparseKvGatherKernel<KV_T, IS_PA, SKGLayoutKernel LAYOUT_Q>::CopyInKvPair(
    LocalTensor<KV_T> &kvUb, uint32_t startRow,
    int64_t t0Idx, int64_t t1Idx, uint32_t bIdx)
{
    int64_t off0 = GetKeyOffset(t0Idx, bIdx);
    int64_t off1 = GetKeyOffset(t1Idx, bIdx);

    if (unlikely(off0 < 0 && off1 < 0)) return 0;

    // Only one valid → pack at startRow
    if (off0 < 0) { CopyInSingleKv(kvUb, startRow, off1); return 1; }
    if (off1 < 0) { CopyInSingleKv(kvUb, startRow, off0); return 1; }

    // Both valid
    int64_t gap = (off0 > off1) ? (off0 - off1) : (off1 - off0);
    int64_t srcStride     = (gap - sparseBlockSize_) * D_NOPE * sizeof(KV_T);
    int64_t ropeSrcStride = (gap - sparseBlockSize_) * D_ROPE * sizeof(KV_T);

    bool canPair = (sparseBlockSize_ == 1) &&
                   (srcStride >= 0) && (srcStride < INT32_MAX);

    if (!canPair) {
        CopyInSingleKv(kvUb, startRow, off0);
        CopyInSingleKv(kvUb, startRow + 1, off1);
        return 2;
    }

    // Fast path: single 2-block DMA
    int64_t minOff = (off0 < off1) ? off0 : off1;

    DataCopyExtParams params;
    params.blockCount = 2;
    params.blockLen   = D_NOPE * sizeof(KV_T);
    params.srcStride  = srcStride;
    params.dstStride  = D_ROPE * sizeof(KV_T) / BYTES_PER_BLOCK;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvUb[startRow * D_COMBINED],
                keyNopeGm_[minOff * D_NOPE], params, padParams);

    params.blockLen   = D_ROPE * sizeof(KV_T);
    params.srcStride  = ropeSrcStride;
    params.dstStride  = D_NOPE * sizeof(KV_T) / BYTES_PER_BLOCK;
    DataCopyPad(kvUb[startRow * D_COMBINED + D_NOPE],
                keyRopeGm_[minOff * D_ROPE], params, padParams);

    return 2;
}

// ==================== Process ====================

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline void
SparseKvGatherKernel<KV_T, IS_PA, SKGLayoutKernel LAYOUT_Q>::ProcessOneGroup(
    uint32_t qIdx)
{
    // --- Determine batch index and S1 offset for this query row ---
    uint32_t bIdx, s1o;
    if constexpr (LAYOUT_Q == SKGLayoutKernel::TND) {
        // TND: qIdx is flat across all batches
        uint32_t lo = 0, hi = batchSize_;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (static_cast<uint32_t>(actSeqLenQGm_.GetValue(mid)) <= qIdx)
                lo = mid + 1;
            else
                hi = mid;
        }
        bIdx = lo;
        int32_t prev = (bIdx == 0) ? 0 : actSeqLenQGm_.GetValue(bIdx - 1);
        s1o = qIdx - static_cast<uint32_t>(prev);
    } else {
        bIdx = qIdx / s1Size_;
        s1o  = qIdx % s1Size_;
    }

    // --- topk base offset ---
    uint64_t topkBase;
    if constexpr (LAYOUT_Q == SKGLayoutKernel::TND) {
        int32_t prefix = (bIdx == 0) ? 0 : actSeqLenQGm_.GetValue(bIdx - 1);
        topkBase = (static_cast<uint64_t>(prefix) + s1o) * topkN_;
    } else {
        topkBase = (static_cast<uint64_t>(bIdx) * s1Size_ + s1o) * topkN_;
    }

    // --- cur_pos for this query row ---
    int32_t curPosVal = -1;
    if (hasCurPos_) {
        curPosVal = curPosGm_.GetValue(qIdx);
    }

    // --- Actual KV length for this batch ---
    uint32_t actS2 = s2Size_;
    if constexpr (IS_PA || LAYOUT_Q == SKGLayoutKernel::TND) {
        if (hasActSeqLenKV_) {
            if constexpr (LAYOUT_Q == SKGLayoutKernel::TND) {
                actS2 = (bIdx == 0)
                    ? actSeqLenKVGm_.GetValue(0)
                    : actSeqLenKVGm_.GetValue(bIdx)
                      - actSeqLenKVGm_.GetValue(bIdx - 1);
            } else {
                actS2 = actSeqLenKVGm_.GetValue(bIdx);
            }
        }
    }

    // --- Inner loop ---
    LocalTensor<KV_T> kvUb = stageBuf_.Get<KV_T>();
    uint64_t kIdx     = 0;
    uint32_t ubRow    = 0;
    uint32_t pp       = 0;
    // Output base for this query row: qIdx * topkN_ (fixed, not stateful)
    uint64_t flatBase = static_cast<uint64_t>(qIdx) * topkN_;

    while (kIdx < topkN_) {
        int64_t tok0, tok1;
        GetTokenPair(tok0, tok1, topkBase, kIdx);
        if (tok0 < 0 && tok1 < 0) break;

        // Apply cur_pos self-masking
        if (hasCurPos_ && curPosVal >= 0) {
            if (tok0 == static_cast<int64_t>(curPosVal)) tok0 = -1;
            if (tok1 == static_cast<int64_t>(curPosVal)) tok1 = -1;
        }

        // Clip to valid KV range
        if (tok0 >= static_cast<int64_t>(actS2)) tok0 = -1;
        if (tok1 >= static_cast<int64_t>(actS2)) tok1 = -1;

        if (tok0 < 0 && tok1 < 0) { kIdx += 2; continue; }

        // DMA-in: GM → UB
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pp]);
        uint32_t copied = CopyInKvPair(kvUb, ubRow, tok0, tok1, bIdx);
        if (copied == 0) {
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pp]);
            kIdx += 2; continue;
        }
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pp]);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[pp]);

        // DMA-out: UB → two separate output tensors.
        // nope rows at flatBase..flatBase+copied-1, rope rows similarly.
        // Invalid slots (beyond valid rows) are zero-initialised by caller.
        DataCopy(outCtkvGm_[flatBase * D_NOPE],
                 kvUb[ubRow * D_COMBINED],
                 copied * D_NOPE);
        DataCopy(outKpeGm_[flatBase * D_ROPE],
                 kvUb[ubRow * D_COMBINED + D_NOPE],
                 copied * D_ROPE);

        flatBase += copied;
        kIdx     += 2;
        ubRow    += copied;
        if (ubRow >= MAX_ROWS_PER_LOOP) ubRow = 0;
        pp ^= 1;
    }
}

template <typename KV_T, bool IS_PA, SKGLayoutKernel LAYOUT_Q>
__aicore__ inline void
SparseKvGatherKernel<KV_T, IS_PA, SKGLayoutKernel LAYOUT_Q>::Process()
{
    if (coreIdx_ >= numCores_) return;

    uint32_t qStart = coreIdx_ * groupsPerCore_;
    uint32_t qEnd   = qStart + groupsPerCore_;
    if (qEnd > numActual_) qEnd = numActual_;

    for (uint32_t q = qStart; q < qEnd; ++q) {
        ProcessOneGroup(q);
    }
}

}  // namespace BaseApi
#endif  // SPARSE_KV_GATHER_KERNEL_H
