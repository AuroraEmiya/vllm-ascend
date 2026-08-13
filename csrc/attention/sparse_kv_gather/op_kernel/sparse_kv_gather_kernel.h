/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef SPARSE_KV_GATHER_KERNEL_H
#define SPARSE_KV_GATHER_KERNEL_H

#include "kernel_operator.h"

namespace BaseApi {

using namespace AscendC;

#ifndef SKG_METADATA_PLAN_ROWS
#define SKG_METADATA_PLAN_ROWS 256
#endif

namespace skg {
constexpr uint32_t CTKV_DIM = 512;
constexpr uint32_t KPE_DIM = 64;
constexpr uint32_t COMBINED_DIM = CTKV_DIM + KPE_DIM;
constexpr uint32_t BLOCK_SIZE = 128;
constexpr uint32_t BLOCK_SHIFT = 7;
constexpr uint32_t BLOCK_MASK = BLOCK_SIZE - 1;
constexpr uint32_t STAGE_ROWS = 16;
constexpr uint32_t PAIR_WIDTH = 2;
constexpr uint32_t STAGE_NUM = 2;
constexpr uint32_t CTKV_STAGE_ELEMS = STAGE_ROWS * CTKV_DIM;
constexpr uint32_t KPE_STAGE_ELEMS = STAGE_ROWS * KPE_DIM;
constexpr uint32_t STAGE_ELEMS = CTKV_STAGE_ELEMS + KPE_STAGE_ELEMS;
constexpr uint32_t CTKV_ROW_BYTES = CTKV_DIM * sizeof(uint16_t);
constexpr uint32_t KPE_ROW_BYTES = KPE_DIM * sizeof(uint16_t);
constexpr uint64_t MAX_DMA_STRIDE = 0xFFFFFFFFULL;
constexpr uint32_t PLAN_ROWS = static_cast<uint32_t>(SKG_METADATA_PLAN_ROWS);
constexpr uint32_t GROUP_MIN_CORE_ROWS = 128;
constexpr uint32_t BLOCK_TABLE_CACHE_ENTRIES = 2048;
constexpr bool ENABLE_PAIR = true;

static_assert(PLAN_ROWS >= STAGE_ROWS, "SKG_METADATA_PLAN_ROWS must be >= 16");
static_assert(PLAN_ROWS % STAGE_ROWS == 0, "SKG_METADATA_PLAN_ROWS must be multiple of 16");
static_assert(PLAN_ROWS <= 2048, "SKG_METADATA_PLAN_ROWS must be <= 2048");
}  // namespace skg

class SparseKvGatherKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t *pagedCtkv, __gm__ uint8_t *pagedKpe,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndices,
        __gm__ uint8_t *curPos, __gm__ uint8_t *outCtkv,
        __gm__ uint8_t *outKpe, uint32_t numBlocks, uint32_t maxBlocks,
        uint32_t topkN, uint64_t totalSlots, uint64_t slotsPerCore,
        uint32_t usedCoreNum, TPipe *pipe);

    __aicore__ inline void Process();

private:
    // Metadata.
    __aicore__ inline int64_t ResolvePhysicalFromGm(uint32_t queryIdx, int64_t logicalToken) const;
    __aicore__ inline int64_t ResolvePhysicalFromUb(
        int64_t logicalToken, const LocalTensor<int32_t> &blockTableUb) const;
    __aicore__ inline void PrefetchTopk(
        uint64_t flatSlot, uint32_t rows, LocalTensor<int32_t> &topkUb);
    __aicore__ inline void PrefetchBlockTable(
        uint32_t queryIdx, LocalTensor<int32_t> &blockTableUb);
    __aicore__ inline bool ResolveChunkGm(
        uint64_t flatSlot, uint32_t queryIdx, uint32_t rows,
        int64_t currentPos, int64_t *physicalTokens) const;
    __aicore__ inline bool ResolveChunkUb(
        uint32_t planOffset, uint32_t rows, int64_t currentPos,
        const LocalTensor<int32_t> &topkUb,
        const LocalTensor<int32_t> &blockTableUb,
        int64_t *physicalTokens) const;

    // DMA primitives.
    __aicore__ inline void WriteZero(
        uint64_t flatSlot, const LocalTensor<uint16_t> &zeroUb) const;
    __aicore__ inline bool CanPair(int64_t p0, int64_t p1) const;
    __aicore__ inline void LoadOne(
        uint32_t row, int64_t physicalToken,
        LocalTensor<uint16_t> &ctkvUb, LocalTensor<uint16_t> &kpeUb) const;
    __aicore__ inline void LoadPair(
        uint32_t row, int64_t p0, int64_t p1,
        LocalTensor<uint16_t> &ctkvUb, LocalTensor<uint16_t> &kpeUb) const;
    __aicore__ inline void GatherOne(
        uint64_t flatSlot, int64_t physicalToken, uint32_t stageIdx,
        LocalTensor<uint16_t> &stageUb);
    __aicore__ inline void GatherPair(
        uint64_t flatSlot, int64_t p0, int64_t p1, uint32_t stageIdx,
        LocalTensor<uint16_t> &stageUb);
    __aicore__ inline void GatherChunk(
        uint64_t flatSlot, uint32_t rows, const int64_t *physicalTokens,
        uint32_t stageIdx, LocalTensor<uint16_t> &stageUb);

    // Shared execution / stage lifecycle.
    __aicore__ inline void AcquireStage(uint32_t stageIdx, const bool *inFlight);
    __aicore__ inline void CommitStage(uint32_t &stageIdx, bool *inFlight);
    __aicore__ inline void DrainStages(const bool *inFlight);
    __aicore__ inline void ExecuteChunk(
        uint64_t flatSlot, uint32_t rows, const int64_t *physicalTokens,
        bool fullValid, uint32_t &stageIdx, bool *inFlight,
        LocalTensor<uint16_t> &stageUb, const LocalTensor<uint16_t> &zeroUb);

    // Scheduling.
    __aicore__ inline bool GetSlotRange(uint64_t &slotStart, uint64_t &slotEnd) const;
    __aicore__ inline void InitZero(LocalTensor<uint16_t> &zeroUb);
    __aicore__ inline void ProcessScalarMetadata();
    __aicore__ inline void ProcessCachedMetadata();

    GlobalTensor<uint16_t> pagedCtkvGm_, pagedKpeGm_, outCtkvGm_, outKpeGm_;
    GlobalTensor<int32_t> blockTableGm_, topkIndicesGm_, curPosGm_;

    uint32_t numBlocks_ = 0;
    uint32_t maxBlocks_ = 0;
    uint32_t topkN_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t coreIdx_ = 0;
    uint64_t totalSlots_ = 0;
    uint64_t slotsPerCore_ = 0;

    TPipe *pipe_ = nullptr;
    TBuf<> stageBuf_, zeroBuf_, topkPlanBuf_, blockTableCacheBuf_;
    TEventID mte2ToMte3_[skg::STAGE_NUM];
    TEventID mte3ToMte2_[skg::STAGE_NUM];
    TEventID vectorToMte3_;
    TEventID mte2ToScalar_;
};

__aicore__ inline void SparseKvGatherKernel::Init(
    __gm__ uint8_t *pagedCtkv, __gm__ uint8_t *pagedKpe,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *topkIndices,
    __gm__ uint8_t *curPos, __gm__ uint8_t *outCtkv,
    __gm__ uint8_t *outKpe, const uint32_t numBlocks,
    const uint32_t maxBlocks, const uint32_t topkN,
    const uint64_t totalSlots, const uint64_t slotsPerCore,
    const uint32_t usedCoreNum, TPipe *pipe)
{
    pipe_ = pipe;
    numBlocks_ = numBlocks;
    maxBlocks_ = maxBlocks;
    topkN_ = topkN;
    totalSlots_ = totalSlots;
    slotsPerCore_ = slotsPerCore;
    usedCoreNum_ = usedCoreNum;
    coreIdx_ = GetBlockIdx();

    pagedCtkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pagedCtkv));
    pagedKpeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pagedKpe));
    outCtkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(outCtkv));
    outKpeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(outKpe));
    blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    topkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(topkIndices));
    curPosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(curPos));

    pipe_->InitBuffer(stageBuf_, skg::STAGE_NUM * skg::STAGE_ELEMS * sizeof(uint16_t));
    pipe_->InitBuffer(zeroBuf_, skg::COMBINED_DIM * sizeof(uint16_t));
    pipe_->InitBuffer(topkPlanBuf_, skg::PLAN_ROWS * sizeof(int32_t));
    pipe_->InitBuffer(
        blockTableCacheBuf_, skg::BLOCK_TABLE_CACHE_ENTRIES * sizeof(int32_t));

    for (uint32_t i = 0; i < skg::STAGE_NUM; ++i) {
        mte2ToMte3_[i] = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2_[i] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }
    vectorToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
    mte2ToScalar_ = pipe_->AllocEventID<HardEvent::MTE2_S>();
}

// ---- metadata ----------------------------------------------------------------

__aicore__ inline int64_t SparseKvGatherKernel::ResolvePhysicalFromGm(
    const uint32_t queryIdx, const int64_t logicalToken) const
{
    if (logicalToken < 0) return -1;

    const uint64_t logicalBlock =
        static_cast<uint64_t>(logicalToken) >> skg::BLOCK_SHIFT;
    if (logicalBlock >= maxBlocks_) return -1;

    const uint64_t tableOffset =
        static_cast<uint64_t>(queryIdx) * maxBlocks_ + logicalBlock;
    const int64_t physicalBlock =
        static_cast<int64_t>(blockTableGm_.GetValue(tableOffset));
    if (physicalBlock < 0 || physicalBlock >= static_cast<int64_t>(numBlocks_)) return -1;

    return physicalBlock * static_cast<int64_t>(skg::BLOCK_SIZE) +
           static_cast<int64_t>(static_cast<uint64_t>(logicalToken) & skg::BLOCK_MASK);
}

__aicore__ inline int64_t SparseKvGatherKernel::ResolvePhysicalFromUb(
    const int64_t logicalToken, const LocalTensor<int32_t> &blockTableUb) const
{
    if (logicalToken < 0) return -1;

    const uint64_t logicalBlock =
        static_cast<uint64_t>(logicalToken) >> skg::BLOCK_SHIFT;
    if (logicalBlock >= maxBlocks_) return -1;

    const int64_t physicalBlock = static_cast<int64_t>(
        blockTableUb.GetValue(static_cast<uint32_t>(logicalBlock)));
    if (physicalBlock < 0 || physicalBlock >= static_cast<int64_t>(numBlocks_)) return -1;

    return physicalBlock * static_cast<int64_t>(skg::BLOCK_SIZE) +
           static_cast<int64_t>(static_cast<uint64_t>(logicalToken) & skg::BLOCK_MASK);
}

__aicore__ inline void SparseKvGatherKernel::PrefetchTopk(
    const uint64_t flatSlot, const uint32_t rows, LocalTensor<int32_t> &topkUb)
{
    DataCopyExtParams p{};
    p.blockCount = 1;
    p.blockLen = rows * sizeof(int32_t);
    DataCopyPadExtParams<int32_t> pad{};
    pad.isPad = false;
    DataCopyPad(topkUb, topkIndicesGm_[flatSlot], p, pad);
    SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
}

__aicore__ inline void SparseKvGatherKernel::PrefetchBlockTable(
    const uint32_t queryIdx, LocalTensor<int32_t> &blockTableUb)
{
    DataCopyExtParams p{};
    p.blockCount = 1;
    p.blockLen = maxBlocks_ * sizeof(int32_t);
    DataCopyPadExtParams<int32_t> pad{};
    pad.isPad = false;
    DataCopyPad(
        blockTableUb,
        blockTableGm_[static_cast<uint64_t>(queryIdx) * maxBlocks_],
        p, pad);
    SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
    WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
}

__aicore__ inline bool SparseKvGatherKernel::ResolveChunkGm(
    const uint64_t flatSlot, const uint32_t queryIdx, const uint32_t rows,
    const int64_t currentPos, int64_t *physicalTokens) const
{
    bool fullValid = true;
    for (uint32_t row = 0; row < rows; ++row) {
        const int64_t logicalToken =
            static_cast<int64_t>(topkIndicesGm_.GetValue(flatSlot + row));
        int64_t physicalToken = -1;
        if (logicalToken >= 0 && logicalToken != currentPos) {
            physicalToken = ResolvePhysicalFromGm(queryIdx, logicalToken);
        }
        physicalTokens[row] = physicalToken;
        fullValid = fullValid && physicalToken >= 0;
    }
    return fullValid;
}

__aicore__ inline bool SparseKvGatherKernel::ResolveChunkUb(
    const uint32_t planOffset, const uint32_t rows, const int64_t currentPos,
    const LocalTensor<int32_t> &topkUb, const LocalTensor<int32_t> &blockTableUb,
    int64_t *physicalTokens) const
{
    bool fullValid = true;
    for (uint32_t row = 0; row < rows; ++row) {
        const int64_t logicalToken =
            static_cast<int64_t>(topkUb.GetValue(planOffset + row));
        int64_t physicalToken = -1;
        if (logicalToken >= 0 && logicalToken != currentPos) {
            physicalToken = ResolvePhysicalFromUb(logicalToken, blockTableUb);
        }
        physicalTokens[row] = physicalToken;
        fullValid = fullValid && physicalToken >= 0;
    }
    return fullValid;
}

// ---- DMA ---------------------------------------------------------------------

__aicore__ inline void SparseKvGatherKernel::WriteZero(
    const uint64_t flatSlot, const LocalTensor<uint16_t> &zeroUb) const
{
    DataCopy(outCtkvGm_[flatSlot * skg::CTKV_DIM], zeroUb, skg::CTKV_DIM);
    DataCopy(
        outKpeGm_[flatSlot * skg::KPE_DIM],
        zeroUb[skg::CTKV_DIM],
        skg::KPE_DIM);
}

__aicore__ inline bool SparseKvGatherKernel::CanPair(
    const int64_t p0, const int64_t p1) const
{
    if (p1 <= p0) return false;
    const uint64_t gap = static_cast<uint64_t>(p1 - p0);
    return (gap - 1U) * skg::CTKV_ROW_BYTES <= skg::MAX_DMA_STRIDE;
}

__aicore__ inline void SparseKvGatherKernel::LoadOne(
    const uint32_t row, const int64_t physicalToken,
    LocalTensor<uint16_t> &ctkvUb, LocalTensor<uint16_t> &kpeUb) const
{
    DataCopy(
        ctkvUb[static_cast<uint64_t>(row) * skg::CTKV_DIM],
        pagedCtkvGm_[static_cast<uint64_t>(physicalToken) * skg::CTKV_DIM],
        skg::CTKV_DIM);
    DataCopy(
        kpeUb[static_cast<uint64_t>(row) * skg::KPE_DIM],
        pagedKpeGm_[static_cast<uint64_t>(physicalToken) * skg::KPE_DIM],
        skg::KPE_DIM);
}

__aicore__ inline void SparseKvGatherKernel::LoadPair(
    const uint32_t row, const int64_t p0, const int64_t p1,
    LocalTensor<uint16_t> &ctkvUb, LocalTensor<uint16_t> &kpeUb) const
{
    const uint64_t gap = static_cast<uint64_t>(p1 - p0);

    DataCopyExtParams ctkv{};
    ctkv.blockCount = skg::PAIR_WIDTH;
    ctkv.blockLen = skg::CTKV_ROW_BYTES;
    ctkv.srcStride = static_cast<uint32_t>((gap - 1U) * skg::CTKV_ROW_BYTES);

    DataCopyExtParams kpe{};
    kpe.blockCount = skg::PAIR_WIDTH;
    kpe.blockLen = skg::KPE_ROW_BYTES;
    kpe.srcStride = static_cast<uint32_t>((gap - 1U) * skg::KPE_ROW_BYTES);

    DataCopyPadExtParams<uint16_t> pad{};
    pad.isPad = false;

    DataCopyPad(
        ctkvUb[static_cast<uint64_t>(row) * skg::CTKV_DIM],
        pagedCtkvGm_[static_cast<uint64_t>(p0) * skg::CTKV_DIM],
        ctkv, pad);
    DataCopyPad(
        kpeUb[static_cast<uint64_t>(row) * skg::KPE_DIM],
        pagedKpeGm_[static_cast<uint64_t>(p0) * skg::KPE_DIM],
        kpe, pad);
}

__aicore__ inline void SparseKvGatherKernel::GatherOne(
    const uint64_t flatSlot, const int64_t physicalToken,
    const uint32_t stageIdx, LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunk = stageUb[stageIdx * skg::STAGE_ELEMS];
    LocalTensor<uint16_t> ctkv = chunk;
    LocalTensor<uint16_t> kpe = chunk[skg::CTKV_STAGE_ELEMS];

    LoadOne(0, physicalToken, ctkv, kpe);
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);

    DataCopy(outCtkvGm_[flatSlot * skg::CTKV_DIM], ctkv, skg::CTKV_DIM);
    DataCopy(outKpeGm_[flatSlot * skg::KPE_DIM], kpe, skg::KPE_DIM);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[stageIdx]);
}

__aicore__ inline void SparseKvGatherKernel::GatherPair(
    const uint64_t flatSlot, const int64_t p0, const int64_t p1,
    const uint32_t stageIdx, LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunk = stageUb[stageIdx * skg::STAGE_ELEMS];
    LocalTensor<uint16_t> ctkv = chunk;
    LocalTensor<uint16_t> kpe = chunk[skg::CTKV_STAGE_ELEMS];

    LoadPair(0, p0, p1, ctkv, kpe);
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);

    DataCopy(
        outCtkvGm_[flatSlot * skg::CTKV_DIM],
        ctkv,
        skg::PAIR_WIDTH * skg::CTKV_DIM);
    DataCopy(
        outKpeGm_[flatSlot * skg::KPE_DIM],
        kpe,
        skg::PAIR_WIDTH * skg::KPE_DIM);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[stageIdx]);
}

__aicore__ inline void SparseKvGatherKernel::GatherChunk(
    const uint64_t flatSlot, const uint32_t rows,
    const int64_t *physicalTokens, const uint32_t stageIdx,
    LocalTensor<uint16_t> &stageUb)
{
    LocalTensor<uint16_t> chunk = stageUb[stageIdx * skg::STAGE_ELEMS];
    LocalTensor<uint16_t> ctkv = chunk;
    LocalTensor<uint16_t> kpe = chunk[skg::CTKV_STAGE_ELEMS];

    uint32_t row = 0;
    while (row + 1U < rows) {
        if (skg::ENABLE_PAIR && CanPair(physicalTokens[row], physicalTokens[row + 1U])) {
            LoadPair(row, physicalTokens[row], physicalTokens[row + 1U], ctkv, kpe);
            row += 2U;
        } else {
            LoadOne(row, physicalTokens[row], ctkv, kpe);
            ++row;
        }
    }
    if (row < rows) LoadOne(row, physicalTokens[row], ctkv, kpe);

    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3_[stageIdx]);

    DataCopy(
        outCtkvGm_[flatSlot * skg::CTKV_DIM],
        ctkv,
        static_cast<uint64_t>(rows) * skg::CTKV_DIM);
    DataCopy(
        outKpeGm_[flatSlot * skg::KPE_DIM],
        kpe,
        static_cast<uint64_t>(rows) * skg::KPE_DIM);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[stageIdx]);
}

// ---- shared executor ---------------------------------------------------------

__aicore__ inline void SparseKvGatherKernel::AcquireStage(
    const uint32_t stageIdx, const bool *inFlight)
{
    if (inFlight[stageIdx]) {
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[stageIdx]);
    }
}

__aicore__ inline void SparseKvGatherKernel::CommitStage(
    uint32_t &stageIdx, bool *inFlight)
{
    inFlight[stageIdx] = true;
    stageIdx ^= 1U;
}

__aicore__ inline void SparseKvGatherKernel::DrainStages(const bool *inFlight)
{
    for (uint32_t i = 0; i < skg::STAGE_NUM; ++i) {
        if (inFlight[i]) WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_[i]);
    }
}

__aicore__ inline void SparseKvGatherKernel::ExecuteChunk(
    const uint64_t flatSlot, const uint32_t rows, const int64_t *physicalTokens,
    const bool fullValid, uint32_t &stageIdx, bool *inFlight,
    LocalTensor<uint16_t> &stageUb, const LocalTensor<uint16_t> &zeroUb)
{
    if (fullValid) {
        AcquireStage(stageIdx, inFlight);
        GatherChunk(flatSlot, rows, physicalTokens, stageIdx, stageUb);
        CommitStage(stageIdx, inFlight);
        return;
    }

    uint32_t row = 0;
    while (row < rows) {
        const bool pair =
            row + 1U < rows && skg::ENABLE_PAIR &&
            physicalTokens[row] >= 0 && physicalTokens[row + 1U] >= 0 &&
            CanPair(physicalTokens[row], physicalTokens[row + 1U]);

        if (pair) {
            AcquireStage(stageIdx, inFlight);
            GatherPair(
                flatSlot + row, physicalTokens[row], physicalTokens[row + 1U],
                stageIdx, stageUb);
            CommitStage(stageIdx, inFlight);
            row += 2U;
        } else {
            if (physicalTokens[row] < 0) {
                WriteZero(flatSlot + row, zeroUb);
            } else {
                AcquireStage(stageIdx, inFlight);
                GatherOne(flatSlot + row, physicalTokens[row], stageIdx, stageUb);
                CommitStage(stageIdx, inFlight);
            }
            ++row;
        }
    }
}

// ---- scheduling --------------------------------------------------------------

__aicore__ inline bool SparseKvGatherKernel::GetSlotRange(
    uint64_t &slotStart, uint64_t &slotEnd) const
{
    slotStart = static_cast<uint64_t>(coreIdx_) * slotsPerCore_;
    slotEnd = slotStart + slotsPerCore_;
    if (slotEnd > totalSlots_) slotEnd = totalSlots_;
    return slotStart < slotEnd;
}

__aicore__ inline void SparseKvGatherKernel::InitZero(LocalTensor<uint16_t> &zeroUb)
{
    Duplicate(zeroUb, static_cast<uint16_t>(0), skg::COMBINED_DIM);
    SetFlag<HardEvent::V_MTE3>(vectorToMte3_);
    WaitFlag<HardEvent::V_MTE3>(vectorToMte3_);
}

__aicore__ inline void SparseKvGatherKernel::ProcessScalarMetadata()
{
    uint64_t slotStart, slotEnd;
    if (!GetSlotRange(slotStart, slotEnd)) return;

    LocalTensor<uint16_t> stageUb = stageBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> zeroUb = zeroBuf_.Get<uint16_t>();
    InitZero(zeroUb);

    bool inFlight[skg::STAGE_NUM] = {false, false};
    uint32_t stageIdx = 0;
    uint32_t queryIdx = static_cast<uint32_t>(slotStart / topkN_);
    uint32_t slotInQuery =
        static_cast<uint32_t>(slotStart - static_cast<uint64_t>(queryIdx) * topkN_);
    int64_t currentPos = static_cast<int64_t>(curPosGm_.GetValue(queryIdx));

    for (uint64_t flatSlot = slotStart; flatSlot < slotEnd;) {
        const uint64_t leftCore = slotEnd - flatSlot;
        const uint32_t leftQuery = topkN_ - slotInQuery;
        uint32_t rows = skg::STAGE_ROWS;
        if (leftCore < rows) rows = static_cast<uint32_t>(leftCore);
        if (leftQuery < rows) rows = leftQuery;

        int64_t physicalTokens[skg::STAGE_ROWS];
        const bool fullValid =
            ResolveChunkGm(flatSlot, queryIdx, rows, currentPos, physicalTokens);
        ExecuteChunk(
            flatSlot, rows, physicalTokens, fullValid,
            stageIdx, inFlight, stageUb, zeroUb);

        flatSlot += rows;
        slotInQuery += rows;
        if (slotInQuery == topkN_ && flatSlot < slotEnd) {
            slotInQuery = 0;
            ++queryIdx;
            currentPos = static_cast<int64_t>(curPosGm_.GetValue(queryIdx));
        }
    }
    DrainStages(inFlight);
}

__aicore__ inline void SparseKvGatherKernel::ProcessCachedMetadata()
{
    uint64_t slotStart, slotEnd;
    if (!GetSlotRange(slotStart, slotEnd)) return;

    LocalTensor<uint16_t> stageUb = stageBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> zeroUb = zeroBuf_.Get<uint16_t>();
    LocalTensor<int32_t> topkUb = topkPlanBuf_.Get<int32_t>();
    LocalTensor<int32_t> blockTableUb = blockTableCacheBuf_.Get<int32_t>();
    InitZero(zeroUb);

    bool inFlight[skg::STAGE_NUM] = {false, false};
    uint32_t stageIdx = 0;
    uint32_t queryIdx = static_cast<uint32_t>(slotStart / topkN_);
    uint32_t slotInQuery =
        static_cast<uint32_t>(slotStart - static_cast<uint64_t>(queryIdx) * topkN_);
    int64_t currentPos = static_cast<int64_t>(curPosGm_.GetValue(queryIdx));
    PrefetchBlockTable(queryIdx, blockTableUb);

    for (uint64_t flatSlot = slotStart; flatSlot < slotEnd;) {
        const uint64_t leftCore = slotEnd - flatSlot;
        const uint32_t leftQuery = topkN_ - slotInQuery;
        uint32_t planRows = skg::PLAN_ROWS;
        if (leftCore < planRows) planRows = static_cast<uint32_t>(leftCore);
        if (leftQuery < planRows) planRows = leftQuery;

        PrefetchTopk(flatSlot, planRows, topkUb);

        for (uint32_t planOffset = 0; planOffset < planRows;) {
            const uint32_t leftPlan = planRows - planOffset;
            const uint32_t rows =
                leftPlan < skg::STAGE_ROWS ? leftPlan : skg::STAGE_ROWS;

            int64_t physicalTokens[skg::STAGE_ROWS];
            const bool fullValid = ResolveChunkUb(
                planOffset, rows, currentPos, topkUb, blockTableUb, physicalTokens);
            ExecuteChunk(
                flatSlot + planOffset, rows, physicalTokens, fullValid,
                stageIdx, inFlight, stageUb, zeroUb);
            planOffset += rows;
        }

        flatSlot += planRows;
        slotInQuery += planRows;
        if (slotInQuery == topkN_ && flatSlot < slotEnd) {
            slotInQuery = 0;
            ++queryIdx;
            currentPos = static_cast<int64_t>(curPosGm_.GetValue(queryIdx));
            PrefetchBlockTable(queryIdx, blockTableUb);
        }
    }
    DrainStages(inFlight);
}

__aicore__ inline void SparseKvGatherKernel::Process()
{
    if (coreIdx_ >= usedCoreNum_) return;

    const bool useCached =
        slotsPerCore_ >= skg::GROUP_MIN_CORE_ROWS &&
        maxBlocks_ <= skg::BLOCK_TABLE_CACHE_ENTRIES;

    if (useCached) ProcessCachedMetadata();
    else ProcessScalarMetadata();
}

}  // namespace BaseApi

#endif  // SPARSE_KV_GATHER_KERNEL_H
