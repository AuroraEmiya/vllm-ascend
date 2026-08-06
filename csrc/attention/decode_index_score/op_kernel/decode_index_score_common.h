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
 * \file decode_index_score_common.h
 * \brief Common constants, tiling mirror (ConstInfo) and the score-range
 *        boundary math shared by the cube and vector services.
 *
 * The boundary math below mirrors `_decode_index_score_kernel` in
 * vllm_ascend/models/minimax_m3/ops/msa_m3_triton.py exactly:
 *   - init range   [chunkStart, initEnd)     -> 1e30 sentinel
 *   - normal range [normalStart, normalFullEnd) full blocks -> dot-max over 128
 *   - partial block (id == fullBlockCount)  -> dot-max over first partialValidPos
 *   - local range  [localBegin, validEnd)   -> 1e29 sentinel
 *   - tail range   [tailBegin, chunkEnd)    -> -inf sentinel
 */

#ifndef DECODE_INDEX_SCORE_COMMON_H_
#define DECODE_INDEX_SCORE_COMMON_H_

#include "kernel_operator.h"

namespace DecodeIndexScoreKernel {
using namespace AscendC;

constexpr uint32_t SPARSE_BLOCK = 128;   // tokens per sparse block (page)
constexpr uint32_t BLOCK_CUBE_SIZE = 16; // cube L0 alignment granularity
constexpr float SCORE_INIT = 1e30f;      // init-block priority sentinel
constexpr float SCORE_LOCAL = 1e29f;     // local-window priority sentinel
constexpr float SCORE_NEG_INF = -3.402823466e+38f; // -inf invalid tail sentinel

// Cross-core sync event ids (AIC <-> AIV). Single hand-off per program.
constexpr uint32_t CROSS_CV_EVENT = 0; // cube -> vector
constexpr uint32_t CROSS_VC_EVENT = 1; // vector -> cube
constexpr uint32_t DI_SYNC_MODE = 2;   // cross-core sync mode (as lightning_indexer)

struct ConstInfo {
    uint32_t totalQ;
    uint32_t numIdxHeads;
    uint32_t headDim;
    uint32_t decodeQueryLen;
    uint32_t blockOffset;
    uint32_t initBlocks;
    uint32_t localBlocks;
    uint32_t scoreBlockStride;   // == SCORE_BLOCK_COUNT
    uint32_t numChunks;
    uint32_t localMaxBlocks;     // block_table row stride
    uint32_t wsBlocksPerChunk;   // workspace: max blocks a chunk can own
    uint32_t wsStripSize;        // workspace: per-block strip = align16(H)*128 floats
};

struct RangeInfo {
    uint32_t requestId;
    uint32_t queryOffset;
    uint32_t localKvLength;
    uint32_t validBlockCount;
    uint32_t fullBlockCount;
    // chunk partition
    uint32_t chunkStart;
    uint32_t chunkEnd;
    // score ranges (shard-local block ids)
    uint32_t initEnd;        // [chunkStart, initEnd)      -> 1e30
    uint32_t normalStart;    // normal full blocks
    uint32_t normalFullEnd;  // [normalStart, normalFullEnd) -> dot-max
    uint32_t normalEnd;      // normal range incl. partial
    uint32_t localBegin;     // [localBegin, validEnd)     -> 1e29
    uint32_t validEnd;
    uint32_t tailBegin;      // [tailBegin, chunkEnd)      -> -inf
    bool hasPartial;         // partial block is inside the normal range
    uint32_t partialValidPos; // number of valid positions in the partial block
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 MinT(T1 a, T2 b)
{
    return (a > b) ? (T1)b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 MaxT(T1 a, T2 b)
{
    return (a > b) ? a : (T1)b;
}

// Mirrors the Triton kernel's scalar boundary computation. All block ids are
// shard-local (global block id = local id + block_offset).
__aicore__ inline RangeInfo ComputeRanges(uint32_t queryId, uint32_t chunkId, const ConstInfo &ci,
                                          int32_t localSeqLen, int32_t globalSeqLen)
{
    RangeInfo r;
    r.requestId = queryId / ci.decodeQueryLen;
    r.queryOffset = queryId % ci.decodeQueryLen;

    int32_t queryPos = globalSeqLen - (int32_t)ci.decodeQueryLen + (int32_t)r.queryOffset;
    int32_t globalKvLength = queryPos + 1;
    globalKvLength = MaxT(globalKvLength, 0);
    int32_t localKvLength = globalKvLength - (int32_t)(ci.blockOffset * SPARSE_BLOCK);
    localKvLength = MaxT(localKvLength, 0);
    localKvLength = MinT(localKvLength, localSeqLen);
    r.localKvLength = (uint32_t)localKvLength;

    r.validBlockCount = (r.localKvLength + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
    r.fullBlockCount = r.localKvLength / SPARSE_BLOCK;
    uint32_t validGlobalBlockCount = ((uint32_t)globalKvLength + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
    uint32_t localStartGlobal = MaxT(validGlobalBlockCount, ci.localBlocks) - ci.localBlocks;

    // exact quotient/remainder chunk partition (differ by at most one block)
    uint32_t blocksPerChunk = ci.scoreBlockStride / ci.numChunks;
    uint32_t extraChunks = ci.scoreBlockStride % ci.numChunks;
    r.chunkStart = chunkId * blocksPerChunk + MinT(chunkId, extraChunks);
    r.chunkEnd = r.chunkStart + blocksPerChunk + (chunkId < extraChunks ? 1u : 0u);

    uint32_t validEnd = MinT(r.chunkEnd, r.validBlockCount);
    r.validEnd = validEnd;

    // init range: global ids [0, min(init_blocks, local_start_global))
    uint32_t initGlobalEnd = MinT(ci.initBlocks, localStartGlobal);
    uint32_t initEnd = (uint32_t)MaxT((int32_t)initGlobalEnd - (int32_t)ci.blockOffset, 0);
    r.initEnd = MinT(validEnd, initEnd);

    // normal range: between init window and local window
    uint32_t normalStart = MaxT(r.chunkStart, (uint32_t)MaxT((int32_t)ci.initBlocks - (int32_t)ci.blockOffset, 0));
    uint32_t normalEnd = MinT(validEnd, (uint32_t)MaxT((int32_t)localStartGlobal - (int32_t)ci.blockOffset, 0));
    r.normalStart = normalStart;
    r.normalEnd = normalEnd;
    r.normalFullEnd = MinT(normalEnd, r.fullBlockCount);

    // local range: last local_blocks valid blocks (1e29 overrides init on overlap)
    uint32_t localBegin = MaxT(r.chunkStart, (uint32_t)MaxT((int32_t)localStartGlobal - (int32_t)ci.blockOffset, 0));
    r.localBegin = localBegin;

    // tail: beyond the valid block count
    r.tailBegin = MaxT(r.chunkStart, r.validBlockCount);

    // partial block (id == fullBlockCount) when the boundary splits a block
    r.partialValidPos = r.localKvLength - r.fullBlockCount * SPARSE_BLOCK;
    r.hasPartial = (r.fullBlockCount < r.validBlockCount) && (r.fullBlockCount >= r.normalStart) &&
                   (r.fullBlockCount < r.normalEnd);
    return r;
}

} // namespace DecodeIndexScoreKernel
#endif // DECODE_INDEX_SCORE_COMMON_H_
