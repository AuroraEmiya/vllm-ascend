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
 * \file decode_index_score_tiling.cpp
 * \brief Tiling for the decode indexer block-score op.
 */

#include "decode_index_score_tiling.h"
#include "../op_kernel/decode_index_score_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t SCORE_ELEM_SIZE = 4; // fp32
constexpr uint32_t BLOCK_TOKEN_SIZE = SPARSE_BLOCK_SIZE;

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

inline uint32_t AlignUp(uint32_t a, uint32_t align)
{
    return ((a + align - 1) / align) * align;
}
} // namespace

static ge::graphStatus TilingForDecodeIndexScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DecodeIndexScore", "Tiling context is null."), return ge::GRAPH_FAILED);

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    const gert::Shape *seqLensShape = context->GetInputShape(SEQ_LENS_INDEX);
    const gert::Shape *blockTableShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLensShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTableShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *decodeQueryLenPtr = attrs->GetInt(ATTR_DECODE_QUERY_LEN_INDEX);
    const int64_t *blockOffsetPtr = attrs->GetInt(ATTR_BLOCK_OFFSET_INDEX);
    const int64_t *initBlocksPtr = attrs->GetInt(ATTR_INIT_BLOCKS_INDEX);
    const int64_t *localBlocksPtr = attrs->GetInt(ATTR_LOCAL_BLOCKS_INDEX);
    const int64_t *scoreBlockStridePtr = attrs->GetInt(ATTR_SCORE_BLOCK_STRIDE_INDEX);
    const int64_t *numChunksPtr = attrs->GetInt(ATTR_NUM_CHUNKS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, decodeQueryLenPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockOffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, initBlocksPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, localBlocksPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreBlockStridePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, numChunksPtr);

    OP_CHECK_IF(queryShape->GetDimNum() != 3, OP_LOGE(context, "query must be [total_q, H, D]."),
                return ge::GRAPH_FAILED);
    const uint32_t totalQ = static_cast<uint32_t>(queryShape->GetDim(0));
    const uint32_t numIdxHeads = static_cast<uint32_t>(queryShape->GetDim(1));
    const uint32_t headDim = static_cast<uint32_t>(queryShape->GetDim(2));
    const uint32_t numReqs = static_cast<uint32_t>(seqLensShape->GetDim(0));
    const uint32_t localMaxBlocks = static_cast<uint32_t>(blockTableShape->GetDim(1));
    const int64_t decodeQueryLen = *decodeQueryLenPtr;
    const int64_t blockOffset = *blockOffsetPtr;
    const int64_t initBlocks = *initBlocksPtr;
    const int64_t localBlocks = *localBlocksPtr;
    const uint32_t scoreBlockStride = static_cast<uint32_t>(*scoreBlockStridePtr);
    const uint32_t numChunks = static_cast<uint32_t>(*numChunksPtr);

    OP_CHECK_IF(decodeQueryLen <= 0 || numChunks == 0 || scoreBlockStride == 0,
                OP_LOGE(context, "decode_query_len/num_chunks/score_block_stride must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalQ != numReqs * static_cast<uint32_t>(decodeQueryLen),
                OP_LOGE(context, "total_q must equal num_reqs * decode_query_len."), return ge::GRAPH_FAILED);

    // grid: (total_q, num_chunks) -> blockDim = total_q * num_chunks
    uint32_t blockDim = totalQ * numChunks;
    context->SetBlockDim(blockDim);

    // workspace: cube -> vector hand-off of per-block mmad results.
    // per program: wsBlocksPerChunk strips of [align16(H), 128] fp32.
    uint32_t wsBlocksPerChunk = CeilDiv(scoreBlockStride, numChunks);
    uint32_t wsStripSize = AlignUp(numIdxHeads, ALIGN_16) * BLOCK_TOKEN_SIZE;
    uint64_t perProgramBytes = static_cast<uint64_t>(wsBlocksPerChunk) * wsStripSize * SCORE_ELEM_SIZE;
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = static_cast<size_t>(blockDim) * perProgramBytes;

    // tiling data
    DecodeIndexScoreTilingData tilingData;
    tilingData.set_totalQ(totalQ);
    tilingData.set_numIdxHeads(numIdxHeads);
    tilingData.set_headDim(headDim);
    tilingData.set_decodeQueryLen(static_cast<uint32_t>(decodeQueryLen));
    tilingData.set_blockOffset(static_cast<uint32_t>(blockOffset));
    tilingData.set_initBlocks(static_cast<uint32_t>(initBlocks));
    tilingData.set_localBlocks(static_cast<uint32_t>(localBlocks));
    tilingData.set_scoreBlockStride(scoreBlockStride);
    tilingData.set_numChunks(numChunks);
    tilingData.set_localMaxBlocks(localMaxBlocks);
    tilingData.set_wsBlocksPerChunk(wsBlocksPerChunk);
    tilingData.set_wsStripSize(wsStripSize);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // tiling key: input dtype pair (query / index-k cache share the dtype)
    const gert::CompileTimeTensorDesc *queryDesc = context->GetInputTensorDesc(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    ge::DataType inputQType = queryDesc->GetDataType();
    uint32_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint32_t>(inputQType), static_cast<uint32_t>(inputQType));
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1); // batchmode

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForDecodeIndexScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DecodeIndexScore)
    .Tiling(TilingForDecodeIndexScore)
    .TilingParse<DecodeIndexScoreCompileInfo>(TilingPrepareForDecodeIndexScore);

} // namespace optiling
