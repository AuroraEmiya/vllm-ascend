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
 * \file decode_index_score_tiling.h
 * \brief Tiling data for the decode indexer block-score op.
 */

#ifndef DECODE_INDEX_SCORE_TILING_H_
#define DECODE_INDEX_SCORE_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"

namespace optiling {

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t INDEX_KV_CACHE_INDEX = 1;
constexpr uint32_t BLOCK_TABLE_INDEX = 2;
constexpr uint32_t SEQ_LENS_INDEX = 3;
constexpr uint32_t GLOBAL_SEQ_LENS_INDEX = 4;
// Outputs Index
constexpr uint32_t SCORE_OUT_INDEX = 0;
// Attributes Index
constexpr uint32_t ATTR_DECODE_QUERY_LEN_INDEX = 0;
constexpr uint32_t ATTR_BLOCK_OFFSET_INDEX = 1;
constexpr uint32_t ATTR_INIT_BLOCKS_INDEX = 2;
constexpr uint32_t ATTR_LOCAL_BLOCKS_INDEX = 3;
constexpr uint32_t ATTR_SCORE_BLOCK_STRIDE_INDEX = 4;
constexpr uint32_t ATTR_NUM_CHUNKS_INDEX = 5;

// sparse block size in tokens (== SPARSE_BLOCK_SIZE in the Triton version)
constexpr uint32_t SPARSE_BLOCK_SIZE = 128;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(DecodeIndexScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalQ)
TILING_DATA_FIELD_DEF(uint32_t, numIdxHeads)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, decodeQueryLen)
TILING_DATA_FIELD_DEF(uint32_t, blockOffset)
TILING_DATA_FIELD_DEF(uint32_t, initBlocks)
TILING_DATA_FIELD_DEF(uint32_t, localBlocks)
TILING_DATA_FIELD_DEF(uint32_t, scoreBlockStride)   // == SCORE_BLOCK_COUNT
TILING_DATA_FIELD_DEF(uint32_t, numChunks)
TILING_DATA_FIELD_DEF(uint32_t, localMaxBlocks)     // block_table row stride
TILING_DATA_FIELD_DEF(uint32_t, wsBlocksPerChunk)   // workspace: max blocks per chunk
TILING_DATA_FIELD_DEF(uint32_t, wsStripSize)        // workspace: per-block strip = align16(H)*128 floats
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(DecodeIndexScore, DecodeIndexScoreTilingData)

// -----------算子CompileInfo定义-------------------
struct DecodeIndexScoreCompileInfo {};

} // namespace optiling
#endif // DECODE_INDEX_SCORE_TILING_H_
