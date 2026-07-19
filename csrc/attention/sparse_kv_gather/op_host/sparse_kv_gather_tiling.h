/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef SPARSE_KV_GATHER_TILING_H
#define SPARSE_KV_GATHER_TILING_H

#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SKG_PAGED_CTKV_IDX   = 0;
constexpr uint32_t SKG_PAGED_KPE_IDX    = 1;
constexpr uint32_t SKG_BLOCK_TABLE_IDX  = 2;
constexpr uint32_t SKG_TOPK_INDICES_IDX = 3;
constexpr uint32_t SKG_CUR_POS_IDX      = 4;

constexpr uint32_t SKG_OUT_CTKV_IDX = 0;
constexpr uint32_t SKG_OUT_KPE_IDX  = 1;

constexpr uint32_t SKG_ATTR_BLOCK_SIZE = 0;

constexpr uint32_t SKG_BLOCK_SIZE = 128;
constexpr uint32_t SKG_CTKV_DIM   = 512;
constexpr uint32_t SKG_KPE_DIM    = 64;
constexpr uint32_t SKG_HEAD_NUM   = 1;

enum class SKGIndexType : uint32_t {
    INT32 = 0,
    INT64 = 1,
};

BEGIN_TILING_DATA_DEF(SparseKvGatherTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, numActual)
    TILING_DATA_FIELD_DEF(uint32_t, maxBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, topkN)
    TILING_DATA_FIELD_DEF(uint64_t, totalSlots)
    TILING_DATA_FIELD_DEF(uint64_t, slotsPerCore)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableType)
    TILING_DATA_FIELD_DEF(uint32_t, topkIndicesType)
    TILING_DATA_FIELD_DEF(uint32_t, curPosType)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseKvGather, SparseKvGatherTilingData)

struct SKGParamInfo {
    const gert::CompileTimeTensorDesc *pagedCtkvDesc = nullptr;
    const gert::StorageShape *pagedCtkvShape = nullptr;

    const gert::CompileTimeTensorDesc *pagedKpeDesc = nullptr;
    const gert::StorageShape *pagedKpeShape = nullptr;

    const gert::CompileTimeTensorDesc *blockTableDesc = nullptr;
    const gert::StorageShape *blockTableShape = nullptr;

    const gert::CompileTimeTensorDesc *topkIndicesDesc = nullptr;
    const gert::StorageShape *topkIndicesShape = nullptr;

    const gert::CompileTimeTensorDesc *curPosDesc = nullptr;
    const gert::StorageShape *curPosShape = nullptr;

    const gert::CompileTimeTensorDesc *outCtkvDesc = nullptr;
    const gert::StorageShape *outCtkvShape = nullptr;

    const gert::CompileTimeTensorDesc *outKpeDesc = nullptr;
    const gert::StorageShape *outKpeShape = nullptr;

    const int64_t *blockSize = nullptr;
};

struct SKGTilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    SKGParamInfo params;

    uint32_t numBlocks = 0;
    uint32_t numActual = 0;
    uint32_t maxBlocks = 0;
    uint32_t topkN = 0;

    uint64_t totalSlots = 0;
    uint64_t slotsPerCore = 0;

    uint32_t aivNum = 0;
    uint32_t usedCoreNum = 0;

    SKGIndexType blockTableType = SKGIndexType::INT32;
    SKGIndexType topkIndicesType = SKGIndexType::INT32;
    SKGIndexType curPosType = SKGIndexType::INT32;
};

class SKGInfoParser {
public:
    explicit SKGInfoParser(const gert::TilingContext *context) : context_(context) {}
    ge::graphStatus Parse(SKGTilingInfo &info);

private:
    ge::graphStatus GetTensorInfo(SKGParamInfo &params) const;
    ge::graphStatus GetAttrs(SKGParamInfo &params) const;
    ge::graphStatus CheckDtypes(SKGTilingInfo &info) const;
    ge::graphStatus CheckShapes(SKGTilingInfo &info) const;

    const gert::TilingContext *context_ = nullptr;
};

class SparseKvGatherTiling {
public:
    explicit SparseKvGatherTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(SKGTilingInfo *info);

private:
    ge::graphStatus GetPlatformInfo(SKGTilingInfo *info) const;
    ge::graphStatus SplitWork(SKGTilingInfo *info) const;
    void FillTilingData(const SKGTilingInfo *info);

    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;

    gert::TilingContext *context_ = nullptr;
    SparseKvGatherTilingData tilingData_;
};

}  // namespace optiling

#endif  // SPARSE_KV_GATHER_TILING_H
