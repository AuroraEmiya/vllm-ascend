/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file sparse_kv_gather_tiling.h
 * \brief Tiling data structures for SparseKvGather operator.
 *
 * Gathers non-contiguous KV-cache entries selected by sparse top-k
 * indices into two contiguous output tensors:
 *   out_ctkv [num_actual, topk_n, 512]
 *   out_kpe  [num_actual, topk_n,  64]
 */

#ifndef SPARSE_KV_GATHER_TILING_H
#define SPARSE_KV_GATHER_TILING_H

#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"

namespace optiling {

// ===================== Input / Output / Attr indices =====================
constexpr uint32_t SKG_SPARSE_INDICES_IDX = 0;
constexpr uint32_t SKG_KEY_NOPE_IDX       = 1;
constexpr uint32_t SKG_KEY_ROPE_IDX       = 2;
constexpr uint32_t SKG_BLOCK_TABLE_IDX    = 3;
constexpr uint32_t SKG_ACT_SEQLEN_Q_IDX   = 4;
constexpr uint32_t SKG_ACT_SEQLEN_KV_IDX  = 5;
constexpr uint32_t SKG_CUR_POS_IDX        = 6;

constexpr uint32_t SKG_OUT_CTKV_IDX = 0;
constexpr uint32_t SKG_OUT_KPE_IDX  = 1;

constexpr uint32_t SKG_ATTR_SPARSE_BLOCK_SIZE = 0;
constexpr uint32_t SKG_ATTR_LAYOUT_QUERY       = 1;
constexpr uint32_t SKG_ATTR_LAYOUT_KV          = 2;

// ===================== Layout enum =====================
enum class SKGLayout : uint32_t {
    BSND    = 0,
    TND     = 1,
    PA_BSND = 2,
};

// ===================== Compile-time constants =====================
constexpr uint32_t SKG_D_NOPE = 512;
constexpr uint32_t SKG_D_ROPE = 64;

// ===================== Tiling Data =====================
BEGIN_TILING_DATA_DEF(SparseKvGatherBaseParams)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize)
    TILING_DATA_FIELD_DEF(uint32_t, s1Size)              // S1 (qSeqSize) or T
    TILING_DATA_FIELD_DEF(uint32_t, s2Size)              // S2 (kvSeqSize, logical max)
    TILING_DATA_FIELD_DEF(uint32_t, numActual)            // N = B*S1 (BSND) or T (TND)
    TILING_DATA_FIELD_DEF(uint32_t, topkN)               // K = sparseBlockCount = topk_n
    TILING_DATA_FIELD_DEF(int64_t,  sparseBlockSize)     // always 1
    TILING_DATA_FIELD_DEF(uint32_t, totalGroups)         // numActual
    TILING_DATA_FIELD_DEF(uint32_t, totalOutputRows)     // numActual * topkN
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore)
    TILING_DATA_FIELD_DEF(uint32_t, groupsPerCore)
    TILING_DATA_FIELD_DEF(int64_t,  blockSize)           // PA block size
    TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, layoutQuery)
    TILING_DATA_FIELD_DEF(uint32_t, layoutKv)
    TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsNull)
    TILING_DATA_FIELD_DEF(uint32_t, isActualLenDimsKVNull)
    TILING_DATA_FIELD_DEF(uint32_t, isPageAttention)
    TILING_DATA_FIELD_DEF(uint32_t, hasCurPos)           // 1 if cur_pos provided
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseKvGatherBaseParamsOp, SparseKvGatherBaseParams)

BEGIN_TILING_DATA_DEF(SparseKvGatherTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(SparseKvGatherBaseParams, baseParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SparseKvGather, SparseKvGatherTilingData)

// ===================== Tiling Parameter Info =====================
struct SKGParamInfo {
    const gert::CompileTimeTensorDesc *sparseIndicesDesc  = nullptr;
    const gert::StorageShape          *sparseIndicesShape = nullptr;
    const gert::CompileTimeTensorDesc *keyNopeDesc        = nullptr;
    const gert::StorageShape          *keyNopeShape       = nullptr;
    const gert::CompileTimeTensorDesc *keyRopeDesc        = nullptr;
    const gert::StorageShape          *keyRopeShape       = nullptr;
    const gert::CompileTimeTensorDesc *blockTableDesc     = nullptr;
    const gert::Tensor                *blockTableTensor   = nullptr;
    const gert::CompileTimeTensorDesc *actSeqLenQDesc     = nullptr;
    const gert::Tensor                *actSeqLenQTensor   = nullptr;
    const gert::CompileTimeTensorDesc *actSeqLenKVDesc    = nullptr;
    const gert::Tensor                *actSeqLenKVTensor  = nullptr;
    const gert::CompileTimeTensorDesc *curPosDesc         = nullptr;
    const gert::Tensor                *curPosTensor       = nullptr;

    const int64_t *sparseBlockSize = nullptr;
    const char    *layoutQuery     = nullptr;
    const char    *layoutKv        = nullptr;
};

// ===================== Tiling Info (parsed) =====================
struct SKGTilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    SKGParamInfo params;

    uint32_t bSize      = 0;
    uint32_t s1Size     = 0;
    uint32_t s2Size     = 0;
    uint32_t n2Size     = 0;

    int64_t  sparseBlockSize  = 1;
    uint32_t sparseBlockCount = 0;

    SKGLayout layoutQ  = SKGLayout::BSND;
    SKGLayout layoutKV = SKGLayout::PA_BSND;
    bool isPa           = false;
    bool hasActSeqLenQ  = false;
    bool hasActSeqLenKV = false;
    bool hasCurPos       = false;

    int64_t  blockSize           = 0;
    uint32_t maxBlockNumPerBatch = 0;

    uint32_t numActual      = 0;  // N (= B*S1 or T)
    uint32_t topkN          = 0;  // K (= sparseBlockCount)
    uint32_t totalOutputRows = 0;
    uint32_t aivNum          = 0;

    uint32_t usedCoreNum  = 0;
    uint32_t rowsPerCore  = 0;
    uint32_t groupsPerCore = 0;
};

// ===================== Tiling Class =====================
class SparseKvGatherTiling {
public:
    explicit SparseKvGatherTiling(gert::TilingContext *context)
        : context_(context) {}
    ge::graphStatus DoOpTiling(SKGTilingInfo *info);

private:
    ge::graphStatus GetPlatformInfo(SKGTilingInfo *info);
    void InitParams(SKGTilingInfo *info);
    void SplitWork(SKGTilingInfo *info);
    void FillTilingData(SKGTilingInfo *info);
    void CalcWorkspace(SKGTilingInfo *info);

    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;

    gert::TilingContext *context_ = nullptr;
    SKGTilingInfo *info_ = nullptr;

    uint32_t blockDim_   = 0;
    uint64_t workspaceSize_ = 0;

    SparseKvGatherTilingData tilingData_;
};

// ===================== Parser =====================
class SKGInfoParser {
public:
    explicit SKGInfoParser(const gert::TilingContext *context)
        : context_(context) {}
    ge::graphStatus Parse(SKGTilingInfo &info);

private:
    ge::graphStatus GetInputs(SKGParamInfo &params);
    ge::graphStatus GetAttrs(SKGParamInfo &params);
    ge::graphStatus GetShapes(SKGTilingInfo &info, const SKGParamInfo &params);
    ge::graphStatus Validate(const SKGTilingInfo &info);

    const gert::TilingContext *context_ = nullptr;
};

// ===================== Validation =====================
class SKGTilingCheck {
public:
    explicit SKGTilingCheck(const SKGTilingInfo &info) : info_(info) {}
    ge::graphStatus Process();

private:
    ge::graphStatus CheckDtypes() const;
    ge::graphStatus CheckShapes() const;
    ge::graphStatus CheckAttrs() const;
    const SKGTilingInfo &info_;
};

}  // namespace optiling

#endif  // SPARSE_KV_GATHER_TILING_H
