/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <string>
#include <algorithm>
#include "sparse_kv_gather_tiling.h"
#include "register/op_def_registry.h"

using std::string;

using namespace ge;
namespace optiling {

static const string OP_NAME_STR = "SparseKvGather";

static SKGLayout ParseLayoutStr(const char *s) {
    string str(s);
    if (str == "TND")      return SKGLayout::TND;
    if (str == "PA_BSND")  return SKGLayout::PA_BSND;
    return SKGLayout::BSND;
}

// ===================== SKGInfoParser =====================

ge::graphStatus SKGInfoParser::GetInputs(SKGParamInfo &params) {
    params.sparseIndicesDesc  = context_->GetInputDesc(SKG_SPARSE_INDICES_IDX);
    params.sparseIndicesShape = context_->GetInputShape(SKG_SPARSE_INDICES_IDX);
    params.keyNopeDesc        = context_->GetInputDesc(SKG_KEY_NOPE_IDX);
    params.keyNopeShape       = context_->GetInputShape(SKG_KEY_NOPE_IDX);
    params.keyRopeDesc        = context_->GetInputDesc(SKG_KEY_ROPE_IDX);
    params.keyRopeShape       = context_->GetInputShape(SKG_KEY_ROPE_IDX);

    params.blockTableTensor   = context_->GetOptionalInputTensor(SKG_BLOCK_TABLE_IDX);
    params.blockTableDesc     = context_->GetOptionalInputDesc(SKG_BLOCK_TABLE_IDX);
    params.actSeqLenQTensor   = context_->GetOptionalInputTensor(SKG_ACT_SEQLEN_Q_IDX);
    params.actSeqLenQDesc     = context_->GetOptionalInputDesc(SKG_ACT_SEQLEN_Q_IDX);
    params.actSeqLenKVTensor  = context_->GetOptionalInputTensor(SKG_ACT_SEQLEN_KV_IDX);
    params.actSeqLenKVDesc    = context_->GetOptionalInputDesc(SKG_ACT_SEQLEN_KV_IDX);
    params.curPosTensor       = context_->GetOptionalInputTensor(SKG_CUR_POS_IDX);
    params.curPosDesc         = context_->GetOptionalInputDesc(SKG_CUR_POS_IDX);

    if (params.sparseIndicesDesc == nullptr || params.keyNopeDesc == nullptr ||
        params.keyRopeDesc == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(),
            "Required inputs (sparse_indices, key_nope, key_rope) must not be null.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGInfoParser::GetAttrs(SKGParamInfo &params) {
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Attributes are null.");
        return ge::GRAPH_FAILED;
    }
    params.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SKG_ATTR_SPARSE_BLOCK_SIZE);
    params.layoutQuery     = attrs->GetStr(SKG_ATTR_LAYOUT_QUERY);
    params.layoutKv        = attrs->GetStr(SKG_ATTR_LAYOUT_KV);

    if (params.sparseBlockSize == nullptr || params.layoutQuery == nullptr ||
        params.layoutKv == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "Required attrs missing.");
        return ge::GRAPH_FAILED;
    }
    if (*params.sparseBlockSize != 1) {
        OP_LOGE(OP_NAME_STR.c_str(),
            "sparseBlockSize only supports 1 (token-wise), got %ld.",
            *params.sparseBlockSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGInfoParser::GetShapes(SKGTilingInfo &info, const SKGParamInfo &params) {
    info.layoutQ  = ParseLayoutStr(params.layoutQuery);
    info.layoutKV = ParseLayoutStr(params.layoutKv);
    info.isPa     = (info.layoutKV == SKGLayout::PA_BSND);
    info.hasActSeqLenQ  = (params.actSeqLenQTensor != nullptr);
    info.hasActSeqLenKV = (params.actSeqLenKVTensor != nullptr);
    info.hasCurPos       = (params.curPosTensor != nullptr);

    // --- Sparse indices shape: last dim = topk_n ---
    auto &siShape = params.sparseIndicesShape->GetStorageShape();
    info.topkN = siShape.GetDim(siShape.GetDimNum() - 1);
    info.sparseBlockCount = info.topkN;
    info.sparseBlockSize  = *params.sparseBlockSize;

    // --- N2 from key ---
    auto &kShape = params.keyNopeShape->GetStorageShape();
    uint32_t n2DimIdx;
    if (info.isPa) {
        n2DimIdx = 2;
        info.blockSize = kShape.GetDim(1);
        info.s2Size    = kShape.GetDim(0) * kShape.GetDim(1);
    } else if (info.layoutKV == SKGLayout::TND) {
        n2DimIdx = 1;
        info.s2Size = kShape.GetDim(0);
    } else {
        n2DimIdx = 2;
        info.s2Size = kShape.GetDim(1);
    }
    info.n2Size = kShape.GetDim(n2DimIdx);

    // --- numActual (N) ---
    // Supports both 2D [N, K] (Triton) and 3D [B, S1, K] (BSND legacy).
    if (info.layoutQ == SKGLayout::TND) {
        info.s1Size = siShape.GetDim(0);
        info.bSize  = info.hasActSeqLenQ
            ? params.actSeqLenQTensor->GetShapeSize() : 1;
        info.numActual = info.s1Size;  // T
    } else {
        uint32_t dimNum = siShape.GetDimNum();
        if (dimNum == 2) {
            // 2D: Triton-compatible [num_actual, topk_n]
            info.bSize  = siShape.GetDim(0);
            info.s1Size = 1;
        } else {
            // 3D: [B, S1, K]
            info.bSize  = siShape.GetDim(0);
            info.s1Size = siShape.GetDim(1);
        }
        info.numActual = info.bSize * info.s1Size;
    }

    info.totalOutputRows = info.numActual * info.topkN;

    // --- PA ---
    if (info.isPa) {
        if (params.blockTableTensor == nullptr) {
            OP_LOGE(OP_NAME_STR.c_str(),
                "block_table is required for PA_BSND layout.");
            return ge::GRAPH_FAILED;
        }
        info.maxBlockNumPerBatch =
            params.blockTableTensor->GetStorageShape().GetDim(1);
    }

    // --- cur_pos shape check ---
    if (info.hasCurPos) {
        auto curPosSize = params.curPosTensor->GetShapeSize();
        if (static_cast<uint32_t>(curPosSize) != info.numActual) {
            OP_LOGE(OP_NAME_STR.c_str(),
                "cur_pos size %ld != num_actual %u.",
                curPosSize, info.numActual);
            return ge::GRAPH_FAILED;
        }
    }

    // --- Head dim validation ---
    uint32_t lastDim = kShape.GetDimNum() - 1;
    uint32_t dNope = kShape.GetDim(lastDim);
    auto &rShape = params.keyRopeShape->GetStorageShape();
    uint32_t dRope = rShape.GetDim(rShape.GetDimNum() - 1);
    if (dNope != SKG_D_NOPE || dRope != SKG_D_ROPE) {
        OP_LOGE(OP_NAME_STR.c_str(),
            "Expected nope_dim=512, rope_dim=64; got %u, %u.", dNope, dRope);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGInfoParser::Validate(const SKGTilingInfo &info) {
    if (info.numActual == 0 || info.topkN == 0) {
        OP_LOGE(OP_NAME_STR.c_str(), "Zero-sized: num_actual=%u topk_n=%u.",
                info.numActual, info.topkN);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SKGInfoParser::Parse(SKGTilingInfo &info) {
    SKGParamInfo params;
    if (GetInputs(params)  != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (GetAttrs(params)   != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (GetShapes(info, params) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (Validate(info)     != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    info.params = params;
    return ge::GRAPH_SUCCESS;
}

// ===================== SKGTilingCheck =====================

ge::graphStatus SKGTilingCheck::Process() {
    if (CheckDtypes() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckShapes() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (CheckAttrs()  != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SKGTilingCheck::CheckDtypes() const { return ge::GRAPH_SUCCESS; }
ge::graphStatus SKGTilingCheck::CheckShapes() const {
    if (info_.totalOutputRows == 0) {
        OP_LOGE(OP_NAME_STR.c_str(), "totalOutputRows is 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SKGTilingCheck::CheckAttrs() const { return ge::GRAPH_SUCCESS; }

// ===================== SparseKvGatherTiling =====================

ge::graphStatus SparseKvGatherTiling::SetBlockDim(uint32_t blockDim) const {
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SparseKvGatherTiling::SetWorkspaceSize(uint64_t size) const {
    size_t *ws = context_->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "workspace size ptr is null.");
        return ge::GRAPH_FAILED;
    }
    ws[0] = size;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SparseKvGatherTiling::SetTilingData(TilingDef &td) const {
    if (context_->GetRawTilingData() == nullptr) {
        OP_LOGE(OP_NAME_STR.c_str(), "RawTilingData is null.");
        return ge::GRAPH_FAILED;
    }
    td.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                    context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(td.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseKvGatherTiling::GetPlatformInfo(SKGTilingInfo *info) {
    auto *plat = info->platformInfo;
    if (plat == nullptr) {
        OP_LOGE(info->opName, "GetPlatformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto ascPlat = platform_ascendc::PlatformAscendC(plat);
    info->aivNum = ascPlat.GetCoreNumAiv();
    if (info->aivNum == 0) {
        OP_LOGE(info->opName, "AIV core count is 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SparseKvGatherTiling::InitParams(SKGTilingInfo *info) {
    info->usedCoreNum  = info->aivNum;
    info->groupsPerCore = (info->numActual + info->usedCoreNum - 1)
                          / info->usedCoreNum;
    info->rowsPerCore   = info->groupsPerCore * info->topkN;
}

void SparseKvGatherTiling::SplitWork(SKGTilingInfo *info) {
    // Static even split; tail handled by kernel clamping.
}

void SparseKvGatherTiling::CalcWorkspace(SKGTilingInfo *info) {
    auto ascPlat = platform_ascendc::PlatformAscendC(info->platformInfo);
    workspaceSize_ = ascPlat.GetLibApiWorkSpaceSize();
}

void SparseKvGatherTiling::FillTilingData(SKGTilingInfo *info) {
    auto &bp = tilingData_.baseParams;
    bp.set_batchSize(info->bSize);
    bp.set_s1Size(info->s1Size);
    bp.set_s2Size(info->s2Size);
    bp.set_numActual(info->numActual);
    bp.set_topkN(info->topkN);
    bp.set_sparseBlockCount(info->sparseBlockCount);
    bp.set_sparseBlockSize(info->sparseBlockSize);
    bp.set_totalGroups(info->numActual);
    bp.set_totalOutputRows(info->totalOutputRows);
    bp.set_rowsPerCore(info->rowsPerCore);
    bp.set_groupsPerCore(info->groupsPerCore);
    bp.set_blockSize(info->blockSize);
    bp.set_maxBlockNumPerBatch(info->maxBlockNumPerBatch);
    bp.set_usedCoreNum(info->usedCoreNum);
    bp.set_layoutQuery(static_cast<uint32_t>(info->layoutQ));
    bp.set_layoutKv(static_cast<uint32_t>(info->layoutKV));
    bp.set_isActualLenDimsNull(info->hasActSeqLenQ ? 0U : 1U);
    bp.set_isActualLenDimsKVNull(info->hasActSeqLenKV ? 0U : 1U);
    bp.set_isPageAttention(info->isPa ? 1U : 0U);
    bp.set_hasCurPos(info->hasCurPos ? 1U : 0U);
}

ge::graphStatus SparseKvGatherTiling::DoOpTiling(SKGTilingInfo *info) {
    info_ = info;
    if (GetPlatformInfo(info) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    InitParams(info);
    SplitWork(info);
    FillTilingData(info);
    CalcWorkspace(info);

    blockDim_ = info->usedCoreNum;

    if (SetBlockDim(blockDim_)          != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    if (SetTilingData(tilingData_)      != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
