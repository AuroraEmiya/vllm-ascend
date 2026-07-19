/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "sparse_kv_gather_tiling.h"

using namespace optiling;

static ge::graphStatus TilingSparseKvGather(
    gert::TilingContext *context)
{
    SKGTilingInfo info;
    info.opName       = context->GetNodeName();
    info.platformInfo = context->GetPlatformInfo();

    SKGInfoParser parser(context);
    if (parser.Parse(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SparseKvGatherTiling tiling(context);
    return tiling.DoOpTiling(&info);
}

static ge::graphStatus TilingPrepareForSparseKvGather(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseKvGather)
    .Tiling(TilingSparseKvGather)
    .TilingParse<int>(TilingPrepareForSparseKvGather);
