/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "sparse_kv_gather_tiling.h"
#include "../op_kernel/sparse_kv_gather_kernel.h"

using namespace AscendC;
using namespace BaseApi;
using namespace optiling;

static ge::graphStatus TilingSparseKvGather(gert::TilingContext *context)
{
    SKGTilingInfo info;
    info.opName = context->GetNodeName();
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

extern "C" __global__ __aicore__ void sparse_kv_gather(
    __gm__ uint8_t *pagedCtkv,
    __gm__ uint8_t *pagedKpe,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *topkIndices,
    __gm__ uint8_t *curPos,
    __gm__ uint8_t *outCtkv,
    __gm__ uint8_t *outKpe,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    (void)workspace;

    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(SparseKvGatherTilingData, tilingData, tiling);

    SparseKvGatherKernel op;
    op.Init(pagedCtkv, pagedKpe, blockTable, topkIndices, curPos,
            outCtkv, outKpe, &tilingData, &pipe);
    op.Process();
}
