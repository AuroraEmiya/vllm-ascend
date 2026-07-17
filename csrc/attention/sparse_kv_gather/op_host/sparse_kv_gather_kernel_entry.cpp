/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "sparse_kv_gather_tiling.h"
#include "../op_kernel/sparse_kv_gather_kernel.h"

using namespace AscendC;
using namespace optiling;
using namespace BaseApi;

// ===================== Tiling entry =====================

static ge::graphStatus TilingSparseKvGather(gert::TilingContext *context) {
    SKGTilingInfo info;
    info.opName       = context->GetNodeName();
    info.platformInfo = context->GetPlatformInfo();

    SKGInfoParser parser(context);
    if (parser.Parse(info) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    SKGTilingCheck checker(info);
    if (checker.Process() != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;

    SparseKvGatherTiling tiling(context);
    return tiling.DoOpTiling(&info);
}

static ge::graphStatus
TilingPrepareForSparseKvGather(gert::TilingParseContext *context) {
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SparseKvGather)
    .Tiling(TilingSparseKvGather)
    .TilingParse<int>(TilingPrepareForSparseKvGather);

// ===================== Kernel entry =====================

extern "C" __global__ __aicore__ void
sparse_kv_gather(__gm__ uint8_t *sparseIndices,
                 __gm__ uint8_t *keyNope,
                 __gm__ uint8_t *keyRope,
                 __gm__ uint8_t *blockTable,
                 __gm__ uint8_t *actSeqLenQ,
                 __gm__ uint8_t *actSeqLenKV,
                 __gm__ uint8_t *curPos,
                 __gm__ uint8_t *outCtkv,
                 __gm__ uint8_t *outKpe,
                 __gm__ uint8_t *workspace,
                 __gm__ uint8_t *tiling)
{
    if ASCEND_IS_AIC { return; }

    TPipe tPipe;
    GET_TILING_DATA_WITH_STRUCT(SparseKvGatherTilingData, tilingData, tiling);

    const auto &bp = tilingData.baseParams;
    bool     isPa  = bp.isPageAttention;
    uint32_t layQ  = bp.layoutQuery;
    using KV_T = ::half;

    if (isPa) {
        if (layQ == static_cast<uint32_t>(SKGLayoutKernel::TND)) {
            SparseKvGatherKernel<KV_T, true, SKGLayoutKernel::TND> op;
            op.Init(sparseIndices, keyNope, keyRope, blockTable,
                    actSeqLenQ, actSeqLenKV, curPos,
                    outCtkv, outKpe, &tilingData, &tPipe);
            op.Process();
        } else {
            SparseKvGatherKernel<KV_T, true, SKGLayoutKernel::BSND> op;
            op.Init(sparseIndices, keyNope, keyRope, blockTable,
                    actSeqLenQ, actSeqLenKV, curPos,
                    outCtkv, outKpe, &tilingData, &tPipe);
            op.Process();
        }
    } else {
        if (layQ == static_cast<uint32_t>(SKGLayoutKernel::TND)) {
            SparseKvGatherKernel<KV_T, false, SKGLayoutKernel::TND> op;
            op.Init(sparseIndices, keyNope, keyRope, blockTable,
                    actSeqLenQ, actSeqLenKV, curPos,
                    outCtkv, outKpe, &tilingData, &tPipe);
            op.Process();
        } else {
            SparseKvGatherKernel<KV_T, false, SKGLayoutKernel::BSND> op;
            op.Init(sparseIndices, keyNope, keyRope, blockTable,
                    actSeqLenQ, actSeqLenKV, curPos,
                    outCtkv, outKpe, &tilingData, &tPipe);
            op.Process();
        }
    }
}
