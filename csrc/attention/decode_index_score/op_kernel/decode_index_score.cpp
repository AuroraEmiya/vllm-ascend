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
 * \file decode_index_score.cpp
 * \brief AscendC kernel entry for the decode indexer block-score op (arch22).
 *
 * Mixed cube/vector kernel (1 AIC : 2 AIV, 910B). The AIC core gathers each
 * normal candidate page's index-K tile, runs a per-block mmad against the index
 * query and fixes the [H, 128] result out to workspace GM. The AIV core folds
 * the per-block max over the 128 positions, applies the partial-block position
 * limit and writes init/local/-inf sentinels to the score buffer. Cross-core
 * hand-off uses a single wait/set flag pair. Only sub-AIV 0 runs the vector
 * service (the per-program vector workload is too small to split).
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "decode_index_score_tiling_key.h"
#include "arch22/decode_index_score_kernel.h"

using namespace DecodeIndexScoreKernel;

template <int DT_Q, int DT_K>
__global__ __aicore__ void decode_index_score(__gm__ uint8_t *query, __gm__ uint8_t *indexKvCache,
                                              __gm__ uint8_t *blockTable, __gm__ uint8_t *seqLens,
                                              __gm__ uint8_t *globalSeqLens, __gm__ uint8_t *score,
                                              __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA_WITH_STRUCT(DecodeIndexScoreTilingData, tiling_data_in, tiling);
    const DecodeIndexScoreTilingData *__restrict tiling_data = &tiling_data_in;

    if constexpr (DT_Q == DI_TPL_FP16 && DT_K == DI_TPL_FP16) {
        DecodeIndexScoreKernelT<half> op;
        op.Init(query, indexKvCache, blockTable, seqLens, globalSeqLens, score, user, tiling_data, &tPipe);
        op.Process();
    } else {
        DecodeIndexScoreKernelT<bfloat16_t> op;
        op.Init(query, indexKvCache, blockTable, seqLens, globalSeqLens, score, user, tiling_data, &tPipe);
        op.Process();
    }
}
