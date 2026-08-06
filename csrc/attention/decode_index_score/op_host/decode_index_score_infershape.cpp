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
 * \file decode_index_score_infershape.cpp
 * \brief Output score: [num_idx_heads, total_q, score_block_stride].
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t SCORE_OUT_INDEX = 0;
constexpr uint32_t ATTR_SCORE_BLOCK_STRIDE_INDEX = 4;
constexpr uint32_t DIM_NUM = 3;

static ge::graphStatus InferShapeDecodeIndexScore(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DecodeIndexScore", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *scoreShape = context->GetOutputShape(SCORE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *scoreBlockStride = attrs->GetInt(ATTR_SCORE_BLOCK_STRIDE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreBlockStride);
    OP_CHECK_IF(*scoreBlockStride <= 0,
                OP_LOGE(context, "score_block_stride must be positive, got %ld.", *scoreBlockStride),
                return ge::GRAPH_FAILED);

    // query: [total_q, num_idx_heads, head_dim] -> score: [num_idx_heads, total_q, score_block_stride]
    scoreShape->SetDimNum(DIM_NUM);
    scoreShape->SetDim(0, queryShape->GetDim(1)); // num_idx_heads
    scoreShape->SetDim(1, queryShape->GetDim(0)); // total_q
    scoreShape->SetDim(2, *scoreBlockStride);
    OP_LOGI(context->GetNodeName(), "DecodeIndexScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDecodeIndexScore(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DecodeIndexScore", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(SCORE_OUT_INDEX, ge::DT_FLOAT);
    OP_LOGI(context->GetNodeName(), "DecodeIndexScore InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DecodeIndexScore)
    .InferShape(InferShapeDecodeIndexScore)
    .InferDataType(InferDataTypeDecodeIndexScore);
} // namespace ops
