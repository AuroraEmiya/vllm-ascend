// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  *context->GetOutputShape(1) = *context->GetInputShape(7);
  *context->GetOutputShape(2) = *context->GetInputShape(8);
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0,ge::DT_BF16);
  context->SetOutputDataType(1,ge::DT_BF16);
  context->SetOutputDataType(2,ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ContactNorm).InferShape(InferShape).InferDataType(InferDataType);
}
