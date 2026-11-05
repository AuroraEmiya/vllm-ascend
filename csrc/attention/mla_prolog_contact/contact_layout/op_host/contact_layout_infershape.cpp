// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& s=*context->GetInputShape(0);
  if (s.GetDimNum()!=3) return ge::GRAPH_FAILED;
  *context->GetOutputShape(0)=gert::Shape({s.GetDim(1),1,s.GetDim(0),s.GetDim(2)});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0,ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ContactLayout).InferShape(InferShape).InferDataType(InferDataType);
}
