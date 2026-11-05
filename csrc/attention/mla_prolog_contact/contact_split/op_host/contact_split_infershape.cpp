// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& s=*context->GetInputShape(0); const auto* a=context->GetAttrs();
  if (s.GetDimNum()!=3 || s.GetDim(1)!=1) return ge::GRAPH_FAILED;
  int64_t b=s.GetDim(0),n=*a->GetInt(0),d=*a->GetInt(1),r=*a->GetInt(2);
  *context->GetOutputShape(0)=gert::Shape({n,b,d});
  *context->GetOutputShape(1)=gert::Shape({b,1,n,r});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0,ge::DT_BF16);
  context->SetOutputDataType(1,ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ContactSplit).InferShape(InferShape).InferDataType(InferDataType);
}
