/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& x = *context->GetInputShape(0);
  const auto requestedMode = *context->GetAttrs()->GetInt(0), c = *context->GetAttrs()->GetInt(1);
  const auto mode = (requestedMode >= 9 && requestedMode <= 11) ? 5 : requestedMode;
  if (x.GetDimNum() != 2 || c != 4) return ge::GRAPH_FAILED;
  *context->GetOutputShape(0) = x;
  if (mode == 2) context->GetOutputShape(0)->SetDim(1, x.GetDim(1) / c);
  if (mode == 3 || mode == 5) context->GetOutputShape(0)->SetDim(1, 320);
  *context->GetOutputShape(1) = mode == 0 ? gert::Shape({x.GetDim(0), c}) : gert::Shape({0});
  bool gate = mode == 3 || mode == 5 || (mode == 2 && context->GetInputShape(3)->GetShapeSize() == x.GetDim(0) * c);
  *context->GetOutputShape(2) = gate ? gert::Shape({x.GetDim(0), mode == 5 ? 16 : c}) : gert::Shape({0});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  context->SetOutputDataType(2, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(XTgrActivation).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
