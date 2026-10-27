// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& kv = *context->GetInputShape(3);
  const auto* attrs = context->GetAttrs();
  *context->GetOutputShape(0) = gert::Shape({*attrs->GetInt(2), kv.GetDim(1) / *attrs->GetInt(1)});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CompressorSerialPool).InferShape(InferShape).InferDataType(InferType);
}  // namespace ops
