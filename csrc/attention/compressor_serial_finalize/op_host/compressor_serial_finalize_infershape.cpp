// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto& pool = *context->GetInputShape(0);
  const auto* attrs = context->GetAttrs();
  *context->GetOutputShape(0) = pool;
  *context->GetOutputShape(1) =
      gert::Shape({*attrs->GetInt(4), *attrs->GetInt(3), 2 * *attrs->GetInt(1) * pool.GetDim(1)});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(11));
  context->SetOutputDataType(1, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(CompressorSerialFinalize).InferShape(InferShape).InferDataType(InferType);
}  // namespace ops
