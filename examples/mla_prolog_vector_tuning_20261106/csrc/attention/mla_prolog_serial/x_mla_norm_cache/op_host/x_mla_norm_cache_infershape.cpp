// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const auto* attrs = context->GetAttrs();
  const int64_t rows = *attrs->GetInt(0), slots = *attrs->GetInt(1);
  if (rows < 1 || rows > 65536 || slots < rows || slots > 1048576) return ge::GRAPH_FAILED;
  *context->GetOutputShape(0) = gert::Shape({rows, 1536});
  *context->GetOutputShape(1) = gert::Shape({slots, 512});
  *context->GetOutputShape(2) = gert::Shape({slots, 64});
  return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  for (size_t i = 0; i < 3; ++i) context->SetOutputDataType(i, ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(XMlaNormCache).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
