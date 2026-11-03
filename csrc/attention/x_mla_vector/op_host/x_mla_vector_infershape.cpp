// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    const int64_t mode = *attrs->GetInt(0), rows = *attrs->GetInt(1);
    const int64_t heads = *attrs->GetInt(2), slots = *attrs->GetInt(3);
    if (mode < 0 || mode > 9 || rows < 1 || heads < 1 || slots < rows) return ge::GRAPH_FAILED;
    const bool mergedNorm = (mode >= 3 && mode <= 5) || mode == 7;
    const bool split = mode == 1 || mode == 6 || mode == 8;
    *context->GetOutputShape(0) = (mode == 0 || mergedNorm) ? gert::Shape({rows, 1536}) :
        (split ? gert::Shape({heads, rows, 128}) : gert::Shape({rows, heads, 512}));
    *context->GetOutputShape(1) = split ? gert::Shape({rows, heads, 64}) : gert::Shape({1});
    *context->GetOutputShape(2) = gert::Shape({slots, 512});
    *context->GetOutputShape(3) = gert::Shape({slots, 64});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    for (size_t i = 0; i < 4; ++i) context->SetOutputDataType(i, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(XMlaVector).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
