/**
 * @file store_kv_decode_proto.cpp
 * @brief InferShape / InferDataType for StoreKVDecode
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeStoreKVDecode(gert::InferShapeContext *context)
{
    gert::Shape *outShape = context->GetOutputShape(0);
    const gert::Shape *cacheShape = context->GetInputShape(1);
    *outShape = *cacheShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeStoreKVDecode(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StoreKVDecode)
    .InferShape(InferShapeStoreKVDecode)
    .InferDataType(InferDataTypeStoreKVDecode);
} // namespace ops
