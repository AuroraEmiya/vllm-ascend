/**
 * @file store_kv_decode_tiling.cpp
 * @brief StoreKVDecode TilingFunc — single core, rowBytes from keyIn shape+dtype
 */

#include "store_kv_decode_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    StoreKVDecodeTilingData tiling;

    auto keyShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t rowElems = static_cast<uint32_t>(keyShape.GetDim(1));

    auto keyInDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t elemBytes = (keyInDtype == ge::DT_FLOAT16 || keyInDtype == ge::DT_BF16) ? 2 : 1;

    tiling.set_rowBytes(rowElems * elemBytes);

    context->SetTilingKey(0);
    context->SetBlockDim(1);  // single core for decode

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForStoreKVDecode(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StoreKVDecode)
    .Tiling(TilingFunc)
    .TilingParse<storeKVDecodeCompileInfo>(TilingPrepareForStoreKVDecode);
}
