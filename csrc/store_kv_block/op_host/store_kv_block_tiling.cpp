/**
 * @file store_kv_block_tiling.cpp
 * @brief StoreKVBlock TilingFunc implementation
 */

#include "store_kv_block_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    StoreKVBlockTilingData tiling;
    auto keyShape = context->GetInputShape(0)->GetStorageShape();
    auto keyCacheShape = context->GetInputShape(1)->GetStorageShape();
    auto groupLenShape = context->GetInputShape(2)->GetStorageShape();

    uint32_t numTokens = static_cast<uint32_t>(keyShape.GetDim(0));
    uint32_t headDim = static_cast<uint32_t>(keyShape.GetDim(1));
    uint32_t numGroups = static_cast<uint32_t>(groupLenShape.GetDim(0));
    uint32_t numCore = ascendcPlatform.GetCoreNumAiv();

    auto attrs = context->GetAttrs();
    int32_t blockSize = *(attrs->GetAttrPointer<int32_t>(0));

    tiling.set_numTokens(numTokens);
    tiling.set_headDim(headDim);
    tiling.set_numGroups(numGroups);
    tiling.set_blockSize(static_cast<uint32_t>(blockSize));
    tiling.set_numCore(numCore);

    context->SetTilingKey(0);
    context->SetBlockDim(numCore);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForStoreKVBlock(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StoreKVBlock)
    .Tiling(TilingFunc)
    .TilingParse<storeKVBlockCompileInfo>(TilingPrepareForStoreKVBlock);
}
