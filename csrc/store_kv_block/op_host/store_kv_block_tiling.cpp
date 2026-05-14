/**
 * @file store_kv_block_tiling.cpp
 * @brief StoreKVBlock TilingFunc — populates tiling data from tensor shapes and attr
 */

#include "store_kv_block_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    StoreKVBlockTilingData tiling;

    // keyIn shape: [numTokens, ...] — rowElems = product of all dims after dim 0
    auto keyShape = context->GetInputShape(0)->GetStorageShape();
    auto groupLenShape = context->GetInputShape(2)->GetStorageShape();

    uint32_t rowElems = 1;
    for (int i = 1; i < keyShape.GetDimNum(); i++) {
        rowElems *= static_cast<uint32_t>(keyShape.GetDim(i));
    }
    uint32_t groupCount = static_cast<uint32_t>(groupLenShape.GetDim(0));
    uint32_t coreCount = ascendcPlatform.GetCoreNumAiv();

    auto attrs = context->GetAttrs();
    int32_t blockSize = *(attrs->GetAttrPointer<int32_t>(0));

    // Determine bytes per element from dtype
    auto keyInDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t elemBytes = (keyInDtype == ge::DT_FLOAT16 || keyInDtype == ge::DT_BF16) ? 2 : 1;

    tiling.set_rowBytes(rowElems * elemBytes);
    tiling.set_maxGroupLen(static_cast<uint32_t>(blockSize));
    tiling.set_groupCount(groupCount);
    tiling.set_coreCount(coreCount);

    context->SetTilingKey(0);
    context->SetBlockDim(coreCount);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForStoreKVBlock(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StoreKVBlock)
    .Tiling(TilingFunc)
    .TilingParse<storeKVBlockCompileInfo>(TilingPrepareForStoreKVBlock);
}
