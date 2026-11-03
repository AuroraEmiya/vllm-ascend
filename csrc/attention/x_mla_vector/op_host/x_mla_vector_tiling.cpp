// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_mla_vector_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    const int64_t mode = *attrs->GetInt(0), rows = *attrs->GetInt(1);
    const int64_t heads = *attrs->GetInt(2), slots = *attrs->GetInt(3);
    const float epsQ = *attrs->GetFloat(4), epsKv = *attrs->GetFloat(5);
    if (mode < 0 || mode > 6 || rows < 1 || rows > 128 || heads < 1 || heads > 128 ||
        (heads & (heads - 1)) || slots < rows || slots > 1048576 ||
        !std::isfinite(epsQ) || !std::isfinite(epsKv) || epsQ <= 0 || epsKv <= 0) return ge::GRAPH_FAILED;
    const auto& shape = context->GetInputShape(0)->GetStorageShape();
    const bool mergedNorm = mode >= 3 && mode <= 5;
    const bool split = mode == 1 || mode == 6;
    const int64_t expectedRows = mode == 2 ? heads * rows : rows;
    const int64_t expectedWidth = mergedNorm ? 2112 : (mode == 0 ? 1536 : (split ? heads * 192 : 512));
    if (shape.GetDimNum() != 2 || shape.GetDim(0) != expectedRows || shape.GetDim(1) != expectedWidth)
        return ge::GRAPH_FAILED;
    if (mode == 0 && (context->GetInputShape(1)->GetStorageShape().GetShapeSize() != rows * 576 ||
        context->GetInputShape(2)->GetStorageShape().GetShapeSize() != 1536 ||
        context->GetInputShape(3)->GetStorageShape().GetShapeSize() != 512)) return ge::GRAPH_FAILED;
    if (mergedNorm && (context->GetInputShape(2)->GetStorageShape().GetShapeSize() != 1536 ||
        context->GetInputShape(3)->GetStorageShape().GetShapeSize() != 512)) return ge::GRAPH_FAILED;
    if (mode != 2 && (context->GetInputShape(4)->GetStorageShape().GetShapeSize() != rows * 64 ||
        context->GetInputShape(5)->GetStorageShape().GetShapeSize() != rows * 64 ||
        context->GetInputShape(6)->GetStorageShape().GetShapeSize() != rows)) return ge::GRAPH_FAILED;
    XMlaVectorTilingData tiling;
    tiling.set_mode(mode); tiling.set_rows(rows); tiling.set_heads(heads); tiling.set_slots(slots);
    const uint32_t headTile = rows >= 32 ? 32 : 16;
    tiling.set_units((mode == 0 || mergedNorm) ? rows * 2 :
        (split ? rows * ((heads + headTile - 1) / headTile) : rows));
    tiling.set_epsQ(epsQ); tiling.set_epsKv(epsKv);
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t requiredUb = mode == 2 ? heads * 512 * sizeof(uint16_t) : 59424;
    if (ubBytes < requiredUb || platform.GetCoreNumAiv() < 1) return ge::GRAPH_FAILED;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(std::min(tiling.get_units(), platform.GetCoreNumAiv()));
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(XMlaVector).Tiling(TilingFunc);
}  // namespace optiling
