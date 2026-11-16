// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_mla_layout_tiling.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto* attrs = context->GetAttrs();
  const int64_t rows = *attrs->GetInt(0), heads = *attrs->GetInt(1);
  if (rows < 1 || rows > 65536 || heads < 1 || heads > 128 || (heads & (heads - 1))) {
    return ge::GRAPH_FAILED;
  }
  const auto& input = context->GetInputShape(0)->GetStorageShape();
  if (input.GetDimNum() != 2 || input.GetDim(0) != heads * rows || input.GetDim(1) != 512) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  const uint32_t cores = platform.GetCoreNumAiv();
  if (!cores || !ubBytes) return ge::GRAPH_FAILED;

  const uint32_t rowTile = rows <= 32 ? 8 : 2;
  const uint32_t rowTiles = (rows + rowTile - 1) / rowTile;
  uint32_t headTile = std::min<int64_t>(heads, 4);
  while (headTile < 32 && 2 * headTile <= heads &&
         rowTiles * ((heads + 2 * headTile - 1) / (2 * headTile)) >= cores) {
    headTile *= 2;
  }
  while (headTile > 1 && 4ull * headTile * rowTile * 512 > ubBytes) headTile /= 2;
  const uint32_t units = rowTiles * ((heads + headTile - 1) / headTile);
  const uint32_t workElements = headTile * rowTile * 512;
  if (4ull * workElements > ubBytes) return ge::GRAPH_FAILED;

  XMlaLayoutTilingData tiling;
  tiling.set_rows(rows);
  tiling.set_heads(heads);
  tiling.set_units(units);
  tiling.set_headTile(headTile);
  tiling.set_rowTile(rowTile);
  tiling.set_workElements(workElements);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(std::min(units, cores));
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(XMlaLayout).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
