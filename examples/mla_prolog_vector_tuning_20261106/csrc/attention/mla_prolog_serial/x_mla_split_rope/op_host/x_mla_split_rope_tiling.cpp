// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_mla_split_rope_tiling.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto* attrs = context->GetAttrs();
  const int64_t rows = *attrs->GetInt(0), heads = *attrs->GetInt(1);
  if (rows < 1 || rows > 65536 || heads < 1 || heads > 128 || (heads & (heads - 1))) {
    return ge::GRAPH_FAILED;
  }
  const auto& input = context->GetInputShape(0)->GetStorageShape();
  if (input.GetDimNum() != 2 || input.GetDim(0) != rows || input.GetDim(1) != heads * 192 ||
      context->GetInputShape(1)->GetStorageShape().GetShapeSize() != rows * 64 ||
      context->GetInputShape(2)->GetStorageShape().GetShapeSize() != rows * 64) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  const uint32_t cores = platform.GetCoreNumAiv();
  if (!cores || !ubBytes) return ge::GRAPH_FAILED;

  // Tiny inputs retain a coarse tile to avoid repeated coefficient loads.
  uint32_t headTile = std::min<int64_t>(heads, 16);
  if (rows >= 32) {
    headTile = 1;
    while (headTile < std::min<int64_t>(heads, 32) &&
           rows * ((heads + 2 * headTile - 1) / (2 * headTile)) >= cores) {
      headTile *= 2;
    }
  }
  const uint32_t units = rows * ((heads + headTile - 1) / headTile);
  const uint32_t workElements = headTile * 64;
  const uint32_t floatElements = std::max(256u, workElements);
  // 192 loads a complete contiguous head; 128 loads NoPE and RoPE separately.
  const uint32_t inputHeadWidth = rows >= 32 ? 192 : 128;
  const uint64_t peakBytes = 4ull * (headTile * inputHeadWidth + 128 + workElements) +
                             20ull * floatElements + 32;
  if (peakBytes > ubBytes) return ge::GRAPH_FAILED;

  XMlaSplitRopeTilingData tiling;
  tiling.set_rows(rows);
  tiling.set_heads(heads);
  tiling.set_units(units);
  tiling.set_headTile(headTile);
  tiling.set_workElements(workElements);
  tiling.set_floatElements(floatElements);
  tiling.set_inputHeadWidth(inputHeadWidth);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(std::min(units, cores));
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(XMlaSplitRope).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
