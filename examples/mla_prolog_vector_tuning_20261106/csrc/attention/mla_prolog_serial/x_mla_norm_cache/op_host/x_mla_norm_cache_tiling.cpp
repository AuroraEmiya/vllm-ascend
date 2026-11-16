// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_mla_norm_cache_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto* attrs = context->GetAttrs();
  const int64_t rows = *attrs->GetInt(0), slots = *attrs->GetInt(1);
  const float epsQ = *attrs->GetFloat(2), epsKv = *attrs->GetFloat(3);
  if (rows < 1 || rows > 65536 || slots < rows || slots > 1048576 ||
      !std::isfinite(epsQ) || !std::isfinite(epsKv) || epsQ <= 0 || epsKv <= 0) {
    return ge::GRAPH_FAILED;
  }
  const auto& input = context->GetInputShape(0)->GetStorageShape();
  if (input.GetDimNum() != 2 || input.GetDim(0) != rows || input.GetDim(1) != 2112 ||
      context->GetInputShape(1)->GetStorageShape().GetShapeSize() != 1536 ||
      context->GetInputShape(2)->GetStorageShape().GetShapeSize() != 512 ||
      context->GetInputShape(3)->GetStorageShape().GetShapeSize() != rows * 64 ||
      context->GetInputShape(4)->GetStorageShape().GetShapeSize() != rows * 64 ||
      context->GetInputShape(5)->GetStorageShape().GetShapeSize() != rows) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ubBytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
  const uint32_t cores = platform.GetCoreNumAiv();
  if (!cores || !ubBytes) return ge::GRAPH_FAILED;

  // Large inputs pair Q/KV branches of each token and keep both gammas.
  // Small inputs expose the two branches independently for more parallelism.
  const uint32_t branchesPerUnit = rows >= 512 ? 2 : 1;
  const uint32_t units = 2 * rows / branchesPerUnit;
  const uint32_t floatElements = branchesPerUnit == 2 ? 2112 : 1600;
  const uint64_t peakBytes = 2ull * (1536 + 1536 + 64 + 2 * 1536) + 20ull * floatElements + 32;
  if (peakBytes > ubBytes) return ge::GRAPH_FAILED;

  XMlaNormCacheTilingData tiling;
  tiling.set_rows(rows);
  tiling.set_slots(slots);
  tiling.set_units(units);
  tiling.set_branchesPerUnit(branchesPerUnit);
  tiling.set_floatElements(floatElements);
  tiling.set_epsQ(epsQ);
  tiling.set_epsKv(epsKv);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(std::min(units, cores));
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(XMlaNormCache).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
