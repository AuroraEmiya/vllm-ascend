// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_mla_split_rope_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto* a = context->GetAttrs();
  const int64_t mode = *a->GetInt(0), rows = *a->GetInt(1), heads = *a->GetInt(2), slots = *a->GetInt(3);
  const float epsQ = *a->GetFloat(4), epsKv = *a->GetFloat(5);
  if (!(mode == 1 || mode == 6 || mode == 8 || mode == 10 || mode == 30 || mode == 31 || mode == 32) || rows < 1 || rows > 65536 || heads < 1 || heads > 128 ||
      (heads & (heads - 1)) || slots < rows || slots > 1048576 || !std::isfinite(epsQ) || !std::isfinite(epsKv) ||
      epsQ <= 0 || epsKv <= 0)
    return ge::GRAPH_FAILED;
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  const uint32_t cores = platform.GetCoreNumAiv();
  if (!cores || !ub) return ge::GRAPH_FAILED;
  const auto& s = context->GetInputShape(0)->GetStorageShape();
  const bool norm = 1 == 0, split = 1 == 1;
  const int64_t inputRows = norm || split ? rows : heads * rows;
  const int64_t inputWidth = norm ? (mode == 0 ? 1536 : 2112) : (split ? heads * 192 : 512);
  if (s.GetDimNum() != 2 || s.GetDim(0) != inputRows || s.GetDim(1) != inputWidth) return ge::GRAPH_FAILED;
  if (norm && (context->GetInputShape(2)->GetStorageShape().GetShapeSize() != 1536 ||
               context->GetInputShape(3)->GetStorageShape().GetShapeSize() != 512))
    return ge::GRAPH_FAILED;
  if (norm && mode == 0 && context->GetInputShape(1)->GetStorageShape().GetShapeSize() != rows * 576)
    return ge::GRAPH_FAILED;
  if (norm || split) {
    if (context->GetInputShape(4)->GetStorageShape().GetShapeSize() != rows * 64 ||
        context->GetInputShape(5)->GetStorageShape().GetShapeSize() != rows * 64 ||
        context->GetInputShape(6)->GetStorageShape().GetShapeSize() != rows)
      return ge::GRAPH_FAILED;
  }
  XMlaSplitRopeTilingData t;
  t.set_mode(mode);
  t.set_rows(rows);
  t.set_heads(heads);
  t.set_slots(slots);
  t.set_epsQ(epsQ);
  t.set_epsKv(epsKv);
  uint32_t headTile = 1, units = 0, work = 0, fp32 = 0;
  uint64_t peak = 0;
  headTile = std::min<int64_t>(heads, mode < 30 && mode != 10 && rows >= 32 ? 32 : 16);
  if (mode >= 31) {
    // Grow the tile only while enough independent row/head units remain
    // to occupy the available AIV cores. Small decode batches expose heads.
    headTile = 1;
    while (headTile < std::min<int64_t>(heads, 32) &&
           rows * ((heads + 2 * headTile - 1) / (2 * headTile)) >= cores)
      headTile *= 2;
  }
  units = rows * ((heads + headTile - 1) / headTile);
  work = headTile * 64;
  fp32 = std::max(256u, work);
  const uint32_t inputHeadWidth = mode >= 32 ? 192 : 128;
  peak = (mode >= 30 ? 4ull : 2ull) * (headTile * inputHeadWidth + 128 + work) + 20ull * fp32 + 32;
  if (peak > ub || !units) return ge::GRAPH_FAILED;
  t.set_units(units);
  t.set_headTile(headTile);
  t.set_workElements(work);
  t.set_floatElements(fp32);
  t.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  context->SetBlockDim(std::min(units, cores));
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(XMlaSplitRope).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
