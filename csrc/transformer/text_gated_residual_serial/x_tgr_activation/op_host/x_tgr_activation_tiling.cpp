// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "x_tgr_activation_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto& s = context->GetInputShape(0)->GetStorageShape();
  const auto* a = context->GetAttrs();
  const int64_t requested = *a->GetInt(0), channels = *a->GetInt(1);
  const uint32_t mode = requested >= 10 && requested <= 11
                            ? 5
                            : (requested == 9 ? 5 : (requested == 8 ? 4 : (requested == 6 ? 2 : requested)));
  const float eps = *a->GetFloat(2);
  if (s.GetDimNum() != 2 || s.GetDim(0) < 1 || s.GetDim(0) > 2048 || channels != 4 || !std::isfinite(eps) || eps <= 0 ||
      !(mode == 1 || mode == 3 || mode == 5))
    return ge::GRAPH_FAILED;
  const uint32_t rows = s.GetDim(0), width = s.GetDim(1);
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  const uint32_t cores = platform.GetCoreNumAiv();
  if (!ub || !cores) return ge::GRAPH_FAILED;
  const auto gamma = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  const auto gate = context->GetInputShape(3)->GetStorageShape().GetShapeSize();
  XTgrActivationTilingData t;
  t.set_rows(rows);
  t.set_width(width);
  t.set_channels(channels);
  t.set_mode(mode);
  t.set_eps(eps);
  t.set_hasGamma(gamma == width);
  t.set_hasGate(mode == 2 && gate == rows * channels);
  uint32_t tile = 1, units = 0, work = 0;
  uint64_t peak = 0;

  if ((mode == 1 && width != 320) || ((mode == 3 || mode == 5) && width != (requested >= 10 ? 640 : 336)))
    return ge::GRAPH_FAILED;
  if (mode == 1) {
    tile = std::min(2560u, std::max(256u, ((rows * width + cores * 32 - 1) / (cores * 32)) * 32));
    work = tile;
    units = (rows * width + tile - 1) / tile;
  } else {
    if (mode == 5 && requested != 10) {
      while (tile < 8 && (rows + 2 * tile - 1) / (2 * tile) >= cores) tile *= 2;
    }
    work = 336 * tile;
    units = (rows + tile - 1) / tile;
  }
  peak = 12ull * work;

  if (!units || peak > ub) return ge::GRAPH_FAILED;
  t.set_tile(tile);
  t.set_units(units);
  t.set_workElements(work);
  t.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  context->SetBlockDim(std::min(units, cores));
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(XTgrActivation).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
