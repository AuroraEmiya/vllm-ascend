// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(XMlaLayoutTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, units);
TILING_DATA_FIELD_DEF(uint32_t, headTile);
TILING_DATA_FIELD_DEF(uint32_t, rowTile);
TILING_DATA_FIELD_DEF(uint32_t, workElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XMlaLayout, XMlaLayoutTilingData)
}  // namespace optiling
