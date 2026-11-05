// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ContactLayoutTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, qRank);
TILING_DATA_FIELD_DEF(uint32_t, kvRank);
TILING_DATA_FIELD_DEF(uint32_t, nope);
TILING_DATA_FIELD_DEF(uint32_t, rope);
TILING_DATA_FIELD_DEF(uint32_t, units);
TILING_DATA_FIELD_DEF(uint32_t, headTile);
TILING_DATA_FIELD_DEF(uint32_t, inElements);
TILING_DATA_FIELD_DEF(uint32_t, outElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ContactLayout, ContactLayoutTilingData)
}
