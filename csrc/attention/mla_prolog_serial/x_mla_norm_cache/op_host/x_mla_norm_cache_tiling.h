// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef VLLM_ASCEND_X_MLA_NORM_CACHE_TILING_H
#define VLLM_ASCEND_X_MLA_NORM_CACHE_TILING_H
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(XMlaNormCacheTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mode);
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, slots);
TILING_DATA_FIELD_DEF(uint32_t, units);
TILING_DATA_FIELD_DEF(uint32_t, headTile);
TILING_DATA_FIELD_DEF(uint32_t, workElements);
TILING_DATA_FIELD_DEF(uint32_t, floatElements);
TILING_DATA_FIELD_DEF(float, epsQ);
TILING_DATA_FIELD_DEF(float, epsKv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XMlaNormCache, XMlaNormCacheTilingData)
}  // namespace optiling
#endif
