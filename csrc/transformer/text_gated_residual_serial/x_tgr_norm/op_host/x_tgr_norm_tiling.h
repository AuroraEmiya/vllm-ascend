/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_X_TGR_NORM_TILING_H
#define VLLM_ASCEND_X_TGR_NORM_TILING_H
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(XTgrNormTilingData)
TILING_DATA_FIELD_DEF(uint32_t, rows);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(uint32_t, channels);
TILING_DATA_FIELD_DEF(uint32_t, mode);
TILING_DATA_FIELD_DEF(uint32_t, units);
TILING_DATA_FIELD_DEF(uint32_t, hasGamma);
TILING_DATA_FIELD_DEF(uint32_t, hasGate);
TILING_DATA_FIELD_DEF(uint32_t, tile);
TILING_DATA_FIELD_DEF(uint32_t, workElements);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(XTgrNorm, XTgrNormTilingData)
}  // namespace optiling
#endif
