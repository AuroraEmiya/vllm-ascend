// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(CompressorSerialPoolTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tokens);
TILING_DATA_FIELD_DEF(uint32_t, groups);
TILING_DATA_FIELD_DEF(uint32_t, batches);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, coff);
TILING_DATA_FIELD_DEF(uint32_t, ratio);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, tableCols);
TILING_DATA_FIELD_DEF(uint32_t, hasUsed);
TILING_DATA_FIELD_DEF(uint32_t, blocks);
TILING_DATA_FIELD_DEF(uint32_t, tileCols);
TILING_DATA_FIELD_DEF(uint32_t, channelTile);
TILING_DATA_FIELD_DEF(uint32_t, stateTile);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CompressorSerialPool, CompressorSerialPoolTilingData)
}  // namespace optiling
