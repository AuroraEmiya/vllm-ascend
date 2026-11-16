// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>

struct XMlaLayoutTilingData {
  uint32_t rows;
  uint32_t heads;
  uint32_t units;
  uint32_t headTile;
  uint32_t rowTile;
  uint32_t workElements;
};
