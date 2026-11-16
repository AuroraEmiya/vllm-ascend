// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>

struct XMlaSplitRopeTilingData {
  uint32_t rows;
  uint32_t heads;
  uint32_t units;
  uint32_t headTile;
  uint32_t workElements;
  uint32_t floatElements;
  uint32_t inputHeadWidth;
};
