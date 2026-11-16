// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>

struct XMlaNormCacheTilingData {
  uint32_t rows;
  uint32_t slots;
  uint32_t units;
  uint32_t branchesPerUnit;
  uint32_t floatElements;
  float epsQ;
  float epsKv;
};
