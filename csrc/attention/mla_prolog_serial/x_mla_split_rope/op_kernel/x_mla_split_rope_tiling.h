// Licensed under CANN Open Software License Agreement Version 2.0.
#ifndef VLLM_ASCEND_X_MLA_SPLIT_ROPE_DEVICE_TILING_H
#define VLLM_ASCEND_X_MLA_SPLIT_ROPE_DEVICE_TILING_H
#include <cstdint>
struct XMlaSplitRopeTilingData {
  uint32_t mode;
  uint32_t rows;
  uint32_t heads;
  uint32_t slots;
  uint32_t units;
  uint32_t headTile;
  uint32_t workElements;
  uint32_t floatElements;
  float epsQ;
  float epsKv;
};
#endif
