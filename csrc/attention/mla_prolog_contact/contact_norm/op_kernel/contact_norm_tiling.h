// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstdint>
struct ContactNormTilingData {
  uint32_t rows;
  uint32_t heads;
  uint32_t qRank;
  uint32_t kvRank;
  uint32_t nope;
  uint32_t rope;
  uint32_t units;
  uint32_t headTile;
  uint32_t inElements;
  uint32_t outElements;
};
