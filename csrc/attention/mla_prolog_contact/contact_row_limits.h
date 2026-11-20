// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ContactSerial {
constexpr uint32_t kBf16Bytes = 2;

inline bool TryGetRowSchedule(int64_t rawRows, uint32_t unitsPerRow, uint32_t cores,
                              uint32_t& rows, uint32_t& units) {
  constexpr uint64_t maxValue = std::numeric_limits<uint32_t>::max();
  if (rawRows <= 0 || static_cast<uint64_t>(rawRows) > maxValue || unitsPerRow == 0 || cores == 0) {
    return false;
  }

  const uint64_t wideUnits = static_cast<uint64_t>(rawRows) * unitsPerRow;
  const uint64_t blockDim = std::min(wideUnits, static_cast<uint64_t>(cores));
  // The kernel forms unit + blockDim even when there is no next task.
  if (wideUnits - 1 + blockDim > maxValue) {
    return false;
  }

  rows = static_cast<uint32_t>(rawRows);
  units = static_cast<uint32_t>(wideUnits);
  return true;
}

inline bool FitsUint32RowOffset(uint32_t rows, uint32_t stride) {
  return rows != 0 && (static_cast<uint64_t>(rows) - 1) * stride <= std::numeric_limits<uint32_t>::max();
}
}  // namespace ContactSerial
