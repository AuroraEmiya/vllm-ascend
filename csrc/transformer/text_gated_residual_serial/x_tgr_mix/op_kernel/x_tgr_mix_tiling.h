/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef X_TGR_MIX_TILING_H
#define X_TGR_MIX_TILING_H
#include <cstdint>

struct XTgrMixTilingData {
  uint32_t rows;
  uint32_t width;
  uint32_t channels;
  uint32_t mode;
  uint32_t units;
  uint32_t hasGamma;
  uint32_t hasGate;
  uint32_t tile;
  uint32_t workElements;
  float eps;
};

#endif  // X_TGR_MIX_TILING_H
