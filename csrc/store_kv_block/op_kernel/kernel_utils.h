/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_UTILS_COMMON_KERNEL_KERNEL_UTILS_H
#define ASCEND_OPS_UTILS_COMMON_KERNEL_KERNEL_UTILS_H
#include "kernel_operator.h"

using AscendC::HardEvent;

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : ((x + y - 1) / y);
}

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
{
    return (x + y - 1) / y * y;
}

__aicore__ inline uint32_t Min(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

__aicore__ inline uint32_t Max(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

#endif
