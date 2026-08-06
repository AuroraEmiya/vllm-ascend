/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file decode_index_score_tiling_key.h
 * \brief Compile-time tiling key: the query/index-k dtype pair (bf16 or fp16).
 */

#ifndef DECODE_INDEX_SCORE_TILING_KEY_H_
#define DECODE_INDEX_SCORE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define DI_TPL_FP16 1   // ge::DT_FLOAT16
#define DI_TPL_BF16 27  // ge::DT_BF16

// Template args supported by the op: DT_Q and DT_K share the input dtype.
ASCENDC_TPL_ARGS_DECL(DecodeIndexScore,
                      ASCENDC_TPL_DTYPE_DECL(DT_Q, DI_TPL_FP16, DI_TPL_BF16),
                      ASCENDC_TPL_DTYPE_DECL(DT_K, DI_TPL_FP16, DI_TPL_BF16), );

// Valid combinations (used by GET_TPL_TILING_KEY on the host to validate keys).
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, DI_TPL_FP16), ASCENDC_TPL_DTYPE_SEL(DT_K, DI_TPL_FP16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, DI_TPL_BF16), ASCENDC_TPL_DTYPE_SEL(DT_K, DI_TPL_BF16)), );

#endif // DECODE_INDEX_SCORE_TILING_KEY_H_
