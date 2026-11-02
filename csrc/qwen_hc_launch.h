// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

struct HcParams {
    uint32_t rows, hidden, groups;
    uint32_t residual_stride, block_stride, injection_stride;
    uint32_t shared_weight, has_injection;
    uint32_t tile, row_batch;
    float eps;
};
void qwen_hc_combine_norm_launch(uint32_t,void*,void*,void*,void*,void*,void*,void*,HcParams);
void qwen_hc_silu_launch(uint32_t,void*,void*,void*,HcParams);
void qwen_hc_gate_mix_launch(uint32_t,void*,void*,void*,void*,HcParams);
