#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Host/device tiling contract for the cv60n packed GEMM kernel.
// The struct is memcpy'd into a byte tensor on the host and copied word-wise
// on the device, exactly like the frozen baseline family does.
struct C2Bf16GemmTiling {
    TCubeTiling cube;
    int64_t rows;
    int64_t tasks;
    int32_t singleM;
    int32_t singleN;
    int32_t blocks;
    int32_t mode;
    int32_t layout;
};
