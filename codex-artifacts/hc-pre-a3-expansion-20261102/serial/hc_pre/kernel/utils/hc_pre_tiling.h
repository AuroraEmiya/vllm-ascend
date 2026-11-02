/*
 * Shared POD tiling header between op_host and the K2 cube kernel.
 * Kept dependency-free (no torch / no kernel_operator includes) so it can be
 * included from both translation units.
 */
#ifndef HC_PRE_TILING_H
#define HC_PRE_TILING_H

#include "adv_api/kernel_tiling.h"  // AscendC::tiling::TCubeTiling (POD)

struct HcPreMatmulTilingHeader {
    AscendC::tiling::TCubeTiling cubeTiling;  // filled by MatmulApiTiling on host
    int32_t ks;      // split-K segments == blockDim == AIC cores used
    int32_t kSeg;    // K elements per segment (max)
    int32_t m;       // M == batch rows
    int32_t n;       // N == hc_mix
    int32_t k;       // full K == fan_in
    int32_t nPad;    // C store width (N aligned up to the fp32 fractal c0=8)
};

#endif  // HC_PRE_TILING_H
