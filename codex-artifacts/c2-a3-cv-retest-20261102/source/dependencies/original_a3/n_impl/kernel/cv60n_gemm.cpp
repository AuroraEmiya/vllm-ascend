// cv60n GEMM kernel: packed C-side projection [T,5120] x [5120,1024] -> pair [T,1024].
// Columns 0-511 are KV, columns 512-1023 are gate scores (packed layout, same
// interface detail the frozen baseline B exposes through its own C kernel).
// FP32 x FP32 -> FP32, HF32 off, NORMAL mad; reduction order fixed by the
// host-side MatmulApiTiling candidate recorded in the tiling byte tensor.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "cv60n_tiling.h"

using namespace AscendC;
using namespace matmul;
constexpr MatmulConfig CV60N_NORM_CONFIG = GetNormalConfig();
constexpr MatmulConfig CV60N_MDL_CONFIG = GetMDLConfig();
using AType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template<const MatmulConfig& Config, CubeFormat BFormat>
__aicore__ inline void Compute(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace,
                               GM_ADDR tiling) {
    if ASCEND_IS_AIV { return; }
    Cv60nGemmTiling t;
    auto src = reinterpret_cast<const __gm__ int32_t*>(tiling);
    auto dst = reinterpret_cast<int32_t*>(&t);
    static_assert(sizeof(Cv60nGemmTiling) % sizeof(int32_t) == 0);
    for (uint32_t i = 0; i < sizeof(Cv60nGemmTiling) / sizeof(int32_t); ++i) dst[i] = src[i];
    TPipe pipe;
    using BType = MatmulType<TPosition::GM, BFormat, float, true>;
    Matmul<AType, BType, CType, CType, Config> mm;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm, &t.cube);
    mm.SetHF32(false, 0);
    mm.SetOrgShape(static_cast<int32_t>(t.rows), 1024, 5120, 5120, 1024);
    GlobalTensor<float> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    const int32_t nTasks = (1024 + t.singleN - 1) / t.singleN;
    const int32_t block = static_cast<int32_t>(GetBlockIdx());
    const int32_t blocks = t.blocks;
    const bool affineRows = blocks % nTasks == 0;
    const int64_t rowStep = affineRows ? static_cast<int64_t>(blocks / nTasks) * static_cast<int64_t>(t.singleM) : 0;
    int64_t m0 = affineRows ? static_cast<int64_t>(block / nTasks) * static_cast<int64_t>(t.singleM) : 0;
    int64_t n0 = affineRows ? static_cast<int64_t>(block % nTasks) * static_cast<int64_t>(t.singleN) : 0;
    for (int64_t task = block; task < t.tasks; task += blocks) {
        if (!affineRows) {
            m0 = (task / nTasks) * static_cast<int64_t>(t.singleM);
            n0 = (task % nTasks) * static_cast<int64_t>(t.singleN);
        }
        const int32_t tailM = static_cast<int32_t>(t.rows - m0 < t.singleM ? t.rows - m0 : t.singleM);
        const int32_t tailN = static_cast<int32_t>(1024 - n0 < t.singleN ? 1024 - n0 : t.singleN);
        mm.SetTail(tailM, tailN, 5120);
        mm.SetTensorA(a[m0 * int64_t(5120)], false);
        // NZ keeps the full original N stride; only its task base changes.
        constexpr int64_t bRowOffset = BFormat == CubeFormat::NZ ? 8 : 5120;
        mm.SetTensorB(b[n0 * bRowOffset], true);
        mm.IterateAll(c[m0 * int64_t(1024) + n0]);
        mm.End();
        if (affineRows) {
            m0 += rowStep;
        }
    }
}

extern "C" __global__ __aicore__ void cv60n_gemm_nd_norm(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    Compute<CV60N_NORM_CONFIG, CubeFormat::ND>(x, w, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void cv60n_gemm_nd_mdl(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    Compute<CV60N_MDL_CONFIG, CubeFormat::ND>(x, w, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void cv60n_gemm_nz_norm(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                         GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    Compute<CV60N_NORM_CONFIG, CubeFormat::NZ>(x, w, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void cv60n_gemm_nz_mdl(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    Compute<CV60N_MDL_CONFIG, CubeFormat::NZ>(x, w, y, workspace, tiling);
}
#ifndef __CCE_KT_TEST__
extern "C" void launch_cv60n_gemm(uint32_t blocks, int32_t mode, int32_t layout, void* stream,
                                  uint8_t* x, uint8_t* w, uint8_t* y, uint8_t* ws, uint8_t* t) {
    if (layout == 0) {
        if (mode == 0) cv60n_gemm_nd_norm<<<blocks, nullptr, stream>>>(x, w, y, ws, t);
        else cv60n_gemm_nd_mdl<<<blocks, nullptr, stream>>>(x, w, y, ws, t);
    } else {
        if (mode == 0) cv60n_gemm_nz_norm<<<blocks, nullptr, stream>>>(x, w, y, ws, t);
        else cv60n_gemm_nz_mdl<<<blocks, nullptr, stream>>>(x, w, y, ws, t);
    }
}
#endif
