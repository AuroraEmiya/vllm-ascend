/*
 * HcPre K2 — cube kernel (AIC only), CANN-reference round.
 *
 * torch_ops/hc_pre.py stage covered: C (HF32 GEMM + inv_rms scaling — the
 * scaling itself lives in K3's epilogue; this kernel emits raw partials).
 *
 * CANN implementation pattern (per-round mandate): MatmulImpl standard usage
 * — archive_tasks/matmul_leakyrelu + ascendc-tiling-design
 * references/matmul/ascendc-api-matmul-tiling.md:
 *   host : MatmulApiTiling::GetTiling(TCubeTiling&) + mandatory overwrite of
 *          M / N / Ka / Kb, TilingHeader uploaded via a blocking copy
 *   kernel: LoadTiling -> mm_.Init(&cubeTiling, pipe) -> per-blockIdx
 *          SetSingleShape + SetTensorA/B + IterateAll, with the mandatory
 *          `if (GetBlockIdx() >= ks) return;` bounds guard.
 *
 * Inputs : x_f32_ws [B, K] fp32 (K = hc_mult * D, produced by K1; bf16-origin
 *          values, exact fp32 cast), hc_fn_t [K, N] fp32 (host-pre-transposed)
 * Output : partial [KS, B, N] fp32, partial[s] = x[:, s*kSeg:(s+1)*kSeg] @
 *          hc_fn_t[s*kSeg:(s+1)*kSeg, :]
 *
 * Shape regime: K = 20480 >> M*N (128*24) -> deterministic split-K (the
 * ascendc-tiling-design Split-K routing: workspace partials + ordered
 * epilogue reduction; no ATOMIC_ADD — deterministic and reproducible).
 * M and N stay single tiles (M <= 128; N = 24 is a multiple of the fp32
 * fractal c0 = 8, no tail). blockDim == KS == AIC cores used.
 *
 * Precision: HF32 cube mode — host SetMadType(MatrixMadType::HF32) +
 * kernel SetHF32(true, 0), the v6-proven configuration (three rounds of
 * evaluate PASS on this platform). Semantics match the reference
 * _to_hf32(x) @ _to_hf32(hc_fn): the A operand is bf16-origin (its HF32
 * truncation is lossless), and the cube unit truncates the fp32 hc_fn
 * operand to the 10-bit HF32 mantissa. Arbitration note: the requirements
 * "SetHF32(true) diverges" constraint traces to the merge-round hand-written
 * Mmad probes (SetHF32Mode(true)+RTZ, previous trace.md), NOT to the
 * MatmulImpl path; the KEEP_FLOAT_DTYPE pair (false,0) was additionally
 * measured this round to be numerically fine but 30% slower (fp32 mad
 * throughput is 1/4 of HF32, turning this bandwidth-bound skinny GEMM
 * compute-bound).
 *
 * Platform workaround (W1, mandatory): the MatmulImpl trans-B (isTrans=true)
 * path produces misaligned output on this platform, so the host pre-transposes
 * hc_fn into hc_fn_t [K, N] (cached by tensor identity) and both operands are
 * consumed no-trans here.
 *
 * Design source: design/tile_level/hc_pre.py::hc_pre_matmul
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "adv_api/matmul/tiling.h"  // CFG_NORM (enUnitFlag = true by default)

#include "hc_pre_tiling.h"

using namespace AscendC;
using namespace matmul;

namespace {

class HcPreMatmulKernel {
public:
    __aicore__ inline HcPreMatmulKernel() {}

    __aicore__ inline void Init(GM_ADDR xF32WsIn, GM_ADDR hcFnTIn, GM_ADDR partialIn,
                                GM_ADDR tilingIn, TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            return;
        }
        LoadTilingFromGM<HcPreMatmulTilingHeader>(tilingIn, header_);
        aBase_ = reinterpret_cast<__gm__ float *>(xF32WsIn);
        bBase_ = reinterpret_cast<__gm__ float *>(hcFnTIn);
        aGm_.SetGlobalBuffer(aBase_);
        bGm_.SetGlobalBuffer(bBase_);
        cGm_.SetGlobalBuffer((__gm__ float *)partialIn);
        mm_.SetSubBlockIdx(0);
        mm_.Init(&header_.cubeTiling, pipe);
        // HF32 cube mode with clearRtol=0, pairing with the host
        // MatrixMadType::HF32 — the v6-proven configuration (three rounds of
        // evaluate PASS, K2 = 10.74us). Arbitration note: the requirements
        // "SetHF32(true) diverges" constraint traces to the merge-round
        // hand-written Mmad path (SetHF32Mode(true)+RTZ probes, trace.md
        // L222), NOT to MatmulImpl; KEEP_FLOAT (false,0) was measured 30%
        // slower this round (fp32 mad throughput = 1/4 of HF32, turning this
        // bandwidth-bound skinny GEMM compute-bound: 13.9us vs 10.7us).
        mm_.SetHF32(true, 0);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;  // pure cube kernel: vector cores exit immediately
        }
        const int32_t blockIdx = GetBlockIdx();
        if (blockIdx >= header_.ks) {
            return;  // mandatory bounds guard (extra cores must exit)
        }
        // Segment-major operands: A = xF32Ws [ks][M][kSeg] and B = hcFnT
        // [ks][kSeg][N], both compact per segment (the tiling Ka/Kb define
        // the operand row strides, so each segment must be a tight matrix).
        // Anchor the GlobalTensors at each segment start.
        const int64_t aOff = static_cast<int64_t>(blockIdx) * header_.m * header_.kSeg;
        const int64_t bOff = static_cast<int64_t>(blockIdx) * header_.kSeg * header_.n;
        aGm_.SetGlobalBuffer(aBase_ + aOff);
        bGm_.SetGlobalBuffer(bBase_ + bOff);
        mm_.SetSingleShape(header_.m, header_.n, header_.kSeg);
        mm_.SetTensorA(aGm_, false);
        mm_.SetTensorB(bGm_, false);
        // IterateAll stores C as a tight [M, N] ND matrix (row stride N), so
        // partial segments are compact [M, N] blocks
        const int64_t cOff = static_cast<int64_t>(blockIdx) * header_.m * header_.n;
        mm_.IterateAll(cGm_[cOff], 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void End() { mm_.End(); }

private:
    template <typename HeaderT>
    __aicore__ inline void LoadTilingFromGM(GM_ADDR tilingGM, HeaderT &dstStack)
    {
        constexpr uint32_t headerInts =
            (sizeof(HeaderT) + sizeof(int32_t) - 1U) / sizeof(int32_t);
        const auto *gmInts = reinterpret_cast<const __gm__ int32_t *>(tilingGM);
        auto *dstInts = reinterpret_cast<int32_t *>(&dstStack);
        for (uint32_t i = 0; i < headerInts; ++i) {
            dstInts[i] = gmInts[i];
        }
    }

    HcPreMatmulTilingHeader header_{};
    using A_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using B_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> mm_;
    GlobalTensor<float> aGm_, bGm_, cGm_;
    __gm__ float *aBase_ = nullptr;
    __gm__ float *bBase_ = nullptr;
};
}  // namespace

extern "C" __global__ __aicore__ void hc_pre_matmul(GM_ADDR x_f32_ws, GM_ADDR hc_fn_t,
                                                    GM_ADDR partial, GM_ADDR tiling)
{
    TPipe pipe;
    HcPreMatmulKernel op;
    op.Init(x_f32_ws, hc_fn_t, partial, tiling, &pipe);
    op.Process();
    op.End();
}
