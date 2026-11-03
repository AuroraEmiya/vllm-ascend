/*
 * HcPre op_host — tiling computation + strictly serial launches of the three
 * CV-serial kernels (CANN-reference round; see design/DESIGN.md):
 *
 *   K1 hc_pre_preprocess [AIV]  : stages A+B+F — x bf16 -> xF32Ws fp32 +
 *                                  invRms + y (canonical TQue double-buffered
 *                                  three-phase pipeline)
 *   K2 hc_pre_matmul     [AIC]  : stage C — deterministic split-K GEMM,
 *                                  KEEP_FLOAT_DTYPE (MatrixMadType::NORMAL)
 *   K3 hc_pre_epilogue   [AIV]  : stages D+E — reduce partials ->
 *                                  pre/post/comb_frag (Sinkhorn)
 *
 * All kernel arguments are named lvalues (EXEC_KERNEL_CMD contract); the only
 * struct crossing the ABI is the K2 TilingHeader (MatmulImpl requires a
 * TCubeTiling), uploaded as a device byte tensor via a blocking H2D copy.
 *
 * Core counts are queried from PlatformAscendC at runtime (no hardcoding).
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/extension.h>

#include "adv_api/matmul/matmul_tiling.h"
#include "adv_api/reduce/reduce_tiling.h"
#include "adv_api/pad/broadcast_tiling.h"
#include "adv_api/activation/sigmoid_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include "torch_kernel_helper.h"
#include "hc_pre_tiling.h"

#include "kernels/aclrtlaunch_hc_pre_preprocess.h"
#include "kernels/aclrtlaunch_hc_pre_matmul.h"
#include "kernels/aclrtlaunch_hc_pre_epilogue.h"

namespace ascend_kernel {
namespace {

int64_t CeilDiv(const int64_t a, const int64_t b)
{
    return (a + b - 1) / b;
}

uint32_t ReduceTmpFor(const AscendC::ReducePattern pattern, const std::vector<int64_t> &shape,
                      const bool innerPad)
{
    // host-side query for ReduceSum/ReduceMax work buffer sizes (fp32)
    std::vector<int64_t> dims(shape);
    const ge::Shape geShape(dims);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    GetReduceSumMaxMinTmpSize(geShape, ge::DT_FLOAT, pattern, innerPad, false, maxValue, minValue);
    uint32_t value = std::max(maxValue, minValue);
    GetReduceMaxMaxMinTmpSize(geShape, ge::DT_FLOAT, pattern, innerPad, false, maxValue, minValue);
    value = std::max({value, maxValue, minValue});
    return value;
}

uint32_t BrcTmpFor(const platform_ascendc::PlatformAscendC &platform, const std::vector<int64_t> &src,
                   const std::vector<int64_t> &dst)
{
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetBroadCastMaxMinTmpSize(platform, ge::Shape(src), ge::Shape(dst), sizeof(float),
                                       false, maxValue, minValue);
    return std::max(maxValue, minValue);
}

uint32_t SigmoidTmpFor(const std::vector<int64_t> &shape)
{
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSigmoidMaxMinTmpSize(ge::Shape(shape), sizeof(float), false, maxValue, minValue);
    return std::max(maxValue, minValue);
}

}  // namespace

// WORKAROUND W1 (platform-mandated, see design/DESIGN.md §6): the MatmulImpl
// trans-B (isTrans=true) path produces misaligned output on this platform, so
// hc_fn [mix, fanIn] is pre-transposed on the host into hcFnT [fanIn, mix] and
// consumed no-trans by K2. hc_fn is a constant weight in inference: the
// transpose is cached by (data_ptr, sizes) so the one-time cost amortizes to
// zero in steady state (same policy as weight-repack caching in conv
// pipelines).
//
// D-fix (precision, mandatory): hcFnT additionally gets its low 13 mantissa
// bits cleared (RTZ truncation), which is bit-exact with the frozen CPU
// oracle's _to_hf32(hc_fn). Rationale: the cube HF32 conversion is a NEAREST
// rounding (ties-even/ties-away per the SetHF32 transMode — measured to
// differ only at tie points, ~1e-7 on contract data), NOT a truncation; on
// hc_fn's all-positive rand data nearest-vs-RTZ leaves a systematic
// per-element bias that accumulates over K=20480 into max pre/post
// deviations of ~3.4e-5/~5.5e-5 vs the oracle — only 1.8x under the 1e-4
// gate, unstable for the 24-case regression. With the pre-truncated operand
// the in-cube HF32 conversion is the identity (mantissa already 10-bit), so
// the GEMM consumes exactly the oracle's operand and the residual collapses
// to sum-order noise.
const at::Tensor &GetHcFnT(const at::Tensor &hc_fn)
{
    struct Key
    {
        const void *ptr;
        int64_t rows;
        int64_t cols;
        bool operator==(const Key &o) const { return ptr == o.ptr && rows == o.rows && cols == o.cols; }
    };
    struct KeyHash
    {
        size_t operator()(const Key &k) const { return std::hash<const void *>{}(k.ptr); }
    };
    static std::unordered_map<Key, std::pair<at::Tensor, at::Tensor>, KeyHash> cache;
    Key key{hc_fn.data_ptr(), hc_fn.size(0), hc_fn.size(1)};
    auto it = cache.find(key);
    if (it == cache.end()) {
        // keep the original alive so the caching allocator never hands its
        // address to a different weight (prevents stale-cache hits)
        at::Tensor hcFnT = hc_fn.t().contiguous();
        // RTZ pre-truncation to the 10-bit HF32 mantissa (bitwise, no fp math)
        at::Tensor mask = at::scalar_tensor(~((1 << 13) - 1), hc_fn.options().dtype(at::kInt));
        hcFnT = at::bitwise_and(hcFnT.view(at::kInt), mask).view(hc_fn.scalar_type());
        it = cache.emplace(key, std::make_pair(hc_fn, std::move(hcFnT))).first;
    }
    return it->second.second;
}

// WORKAROUND W3 (platform-mandated, precision-critical): [1,1,1,1,0,0,0,0]
// constant mask. GM->UB 16B block copies zero tail columns at 32B write
// granularity; exp() would turn those zero pads into 1s and the +eps would
// leak eps into the pad lanes, permanently biasing the Sinkhorn denominators.
// K3 multiplies this mask right after exp() and again after the +eps. There
// is no standard-API path that avoids the 32B-granularity tail zeroing (the
// band loads are inherently 16B), so the mask is mandatory.
const at::Tensor &GetMask8(const at::Device &device)
{
    static at::Tensor cache;
    if (!cache.defined()) {
        at::Tensor cpu = at::tensor({1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                    at::device(at::kCPU).dtype(at::kFloat));
        cache = cpu.to(device);
    }
    return cache;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hc_pre(
    const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
    const at::Tensor &hc_base, const at::Tensor &pre_mix, int64_t hc_mult,
    int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps)
{
    TORCH_CHECK(x.dim() == 3, "x must be [B, hc_mult, D]");
    const int64_t B = x.size(0);
    const int64_t hc = x.size(1);
    const int64_t D = x.size(2);
    const int64_t fanIn = hc * D;
    const int64_t mix = hc_fn.size(0);  // 24 = 2*hc + hc*hc
    TORCH_CHECK(hc_fn.dim() == 2 && hc_fn.size(1) == fanIn, "hc_fn must be [hc_mix, fan_in]");
    TORCH_CHECK(mix == 2 * hc + hc * hc, "hc_mix must equal 2*hc + hc^2");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16 (target shapes)");
    TORCH_CHECK(hc_fn.scalar_type() == at::kFloat && hc_scale.scalar_type() == at::kFloat &&
                    hc_base.scalar_type() == at::kFloat && pre_mix.scalar_type() == at::kFloat,
                "hc_fn/hc_scale/hc_base/pre_mix must be fp32");
    TORCH_CHECK(x.is_contiguous() && hc_fn.is_contiguous() && pre_mix.is_contiguous(),
                "inputs must be contiguous");

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get PlatformAscendC");
    const int64_t aivCoreNum = platform->GetCoreNumAiv();
    const int64_t aicCoreNum = platform->GetCoreNumAic();

    const auto outOpt = x.options().dtype(at::kFloat);
    const auto outBfOpt = x.options();

    at::Tensor y = at::empty({B, D}, outBfOpt);
    at::Tensor post = at::empty({B, hc}, outOpt);
    at::Tensor combFrag = at::empty({B, hc, hc}, outOpt);
    at::Tensor pre = at::empty({B, hc}, outOpt);

    at::Tensor xF32Ws = at::empty({B, fanIn}, outOpt);
    at::Tensor invRms = at::empty({B}, outOpt);
    at::Tensor mixesWs = at::empty({B, (mix + 7) / 8 * 8}, outOpt);

    const float normEpsF = static_cast<float>(norm_eps);
    const float hcEpsF = static_cast<float>(hc_eps);

    // ---------------- K2 tiling (MatmulApiTiling + split-K) ----------------
    // Derived first: its kSeg also defines the K1 ws segment layout.
    // Segment-major operands need fanIn % kSeg == 0 and kSeg a multiple of 64
    // (cube fractal). Use the largest divisor of fanIn that is <= aicCoreNum
    // and a multiple of 64 (target: 20480 -> kSeg 1024, ks 20 — the v6-proven
    // optimal family; K = 20480 >> M*N so split-K is mandatory for core use).
    int64_t kSeg = fanIn;
    for (int64_t cand = aicCoreNum; cand >= 1; --cand) {
        if (fanIn % cand == 0 && (fanIn / cand) % 64 == 0) {
            kSeg = fanIn / cand;
            break;
        }
    }
    const int64_t ks = CeilDiv(fanIn, kSeg);

    HcPreMatmulTilingHeader mmHeader{};
    {
        namespace mt = matmul_tiling;
        mt::MatmulApiTiling mmTiling(*platform);
        mmTiling.SetAType(mt::TPosition::GM, mt::CubeFormat::ND, mt::DataType::DT_FLOAT, false);
        mmTiling.SetBType(mt::TPosition::GM, mt::CubeFormat::ND, mt::DataType::DT_FLOAT, false);
        mmTiling.SetShape(B, mix, kSeg);
        // HF32 mad type, pairing with the kernel's SetHF32(true, 0) — the
        // v6-proven configuration (see the arbitration note in
        // op_kernel/hc_pre_matmul.cpp: the "SetHF32(true) diverges" platform
        // note traces to the merge-round hand-written Mmad probes, not to
        // MatmulImpl; KEEP_FLOAT false,0 was measured 30% slower this round)
        mmTiling.SetMadType(mt::MatrixMadType::HF32);
        TORCH_CHECK(mmTiling.GetTiling(mmHeader.cubeTiling) == 0, "MatmulApiTiling failed");
        // mandatory overwrite (ascendc-api-matmul-tiling.md checklist): M/N/Ka/Kb
        mmHeader.cubeTiling.M = static_cast<uint32_t>(B);
        mmHeader.cubeTiling.N = static_cast<uint32_t>(mix);
        mmHeader.cubeTiling.Ka = static_cast<uint32_t>(kSeg);
        mmHeader.cubeTiling.Kb = static_cast<uint32_t>(kSeg);
        mmHeader.ks = static_cast<int32_t>(ks);
        mmHeader.kSeg = static_cast<int32_t>(kSeg);
        mmHeader.m = static_cast<int32_t>(B);
        mmHeader.n = static_cast<int32_t>(mix);
        mmHeader.k = static_cast<int32_t>(fanIn);
        mmHeader.nPad = static_cast<int32_t>((mix + 7) / 8 * 8);
    }

    at::Tensor partial = at::empty({ks, B, mix}, outOpt);

    at::Tensor mmTilingCpu = at::empty({static_cast<int64_t>(sizeof(HcPreMatmulTilingHeader))},
                                       at::device(at::kCPU).dtype(at::kByte));
    std::memcpy(mmTilingCpu.data_ptr(), &mmHeader, sizeof(mmHeader));
    at::Tensor mmTilingNpu = mmTilingCpu.to(at::kPrivateUse1);  // blocking copy

    // ---------------- K1 tiling ----------------
    // Whole-row grid-stride schedule (one core owns whole rows so the per-row
    // RMS reduction stays in-core); rows stream as hc [D] blocks through the
    // canonical double-buffered TQue pipeline. The ws segment size equals the
    // K2 split-K segment (kSeg) — CopyOut stores 5x4KB blocks per sub-row.
    TORCH_CHECK(D % kSeg == 0, "D must be divisible by the split-K segment (target shapes)");
    const int64_t k1Block = std::min(B, aivCoreNum);
    const int64_t dChunks = CeilDiv(D, kSeg);

    uint32_t k1RedMax = 0;
    uint32_t k1RedMin = 0;
    {
        // K1 reduces the folded [hc * 80] square-sum slab once (Add-tree
        // folding handles the 5120 -> 80 part, no wide AR reduce is issued)
        const ge::Shape redShape(std::vector<int64_t>{1, hc * 80});
        GetReduceSumMaxMinTmpSize(redShape, ge::DT_FLOAT, AscendC::ReducePattern::AR, false, false,
                                  k1RedMax, k1RedMin);
    }
    // UB budget with double-buffered queues: inQueueX 2x10KB + outQueueXf
    // 2x20KB + t1/yAcc/yUb 50KB ~ 130KB, so take the API minimum (still
    // legal) for the reduce work buffer.
    const int64_t k1RedTmp = static_cast<int64_t>(k1RedMin);

    // ---------------- K3 tiling ----------------
    // rowsPerCore adapts to B: large B keeps 2*HC (16 cores at B=128 — more
    // cores measurably raise the AIV raster/launch tax and hurt), small B uses
    // HC (8 cores at B=32 — halves the per-core IO of the serial Sinkhorn
    // chain). [rows,1] broadcast sources are 16B/32B (verified family).
    TORCH_CHECK(B % hc == 0, "B must be divisible by hc_mult (target shapes)");
    const int64_t k3RowsPC = (B >= 64) ? (2 * hc) : hc;
    const int64_t k3Block = CeilDiv(B, k3RowsPC);
    const int64_t nRows = k3RowsPC * hc;

    uint32_t k3Red = ReduceTmpFor(AscendC::ReducePattern::RA, {ks, k3RowsPC * 2 * hc}, false);
    k3Red = std::max(k3Red, ReduceTmpFor(AscendC::ReducePattern::RA, {ks, k3RowsPC * hc * hc}, false));
    k3Red = std::max(k3Red, ReduceTmpFor(AscendC::ReducePattern::AR, {k3RowsPC, 2 * hc}, false));
    k3Red = std::max(k3Red, ReduceTmpFor(AscendC::ReducePattern::AR, {nRows, 8}, false));
    k3Red = std::max(k3Red, ReduceTmpFor(AscendC::ReducePattern::RA, {hc, k3RowsPC * 8}, false));
    uint32_t k3Op = BrcTmpFor(*platform, {k3RowsPC, 1}, {k3RowsPC, 2 * hc});
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {1, 2 * hc}, {k3RowsPC, 2 * hc}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {k3RowsPC, 1}, {k3RowsPC, hc * hc}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {1, hc * hc}, {k3RowsPC, hc * hc}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {nRows, 1}, {nRows, 8}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {1, 8}, {nRows, 8}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {1, k3RowsPC * 8}, {hc, k3RowsPC * 8}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {k3RowsPC, 1}, {k3RowsPC, hc}));
    k3Op = std::max(k3Op, BrcTmpFor(*platform, {1, hc}, {k3RowsPC, hc}));
    k3Op = std::max(k3Op, SigmoidTmpFor({k3RowsPC * 8}));
    const int64_t k3RedTmp = static_cast<int64_t>(k3Red);
    const int64_t k3OpTmp = static_cast<int64_t>(k3Op);

    // ---------------- strictly serial launches (W5: no MIX merging) ----------------
    // debug switch: HC_PRE_STAGES=1 (K1 only), 12 (K1+K2), 123/empty (all)
    const char *stagesEnv = std::getenv("HC_PRE_STAGES");
    const std::string stages = (stagesEnv != nullptr && stagesEnv[0] != '\0') ? stagesEnv : "123";
    uint32_t k1BlockDim = static_cast<uint32_t>(k1Block);
    uint32_t k2BlockDim = static_cast<uint32_t>(ks);
    uint32_t k3BlockDim = static_cast<uint32_t>(k3Block);
    int64_t bsL = B;
    int64_t dL = D;
    int64_t segLenL = kSeg;
    int64_t dChunksL = dChunks;
    int64_t hcMultL = hc_mult;
    int64_t k1RedL = k1RedTmp;
    float normEpsL = normEpsF;

    EXEC_KERNEL_CMD(hc_pre_preprocess, k1BlockDim, x, pre_mix, xF32Ws, invRms, y, bsL, dL,
                    segLenL, dChunksL, hcMultL, normEpsL, k1RedL);

    // D-fix debug hook: dump the K1 GM-side intermediates for the precision
    // investigation. Enabled ONLY via HC_PRE_DUMP_DIR; production path is
    // untouched (no extra sync cost when the env var is absent).
    if (const char *dumpDir = std::getenv("HC_PRE_DUMP_DIR")) {
        auto dumpF32 = [&](const char *name, const at::Tensor &t) {
            at::Tensor cpu = t.to(at::kCPU).contiguous();
            std::string path = std::string(dumpDir) + "/" + name + ".f32";
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char *>(cpu.data_ptr()),
                    cpu.numel() * sizeof(float));
        };
        dumpF32("inv_rms", invRms);
        dumpF32("x_f32_ws", xF32Ws);
    }

    int64_t ksL = ks;
    int64_t mixL = mix;
    int64_t k3RowsL = k3RowsPC;
    int64_t itersL = hc_sinkhorn_iters;
    int64_t k3RedL = k3RedTmp;
    int64_t k3OpL = k3OpTmp;
    float hcEpsL = hcEpsF;

    if (stages.find('2') != std::string::npos) {
        at::Tensor hcFnT = GetHcFnT(hc_fn);
        EXEC_KERNEL_CMD(hc_pre_matmul, k2BlockDim, xF32Ws, hcFnT, partial, mmTilingNpu);
    }
    if (stages.find('3') != std::string::npos) {
        at::Tensor mask8 = GetMask8(x.device());
        EXEC_KERNEL_CMD(hc_pre_epilogue, k3BlockDim, partial, mixesWs, invRms, hc_scale, hc_base,
                        pre, post, combFrag, mask8, bsL, ksL, mixL, k3RowsL, itersL, hcEpsL,
                        k3RedL, k3OpL);
    }

    return {y, post, combFrag, pre};
}

}  // namespace ascend_kernel
