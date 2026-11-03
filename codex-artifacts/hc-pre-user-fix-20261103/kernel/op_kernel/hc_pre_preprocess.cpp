/*
 * HcPre K1 — preprocess vector kernel (AIV only), CANN-reference round.
 *
 * torch_ops/hc_pre.py stages covered: A (cast / flatten) + B (RMS) + F (y).
 *
 * CANN implementation pattern (per-round mandate):
 *   canonical three-phase CopyIn -> Compute -> CopyOut loop over whole rows,
 *   with the row block flowing through two TQue queues (the standard
 *   asc-devkit/examples/00_introduction/01_add pipeline structure):
 *     inQueueX  : TQue<VECIN,  BUFFER_NUM=1> [HC, D] bf16  (GM -> UB)
 *     outQueueXf: TQue<VECOUT, BUFFER_NUM=1> [HC, D] fp32  (compute result
 *                 staged back out; EnQue/DeQue manage the V->MTE3 dependency)
 *
 * Single-buffer justification (measured, see design/PERF_DESIGN.md V1):
 * the whole-row block needs 40KB (bf16 in) + 80KB (fp32 out) per slot, so a
 * BUFFER_NUM=2 double-buffered pair would need 2 x 120KB = 240KB > the 192KB
 * UB budget. A k-sub-row streamed double buffer was tried this round and
 * measured 3.1us SLOWER at B=128 (17.8 vs 14.7us): K1 moves 17MB per B=128
 * pass and the v6 whole-row schedule already saturates the measured
 * ~1.17TB/s bandwidth wall (99%), so pipelining has no physical headroom
 * while the per-row queue event traffic (4x per row) adds pure overhead.
 * The canonical single-buffer three-phase form keeps the TQue structure with
 * the same two event interactions per row as v6.
 *
 * Inputs : x [B, HC, D] bf16, pre_mix [B, HC] fp32
 * Outputs: x_f32_ws segment-major [ks][B][kSeg] fp32 (exact bf16->fp32 cast),
 *          inv_rms [B] fp32, y [B, D] bf16 (sum_k pre_mix[b,k]*x[b,k,:]).
 *
 * Compute (all vectorized, no per-element scalar loops):
 *   Cast(xf, x) [CAST_NONE] -> per-k Axpy(yAcc += w_k * xf) [fused mul-add]
 *   -> per-k Mul(t1, xf, xf) + 6 halving Adds folding [D]->[80] -> Add into
 *   sqAcc[k]. Row tail (once per row): ReduceSum(AR [1, hc*80]) — the CANN
 *   high-order reduction with its work buffer sized on the host via
 *   GetReduceSumMaxMinTmpSize — then the scalar chain
 *   Muls(1/fanIn) -> Adds(norm_eps) -> Sqrt -> Div(1/sqrt) [D-fix: the float
 *   Rsqrt intrinsic misses the 1e-4 gate, see RowTail note] ->
 *   Cast(yUb, yAcc, CAST_RINT).
 *
 * Row-tail outputs use one V_MTE3/MTE3_V event pair: y/inv_rms are emitted
 * once per row, so queueing them through a third TQue buys nothing (the
 * next row's Duplicate/CopyIn does not touch yUb/scal; the next writer of
 * scal is the following row's tail, a full row-pipeline away).
 *
 * Workaround note: none. The 4 scalar GetValue reads of the pre_mix weights
 * (one-time per row, UB-resident) are the standard way to feed Axpy scalars.
 *
 * Design source: design/tile_level/hc_pre.py::hc_pre_preprocess
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

// single-buffered TQue (justification in the header comment)
constexpr int32_t BUFFER_NUM = 1;
// square-sum folding: D=5120 -> 80 (6 halving Adds per k), final AR over [hc*80]
constexpr int32_t SQ_TILE = 80;

class HcPrePreprocess {
public:
    __aicore__ inline HcPrePreprocess() {}

    __aicore__ inline void Init(GM_ADDR xIn, GM_ADDR preMixIn, GM_ADDR wsOut, GM_ADDR invRmsOut,
                                GM_ADDR yOut, int64_t bs, int64_t d, int64_t segLen,
                                int64_t dChunkNum, int64_t hcMult, float normEps,
                                int64_t redTmpSize)
    {
        bs_ = bs;
        d_ = d;
        hc_ = hcMult;
        fanIn_ = hcMult * d;
        segLen_ = segLen;      // K2 split-K segment size (ws layout unit)
        dChunks_ = dChunkNum;  // D / segLen
        normEps_ = normEps;
        blockDim_ = GetBlockNum();

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)xIn);
        preMixGm_.SetGlobalBuffer((__gm__ float *)preMixIn);
        wsGm_.SetGlobalBuffer((__gm__ float *)wsOut);
        invRmsGm_.SetGlobalBuffer((__gm__ float *)invRmsOut);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t *)yOut);

        // canonical TQue allocations (UB): whole-row in/out queues + compute
        // buffers (see design/DESIGN.md §3.3). TQue form is the 3-arg
        // InitBuffer(que, num, lenPerBuffer).
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, hc_ * d_ * sizeof(bfloat16_t));  // 40KB
        pipe_.InitBuffer(outQueueXf_, BUFFER_NUM, hc_ * d_ * sizeof(float));     // 80KB
        pipe_.InitBuffer(t1Buf_, d_ * sizeof(float));                       // mul scratch
        pipe_.InitBuffer(yAccBuf_, d_ * sizeof(float));                     // row accum
        pipe_.InitBuffer(yUbBuf_, d_ * sizeof(bfloat16_t));                 // y stage
        pipe_.InitBuffer(sqAccBuf_, hc_ * SQ_TILE * sizeof(float));         // folded sq
        pipe_.InitBuffer(wBuf_, 8 * sizeof(float));                         // pre_mix row
        pipe_.InitBuffer(scalBuf_, 8 * sizeof(float));                      // 32B-aligned
        pipe_.InitBuffer(onesBuf_, 8 * sizeof(float));                      // Div numerator
        pipe_.InitBuffer(redTmpBuf_, redTmpSize);                           // ReduceSum work
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = GetBlockIdx(); row < bs_; row += blockDim_) {
            RunForRow(row);
        }
    }

private:
    __aicore__ inline LocalTensor<float> t1() { return t1Buf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> yAcc() { return yAccBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<bfloat16_t> yUb() { return yUbBuf_.Get<bfloat16_t>(); }
    __aicore__ inline LocalTensor<float> sqAcc() { return sqAccBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> w8() { return wBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> scal() { return scalBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> ones() { return onesBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<uint8_t> redTmp() { return redTmpBuf_.Get<uint8_t>(); }

    __aicore__ inline void Mte2ToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(id);
        WaitFlag<HardEvent::MTE2_V>(id);
    }
    __aicore__ inline void VToMte3()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
    }
    __aicore__ inline void Mte3ToV()
    {
        event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(id);
        WaitFlag<HardEvent::MTE3_V>(id);
    }

    // ---- canonical three phases (whole-row block) ----

    __aicore__ inline void CopyIn(const int64_t row)
    {
        LocalTensor<bfloat16_t> xUb = inQueueX_.AllocTensor<bfloat16_t>();
        DataCopyExtParams inExt;
        inExt.blockCount = 1;
        inExt.blockLen = static_cast<uint32_t>(hc_ * d_ * sizeof(bfloat16_t));
        inExt.srcStride = 0;
        inExt.dstStride = 0;
        DataCopyPadExtParams<bfloat16_t> padExt;
        padExt.isPad = false;
        DataCopyPad(xUb, xGm_[row * fanIn_], inExt, padExt);  // contiguous row
        inQueueX_.EnQue(xUb);
    }

    __aicore__ inline void Compute(const float *wVal)
    {
        const int32_t dCnt = static_cast<int32_t>(d_);
        LocalTensor<bfloat16_t> xUb = inQueueX_.DeQue<bfloat16_t>();
        LocalTensor<float> xf = outQueueXf_.AllocTensor<float>();

        Cast(xf, xUb, RoundMode::CAST_NONE, hc_ * dCnt);  // exact bf16 -> fp32
        PipeBarrier<PIPE_V>();
        inQueueX_.FreeTensor(xUb);

        for (int64_t k = 0; k < hc_; ++k) {
            // y accumulation (Stage F): yAcc += w_k * xf_k, fused mul-add
            Axpy(yAcc(), xf[k * d_], wVal[k], dCnt);
            PipeBarrier<PIPE_V>();

            // RMS partial (Stage B): fold xf_k^2 from [D] down to [SQ_TILE]
            Mul(t1(), xf[k * d_], xf[k * d_], dCnt);
            PipeBarrier<PIPE_V>();
            int32_t n = dCnt;  // 5120 -> 80 (d_=5120 keeps every step even)
            while (n > SQ_TILE) {
                const int32_t half = n / 2;
                Add(t1(), t1(), t1()[half], half);
                PipeBarrier<PIPE_V>();
                n = half;
            }
            Add(sqAcc()[k * SQ_TILE], sqAcc()[k * SQ_TILE], t1(), SQ_TILE);
            PipeBarrier<PIPE_V>();
        }
        outQueueXf_.EnQue(xf);  // hands xf to the MTE3 CopyOut phase
    }

    __aicore__ inline void CopyOut(const int64_t row)
    {
        LocalTensor<float> xf = outQueueXf_.DeQue<float>();
        // ws segment store: sub-row (row, k) covers segments
        // seg = k * dChunks_ + c, each a contiguous [segLen] block at
        // kOff = seg * bs_ * segLen_ + row * segLen_ (split-K layout for K2)
        DataCopyExtParams segExt;
        segExt.blockCount = 1;
        segExt.blockLen = static_cast<uint32_t>(segLen_ * sizeof(float));
        segExt.srcStride = 0;
        segExt.dstStride = 0;
        for (int64_t k = 0; k < hc_; ++k) {
            for (int64_t c = 0; c < dChunks_; ++c) {
                const int64_t seg = k * dChunks_ + c;
                const int64_t kOff = seg * bs_ * segLen_ + row * segLen_;
                DataCopyPad(wsGm_[kOff], xf[k * d_ + c * segLen_], segExt);
            }
        }
        outQueueXf_.FreeTensor(xf);
    }

    __aicore__ inline void RowTail(const int64_t row)
    {
        const int32_t dCnt = static_cast<int32_t>(d_);
        // single AR reduce over the folded partials (CANN high-order API)
        uint32_t sqShape[2] = {1U, static_cast<uint32_t>(hc_ * SQ_TILE)};
        ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(scal(), sqAcc(), redTmp(), sqShape,
                                                              false);
        PipeBarrier<PIPE_V>();

        // inv_rms = rsqrt(sumSq / fanIn + eps) on the scalar lane.
        // D-fix (root cause): the float Rsqrt intrinsic does NOT meet the
        // 1e-4 comparison requirement (asc-devkit Rsqrt.md "精度误差" note:
        // "float的算子结果对比误差不满足双万分之一的要求，如果需要高精度，
        // 建议使用Div和Sqrt替代实现"). Measured on this SoC the dumped
        // inv_rms was quantized to a ~2^-9 grid (per-row deviation up to
        // 1.3e-3 rel), which multiplied into every mixes element and broke
        // the pre/post 1e-4/99.5% gate. Official high-precision replacement:
        Muls(scal(), scal(), 1.0f / static_cast<float>(fanIn_), 1);
        PipeBarrier<PIPE_V>();
        Adds(scal(), scal(), normEps_, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(scal(), scal(), 1);
        PipeBarrier<PIPE_V>();
        Duplicate(ones(), 1.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(scal(), ones(), scal(), 1);
        PipeBarrier<PIPE_V>();

        Cast(yUb(), yAcc(), RoundMode::CAST_RINT, dCnt);
        PipeBarrier<PIPE_V>();

        VToMte3();
        DataCopyExtParams yExt;
        yExt.blockCount = 1;
        yExt.blockLen = static_cast<uint32_t>(d_ * sizeof(bfloat16_t));
        yExt.srcStride = 0;
        yExt.dstStride = 0;
        DataCopyPad(yGm_[row * d_], yUb(), yExt);
        DataCopyExtParams irExt;
        irExt.blockCount = 1;
        irExt.blockLen = static_cast<uint32_t>(sizeof(float));
        irExt.srcStride = 0;
        irExt.dstStride = 0;
        DataCopyPad(invRmsGm_[row], scal(), irExt);
        Mte3ToV();
    }

    __aicore__ inline void RunForRow(const int64_t row)
    {
        // per-row mixing weights: one 16B GM->UB block, read as Axpy scalars
        DataCopyExtParams wExt;
        wExt.blockCount = 1;
        wExt.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));
        wExt.srcStride = 0;
        wExt.dstStride = 0;
        DataCopyPadExtParams<float> padExt;
        padExt.isPad = false;
        DataCopyPad(w8(), preMixGm_[row * hc_], wExt, padExt);
        Mte2ToV();
        float wVal[4];
        for (int64_t k = 0; k < hc_; ++k) {
            wVal[k] = w8().GetValue(k);
        }

        Duplicate(yAcc(), 0.0f, static_cast<int32_t>(d_));
        PipeBarrier<PIPE_V>();
        Duplicate(sqAcc(), 0.0f, hc_ * SQ_TILE);
        PipeBarrier<PIPE_V>();

        // canonical three-phase row pipeline
        CopyIn(row);
        Compute(wVal);
        CopyOut(row);
        RowTail(row);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueXf_;
    TBuf<TPosition::VECCALC> t1Buf_, yAccBuf_, yUbBuf_, sqAccBuf_, wBuf_, scalBuf_, onesBuf_,
        redTmpBuf_;
    GlobalTensor<bfloat16_t> xGm_, yGm_;
    GlobalTensor<float> preMixGm_, wsGm_, invRmsGm_;
    int64_t bs_ = 0, d_ = 0, hc_ = 0, fanIn_ = 0, segLen_ = 0, dChunks_ = 0;
    int64_t blockDim_ = 1;
    float normEps_ = 0.0f;
};
}  // namespace

extern "C" __global__ __aicore__ void hc_pre_preprocess(GM_ADDR x, GM_ADDR pre_mix, GM_ADDR x_f32_ws,
                                                        GM_ADDR inv_rms, GM_ADDR y, int64_t bs,
                                                        int64_t d, int64_t seg_len,
                                                        int64_t d_chunk_num, int64_t hc_mult,
                                                        float norm_eps, int64_t red_tmp_size)
{
    HcPrePreprocess op;
    op.Init(x, pre_mix, x_f32_ws, inv_rms, y, bs, d, seg_len, d_chunk_num, hc_mult, norm_eps,
            red_tmp_size);
    op.Process();
}
