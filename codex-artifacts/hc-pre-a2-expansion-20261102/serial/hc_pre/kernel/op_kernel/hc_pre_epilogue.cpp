/*
 * HcPre K3 — epilogue vector kernel (AIV only), CANN-reference round.
 *
 * torch_ops/hc_pre.py stages covered: D (split + sigmoid activation) +
 * E (softmax + 20-iteration Sinkhorn).
 *
 * CANN implementation pattern (per-round mandate): the high-order
 * reduce-broadcast chain template of the CANN Softmax/LayerNorm family —
 * every normalization step is ReduceSum(AR/RA) -> +eps -> Broadcast -> Div,
 * every activation band is Mul(scale) -> Broadcast(bias) -> Add ->
 * Sigmoid (api-reduce.md / api-reduce-pattern.md / api-arithmetic.md);
 * all work buffers are sized on the host via the Get*MaxMinTmpSize APIs.
 *
 * Inputs : partial [KS, B, MIX] fp32, mixes_ws [B, MIX] fp32 (GM scratch),
 *          inv_rms [B] fp32, hc_scale [3] fp32, hc_base [24] fp32,
 *          mask8 [8] fp32 ([1,1,1,1,0,0,0,0] constant)
 * Outputs: pre [B, HC] fp32, post [B, HC] fp32, comb_frag [B, HC, HC] fp32
 *
 * Stage 1 (mixes): [KS][B][MIX] band load (uniform multi-block stride) ->
 * ReduceSum(RA) -> Mul(invRms broadcast) -> GM staging in mixes_ws. The
 * [KS][B][MIX] layout admits no uniform multi-block band extraction, so the
 * reduction result round-trips through GM once (layout-inherent, not a hack)
 * and every later band load reads the row-major [B, MIX] buffer; a
 * PipeBarrier<PIPE_ALL> guards the MTE3->MTE2 GM visibility.
 *
 * Stage 2 (pre/post): 16B GM band loads auto-padded into pre-zeroed 32B
 * [rows, 8] domains; sigmoid chain; per-row 16B stores (the [B, 4] output
 * row stride of 16B is not 32B-expressible — output-layout-inherent).
 *
 * Stage 3 (comb): i-outer band loads -> accC [HC][rows][8]; in-kernel comb
 * bias bank; softmax (no max-subtraction; operands are O(1) after scale) +
 * Sinkhorn (row norm AR + col norm RA); per-row store of HC tight 16B blocks.
 *
 * Workarounds (all platform-mandated, see design/DESIGN.md §6):
 *  W3 mask8 — GM->UB 16B block copies zero the tail columns at 32B write
 *     granularity, which exp() would turn into 1s and +eps would leak into
 *     the Sinkhorn denominators. The [1,1,1,1,0,0,0,0] mask is applied right
 *     after exp() and again after the +eps (precision-critical, mandatory).
 *     The same platform semantics is used constructively to build the comb
 *     bias bank IN-KERNEL (below) — replacing the v6 host-side pre-assembled
 *     bias bank, which this round eliminates.
 *  s0/s1/s2 — three one-shot scalar reads of hc_scale (GM GetValue). These
 *     are kernel-side scalar parameter reads (nanosecond-scale, K3 is
 *     latency- not bandwidth-bound); the [3]-element tensor cannot form a
 *     legal 16B-aligned block on its own.
 *
 * Design source: design/tile_level/hc_pre.py::hc_pre_epilogue
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

class HcPreEpilogue {
public:
    __aicore__ inline HcPreEpilogue() {}

    __aicore__ inline void Init(GM_ADDR partialIn, GM_ADDR mixesWsIn, GM_ADDR invRmsIn,
                                GM_ADDR hcScaleIn, GM_ADDR hcBaseIn, GM_ADDR preOut,
                                GM_ADDR postOut, GM_ADDR combOut, GM_ADDR mask8In,
                                int64_t bs, int64_t ks, int64_t mix, int64_t rowsPerCore,
                                int64_t iters, float hcEps, int64_t redTmpSize,
                                int64_t opTmpSize)
    {
        bs_ = bs;
        ks_ = ks;
        mix_ = mix;
        hc_ = mix / 6;
        iters_ = iters;
        eps_ = hcEps;
        row0_ = GetBlockIdx() * rowsPerCore;
        if (row0_ < bs_) {
            rows_ = (bs_ - row0_ < rowsPerCore) ? (bs_ - row0_) : rowsPerCore;
        }

        partialGm_.SetGlobalBuffer((__gm__ float *)partialIn);
        mixesGm_.SetGlobalBuffer((__gm__ float *)mixesWsIn);
        invRmsGm_.SetGlobalBuffer((__gm__ float *)invRmsIn);
        scaleGm_.SetGlobalBuffer((__gm__ float *)hcScaleIn);
        baseGm_.SetGlobalBuffer((__gm__ float *)hcBaseIn);
        preGm_.SetGlobalBuffer((__gm__ float *)preOut);
        postGm_.SetGlobalBuffer((__gm__ float *)postOut);
        combGm_.SetGlobalBuffer((__gm__ float *)combOut);
        mask8Gm_.SetGlobalBuffer((__gm__ float *)mask8In);

        mixPad_ = (mix_ + 7) / 8 * 8;
        pipe_.InitBuffer(ppBuf_, ks_ * rowsPerCore * mixPad_ * sizeof(float));
        pipe_.InitBuffer(accAllBuf_, rowsPerCore * mixPad_ * sizeof(float));
        pipe_.InitBuffer(accPPBuf_, rowsPerCore * 8 * sizeof(float));
        pipe_.InitBuffer(accCBuf_, rowsPerCore * hc_ * 8 * sizeof(float));
        pipe_.InitBuffer(biasBankBuf_, hc_ * hc_ * 8 * sizeof(float));
        pipe_.InitBuffer(irBuf_, rowsPerCore * sizeof(float));
        pipe_.InitBuffer(ir8Buf_, rowsPerCore * 8 * sizeof(float));
        pipe_.InitBuffer(irMBuf_, rowsPerCore * mix_ * sizeof(float));
        pipe_.InitBuffer(basePreBuf_, 8 * sizeof(float));
        pipe_.InitBuffer(basePstBuf_, 8 * sizeof(float));
        pipe_.InitBuffer(mask8Buf_, 8 * sizeof(float));
        pipe_.InitBuffer(t1Buf_, rowsPerCore * hc_ * 8 * sizeof(float));
        pipe_.InitBuffer(rsBuf_, rowsPerCore * hc_ * sizeof(float));
        pipe_.InitBuffer(csBuf_, rowsPerCore * 8 * sizeof(float));
        pipe_.InitBuffer(redTmpBuf_, redTmpSize);
        pipe_.InitBuffer(opTmpBuf_, opTmpSize);
        opTmpL_ = opTmpBuf_.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (row0_ >= bs_) {
            return;
        }
        LoadSmallInputs();
        BuildMixes();
        ComputePre();
        ComputePost();
        ComputeComb();
    }

private:
    __aicore__ inline LocalTensor<float> pp() { return ppBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> accAll() { return accAllBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> accPP() { return accPPBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> accC() { return accCBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> biasBank() { return biasBankBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> ir() { return irBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> ir8() { return ir8Buf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> irM() { return irMBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> basePre() { return basePreBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> basePst() { return basePstBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> mask8() { return mask8Buf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> t1() { return t1Buf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> rs() { return rsBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<float> cs() { return csBuf_.Get<float>(); }
    __aicore__ inline LocalTensor<uint8_t> redTmp() { return redTmpBuf_.Get<uint8_t>(); }
    __aicore__ inline LocalTensor<uint8_t> &opTmp() { return opTmpL_; }

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

    __aicore__ inline void LoadSmallInputs()
    {
        // 32B-aligned small banks: base[0:8), base[4:12), mask8
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padExt;
        padExt.isPad = false;
        ext.blockLen = static_cast<uint32_t>(8 * sizeof(float));
        DataCopyPad(basePre(), baseGm_[0], ext, padExt);
        DataCopyPad(basePst(), baseGm_[hc_], ext, padExt);
        DataCopyPad(mask8(), mask8Gm_[0], ext, padExt);
        Mte2ToV();
        s0_ = scaleGm_.GetValue(0);
        s1_ = scaleGm_.GetValue(1);
        s2_ = scaleGm_.GetValue(2);

        // comb bias bank, built IN-KERNEL (replaces the v6 host-side
        // pre-assembly): four 16B GM->UB block copies, one per slot i. The
        // 32B write granularity of a 16B copy zeroes each slot's tail four
        // lanes (platform semantics W3), producing exactly the
        // [4 valid + 4 zero] 32B-aligned [1,8] sources the Broadcast below
        // needs. Source addresses 32+16i B are 16B aligned; slot addresses
        // i*32B are 32B aligned.
        ext.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));  // 16B
        for (int64_t i = 0; i < hc_; ++i) {
            DataCopyPad(biasBank()[i * 8], baseGm_[2 * hc_ + i * hc_], ext, padExt);
        }

        ext.blockLen = static_cast<uint32_t>(8 * sizeof(float));
        DataCopyPad(ir(), invRmsGm_[row0_], ext, padExt);
        Mte2ToV();
        uint32_t dst8[2] = {static_cast<uint32_t>(rows_), 8U};
        uint32_t srcR[2] = {static_cast<uint32_t>(rows_), 1U};
        Broadcast<float, 2, 1, false>(ir8(), ir(), dst8, srcR, opTmp());
        PipeBarrier<PIPE_V>();
        uint32_t dstM[2] = {static_cast<uint32_t>(rows_), static_cast<uint32_t>(mix_)};
        Broadcast<float, 2, 1, false>(irM(), ir(), dstM, srcR, opTmp());
        PipeBarrier<PIPE_V>();
    }

    // Stage 1: reduce partials into mixes (with inv_rms) and stage through GM.
    __aicore__ inline void BuildMixes()
    {
        DataCopyExtParams ext;
        ext.blockCount = static_cast<uint16_t>(ks_);
        ext.blockLen = static_cast<uint32_t>(rows_ * mix_ * sizeof(float));
        ext.srcStride = static_cast<uint32_t>((bs_ - rows_) * mix_ * sizeof(float));
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padExt;
        padExt.isPad = false;
        DataCopyPad(pp(), partialGm_[row0_ * mix_], ext, padExt);
        Mte2ToV();
        uint32_t shp[2] = {static_cast<uint32_t>(ks_), static_cast<uint32_t>(rows_ * mix_)};
        ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(accAll(), pp(), redTmp(), shp, false);
        PipeBarrier<PIPE_V>();
        Mul(accAll(), accAll(), irM(), rows_ * mix_);
        PipeBarrier<PIPE_V>();
        VToMte3();
        // plain 32-wide passthrough staging (original column order); band
        // loads below use per-row single blocks through the UB non-aligned
        // dummy path for the 4-wide offsets
        DataCopyExtParams outExt;
        outExt.blockCount = 1;
        outExt.blockLen = static_cast<uint32_t>(rows_ * mix_ * sizeof(float));
        outExt.srcStride = 0;
        outExt.dstStride = 0;
        DataCopyPad(mixesGm_[row0_ * mixPad_], accAll(), outExt);
        // full pipeline barrier: the MTE3 store must be globally visible
        // before the in-kernel MTE2 loads below read the same GM addresses
        PipeBarrier<PIPE_ALL>();
    }

    // Load a 4-wide band from the GM mixes buffer into a pre-zeroed
    // [rows, 8] domain (16B blocks auto-padded to 32B slots). Uniform stride.
    __aicore__ inline void LoadMixBand4(const int64_t j0, const LocalTensor<float> &dst)
    {
        Duplicate(dst, 0.0f, rows_ * 8);
        // V -> MTE2 event: the zero fill must retire before the copy lands in
        // the same region (otherwise the async Duplicate overwrites the
        // loaded data with zeros)
        {
            event_t idv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(idv);
            WaitFlag<HardEvent::V_MTE2>(idv);
        }
        DataCopyExtParams ext;
        ext.blockCount = static_cast<uint16_t>(rows_);
        ext.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));
        ext.srcStride = static_cast<uint32_t>((mixPad_ - hc_) * sizeof(float));
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padExt;
        padExt.isPad = false;
        DataCopyPad(dst, mixesGm_[row0_ * mixPad_ + j0], ext, padExt);
        Mte2ToV();
    }

    __aicore__ inline void SigmoidBand(const LocalTensor<float> &acc, const float scale,
                                       const LocalTensor<float> &bias, const float postScale,
                                       const float addEps)
    {
        Muls(acc, acc, scale, rows_ * 8);
        PipeBarrier<PIPE_V>();
        uint32_t dst[2] = {static_cast<uint32_t>(rows_), 8U};
        uint32_t one[2] = {1U, 8U};
        Broadcast<float, 2, 0, false>(t1(), bias, dst, one, opTmp());
        PipeBarrier<PIPE_V>();
        Add(acc, acc, t1(), rows_ * 8);
        PipeBarrier<PIPE_V>();
        Sigmoid<float, false>(acc, acc, opTmp(), rows_ * 8);
        PipeBarrier<PIPE_V>();
        if (postScale != 1.0f) {
            Muls(acc, acc, postScale, rows_ * 8);
            PipeBarrier<PIPE_V>();
        }
        if (addEps != 0.0f) {
            Adds(acc, acc, addEps, rows_ * 8);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputePre()
    {
        LoadMixBand4(0, accPP());
        SigmoidBand(accPP(), s0_, basePre(), 1.0f, eps_);
        StoreBand4(preGm_, accPP());
    }

    __aicore__ inline void ComputePost()
    {
        // passthrough staging keeps original column order: post band is the
        // original columns [4:8) (per-row single blocks handle the 16B offset)
        LoadMixBand4(hc_, accPP());
        SigmoidBand(accPP(), s1_, basePst(), 2.0f, 0.0f);
        StoreBand4(postGm_, accPP());
    }

    __aicore__ inline void StoreBand4(GlobalTensor<float> &gm, const LocalTensor<float> &src)
    {
        VToMte3();
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));
        ext.srcStride = 0;
        ext.dstStride = 0;
        for (int64_t r = 0; r < rows_; ++r) {
            // one 16B block per row; the 16B intra-UB gap between valid bands
            // is half a dataBlock, so multi-block stride cannot express it.
            DataCopyPad(gm[(row0_ + r) * hc_], src[r * 8], ext);
        }
        Mte3ToV();
    }

    __aicore__ inline void ComputeComb()
    {
        const int64_t n = rows_ * hc_;
        // per-i loads straight from the 32-wide staging buffer: block (r)
        // covers columns [16 + i*4, +4), uniform 112B source stride.
        // submit all 16 copies first, then a single MTE2->V wait.
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padExt;
        padExt.isPad = false;
        Duplicate(accC(), 0.0f, hc_ * rows_ * 8);
        {
            event_t idv = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(idv);
            WaitFlag<HardEvent::V_MTE2>(idv);
        }
        for (int64_t i = 0; i < hc_; ++i) {
            for (int64_t r = 0; r < rows_; ++r) {
                DataCopyPad(accC()[(i * rows_ + r) * 8],
                            mixesGm_[(row0_ + r) * mixPad_ + 2 * hc_ + i * hc_], ext, padExt);
            }
        }
        Mte2ToV();
        for (int64_t i = 0; i < hc_; ++i) {
            Muls(accC()[i * rows_ * 8], accC()[i * rows_ * 8], s2_, rows_ * 8);
            PipeBarrier<PIPE_V>();
        }
        // comb bias: per-i [1,8] slots from the in-kernel bank (4 valid +
        // 4 zero tails — see LoadSmallInputs)
        for (int64_t i = 0; i < hc_; ++i) {
            uint32_t dst[2] = {static_cast<uint32_t>(rows_), 8U};
            uint32_t one[2] = {1U, 8U};
            Broadcast<float, 2, 0, false>(t1(), biasBank()[i * 8], dst, one, opTmp());
            PipeBarrier<PIPE_V>();
            Add(accC()[i * rows_ * 8], accC()[i * rows_ * 8], t1(), rows_ * 8);
            PipeBarrier<PIPE_V>();
        }

        // softmax: exp / (rowsum + eps), then + eps.
        // The GM->UB 16B band copies zero the pad columns [4:8) of each slot
        // (32B write granularity), which exp() would turn into 1s — multiply
        // a [1,1,1,1,0,0,0,0] mask right after exp so pads stay exact zeros;
        // the subsequent +eps then matches the reference (eps on every
        // element).
        Exp<float, false>(accC(), accC(), n * 8);
        PipeBarrier<PIPE_V>();
        {
            uint32_t mDst[2] = {static_cast<uint32_t>(n), 8U};
            uint32_t mSrc[2] = {1U, 8U};
            Broadcast<float, 2, 0, false>(t1(), mask8(), mDst, mSrc, opTmp());
            PipeBarrier<PIPE_V>();
            Mul(accC(), accC(), t1(), n * 8);
            PipeBarrier<PIPE_V>();
        }
        RowNorm();
        Adds(accC(), accC(), eps_, n * 8);
        PipeBarrier<PIPE_V>();
        // re-mask: the +eps dirtied the pad columns (eps), which the first
        // ColNorm would blow up to eps/(HC*eps) = 1/HC and permanently bias
        // every later row-normalization denominator
        {
            uint32_t mDst[2] = {static_cast<uint32_t>(n), 8U};
            uint32_t mSrc[2] = {1U, 8U};
            Broadcast<float, 2, 0, false>(t1(), mask8(), mDst, mSrc, opTmp());
            PipeBarrier<PIPE_V>();
            Mul(accC(), accC(), t1(), n * 8);
            PipeBarrier<PIPE_V>();
        }
        ColNorm();
        for (int64_t it = 0; it < iters_ - 1; ++it) {
            RowNorm();
            ColNorm();
        }
        StoreComb();
    }

    __aicore__ inline void RowNorm()
    {
        const int64_t n = rows_ * hc_;
        uint32_t mat[2] = {static_cast<uint32_t>(n), 8U};
        uint32_t col1[2] = {static_cast<uint32_t>(n), 1U};
        ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(rs(), accC(), redTmp(), mat, false);
        PipeBarrier<PIPE_V>();
        Adds(rs(), rs(), eps_, n);
        PipeBarrier<PIPE_V>();
        Broadcast<float, 2, 1, false>(t1(), rs(), mat, col1, opTmp());
        PipeBarrier<PIPE_V>();
        Div(accC(), accC(), t1(), n * 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ColNorm()
    {
        uint32_t red[2] = {static_cast<uint32_t>(hc_), static_cast<uint32_t>(rows_ * 8)};
        ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(cs(), accC(), redTmp(), red, false);
        PipeBarrier<PIPE_V>();
        Adds(cs(), cs(), eps_, rows_ * 8);
        PipeBarrier<PIPE_V>();
        uint32_t dst[2] = {static_cast<uint32_t>(hc_), static_cast<uint32_t>(rows_ * 8)};
        uint32_t one[2] = {1U, static_cast<uint32_t>(rows_ * 8)};
        Broadcast<float, 2, 0, false>(t1(), cs(), dst, one, opTmp());
        PipeBarrier<PIPE_V>();
        Div(accC(), accC(), t1(), rows_ * hc_ * 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreComb()
    {
        VToMte3();
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(hc_ * sizeof(float));
        ext.srcStride = 0;
        ext.dstStride = 0;
        for (int64_t r = 0; r < rows_; ++r) {
            for (int64_t i = 0; i < hc_; ++i) {
                // src row (i, r) of the i-outer domain sits at a 32B-aligned
                // slot; dst is a tight 16B block in the [r][i] row.
                DataCopyPad(combGm_[(row0_ + r) * hc_ * hc_ + i * hc_],
                            accC()[(i * rows_ + r) * 8], ext);
            }
        }
        Mte3ToV();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ppBuf_, accAllBuf_, accPPBuf_, accCBuf_, biasBankBuf_, irBuf_;
    TBuf<TPosition::VECCALC> ir8Buf_, irMBuf_;
    TBuf<TPosition::VECCALC> basePreBuf_, basePstBuf_, mask8Buf_, t1Buf_, rsBuf_, csBuf_;
    TBuf<TPosition::VECCALC> redTmpBuf_, opTmpBuf_;
    LocalTensor<uint8_t> opTmpL_;
    GlobalTensor<float> partialGm_, mixesGm_, invRmsGm_, scaleGm_, baseGm_, preGm_, postGm_;
    GlobalTensor<float> combGm_, mask8Gm_;
    int64_t bs_ = 0, ks_ = 0, mix_ = 0, mixPad_ = 0, hc_ = 0, iters_ = 0;
    int64_t row0_ = 0, rows_ = 0;
    float eps_ = 0.0f, s0_ = 0.0f, s1_ = 0.0f, s2_ = 0.0f;
};
}  // namespace

extern "C" __global__ __aicore__ void hc_pre_epilogue(GM_ADDR partial, GM_ADDR mixes_ws,
                                                      GM_ADDR inv_rms, GM_ADDR hc_scale,
                                                      GM_ADDR hc_base, GM_ADDR pre, GM_ADDR post,
                                                      GM_ADDR comb_frag, GM_ADDR mask8,
                                                      int64_t bs, int64_t ks, int64_t mix,
                                                      int64_t rows_per_core, int64_t iters,
                                                      float hc_eps, int64_t red_tmp_size,
                                                      int64_t op_tmp_size)
{
    HcPreEpilogue op;
    op.Init(partial, mixes_ws, inv_rms, hc_scale, hc_base, pre, post, comb_frag, mask8, bs, ks,
            mix, rows_per_core, iters, hc_eps, red_tmp_size, op_tmp_size);
    op.Process();
}
