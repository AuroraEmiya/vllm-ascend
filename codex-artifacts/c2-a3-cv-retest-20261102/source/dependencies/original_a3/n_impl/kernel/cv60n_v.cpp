// cv60n V kernel: fused pool + RMSNorm + ring update over the packed pair buffer.
// One AIV per latent-row slice; every row of the output is written exactly once
// (completion value or bitwise +0), and every ring tail row is written exactly
// once by the core owning its latent row.
//
// Semantics follow the frozen reference family exactly:
//  - ratio=2 pooling, globally position-aligned pairs (2g, 2g+1)
//  - odd-start requests read the (start-1)%32 ring residual for the even
//    member of their first pair (that read is NOT page-gated)
//  - completion rows additionally require a valid page (0 < page < P)
//  - pooled value is rounded to BF16 BEFORE the RMSNorm reduction
//  - single-Exp stable two-input softmax, 1/512 mean, Rsqrt, gamma in FP32
//  - ring tail = min(used,32) rows of raw FP32 pair data; page-gated
//  - requests failing validation (start<0, used<=0, used>maxq, bad base,
//    page outside (0,P)) contribute nothing at all
//
// Ring self-collision: when start is odd and used >= 32, the write window
// covers all 32 slots including the residual slot (start-1)%32. The row whose
// seg maps to that slot (seg = 31 + 32*ceil((used-32-31)/32), clamped) skips
// its own write; the core owning the request's first completion row (row
// `base`) performs that write after its residual read. Every core derives the
// same rule from metadata alone - no cross-core synchronization.
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

namespace {
constexpr int32_t D = 512;             // head dim (floats per kv / score half)
constexpr int32_t PAIR_COLS = 1024;    // floats per pair row (kv | score)
constexpr int32_t RING = 32;           // ring slots per page

template<int CHUNK_TILE>
class Cv60nVectorV {
    // Residual staging rows per batch: sized to the tile so every residual of
    // a chunk is staged in the single pre-ring load pass (the ownership-
    // transferred write of the same chunk must not overtake any residual read
    // of that chunk).
    static constexpr int32_t RMAX = CHUNK_TILE;
public:
    __aicore__ inline void Init(GM_ADDR pair, GM_ADDR state, GM_ADDR metadata,
                                GM_ADDR gamma, GM_ADDR out,
                                int64_t rows, int64_t pages, int32_t batches,
                                float eps, int64_t maxq) {
        rows_ = rows; pages_ = pages; batches_ = batches; eps_ = eps; maxq_ = maxq;
        pair_.SetGlobalBuffer((__gm__ float*)pair);
        state_.SetGlobalBuffer((__gm__ float*)state);
        meta_.SetGlobalBuffer((__gm__ int32_t*)metadata);
        gamma_.SetGlobalBuffer((__gm__ bfloat16_t*)gamma);
        out_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
        begin_ = rows_ * static_cast<int64_t>(GetBlockIdx()) / GetBlockNum();
        end_ = rows_ * (static_cast<int64_t>(GetBlockIdx()) + 1) / GetBlockNum();
        const uint32_t metaBytes = batches_ > 0 ? ((20u * batches_ + 31u) / 32u) * 32u : 32u;
        pipe_.InitBuffer(metaBuf_, metaBytes);
        metaLocal_ = metaBuf_.Get<int32_t>();
        if (batches_ > 0 && begin_ < end_) {
            DataCopyExtParams params{1, static_cast<uint32_t>(20 * batches_), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(metaLocal_, meta_, params, pad);
            Sync<HardEvent::MTE2_S>();
        }
        // Path selection: a core whose whole range writes no state rows
        // (no ring tail, no odd-start residual/transfer row) can run the
        // double-buffered pipeline; anything else keeps the serial chunk
        // path whose ordering the ring/residual semantics rely on.
        usePipe_ = AllPlain();
        if (usePipe_) {
            // Two planar window slots (kv plane + score plane, rows a..c1-1)
            // plus two output slots: tile i+1 streams in while tile i
            // computes and tile i-1 stores. The float scratch serves the
            // batched chunk compute (two full-width work vectors plus one
            // expansion row).
            pipe_.InitBuffer(pairBuf_, 4 * (CHUNK_TILE + 1) * D * sizeof(float));
            pipe_.InitBuffer(outBuf_, 2 * CHUNK_TILE * D * sizeof(bfloat16_t));
            pipe_.InitBuffer(fpBuf_, 7 * D * sizeof(float));
            pipe_.InitBuffer(bfBuf_, CHUNK_TILE * D * sizeof(bfloat16_t));
            pipe_.InitBuffer(wBuf_, (2 * CHUNK_TILE + 1) * D * sizeof(float));
            pairLocal_ = pairBuf_.Get<float>();
            outLocal_ = outBuf_.Get<bfloat16_t>();
            for (int32_t s = 0; s < 2; ++s) {
                kvW_[s] = pairLocal_[s * 2 * (CHUNK_TILE + 1) * D];
                scW_[s] = kvW_[s][(CHUNK_TILE + 1) * D];
                outSlot_[s] = outLocal_[s * CHUNK_TILE * D];
            }
        } else {
            pipe_.InitBuffer(pairBuf_, (CHUNK_TILE + 1) * PAIR_COLS * sizeof(float));
            pipe_.InitBuffer(resBuf_, RMAX * PAIR_COLS * sizeof(float));
            pipe_.InitBuffer(xferBuf_, PAIR_COLS * sizeof(float));
            pipe_.InitBuffer(outBuf_, CHUNK_TILE * D * sizeof(bfloat16_t));
            pipe_.InitBuffer(fpBuf_, 7 * D * sizeof(float));
            pipe_.InitBuffer(bfBuf_, 3 * D * sizeof(bfloat16_t));
            pairLocal_ = pairBuf_.Get<float>();
            resLocal_ = resBuf_.Get<float>();
            xferLocal_ = xferBuf_.Get<float>();
            outLocal_ = outBuf_.Get<bfloat16_t>();
        }
        fp_ = fpBuf_.Get<float>();
        bf_ = bfBuf_.Get<bfloat16_t>();
        if (batches_ > 0 && begin_ < end_) {
            DataCopy(bf_[D], gamma_, D);
            Sync<HardEvent::MTE2_V>();
            Cast(fp_[0], bf_[D], RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Run() {
        if (begin_ >= end_) return;
        BuildList();
        if (usePipe_) {
            RunPipelined();
            return;
        }
        for (int64_t c0 = begin_; c0 < end_; c0 += CHUNK_TILE) {
            ProcessChunk(c0, (end_ - c0 < CHUNK_TILE ? end_ - c0 : CHUNK_TILE) + c0);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    // True when no chunk of this core's range performs a state write or a
    // residual read: no valid odd-start request owns its base row here
    // (residual / transfer row) and no ring tail intersects the range. Such
    // ranges run the double-buffered pipeline; every other range keeps the
    // serial chunk path (identical to the frozen semantics ordering).
    __aicore__ inline bool AllPlain() {
        if (batches_ <= 0 || begin_ >= end_) return false;
        if (CHUNK_TILE != 8 || batches_ > 64) return false;  // UB budget gate
        for (int32_t b = 0; b < batches_; ++b) {
            int64_t s, u, base, page;
            if (!RequestValid(b, s, u, base, page)) continue;
            if (base + u <= begin_ || base >= end_) continue;
            if ((s & 1) != 0 && base >= begin_) return false;
            const int64_t tail = u < RING ? u : RING;
            if (base + u - tail < end_) return false;
        }
        return true;
    }

    // Plain-chunk scan (pipelined path only): every completion reads its own
    // pair row and its r-1 neighbour from the staged block; no residual /
    // ring / transfer bookkeeping exists on this path by AllPlain().
    __aicore__ inline int32_t ScanPlain(int64_t c0, int64_t c1, int64_t* compRow) {
        int32_t compCount = 0;
        for (int32_t i = 0; i < listLen_; ++i) {
            const int32_t b = metaLocal_.GetValue(2 * batches_ + i);
            int64_t s, u, base, page;
            RequestValid(b, s, u, base, page);
            const int64_t jlo = base > c0 ? base : c0;
            const int64_t jhi = base + u < c1 ? u : c1 - base;
            for (int64_t j = (jlo - base); j < jhi; ++j) {
                if (((s + j) & 1) == 0) continue;
                compRow[compCount++] = base + j;
            }
        }
        return compCount;
    }

    // Loads the chunk window [a, c1) (a = c0-1) as two planar streams: the
    // kv halves of all rows into kvW_[slot], the score halves into
    // scW_[slot] (rows contiguous in each plane). Same de-interleaving
    // strided-pad recipe as the frozen baseline's pool kernel.
    __aicore__ inline void LoadPlain(int64_t c0, int32_t slot) {
        const int64_t a = c0 > 0 ? c0 - 1 : 0;
        const int64_t c1 = (end_ - c0 < CHUNK_TILE ? end_ - c0 : CHUNK_TILE) + c0;
        DataCopyExtParams cp{static_cast<uint16_t>(c1 - a), D * sizeof(float),
                             (PAIR_COLS - D) * sizeof(float), 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(kvW_[slot], pair_[a * PAIR_COLS], cp, pad);
        DataCopyPad(scW_[slot], pair_[a * PAIR_COLS + D], cp, pad);
    }

    // One chunk, fully batched over the R = W-1 window rows that have a left
    // neighbour: the whole weight/pool chain runs on contiguous row-shifted
    // slices (scores[r] vs scores[r-1], kv[r] vs kv[r-1]) - the identical
    // elementwise sequence and reduction tree as the serial per-row
    // ComputeRow, just R*D elements at once - then only completion rows are
    // normalized and stored. Non-completion rows keep their zero-fill.
    __aicore__ inline void ComputeChunk(int32_t slot, int64_t c0, int64_t c1,
                                        const int64_t* compRow, int32_t compCount,
                                        LocalTensor<bfloat16_t> outSlot) {
        const int64_t a = c0 > 0 ? c0 - 1 : 0;
        const int32_t r = static_cast<int32_t>(c1 - a - 1);  // rows with a left neighbour
        const int32_t count = r * D;
        // Planar planes: row w lives at offset w*D, so the row-shifted spans
        // are contiguous [r*D] slices - the same shape the frozen baseline's
        // batched pool chain runs on.
        auto kv0 = kvW_[slot];             // kv rows a .. a+r-1
        auto kv1 = kvW_[slot][D];          // kv rows a+1 .. a+r
        auto sm0 = scW_[slot];             // score rows a .. a+r-1
        auto sm1 = scW_[slot][D];          // score rows a+1 .. a+r
        (void)count; (void)kv0; (void)kv1; (void)sm0; (void)sm1; (void)r;
        // BISECT 2: per-row serial ComputeRow straight off the planar planes.
        for (int32_t i = 0; i < compCount; ++i) {
            const int64_t gmR = compRow[i];
            const int32_t wr = static_cast<int32_t>(gmR - a);  // window row, >= 1
            ComputeRow(kvW_[slot][(wr - 1) * D], kvW_[slot][wr * D],
                       scW_[slot][(wr - 1) * D], scW_[slot][wr * D],
                       outSlot[(gmR - c0) * D]);
        }
        PipeBarrier<PIPE_V>();
        (void)a; (void)c0; (void)c1;
    }

    // Double-buffered chunk pipeline (plain ranges only): chunk i+1 streams
    // in from GM while chunk i computes and chunk i-1 stores out. Same
    // event pattern as the frozen baseline's pipelined pool kernel.
    __aicore__ inline void RunPipelined() {
        for (int32_t i = 0; i < 2; ++i) {
            evLoad_[i] = pipe_.AllocEventID<HardEvent::MTE2_V>();
            evIn_[i] = pipe_.AllocEventID<HardEvent::V_MTE2>();
            evOut_[i] = pipe_.AllocEventID<HardEvent::MTE3_V>();
        }
        int64_t curRow[CHUNK_TILE];
        int64_t nxtRow[CHUNK_TILE];
        int64_t c0 = begin_;
        int32_t slot = 0;
        bool used[2] = {false, false};
        bool outUsed[2] = {false, false};
        int64_t c1 = (end_ - c0 < CHUNK_TILE ? end_ - c0 : CHUNK_TILE) + c0;
        int32_t curCount = ScanPlain(c0, c1, curRow);
        LoadPlain(c0, slot);
        SetFlag<HardEvent::MTE2_V>(evLoad_[slot]);
        while (true) {
            WaitFlag<HardEvent::MTE2_V>(evLoad_[slot]);
            const bool hasNext = c1 < end_;
            int32_t nextSlot = 1 - slot;
            int32_t nextCount = 0;
            int64_t nextC0 = 0, nextC1 = 0;
            if (hasNext) {
                nextC0 = c1;
                nextC1 = (end_ - nextC0 < CHUNK_TILE ? end_ - nextC0 : CHUNK_TILE) + nextC0;
                nextCount = ScanPlain(nextC0, nextC1, nxtRow);
                if (used[nextSlot]) WaitFlag<HardEvent::V_MTE2>(evIn_[nextSlot]);
                LoadPlain(nextC0, nextSlot);
                SetFlag<HardEvent::MTE2_V>(evLoad_[nextSlot]);
            }
            if (outUsed[slot]) WaitFlag<HardEvent::MTE3_V>(evOut_[slot]);
            // ---- compute (identical per-row sequence to the serial path) --
            const int32_t n = static_cast<int32_t>(c1 - c0);
            Duplicate(outSlot_[slot].ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                      static_cast<uint32_t>(n * D));
            PipeBarrier<PIPE_V>();
            ComputeChunk(slot, c0, c1, curRow, curCount, outSlot_[slot]);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(evIn_[slot]);
            Sync<HardEvent::V_MTE3>();
            DataCopy(out_[c0 * D], outSlot_[slot], static_cast<uint32_t>(n * D));
            SetFlag<HardEvent::MTE3_V>(evOut_[slot]);
            outUsed[slot] = true;
            used[slot] = true;
            if (!hasNext) break;
            for (int32_t i = 0; i < nextCount; ++i) curRow[i] = nxtRow[i];
            curCount = nextCount;
            c0 = nextC0;
            c1 = nextC1;
            slot = nextSlot;
        }
        for (int32_t i = 0; i < 2; ++i) {
            if (used[i]) WaitFlag<HardEvent::V_MTE2>(evIn_[i]);
            if (outUsed[i]) WaitFlag<HardEvent::MTE3_V>(evOut_[i]);
        }
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < 2; ++i) {
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(evLoad_[i]);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(evIn_[i]);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(evOut_[i]);
        }
    }

    // Same per-request gate as the frozen baseline family.
    __aicore__ inline bool RequestValid(int32_t b, int64_t& s, int64_t& u,
                                        int64_t& base, int64_t& page) {
        s = metaLocal_.GetValue(b);
        u = metaLocal_.GetValue(batches_ + b);
        base = metaLocal_.GetValue(3 * batches_ + b);
        page = metaLocal_.GetValue(4 * batches_ + b);
        return maxq_ > 0 && s >= 0 && u > 0 && u <= maxq_ && base >= 0 &&
               base + u <= rows_ && page > 0 && page < pages_;
    }

    // Compact per-core request list in the unused third metadata row of the
    // UB copy (metadata row 2 is not used by this operator). Large request
    // counts vectorize the row-overlap pre-filter: the scalar scan costs
    // ~80 cycles per request on EVERY core and dominated the dense
    // many-single-request shapes (B=1440: ~75us of the ~90us case34 gap to
    // the frozen baseline). base/used are exact FP32 integers for
    // rows <= 2^24; padded lanes duplicate base=ends=0 and fail ends>begin.
    __aicore__ inline void BuildList() {
        listLen_ = 0;
        const int32_t aligned = ((batches_ + 63) / 64) * 64;
        // batches_ % 32 keeps the Cast source metaLocal_[3*B] (byte 12*B)
        // 32-byte aligned, mirroring the frozen baseline's vector-filter gate.
        if (batches_ < 16 || batches_ % 32 != 0 || rows_ > (1LL << 24) ||
            D + 2 * aligned + aligned / 16 > 7 * D) {
            for (int32_t b = 0; b < batches_; ++b) {
                int64_t s, u, base, page;
                if (!RequestValid(b, s, u, base, page)) continue;
                if (base + u <= begin_ || base >= end_) continue;
                metaLocal_.SetValue(2 * batches_ + listLen_, b);
                ++listLen_;
            }
            return;
        }
        auto bases = fp_[D];
        auto ends = fp_[D + aligned];
        // Mask tensors need 32-byte-aligned UB (vec alignment); round the
        // second mask's float offset up to the next 8-float boundary.
        const int32_t mask1Off = D + 2 * aligned + (aligned / 32 + 7) / 8 * 8;
        auto mask0 = fp_[D + 2 * aligned].ReinterpretCast<uint8_t>();
        auto mask1 = fp_[mask1Off].ReinterpretCast<uint8_t>();
        Duplicate(bases, 0.0f, aligned);
        Duplicate(ends, 0.0f, aligned);
        PipeBarrier<PIPE_V>();
        Cast(bases, metaLocal_[3 * batches_], RoundMode::CAST_NONE, batches_);
        Cast(ends, metaLocal_[batches_], RoundMode::CAST_NONE, batches_);
        PipeBarrier<PIPE_V>();
        Add(ends, ends, bases, aligned); PipeBarrier<PIPE_V>();
        Compares(mask0, bases, static_cast<float>(end_), CMPMODE::LT, aligned);
        Compares(mask1, ends, static_cast<float>(begin_), CMPMODE::GT, aligned);
        PipeBarrier<PIPE_V>();
        And(mask0.ReinterpretCast<uint16_t>(), mask0.ReinterpretCast<uint16_t>(),
            mask1.ReinterpretCast<uint16_t>(), aligned / 16);
        Sync<HardEvent::V_S>();
        auto words = mask0.ReinterpretCast<uint32_t>();
        for (int32_t word = 0; word < aligned / 32; ++word) {
            uint32_t bits = words.GetValue(word);
            int32_t index = word * 32;
            while (bits != 0) {
                if (bits & 1u) {
                    int64_t s, u, base, page;
                    if (RequestValid(index, s, u, base, page) &&
                        base < end_ && base + u > begin_) {
                        metaLocal_.SetValue(2 * batches_ + listLen_, index);
                        ++listLen_;
                    }
                }
                bits >>= 1;
                ++index;
            }
        }
    }

    // Ownership-transfer row of one request: the seg in the write window whose
    // slot equals the residual slot (start-1)%32, i.e. seg % 32 == 31.
    __aicore__ inline int64_t TransferSeg(int64_t used) const {
        const int64_t tail = used < RING ? used : RING;
        const int64_t seg = 31 + 32 * ((used - tail) / RING);
        if (seg >= used - tail && seg <= used - 1) return seg;
        return -1;
    }

    __aicore__ inline void ProcessChunk(int64_t c0, int64_t c1) {
        // ---- scalar scan -------------------------------------------------
        int32_t compCount = 0;
        int32_t resCount = 0;
        int64_t compRow[CHUNK_TILE];
        int32_t compRes[CHUNK_TILE];      // resBuf slot, -1 = plain pair rows
        int64_t resRow[RMAX];
        int64_t resAddr[RMAX];            // state GM offset (floats)
        bool needPair = false;
        int64_t loadFrom = c0;
        bool ringHere = false;
        bool xferHere = false;            // I own the reader row of an active transfer
        int64_t xferSrc = 0;              // pair GM offset of the transferred row
        int64_t xferDst = 0;              // state GM offset of the residual slot
        for (int32_t i = 0; i < listLen_; ++i) {
            const int32_t b = metaLocal_.GetValue(2 * batches_ + i);
            int64_t s, u, base, page;
            RequestValid(b, s, u, base, page);
            const int64_t jlo = base > c0 ? base : c0;
            const int64_t jhi = base + u < c1 ? u : c1 - base;
            for (int64_t j = (jlo - base); j < jhi; ++j) {
                if (((s + j) & 1) == 0) continue;
                const int64_t r = base + j;
                const bool residual = (j == 0) && ((s & 1) != 0);
                // Every completion reads its own (odd-member) pair row.
                needPair = true;
                compRow[compCount] = r;
                compRes[compCount] = -1;
                if (residual) {
                    compRes[compCount] = resCount;
                    resRow[resCount] = r;
                    resAddr[resCount] = (page * RING + (s - 1) % RING) * PAIR_COLS;
                    ++resCount;
                } else {
                    const int64_t from = r > 0 ? r - 1 : 0;
                    if (from < loadFrom) loadFrom = from;
                }
                ++compCount;
            }
            // Transfer ownership: this core owns the reader row `base` of an
            // active transfer and performs the colliding slot write itself.
            // Only an odd start reads the residual, so only then can a
            // same-round write to slot (start-1)%32 race that read.
            const int64_t seg = (s & 1) ? TransferSeg(u) : -1;
            if (seg >= 0 && base >= c0 && base < c1) {
                xferHere = true;
                xferSrc = (base + seg) * PAIR_COLS;
                xferDst = (page * RING + (s - 1) % RING) * PAIR_COLS;
            }
            if (base + u > c0 && base < c1) ringHere = true;
        }
        // All residuals of a chunk fit one staging batch (RMAX == CHUNK_TILE),
        // so every residual read is issued before the transferred write below.
        // ---- MTE2 loads ---------------------------------------------------
        // Ring writes also source their rows from the staged pair block, so a
        // chunk with ring tail rows must load them even when it hosts no
        // completion (pure-tail chunks otherwise copy stale UB into state).
        if (needPair || ringHere) {
            DataCopy(pairLocal_, pair_[loadFrom * PAIR_COLS],
                     static_cast<uint32_t>((c1 - loadFrom) * PAIR_COLS));
        }
        for (int32_t i = 0; i < resCount; ++i) {
            DataCopy(resLocal_[i * PAIR_COLS], state_[resAddr[i]], PAIR_COLS);
        }
        if (xferHere) {
            DataCopy(xferLocal_, pair_[xferSrc], PAIR_COLS);
        }
        // ---- ring writes (MTE3), reading the freshly loaded pair rows -----
        if (ringHere || xferHere) {
            Sync<HardEvent::MTE2_MTE3>();
            for (int32_t i = 0; i < listLen_; ++i) {
                const int32_t b = metaLocal_.GetValue(2 * batches_ + i);
                int64_t s, u, base, page;
                RequestValid(b, s, u, base, page);
                const int64_t tail = u < RING ? u : RING;
                int64_t lo = base + u - tail;
                if (lo < c0) lo = c0;
                int64_t hi = base + u < c1 ? base + u : c1;
                const int64_t seg = (s & 1) ? TransferSeg(u) : -1;
                const int64_t skip = seg >= 0 ? base + seg : -1;
                if (skip >= lo && skip < hi) {
                    WriteRunOne(lo, skip, s, base, page, loadFrom);
                    WriteRunOne(skip + 1, hi, s, base, page, loadFrom);
                } else {
                    WriteRunOne(lo, hi, s, base, page, loadFrom);
                }
            }
            if (xferHere) {
                DataCopy(state_[xferDst], xferLocal_, PAIR_COLS);
            }
        }
        // ---- vector compute ------------------------------------------------
        Sync<HardEvent::MTE3_V>();   // prior out copy + ring writes drained
        Sync<HardEvent::MTE2_V>();   // pair / residual rows visible to V
        const int32_t n = static_cast<int32_t>(c1 - c0);
        Duplicate(outLocal_.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                  static_cast<uint32_t>(n * D));
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < compCount; ++i) {
            const int64_t r = compRow[i];
            if (compRes[i] >= 0) {
                auto k0 = resLocal_[compRes[i] * PAIR_COLS];
                auto s0 = resLocal_[compRes[i] * PAIR_COLS + D];
                auto k1 = pairLocal_[(r - loadFrom) * PAIR_COLS];
                auto s1 = pairLocal_[(r - loadFrom) * PAIR_COLS + D];
                ComputeRow(k0, k1, s0, s1, outLocal_[(r - c0) * D]);
            } else {
                auto k0 = pairLocal_[(r - 1 - loadFrom) * PAIR_COLS];
                auto s0 = pairLocal_[(r - 1 - loadFrom) * PAIR_COLS + D];
                auto k1 = pairLocal_[(r - loadFrom) * PAIR_COLS];
                auto s1 = pairLocal_[(r - loadFrom) * PAIR_COLS + D];
                ComputeRow(k0, k1, s0, s1, outLocal_[(r - c0) * D]);
            }
        }
        Sync<HardEvent::V_MTE3>();
        DataCopy(out_[c0 * D], outLocal_, static_cast<uint32_t>(n * D));
        Sync<HardEvent::MTE3_MTE2>();
    }

    // Ring rows [lo, hi) of one request: source rows are contiguous in the
    // staged pair block; destination slots are consecutive with at most one
    // wrap around the 32-slot page.
    __aicore__ inline void WriteRunOne(int64_t lo, int64_t hi, int64_t start,
                                       int64_t base, int64_t page, int64_t loadFrom) {
        if (lo >= hi) return;
        const int64_t count = hi - lo;
        const int64_t slot0 = (start + lo - base) % RING;
        const int64_t first = count < RING - slot0 ? count : RING - slot0;
        auto src = pairLocal_[(lo - loadFrom) * PAIR_COLS];
        DataCopy(state_[(page * RING + slot0) * PAIR_COLS], src,
                 static_cast<uint32_t>(first * PAIR_COLS));
        if (first < count) {
            DataCopy(state_[page * RING * PAIR_COLS], src[first * PAIR_COLS],
                     static_cast<uint32_t>((count - first) * PAIR_COLS));
        }
    }

    // Per-row pool + BF16 round + RMSNorm; identical instruction sequence to
    // the frozen baseline's compute (single-Exp stable softmax, 64x8 reduce).
    __aicore__ inline void ComputeRow(LocalTensor<float> k0, LocalTensor<float> k1,
                                      LocalTensor<float> s0, LocalTensor<float> s1,
                                      LocalTensor<bfloat16_t> outRow) {
        auto m = fp_[D]; auto denom = fp_[2 * D]; auto q = fp_[3 * D];
        auto square = fp_[4 * D]; auto scratch = fp_[5 * D];
        auto gamma = fp_[0]; auto red = fp_[6 * D];
        auto mask = bf_.ReinterpretCast<uint8_t>();
        Compare(mask, s0, s1, CMPMODE::GE, D); PipeBarrier<PIPE_V>();
        Sub(m, s0, s1, D); PipeBarrier<PIPE_V>();
        Abs(m, m, D); PipeBarrier<PIPE_V>();
        Muls(m, m, -1.0f, D); PipeBarrier<PIPE_V>();
        Exp(m, m, D); PipeBarrier<PIPE_V>();
        Adds(denom, m, 1.0f, D); PipeBarrier<PIPE_V>();
        Reciprocal(denom, denom, D); PipeBarrier<PIPE_V>();
        Mul(s0, m, denom, D); PipeBarrier<PIPE_V>();
        Select(m, mask, denom, s0, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        PipeBarrier<PIPE_V>();
        Select(s1, mask, s0, denom, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        PipeBarrier<PIPE_V>();
        Mul(k0, k0, m, D); Mul(k1, k1, s1, D); PipeBarrier<PIPE_V>();
        Add(q, k0, k1, D); PipeBarrier<PIPE_V>();
        // Match the production pooling output's BF16 rounding before RMS.
        Cast(bf_[2 * D], q, RoundMode::CAST_RINT, D); PipeBarrier<PIPE_V>();
        Cast(q, bf_[2 * D], RoundMode::CAST_NONE, D); PipeBarrier<PIPE_V>();
        Mul(square, q, q, D); PipeBarrier<PIPE_V>();
        ReduceRepeat<ReduceType::SUM>(scratch, square, 64, 8, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ReduceDataBlock<ReduceType::SUM>(red, scratch, 8, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(red, red, 1.0f / 512.0f, 1); PipeBarrier<PIPE_V>();
        Adds(red, red, eps_, 1); PipeBarrier<PIPE_V>();
        Rsqrt(red, red, 1); PipeBarrier<PIPE_V>();
        Brcb(scratch, red, 1, {1, 8}); PipeBarrier<PIPE_V>();
        Copy(m, scratch, 64, 8, {1, 0, 8, 0}); PipeBarrier<PIPE_V>();
        Mul(q, q, m, D); PipeBarrier<PIPE_V>();
        Mul(q, q, gamma, D); PipeBarrier<PIPE_V>();
        Cast(outRow, q, RoundMode::CAST_RINT, D);
        PipeBarrier<PIPE_V>();
    }

    template <HardEvent event> __aicore__ inline void Sync() {
        const TEventID id = pipe_.FetchEventID(event);
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    TPipe pipe_;
    TBuf<TPosition::VECIN> metaBuf_, pairBuf_, resBuf_, xferBuf_;
    TBuf<TPosition::VECOUT> outBuf_;
    TBuf<TPosition::VECCALC> fpBuf_, bfBuf_, wBuf_;
    GlobalTensor<float> pair_, state_;
    GlobalTensor<int32_t> meta_;
    GlobalTensor<bfloat16_t> gamma_, out_;
    LocalTensor<int32_t> metaLocal_;
    LocalTensor<float> pairLocal_, resLocal_, xferLocal_, fp_;
    LocalTensor<float> kvW_[2], scW_[2];
    LocalTensor<bfloat16_t> outLocal_, bf_, outSlot_[2];
    TEventID evLoad_[2], evIn_[2], evOut_[2];
    bool usePipe_ = false;
    int64_t rows_, pages_, maxq_;
    int32_t batches_, listLen_;
    float eps_;
    int64_t begin_, end_;
};
}  // namespace

#define CV60N_V_ARGS GM_ADDR pair, GM_ADDR state, GM_ADDR metadata, GM_ADDR gamma, GM_ADDR out, \
    int64_t rows, int64_t pages, int32_t batches, float eps, int64_t maxq
#define CV60N_V_PASS pair, state, metadata, gamma, out, rows, pages, batches, eps, maxq

extern "C" __global__ __aicore__ void cv60n_v_c8(CV60N_V_ARGS) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cv60nVectorV<8> op;
    op.Init(CV60N_V_PASS);
    op.Run();
}
extern "C" __global__ __aicore__ void cv60n_v_c16(CV60N_V_ARGS) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cv60nVectorV<16> op;
    op.Init(CV60N_V_PASS);
    op.Run();
}

#ifndef __CCE_KT_TEST__
extern "C" void launch_cv60n_v(uint32_t blocks, int32_t chunk, void* stream,
                               uint8_t* pair, uint8_t* state, uint8_t* metadata,
                               uint8_t* gamma, uint8_t* out,
                               int64_t rows, int64_t pages, int32_t batches,
                               float eps, int64_t maxq) {
    if (chunk <= 8) {
        cv60n_v_c8<<<blocks, nullptr, stream>>>(pair, state, metadata, gamma, out,
            rows, pages, batches, eps, maxq);
    } else {
        cv60n_v_c16<<<blocks, nullptr, stream>>>(pair, state, metadata, gamma, out,
            rows, pages, batches, eps, maxq);
    }
}
#endif
