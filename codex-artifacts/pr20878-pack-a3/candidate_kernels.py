"""PR20878 A3 algorithm candidates; iteration 1 per kernel, baseline stays frozen."""
import triton
import triton.language as tl
from baseline_kernels import _radix_sort_candidate_pairs, _encode_lse_fields


@triton.jit(do_not_specialize=['ROWS'])
def _pack_sorted_candidates_radix(
    indices, scores, packed, ROWS,
    INDEX_ROW: tl.constexpr, INDEX_COL: tl.constexpr,
    SCORE_ROW: tl.constexpr, SCORE_COL: tl.constexpr,
    WORLD: tl.constexpr, RANK: tl.constexpr, INTERLEAVE: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
):
    BLOCK: tl.constexpr = 1 << LOG_BLOCK
    col = tl.arange(0, BLOCK)
    lanes = tl.arange(0, 2 * BLOCK)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        local = tl.load(indices + row * INDEX_ROW + col * INDEX_COL)
        score = tl.load(scores + row * SCORE_ROW + col * SCORE_COL).to(tl.float32)
        valid = (local.to(tl.float32) >= 0) & (tl.abs(score) < float('inf'))
        keys = tl.where(valid, local.to(tl.float32), 0.0)
        score = tl.where(valid, score, -float('inf'))
        if scores.dtype.element_ty == tl.bfloat16:
            DIGIT_BITS: tl.constexpr = 24 - LOG_BLOCK
            SHORT_BITS: tl.constexpr = ((25 + 16 + DIGIT_BITS - 1) // DIGIT_BITS - 1) * DIGIT_BITS - 16
            if tl.max(keys, 0) < (1 << SHORT_BITS):
                score, keys = _radix_sort_candidate_pairs(score, keys, col, LOG_BLOCK, SHORT_BITS, 16)
            else:
                score, keys = _radix_sort_candidate_pairs(score, keys, col, LOG_BLOCK, 25, 16)
        else:
            # Full 32 score bits: no BF16 cast, no approximate/tolerance-based sort.
            score, keys = _radix_sort_candidate_pairs(score, keys, col, LOG_BLOCK, 25, 32)
        ordered = tl.where(tl.abs(score) < float('inf'), keys, 0).to(tl.int32)
        mapped = ordered + (ordered // INTERLEAVE * (WORLD - 1) + RANK) * INTERLEAVE
        ids = tl.where(tl.abs(score) < float('inf'), mapped, -1)
        pair = tl.reshape(tl.join(ids, score.to(tl.int32, bitcast=True)), (2 * BLOCK,))
        tl.store(packed + row * BLOCK * 2 + lanes, pair)


@triton.jit
def _pack_dcp_packets(
    output, lse, sums, send,
    ST: tl.constexpr, SH: tl.constexpr, SD: tl.constexpr,
    LT: tl.constexpr, LH: tl.constexpr, UT: tl.constexpr, UH: tl.constexpr,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    FROM_STATS: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr, BP: tl.constexpr,
):
    token_tiles: tl.constexpr = triton.cdiv(T, BR)
    rr = tl.arange(0, BR)
    dd = tl.arange(0, BD)
    lane = tl.arange(0, BP)
    packet_row = tl.minimum(lane // (D + 4), BR - 1)
    packet_col = lane % (D + 4)
    for tile in range(tl.program_id(0), H * token_tiles, tl.num_programs(0)):
        head = tile // token_tiles
        start = tile % token_tiles * BR
        token = start + rr
        # Affine input tile; offsets stay vector int32, pointers promote as needed.
        values = tl.load(output + token[:, None] * ST + head * SH + dd[None, :] * SD,
                         (token[:, None] < T) & (dd[None, :] < D), 0)
        logs = tl.load(lse + token * LT + head * LH, token < T, 0).to(tl.float32)
        if FROM_STATS:
            total = tl.load(sums + token * UT + head * UH, token < T, 1).to(tl.float32)
            logs = logs + tl.log(total)
        code, hi, mid, lo = _encode_lse_fields(logs)
        code = tl.gather(code.to(tl.bfloat16), packet_row, 0)
        hi = tl.gather(hi.to(tl.bfloat16), packet_row, 0)
        mid = tl.gather(mid.to(tl.bfloat16), packet_row, 0)
        lo = tl.gather(lo.to(tl.bfloat16), packet_row, 0)
        copied = tl.gather(tl.reshape(values, (BR * BD,)),
                           packet_row * BD + tl.minimum(packet_col, D - 1), 0)
        fields = tl.where(packet_col == D, code,
                         tl.where(packet_col == D + 1, hi, tl.where(packet_col == D + 2, mid, lo)))
        packed = tl.where(packet_col < D, copied, fields)
        active = (lane < BR * (D + 4)) & (start + lane // (D + 4) < T)
        tl.store(send + (head * T + start) * (D + 4) + lane, packed, active)
