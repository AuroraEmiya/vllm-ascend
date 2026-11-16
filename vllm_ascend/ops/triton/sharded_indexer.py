# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate transport and causal metadata for a token-sharded indexer."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.hardware_profile import get_current_hardware_profile
from vllm_ascend.ops.triton.sparse_index_remap import _sort
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Bound the paired-sort live set to the Kimi Top-K shape.
MAX_FUSED_TOPK = 2048
PACK_ROWS_PER_PROGRAM = 8
PACK_COL_BLOCK = 256
MERGE_RANKS_PER_GROUP = 4
MIN_GROUPED_MERGE_SAVING_PERCENT = 10
MIN_FUSED_SORT_ROW_PERCENT = 75


@triton.jit(do_not_specialize=["ROWS"])
def _pack_candidates(
    indices,
    scores,
    packed,
    ROWS,
    index_row_stride: tl.constexpr,
    score_row_stride: tl.constexpr,
    index_col_stride: tl.constexpr,
    score_col_stride: tl.constexpr,
    K: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)[None, :]
    lanes = tl.program_id(1) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)[None, :]
    row = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    mask = (col < K) & (row < ROWS)
    local = tl.load(indices + row * index_row_stride + col * index_col_stride, mask, -1)
    # A5 exposes its native BF16 comparator; A3 already produces FP32.
    score = tl.load(scores + row * score_row_stride + col * score_col_stride, mask, -float("inf")).to(tl.float32)
    valid = (local >= 0) & (tl.abs(score) < float("inf"))
    local = tl.where(valid, local, 0).to(tl.int32)
    # Recover the remainder algebraically so mapping needs only one division.
    local_block = local // INTERLEAVE
    global_id = local + (local_block * (WORLD - 1) + RANK) * INTERLEAVE
    global_id = tl.where(valid, global_id, -1)
    score = tl.where(valid, score, -float("inf"))
    pair = tl.reshape(tl.join(global_id, score.to(tl.int32, bitcast=True)), (ROWS_PER_PROGRAM, 2 * BLOCK))
    tl.store(packed + row * K * 2 + lanes, pair, (lanes < 2 * K) & (row < ROWS))


@triton.jit
def _unpack_candidates(
    packed, id_keys, scores, ROWS: tl.constexpr, K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    tiles = tl.cdiv(K, BLOCK)
    rank = tl.program_id(1) // tiles
    tile = tl.program_id(1) - rank * tiles
    col = tile * BLOCK + tl.arange(0, BLOCK)
    lanes = tile * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    source = (rank * ROWS + row) * (2 * K) + lanes
    pair = tl.load(packed + source, lanes < K * 2, 0)
    idx, bits = tl.split(tl.reshape(pair, (BLOCK, 2)))
    score = bits.to(tl.float32, bitcast=True)
    # Model length is guarded at 2**24 by AscendSFAIndexer. Every valid ID
    # is therefore exact in the FP32 sort key; invalid entries sort last.
    key = tl.where(idx.to(tl.float32) >= 0, idx.to(tl.float32), float("inf"))
    tl.store(id_keys + row * WORLD * K + rank * K + col, key, col < K)
    tl.store(scores + row * WORLD * K + rank * K + col, score, col < K)


@triton.jit
def _finalize_candidates(
    idx,
    score,
    col,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = -1,
    INTERLEAVE: tl.constexpr = 1,
):
    valid = (col < K) & (tl.abs(score) < float("inf"))
    # Avoid converting the invalid +inf sentinel to an integer.
    result = tl.where(valid, idx, -1).to(tl.int32)
    if RANK >= 0:
        # Global score/ID selection is already complete. Sort unique position
        # keys only to compact this owner's winners without reordering ties.
        block = result // INTERLEAVE
        cycle = block // WORLD
        owner = block - cycle * WORLD
        keep = (col < K) & valid & (result >= 0) & (owner == RANK)
        local = cycle * INTERLEAVE + (result - block * INTERLEAVE)
        order = _sort(tl.where(keep, col.to(tl.float32), float("inf")), descending=False)
        lookup = tl.minimum(order, BLOCK - 1).to(tl.int32)
        selected = tl.gather(local.to(tl.float32, bitcast=True), lookup, 0).to(tl.int32, bitcast=True)
        result = tl.where(order < float("inf"), selected, -1)
    return result


@triton.jit
def _select_candidates(
    selected_keys,
    sorted_scores,
    output,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    WORLD: tl.constexpr = 1,
    RANK: tl.constexpr = -1,
    INTERLEAVE: tl.constexpr = 1,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    score = tl.load(sorted_scores + row * N + col, col < K, -float("inf"))
    idx = tl.load(selected_keys + row * K + col, col < K, 0)
    result = _finalize_candidates(idx, score, col, K, BLOCK, WORLD, RANK, INTERLEAVE)
    tl.store(output + row * K + col, result, col < K)


@triton.jit
def _sort_candidate_pairs(scores, ids, LOG_BLOCK: tl.constexpr, FIRST_STAGE: tl.constexpr = 1):
    """Exact score-desc/ID-asc order; the value-only A5 sort loses ID ties."""
    col = tl.arange(0, 1 << LOG_BLOCK)
    # Keep compare/exchange buffers loop-carried. Fully unrolling the sorting
    # network makes the A5 compiler retain >1 MiB of stage temporaries.
    for stage in tl.range(FIRST_STAGE, LOG_BLOCK + 1, loop_unroll_factor=1):
        for step in tl.range(stage - 1, -1, -1, loop_unroll_factor=1):
            partner = col ^ (1 << step)
            other_scores = tl.gather(scores, partner, 0)
            other_ids = tl.gather(ids, partner, 0)
            other_better = (other_scores > scores) | ((other_scores == scores) & (other_ids < ids))
            keep_better = ((col & (1 << stage)) == 0) == ((col & (1 << step)) == 0)
            # Equal score/ID pairs are interchangeable, including signed zero.
            swap = other_better == keep_better
            scores = tl.where(swap, other_scores, scores)
            ids = tl.where(swap, other_ids, ids)
    return scores, ids


@triton.jit
def _radix_sort_candidate_pairs(
    scores, ids, col, LOG_BLOCK: tl.constexpr, ID_BITS: tl.constexpr, SCORE_BITS: tl.constexpr
):
    # A digit plus a unique current-position tag fits exactly in FP32. The tag
    # makes each value-only native sort stable, independent of its tie policy.
    DIGIT_BITS: tl.constexpr = 24 - LOG_BLOCK
    for start in tl.static_range(0, ID_BITS + SCORE_BITS, DIGIT_BITS):
        raw = scores.to(tl.int32, bitcast=True)
        # +/-0 compare equal; keep the original score bits in the payload.
        raw = tl.where((raw & 2147483647) == 0, 0, raw)
        word = tl.where(raw < 0, raw, ~(raw ^ (-2147483648))) >> (32 - SCORE_BITS)
        if start >= ID_BITS:
            digit = word >> (start - ID_BITS)
        elif start + DIGIT_BITS <= ID_BITS:
            digit = ids.to(tl.int32) >> start
        else:
            digit = (ids.to(tl.int32) >> start) | (word << (ID_BITS - start))
        digit &= (1 << min(DIGIT_BITS, ID_BITS + SCORE_BITS - start)) - 1
        key = digit.to(tl.float32) * (1 << LOG_BLOCK) + col.to(tl.float32)
        order = _sort(key, descending=False)
        lookup = (order - tl.floor(order * (1.0 / (1 << LOG_BLOCK))) * (1 << LOG_BLOCK)).to(tl.int32)
        scores = tl.gather(scores, lookup, 0)
        ids = tl.gather(ids, lookup, 0)
    return scores, ids


@triton.jit(do_not_specialize=["ROWS"])
def _pack_sorted_candidates(
    indices,
    scores,
    packed,
    ROWS,
    INDEX_ROW: tl.constexpr,
    INDEX_COL: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    SCORE_COL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
):
    BLOCK: tl.constexpr = 1 << LOG_BLOCK
    col = tl.arange(0, BLOCK)
    lanes = tl.arange(0, 2 * BLOCK)
    # Flat, loop-carried rows avoid the larger live set of a row/column tile.
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        local = tl.load(indices + row * INDEX_ROW + col * INDEX_COL)
        score = tl.load(scores + row * SCORE_ROW + col * SCORE_COL).to(tl.float32)
        valid = (local.to(tl.float32) >= 0) & (tl.abs(score) < float("inf"))
        # This rank's local-to-global map is strictly increasing. Sorting
        # local keys preserves global-ID ties without rounding mapped IDs.
        keys = tl.where(valid, local.to(tl.float32), 0.0 if scores.dtype.element_ty == tl.bfloat16 else float("inf"))
        score = tl.where(valid, score, -float("inf"))
        if scores.dtype.element_ty == tl.bfloat16:
            SCORE_BITS: tl.constexpr = 16
            DIGIT_BITS: tl.constexpr = 24 - LOG_BLOCK
            # Include the exact 2**24 boundary ID. An on-device range check
            # saves one pass where possible, without imposing a support limit.
            ID_BITS: tl.constexpr = 25
            SHORT_BITS: tl.constexpr = (
                (ID_BITS + SCORE_BITS + DIGIT_BITS - 1) // DIGIT_BITS - 1
            ) * DIGIT_BITS - SCORE_BITS
            if tl.max(keys, 0) < (1 << SHORT_BITS):
                score, keys = _radix_sort_candidate_pairs(score, keys, col, LOG_BLOCK, SHORT_BITS, SCORE_BITS)
            else:
                score, keys = _radix_sort_candidate_pairs(score, keys, col, LOG_BLOCK, ID_BITS, SCORE_BITS)
        else:
            score, keys = _sort_candidate_pairs(score, keys, LOG_BLOCK)
        ordered = tl.where(tl.abs(score) < float("inf"), keys, 0).to(tl.int32)
        mapped = ordered + (ordered // INTERLEAVE * (WORLD - 1) + RANK) * INTERLEAVE
        ids = tl.where(tl.abs(score) < float("inf"), mapped, -1)
        pair = tl.reshape(tl.join(ids, score.to(tl.int32, bitcast=True)), (2 * BLOCK,))
        tl.store(packed + row * BLOCK * 2 + lanes, pair)


@triton.jit
def _owner_cached_boundaries(
    packed_ptr, peer_base, lower, upper, pivot, K: tl.constexpr, TILE: tl.constexpr, BY_ID: tl.constexpr
):
    # Binary-search contiguous tile endpoints, then count only the boundary
    # tile. The cached per-peer interval excludes every previously crossed group.
    a_lo = tl.zeros(lower.shape, tl.int32)
    a_hi = tl.cdiv(upper - lower, TILE)
    b_lo, b_hi = a_lo, a_hi
    while tl.sum(((a_lo < a_hi) | (b_lo < b_hi)).to(tl.int32), 0) > 0:
        a_mid, b_mid = (a_lo + a_hi) // 2, (b_lo + b_hi) // 2
        a_pos = tl.maximum(0, tl.minimum(lower + (a_mid + 1) * TILE - 1, upper - 1))
        b_pos = tl.maximum(0, tl.minimum(lower + (b_mid + 1) * TILE - 1, upper - 1))
        if BY_ID:
            a_value = tl.load(packed_ptr + peer_base + a_pos * 2)
            b_value = tl.load(packed_ptr + peer_base + b_pos * 2)
            before_a, before_b = a_value < pivot, b_value <= pivot
        else:
            a_value = tl.load(packed_ptr + peer_base + a_pos * 2 + 1).to(tl.float32, bitcast=True)
            b_value = tl.load(packed_ptr + peer_base + b_pos * 2 + 1).to(tl.float32, bitcast=True)
            before_a, before_b = a_value > pivot, b_value >= pivot
        a_active, b_active = a_lo < a_hi, b_lo < b_hi
        a_lo = tl.where(a_active & before_a, a_mid + 1, a_lo)
        a_hi = tl.where(a_active & ~before_a, a_mid, a_hi)
        b_lo = tl.where(b_active & before_b, b_mid + 1, b_lo)
        b_hi = tl.where(b_active & ~before_b, b_mid, b_hi)
    a_start = tl.minimum(lower + a_lo * TILE, upper)
    b_start = tl.minimum(lower + b_lo * TILE, upper)
    col = tl.arange(0, TILE)
    a_pos = a_start[:, None] + col[None, :]
    b_pos = b_start[:, None] + col[None, :]
    if BY_ID:
        a_value = tl.load(packed_ptr + peer_base[:, None] + tl.minimum(a_pos, K - 1) * 2)
        b_value = tl.load(packed_ptr + peer_base[:, None] + tl.minimum(b_pos, K - 1) * 2)
        before_a, before_b = a_value < pivot, b_value <= pivot
    else:
        a_value = tl.load(packed_ptr + peer_base[:, None] + tl.minimum(a_pos, K - 1) * 2 + 1).to(
            tl.float32, bitcast=True
        )
        b_value = tl.load(packed_ptr + peer_base[:, None] + tl.minimum(b_pos, K - 1) * 2 + 1).to(
            tl.float32, bitcast=True
        )
        before_a, before_b = a_value > pivot, b_value >= pivot
    a = a_start + tl.sum(((a_pos < upper[:, None]) & before_a).to(tl.int32), 1)
    b = b_start + tl.sum(((b_pos < upper[:, None]) & before_b).to(tl.int32), 1)
    return a, b


@triton.jit(do_not_specialize=["num_rows"])
def _merge_owner_prefix(
    packed_ptr,
    output_ptr,
    num_rows,
    WORLD: tl.constexpr,
    OWNER: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
    PARALLEL_PEERS: tl.constexpr = False,
):
    """Close this owner's prefix using cached score and exact int32 ID bounds."""
    K: tl.constexpr = 1 << LOG_BLOCK
    BOUND_TILE: tl.constexpr = min(TILE, 64)
    col = tl.arange(0, K)
    ranks = tl.arange(0, BLOCK_RANKS)
    peer_rank = tl.minimum(ranks, WORLD - 1)
    for row in range(tl.program_id(0), num_rows, tl.num_programs(0)):
        peer_base = (peer_rank * num_rows + row) * K * 2
        owner_base = (OWNER * num_rows + row) * K * 2
        tail_id = tl.load(packed_ptr + peer_base + (K - 1) * 2)
        tail_score = tl.load(packed_ptr + peer_base + (K - 1) * 2 + 1).to(tl.float32, bitcast=True)
        full = (ranks < WORLD) & (tail_id >= 0) & (tl.abs(tail_score) < float("inf"))
        valid_lo = tl.where(full, K, 0)
        valid_hi = tl.where(ranks < WORLD, K, 0)
        while tl.sum((valid_lo < valid_hi).to(tl.int32), 0) > 0:
            mid = (valid_lo + valid_hi) // 2
            idx = tl.load(packed_ptr + peer_base + tl.minimum(mid, K - 1) * 2)
            score = tl.load(packed_ptr + peer_base + tl.minimum(mid, K - 1) * 2 + 1).to(tl.float32, bitcast=True)
            valid = (idx >= 0) & (tl.abs(score) < float("inf"))
            active = valid_lo < valid_hi
            valid_lo = tl.where(active & valid, mid + 1, valid_lo)
            valid_hi = tl.where(active & ~valid, mid, valid_hi)
        own_valid = tl.sum(tl.where(ranks == OWNER, valid_lo, 0), 0)
        selected_count = own_valid
        own_tail_pos = tl.maximum(own_valid - 1, 0)
        own_tail_id = tl.load(packed_ptr + owner_base + own_tail_pos * 2)
        own_tail = tl.load(packed_ptr + owner_base + own_tail_pos * 2 + 1).to(tl.float32, bitcast=True)
        own_head = tl.load(packed_ptr + owner_base + 1).to(tl.float32, bitcast=True)
        head_id = tl.load(packed_ptr + peer_base)
        head_score = tl.load(packed_ptr + peer_base + 1).to(tl.float32, bitcast=True)
        valid_peer = (ranks < WORLD) & (ranks != OWNER) & (valid_lo > 0)
        better_tail = valid_peer & ((head_score > own_tail) | ((head_score == own_tail) & (head_id < own_tail_id)))
        peer_peak = tl.max(tl.where(valid_peer, head_score, -float("inf")), 0)
        skewed = peer_peak - own_head > own_head - own_tail
        if (own_valid > 0) & (tl.sum(valid_lo, 0) > K) & (tl.sum(better_tail.to(tl.int32), 0) > 0):
            lower = tl.zeros((BLOCK_RANKS,), tl.int32)
            upper = valid_lo
            own_lower, own_upper = tl.full((), 0, tl.int32), own_valid
            found = tl.full((), False, tl.int1)
            iteration = tl.full((), 0, tl.int32)
            a, b = lower, upper
            while (own_lower < own_upper) & ~found:
                pivot_pos = (own_lower + own_upper) // 2
                if iteration == 0:
                    pivot_pos = tl.where(skewed, 0, tl.minimum(own_upper - 1, K // WORLD))
                pivot = tl.load(packed_ptr + owner_base + pivot_pos * 2 + 1).to(tl.float32, bitcast=True)
                a, b = _owner_cached_boundaries(packed_ptr, peer_base, lower, upper, pivot, K, BOUND_TILE, False)
                greater, through = tl.sum(a, 0), tl.sum(b, 0)
                found = (greater < K) & (through >= K)
                upper = tl.where(greater >= K, a, upper)
                lower = tl.where(through < K, b, lower)
                own_lower = tl.sum(tl.where(ranks == OWNER, lower, 0), 0)
                own_upper = tl.sum(tl.where(ranks == OWNER, upper, 0), 0)
                iteration += 1
            selected_count = own_lower
            if found:
                selected_count = tl.sum(tl.where(ranks == OWNER, b, 0), 0)
                own_a = tl.sum(tl.where(ranks == OWNER, a, 0), 0)
                own_b = selected_count
                if (tl.sum(b, 0) > K) & (own_a < own_b):
                    lower, upper = a, b
                    own_lower, own_upper = own_a, own_b
                    found = tl.full((), False, tl.int1)
                    while (own_lower < own_upper) & ~found:
                        mid = (own_lower + own_upper) // 2
                        pivot_id = tl.load(packed_ptr + owner_base + mid * 2)
                        a, b = _owner_cached_boundaries(
                            packed_ptr, peer_base, lower, upper, pivot_id, K, BOUND_TILE, True
                        )
                        less, through = tl.sum(a, 0), tl.sum(b, 0)
                        found = (less < K) & (through >= K)
                        upper = tl.where(less >= K, a, upper)
                        lower = tl.where(through < K, b, lower)
                        own_lower = tl.sum(tl.where(ranks == OWNER, lower, 0), 0)
                        own_upper = tl.sum(tl.where(ranks == OWNER, upper, 0), 0)
                    selected_count = tl.where(found, tl.sum(tl.where(ranks == OWNER, b, 0), 0), own_lower)
        own_ids = tl.load(packed_ptr + owner_base + col * 2)
        block = own_ids // INTERLEAVE
        local = block // WORLD * INTERLEAVE + own_ids - block * INTERLEAVE
        tl.store(output_ptr + row * K + col, tl.where(col < selected_count, local, -1))


@triton.jit(do_not_specialize=["ROWS"])
def _merge_candidates(
    packed,
    output,
    ROWS,
    LOCAL_K: tl.constexpr,
    K: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    PRESORTED: tl.constexpr = False,
    GROUP_SIZE: tl.constexpr = 0,
    OWNER_WORLD: tl.constexpr = 0,
):
    # Each launch assigns a complete row or rank-group row to one program.
    # The next launch consumes partial winners after the stream boundary;
    # programs within a launch never read one another's intermediate state.
    BLOCK: tl.constexpr = 1 << LOG_BLOCK
    col = tl.arange(0, BLOCK)
    GROUPS: tl.constexpr = tl.cdiv(WORLD, GROUP_SIZE) if GROUP_SIZE else 1
    for work in range(tl.program_id(0), ROWS * GROUPS, tl.num_programs(0)):
        if GROUP_SIZE:
            group = work // ROWS
            row = work - group * ROWS
            first_rank = group * GROUP_SIZE
            end_rank = tl.minimum(first_rank + GROUP_SIZE, WORLD)
        else:
            row = work
            first_rank = 0
            end_rank = WORLD
        best_ids = tl.full((BLOCK,), float("inf"), tl.float32)
        best_scores = tl.full((BLOCK,), -float("inf"), tl.float32)
        for rank in range(first_rank, end_rank):
            for start in range(0, LOCAL_K, BLOCK):
                lanes = tl.arange(0, 2 * BLOCK)
                pair = tl.load(
                    packed + (rank * ROWS + row) * (2 * LOCAL_K) + start * 2 + lanes, lanes < (LOCAL_K - start) * 2, 0
                )
                raw_ids, bits = tl.split(tl.reshape(pair, (BLOCK, 2)))
                ids = tl.where(start + col < LOCAL_K, raw_ids, -1).to(tl.float32)
                scores = bits.to(tl.float32, bitcast=True)
                valid = (ids >= 0) & (tl.abs(scores) < float("inf"))
                ids = tl.where(valid, ids, float("inf"))
                scores = tl.where(valid, scores, -float("inf"))
                if not PRESORTED:
                    scores, ids = _sort_candidate_pairs(scores, ids, LOG_BLOCK)
                if rank == first_rank and start == 0:
                    # The first sorted tile already is the exact current Top-K.
                    best_scores, best_ids = scores, ids
                else:
                    # The first bitonic compare discards the losing half. Its
                    # K winners remain bitonic; sorting the discarded K is wasted.
                    reverse = BLOCK - 1 - col
                    incoming_scores = tl.gather(scores, reverse, 0)
                    incoming_ids = tl.gather(ids, reverse, 0)
                    take = (incoming_scores > best_scores) | (
                        (incoming_scores == best_scores) & (incoming_ids < best_ids)
                    )
                    scores = tl.where(take, incoming_scores, best_scores)
                    ids = tl.where(take, incoming_ids, best_ids)
                    best_scores, best_ids = _sort_candidate_pairs(scores, ids, LOG_BLOCK, LOG_BLOCK)
        if GROUP_SIZE:
            # A group may discard everything below its own K winners: each
            # discarded entry already has K better entries in the same group.
            ids = tl.where(tl.abs(best_scores) < float("inf"), best_ids, -1).to(tl.int32)
            pair = tl.reshape(tl.join(ids, best_scores.to(tl.int32, bitcast=True)), (2 * BLOCK,))
            lanes = tl.arange(0, 2 * BLOCK)
            tl.store(output + work * K * 2 + lanes, pair, lanes < 2 * K)
        else:
            owners: tl.constexpr = OWNER_WORLD if OWNER_WORLD else WORLD
            result = _finalize_candidates(best_ids, best_scores, col, K, BLOCK, owners, RANK, INTERLEAVE)
            tl.store(output + row * K + col, result, col < K)


def select_local_candidates(indices, scores, world_size: int, rank: int, interleave: int, *, out=None):
    """Fuse final score masking, local remap and stable compaction."""
    output = torch.empty_like(indices, dtype=torch.int32) if out is None else out
    if indices.numel() == 0:
        return output
    topk = indices.shape[-1]
    rows = indices.numel() // topk
    _select_candidates[(rows, 1)](
        indices,
        scores,
        output,
        scores.numel() // rows,
        topk,
        triton.next_power_of_2(topk),
        WORLD=world_size,
        RANK=rank,
        INTERLEAVE=interleave,
    )
    return output


@triton.jit(do_not_specialize=["ROWS"])
def _prepare_local_sort_inputs(
    indices,
    scores,
    keys,
    values,
    ROWS,
    INDEX_ROW: tl.constexpr,
    INDEX_COL: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    SCORE_COL: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col = tl.arange(0, BLOCK)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        local = tl.load(indices + row * INDEX_ROW + col * INDEX_COL, col < K, -1)
        score = tl.load(scores + row * SCORE_ROW + col * SCORE_COL, col < K, -float("inf")).to(tl.float32)
        valid = (local >= 0) & (tl.abs(score) < float("inf"))
        tl.store(keys + row * K + col, tl.where(valid, local.to(tl.float32), float("inf")), col < K)
        tl.store(values + row * K + col, tl.where(valid, score, -float("inf")), col < K)


def _local_sort_inputs(indices, scores):
    keys = torch.empty(indices.shape, dtype=torch.float32, device=indices.device)
    values = torch.empty_like(keys)
    rows, _, topk = indices.shape
    if rows and topk:
        init_device_properties_triton()
        _prepare_local_sort_inputs[(min(rows, get_vectorcore_num()),)](
            indices,
            scores,
            keys,
            values,
            rows,
            indices.stride(0),
            indices.stride(-1),
            scores.stride(0),
            scores.stride(-1),
            topk,
            1 << (topk - 1).bit_length(),
        )
    return keys, values


def _sort_local_candidates(indices, scores):
    """Return exact FP32 ID keys and scores in score-desc/ID-asc order."""
    # For one owner the local-to-global mapping is strictly increasing.
    # The backend's 2**24 length bound makes these FP32 ID keys exact.
    keys, values = _local_sort_inputs(indices, scores)
    sorted_keys, id_order = torch.sort(keys, dim=-1, stable=True)
    values = torch.gather(values, -1, id_order)
    ordered_scores, score_order = torch.sort(values, dim=-1, descending=True, stable=True)
    # Reuse sort values instead of gathering INT64 permutations. Pack restores
    # INT32 IDs and masks the +inf keys belonging to invalid scores.
    return torch.gather(sorted_keys, -1, score_order), ordered_scores


def pack_local_candidates(
    indices, scores, world_size: int, rank: int, interleave: int, *, sort_candidates=False, fuse_sort=False
):
    """Transport exact pairs; fuse_sort permits fusion when its work amortizes."""
    rows, _, topk = indices.shape
    packed = torch.empty((rows, topk, 2), dtype=torch.int32, device=indices.device)
    if rows:
        if sort_candidates and fuse_sort and topk == MAX_FUSED_TOPK:
            init_device_properties_triton()
            cores = get_vectorcore_num()
            log_block = MAX_FUSED_TOPK.bit_length() - 1
            local_sort_levels = log_block * (log_block + 1) // 2
            serial_merge_levels = (world_size - 1) * log_block
            # The A3/A5 sweep finds that fused sorting needs both row
            # occupancy and enough downstream merge work to pay off.
            # Small workloads retain native preparation but still use the
            # exact owner-prefix merge. No device-model whitelist is needed.
            if rows * 100 >= cores * MIN_FUSED_SORT_ROW_PERCENT and serial_merge_levels >= local_sort_levels:
                _pack_sorted_candidates[(min(rows, cores),)](
                    indices,
                    scores,
                    packed,
                    rows,
                    indices.stride(0),
                    indices.stride(-1),
                    scores.stride(0),
                    scores.stride(-1),
                    world_size,
                    rank,
                    interleave,
                    log_block,
                    multibuffer=False,
                )
                return packed
        if sort_candidates:
            indices, scores = _sort_local_candidates(indices, scores)
        block = PACK_COL_BLOCK
        rows_per_program = PACK_ROWS_PER_PROGRAM
        _pack_candidates[(triton.cdiv(rows, rows_per_program), triton.cdiv(topk, block))](
            indices,
            scores,
            packed,
            rows,
            indices.stride(0),
            scores.stride(0),
            indices.stride(-1),
            scores.stride(-1),
            topk,
            world_size,
            rank,
            interleave,
            block,
            rows_per_program,
        )
    return packed


def merge_packed_candidates(
    packed,
    rows: int,
    world_size: int,
    topk: int,
    *,
    local_rank: int | None = None,
    interleave: int = 1,
    out=None,
    presorted=False,
    owner_sharded=False,
):
    """Fuse exact Top-K and owner compaction.

    ``owner_sharded`` additionally promises that rank r contains only owner r
    IDs, as guaranteed by pack_local_candidates. Presorted alone is weaker.
    """
    local_k = packed.shape[1]
    output = torch.empty((rows, 1, topk), dtype=torch.int32, device=packed.device) if out is None else out
    if rows == 0:
        return output
    # Smaller K can trigger consolidateToTail in the A5 VF compiler.
    # Keep the existing fallback outside the validated Kimi Top-K shape.
    if topk == MAX_FUSED_TOPK:
        init_device_properties_triton()
        cores = get_vectorcore_num()
        programs = min(rows, cores)
        block = MAX_FUSED_TOPK  # The supported fused shape already is a power of two.
        if presorted and owner_sharded and local_rank is not None and 0 <= local_rank < world_size and local_k == topk:
            tile = min(block, triton.next_power_of_2(max(1, 2 * block // world_size)))
            # Parallel setup amortizes across several peers and multiple rows/core.
            # Small worlds, single-wave batches and other policies keep the original.
            parallel_peers = (
                get_current_hardware_profile().prefer_parallel_owner_prefix and world_size > 4 and rows > cores
            )
            if parallel_peers:
                tile = min(block, triton.next_power_of_2(max(1, block // world_size)))
            _merge_owner_prefix[(programs,)](
                packed,
                output,
                rows,
                world_size,
                local_rank,
                interleave,
                block.bit_length() - 1,
                tile,
                triton.next_power_of_2(world_size),
                PARALLEL_PEERS=parallel_peers,
                multibuffer=False,
            )
            return output
        owners = world_size
        # These are host integers. Triton's constexpr wrappers add measurable
        # Python dispatch overhead to small eager batches.
        groups = (world_size + MERGE_RANKS_PER_GROUP - 1) // MERGE_RANKS_PER_GROUP
        row_waves = (rows + programs - 1) // programs
        group_waves = (rows * groups + cores - 1) // cores
        serial_work = row_waves * (world_size - 1)
        grouped_work = group_waves * (MERGE_RANKS_PER_GROUP - 1) + row_waves * (groups - 1)
        # Extra scratch traffic and a second launch need enough saved merge
        # waves to pay off. The Q1..512 sweep supports this 10% margin; large
        # batches with already balanced row ownership keep the single pass.
        if presorted and groups > 1 and grouped_work * 100 <= serial_work * (100 - MIN_GROUPED_MERGE_SAVING_PERCENT):
            partial = torch.empty((groups * rows, topk, 2), dtype=torch.int32, device=packed.device)
            _merge_candidates[(min(rows * groups, cores),)](
                packed,
                partial,
                rows,
                local_k,
                topk,
                world_size,
                -1,
                interleave,
                block.bit_length() - 1,
                True,
                GROUP_SIZE=MERGE_RANKS_PER_GROUP,
                multibuffer=False,
            )
            packed, world_size, local_k = partial, groups, topk
        _merge_candidates[(programs,)](
            packed,
            output,
            rows,
            local_k,
            topk,
            world_size,
            -1 if local_rank is None else local_rank,
            interleave,
            block.bit_length() - 1,
            presorted,
            OWNER_WORLD=owners,
            multibuffer=False,
        )
        return output
    count = world_size * local_k
    keys = torch.empty((rows, count), dtype=torch.float32, device=packed.device)
    values = torch.empty_like(keys)
    block = min(2048, triton.next_power_of_2(local_k))
    _unpack_candidates[(rows, world_size * triton.cdiv(local_k, block))](
        packed, keys, values, rows, local_k, world_size, block
    )
    sorted_keys, id_order = torch.sort(keys, dim=-1, stable=True)
    sorted_scores = torch.gather(values, -1, id_order)
    ordered_scores, score_order = torch.sort(sorted_scores, dim=-1, descending=True, stable=True)
    # Reuse both sorts' values instead of following two dependent random GM
    # indices in Triton. The native gather vectorizes the remaining lookup;
    # the final kernel only reads contiguous IDs and comparator scores.
    selected_keys = torch.gather(sorted_keys, -1, score_order[:, :topk])
    if local_rank is not None:
        select_local_candidates(
            selected_keys, ordered_scores, world_size, local_rank, interleave, out=output.view(rows, topk)
        )
        return output
    _select_candidates[(rows, triton.cdiv(topk, 256))](
        selected_keys,
        ordered_scores,
        output,
        count,
        topk,
        256,
    )
    return output


@triton.jit(do_not_specialize=["REQUESTS", "ROWS"])
def _local_lengths_small_batch(
    seq_lens,
    query_ends,
    key_lengths,
    visible_lengths,
    REQUESTS,
    ROWS,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK_REQUESTS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    request = tl.arange(0, BLOCK_REQUESTS)
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    seq = tl.load(seq_lens + request, request < REQUESTS, 0).to(tl.int32)
    end = tl.load(query_ends + request, request < REQUESTS, 0).to(tl.int32)
    start = tl.load(query_ends + request - 1, (request > 0) & (request < REQUESTS), 0).to(tl.int32)
    if tl.program_id(0) == 0:
        local = seq // (WORLD * INTERLEAVE) * INTERLEAVE
        local += tl.minimum(tl.maximum(seq % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
        tl.store(key_lengths + request, local, request < REQUESTS)
    # All request bounds fit in UB for the common decode batches. A bounded
    # row/request mask avoids dependent random GM loads during binary search.
    own = (row[:, None] >= start[None, :]) & (row[:, None] < end[None, :])
    own &= request[None, :] < REQUESTS
    visible = tl.maximum(seq[None, :] - end[None, :] + row[:, None] + 1, 0)
    visible = tl.sum(tl.where(own, visible, 0), 1)
    local = visible // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(visible % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(visible_lengths + row, local, row < ROWS)


@triton.jit(do_not_specialize=["REQUESTS", "ROWS"])
def _local_lengths(
    seq_lens,
    query_ends,
    key_lengths,
    visible_lengths,
    REQUESTS,
    ROWS,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(seq_lens + offsets, offsets < REQUESTS, 0)
    local = length // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(length % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(key_lengths + offsets, local, offsets < REQUESTS)
    # Upper_bound on device query ends also handles empty requests. Values
    # change on graph replay without rebuilding or copying a CPU row map.
    lower = tl.full((BLOCK,), 0, tl.int32)
    upper = tl.full((BLOCK,), 0, tl.int32) + REQUESTS
    for _ in range(STEPS):
        mid = (lower + upper) // 2
        end = tl.load(query_ends + mid, mid < REQUESTS, 2147483647)
        right = (mid < REQUESTS) & (end <= offsets)
        lower = tl.where(right, mid + 1, lower)
        upper = tl.where(right, upper, mid)
    request = tl.minimum(lower, REQUESTS - 1)
    seq = tl.load(seq_lens + request)
    end = tl.load(query_ends + request)
    total = tl.load(query_ends + REQUESTS - 1)
    visible = tl.maximum(seq - end + offsets + 1, 0)
    visible = tl.where(offsets < total, visible, 0)
    local = visible // (WORLD * INTERLEAVE) * INTERLEAVE
    local += tl.minimum(tl.maximum(visible % (WORLD * INTERLEAVE) - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(visible_lengths + offsets, local, offsets < ROWS)


@triton.jit(do_not_specialize=["ROWS"])
def _zero_visible_lengths(visible_lengths, ROWS, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(visible_lengths + row, 0, row < ROWS)


def build_local_lengths(
    seq_lens, query_ends, key_lengths, visible_lengths, world_size: int, rank: int, interleave: int
):
    """Write stable metadata buffers used by native causal local QLI."""
    requests, rows = seq_lens.numel(), visible_lengths.numel()
    if requests == 0:
        if rows:
            _zero_visible_lengths[(triton.cdiv(rows, 256),)](visible_lengths, rows, 256)
        return
    if requests <= 32 and rows:
        _local_lengths_small_batch[(triton.cdiv(rows, 32),)](
            seq_lens,
            query_ends,
            key_lengths,
            visible_lengths,
            requests,
            rows,
            world_size,
            rank,
            interleave,
            triton.next_power_of_2(requests),
            32,
        )
        return
    _local_lengths[(triton.cdiv(max(requests, rows), 128),)](
        seq_lens,
        query_ends,
        key_lengths,
        visible_lengths,
        requests,
        rows,
        world_size,
        rank,
        interleave,
        128,
        requests.bit_length(),
    )


@triton.jit(do_not_specialize=["REQUESTS"])
def _local_key_lengths(
    seq_lens,
    key_lengths,
    REQUESTS,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(seq_lens + request, request < REQUESTS, 0).to(tl.int32)
    cycle = WORLD * INTERLEAVE
    local = seq // cycle * INTERLEAVE
    local += tl.minimum(tl.maximum(seq % cycle - RANK * INTERLEAVE, 0), INTERLEAVE)
    tl.store(key_lengths + request, local, request < REQUESTS)


def build_local_key_lengths(seq_lens, key_lengths, world_size: int, rank: int, interleave: int) -> None:
    """Build local cache lengths when QLI computes causal rows internally."""
    requests = seq_lens.numel()
    if requests:
        _local_key_lengths[(triton.cdiv(requests, 128),)](
            seq_lens, key_lengths, requests, world_size, rank, interleave, 128
        )


@triton.jit(do_not_specialize=["REQUESTS"])
def _normalize_dcp_candidates(indices, scores, query_start_loc, REQUESTS, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active_rows = tl.load(query_start_loc + REQUESTS)
    active = (row < active_rows) & (col < K)
    idx = tl.load(indices + row * K + col, active, -1)
    score = tl.load(scores + row * K + col, active, -float("inf")).to(tl.float32)
    valid = active & (idx >= 0) & (tl.abs(score) < float("inf"))
    tl.store(indices + row * K + col, tl.where(valid, idx, -1), col < K)
    tl.store(scores + row * K + col, tl.where(valid, score, -float("inf")), col < K)


def normalize_dcp_candidates(indices, scores, query_start_loc):
    """Clear invalid candidates and graph padding in native FP32 outputs in place.

    QLI does not initialize rows beyond the cumulative query endpoint. Each
    kernel lane owns its index/score slot, so no extra transport buffer is needed.
    """
    if scores.dtype != torch.float32:
        raise TypeError("DCP candidate normalization requires native FP32 QLI scores")
    rows, _, topk = indices.shape
    if rows:
        _normalize_dcp_candidates[(rows, triton.cdiv(topk, 256))](
            indices, scores, query_start_loc, query_start_loc.numel() - 1, topk, 256
        )
    return indices, scores
