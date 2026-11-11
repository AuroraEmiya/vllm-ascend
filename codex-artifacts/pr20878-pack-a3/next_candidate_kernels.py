from vllm.triton_utils import triton, tl

@triton.jit(do_not_specialize=["ROWS"])
def _merge_owner_adaptive(
    packed,
    output,
    ROWS,
    WORLD: tl.constexpr,
    OWNER: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    LOG_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
):
    """Select an owner's exact sorted prefix without sorting other owners."""
    K: tl.constexpr = 1 << LOG_BLOCK
    col = tl.arange(0, K)
    lanes = tl.arange(0, 2 * TILE)
    SKEW_TILE: tl.constexpr = min(K, triton.next_power_of_2(2 * K // WORLD))
    skew_lanes = tl.arange(0, 2 * SKEW_TILE)
    ranks = tl.arange(0, BLOCK_RANKS)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        own = packed + (OWNER * ROWS + row) * K * 2
        own_pair = tl.load(own + tl.arange(0, 2 * K))
        own_ids, own_bits = tl.split(tl.reshape(own_pair, (K, 2)))
        own_scores = own_bits.to(tl.float32, bitcast=True)
        own_valid = (own_ids.to(tl.float32) >= 0) & (tl.abs(own_scores) < float("inf"))
        upper = tl.sum(own_valid.to(tl.float32), 0).to(tl.int32)
        tail_id = tl.load(own + (upper - 1) * 2, upper > 0, -1).to(tl.float32)
        tail_score = tl.load(own + (upper - 1) * 2 + 1, upper > 0, 0).to(tl.float32, bitcast=True)
        head_score = tl.load(own + 1, upper > 0, 0).to(tl.float32, bitcast=True)
        equal_scores = (upper > 0) & (head_score == tail_score)
        head_mask = (ranks.to(tl.float32) < WORLD) & (ranks != OWNER) & (upper > 0)
        # A3's strided masked loads can still fault on padded peer addresses.
        # Bound the addresses as well as the mask for non-power-of-two worlds.
        peer_rank = tl.minimum(ranks, WORLD - 1)
        peer_id = tl.load(packed + (peer_rank * ROWS + row) * K * 2, head_mask, -1).to(tl.float32)
        peer_score = tl.load(packed + (peer_rank * ROWS + row) * K * 2 + 1, head_mask, 0).to(tl.float32, bitcast=True)
        peer_valid = (peer_id >= 0) & (tl.abs(peer_score) < float("inf"))
        before_tail = peer_valid & ((peer_score > tail_score) | ((peer_score == tail_score) & (peer_id < tail_id)))
        # If even the worst own candidate precedes every peer head, all own
        # valid entries are winners. Other rows search their exact cutoff.
        lower = tl.where(tl.sum(before_tail.to(tl.float32), 0) == 0.0, upper, 0)
        peer_peak = tl.max(tl.where(peer_valid, peer_score, -float("inf")), 0)
        # A peer head beyond the owner's observed score span hints at skew.
        # This chooses a probe only; exact counts still establish all bounds.
        skewed = (peer_peak - head_score) > (head_score - tail_score)
        pivot = tl.where(skewed, 0, tl.minimum(tl.full((), K // WORLD, tl.int32), upper - 1))
        # Three interpolation probes are followed by enough bisections for
        # every possible boundary. Estimates affect work only, never selection.
        iteration = tl.full((), 0, tl.int32)
        while (lower < upper) & (iteration < LOG_BLOCK + 4):
            pivot_id = tl.load(own + pivot * 2).to(tl.float32)
            pivot_score = tl.load(own + pivot * 2 + 1).to(tl.float32, bitcast=True)
            pivot_valid = (pivot_id >= 0) & (tl.abs(pivot_score) < float("inf"))
            count = pivot.to(tl.float32)
            if (WORLD <= 4) | skewed:
                for rank in range(WORLD):
                    if rank != OWNER:
                        start = tl.full((), 0, tl.int32)
                        amount = tl.full((), SKEW_TILE, tl.float32)
                        # Sorted peers cannot have a better candidate after
                        # the first tile containing any non-better candidate.
                        while (start < K) & (amount == SKEW_TILE):
                            pair = tl.load(packed + (rank * ROWS + row) * K * 2 + start * 2 + skew_lanes)
                            raw_ids, bits = tl.split(tl.reshape(pair, (SKEW_TILE, 2)))
                            ids = raw_ids.to(tl.float32)
                            scores = bits.to(tl.float32, bitcast=True)
                            valid = (ids >= 0) & (tl.abs(scores) < float("inf"))
                            better = valid & ((scores > pivot_score) | ((scores == pivot_score) & (ids < pivot_id)))
                            amount = tl.sum(better.to(tl.float32), 0)
                            count += amount
                            start += SKEW_TILE
            else:
                # All peers evaluate this pivot together. Inactive peers stop counting
                # after their first partial tile; bounded loads still cover all peers.
                active = (ranks < WORLD) & (ranks != OWNER)
                start = tl.full((), 0, tl.int32)
                peer_count = tl.zeros((BLOCK_RANKS,), tl.float32)
                while (start < K) & (tl.sum(active.to(tl.int32), 0) > 0):
                    offsets = ((peer_rank[:, None] * ROWS + row) * K + start) * 2 + lanes[None, :]
                    pair = tl.load(packed + offsets)
                    raw_ids, bits = tl.split(tl.reshape(pair, (BLOCK_RANKS, TILE, 2)))
                    ids = raw_ids.to(tl.float32)
                    scores = bits.to(tl.float32, bitcast=True)
                    valid = active[:, None] & (ids >= 0) & (tl.abs(scores) < float("inf"))
                    better = valid & ((scores > pivot_score) | ((scores == pivot_score) & (ids < pivot_id)))
                    amount = tl.sum(better.to(tl.float32), 1)
                    peer_count += amount
                    active = active & (amount == TILE)
                    start += TILE
                count += tl.sum(peer_count, 0)
            selected = pivot_valid & (count < K)
            # Let S be the number of better peer entries. Monotonicity
            # makes K-S an upper bound after a selected pivot, or a lower
            # bound after a rejected valid pivot. Invalid pivots only
            # reduce the upper bound.
            jump = (K - count + pivot.to(tl.float32)).to(tl.int32)
            lower = tl.where(selected, pivot + 1, tl.where(pivot_valid, tl.maximum(lower, jump), lower))
            upper = tl.where(selected, tl.minimum(upper, jump), pivot)
            if equal_scores & pivot_valid & (lower < upper):
                block_begin = pivot_id.to(tl.int32) // INTERLEAVE * INTERLEAVE
                block_last = block_begin + INTERLEAVE - 1
                block_start = tl.sum(
                    (own_valid & (own_ids.to(tl.float32) < block_begin.to(tl.float32))).to(tl.float32), 0
                ).to(tl.int32)
                block_end = tl.sum(
                    (own_valid & (own_ids.to(tl.float32) <= block_last.to(tl.float32))).to(tl.float32), 0
                ).to(tl.int32)
                # No peer can insert an equal-score ID inside this owner's
                # interleave block. K-S either solves the boundary in this
                # block or proves that the entire block can be skipped.
                if (jump >= block_start) & (jump <= block_end):
                    lower, upper = jump, jump
                else:
                    lower = tl.where(jump > block_end, tl.maximum(lower, block_end), lower)
                    upper = tl.where(jump < block_start, tl.minimum(upper, block_start), upper)
            if iteration < 2:
                estimate = (pivot.to(tl.float32) + (K - count) / WORLD).to(tl.int32)
                pivot = tl.minimum(tl.maximum(estimate, lower), upper - 1)
            else:
                pivot = (lower + upper) // 2
            iteration += 1
        block = own_ids // INTERLEAVE
        local = block // WORLD * INTERLEAVE + own_ids - block * INTERLEAVE
        result = tl.where(col.to(tl.float32) < lower.to(tl.float32), local, -1)
        tl.store(output + row * K + col, result)


@triton.jit
def _fused_dcp_lse_combine_rank_reduce(
    recv_ptr,
    output_ptr,
    local_output_ptr,
    local_lse_ptr,
    local_output_stride_t,
    local_output_stride_h,
    local_output_stride_d,
    local_lse_stride_t,
    local_lse_stride_h,
    recv_stride_rank,
    recv_stride_scatter,
    recv_stride_replicated,
    recv_stride_d,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    head_dim,
    num_heads,
    total_rows,
    DCP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True,
    RANK_UNROLL: tl.constexpr = 1,
):
    # Parallel rank reduction: decode once and remove per-rank scalar extracts.
    # Bound the wide FP32 tile independently of traced T/H: at most8192 values.
    BLOCK_RANKS: tl.constexpr = triton.next_power_of_2(DCP_SIZE)
    TILE_ROWS: tl.constexpr = min(BLOCK_ROWS, max(1, 8192 // (BLOCK_RANKS * BLOCK_D)))
    ranks = tl.arange(0, BLOCK_RANKS)
    bounded_ranks = tl.minimum(ranks, DCP_SIZE - 1)
    d = tl.arange(0, BLOCK_D)
    for row_start in range(tl.program_id(0) * TILE_ROWS, total_rows, tl.num_programs(0) * TILE_ROWS):
        rows = row_start + tl.arange(0, TILE_ROWS)
        row_mask = rows < total_rows
        tokens = (rows // num_heads).to(tl.int64)
        heads = (rows % num_heads).to(tl.int64)
        bases = bounded_ranks[:, None] * recv_stride_rank + heads[None, :] * recv_stride_scatter + tokens[None, :] * recv_stride_replicated
        rank_row_mask = (ranks[:, None] < DCP_SIZE) & row_mask[None, :]
        packed = tl.load(
            recv_ptr + bases[:, :, None] + (head_dim + tl.arange(0, 4)[None, None, :]) * recv_stride_d,
            rank_row_mask[:, :, None], 0.0,
        ).to(tl.float32)
        even, odd = tl.split(tl.reshape(packed, (BLOCK_RANKS, TILE_ROWS, 2, 2)))
        code, mid = tl.split(even)
        hi, lo = tl.split(odd)
        significand = hi * 65536.0 + mid * 256.0 + lo
        lse_bits = (tl.maximum(tl.abs(code).to(tl.int32) - 2, 0) << 23) + significand.to(tl.int32)
        lse_bits = tl.where(significand != 0.0, lse_bits, 0)
        lses = tl.where(code < 0.0, -1.0, 1.0) * lse_bits.to(tl.float32, bitcast=True)
        valid = rank_row_mask & (code != 0.0) & (tl.abs(lses) < float("inf"))
        maximum = tl.max(tl.where(valid, lses, -float("inf")), 0)
        if HAS_LOCAL:
            local_lse = tl.load(local_lse_ptr + tokens * local_lse_stride_t + heads * local_lse_stride_h, row_mask, 0.0).to(tl.float32)
            local_valid = tl.abs(local_lse) < float("inf")
            maximum = tl.maximum(maximum, tl.where(local_valid, local_lse, -float("inf")))
        maximum = tl.where(maximum != -float("inf"), maximum, 0.0)
        weights = tl.where(valid, tl.exp(lses - maximum[None, :]), 0.0)
        values = tl.load(recv_ptr + bases[:, :, None] + d[None, None, :] * recv_stride_d,
                         rank_row_mask[:, :, None] & (d[None, None, :] < head_dim), 0.0).to(tl.float32)
        numerator = tl.sum(tl.where(valid[:, :, None], values, 0.0) * weights[:, :, None], 0)
        denominator = tl.sum(weights, 0)
        if HAS_LOCAL:
            local_weight = tl.where(local_valid, tl.exp(local_lse - maximum), 0.0)
            local_values = tl.load(local_output_ptr + tokens[:, None] * local_output_stride_t + heads[:, None] * local_output_stride_h + d[None, :] * local_output_stride_d,
                                   row_mask[:, None] & (d[None, :] < head_dim), 0.0).to(tl.float32)
            numerator += tl.where(local_valid[:, None], local_values, 0.0) * local_weight[:, None]
            denominator += local_weight
        result = numerator / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
        out_offsets = tokens[:, None] * output_stride_t + heads[:, None] * output_stride_h + d[None, :] * output_stride_d
        tl.store(output_ptr + out_offsets, result, row_mask[:, None] & (d[None, :] < head_dim))
