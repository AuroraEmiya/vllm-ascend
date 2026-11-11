from vllm.triton_utils import triton, tl

@triton.jit(do_not_specialize=["ROWS"])
def _merge_owner_parallel(
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
            # All peers evaluate this pivot together. Inactive peers stop loading
            # after their first partial tile; the shared cursor remains bounded.
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
def _fused_dcp_lse_combine_cached(
    recv_ptr, output_ptr, local_output_ptr, local_lse_ptr,
    local_output_stride_t, local_output_stride_h, local_output_stride_d: tl.constexpr,
    local_lse_stride_t: tl.constexpr, local_lse_stride_h: tl.constexpr,
    recv_stride_rank: tl.constexpr, recv_stride_scatter: tl.constexpr,
    recv_stride_replicated: tl.constexpr, recv_stride_d: tl.constexpr,
    output_stride_t: tl.constexpr, output_stride_h: tl.constexpr, output_stride_d: tl.constexpr,
    head_dim: tl.constexpr, num_heads: tl.constexpr, total_rows,
    DCP_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True, RANK_UNROLL: tl.constexpr = 1,
):
    # Cache only the small LSE matrix; keep the wide output streaming by rank.
    BLOCK_RANKS: tl.constexpr = triton.next_power_of_2(DCP_SIZE)
    ranks = tl.arange(0, BLOCK_RANKS)[None, :]
    d_offsets = tl.arange(0, BLOCK_D)[None, :]
    for row_start in range(tl.program_id(0) * BLOCK_ROWS, total_rows, tl.num_programs(0) * BLOCK_ROWS):
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        tokens = rows // num_heads
        heads = rows % num_heads
        mask = rows < total_rows
        bases = ranks * recv_stride_rank + heads[:, None] * recv_stride_scatter + tokens[:, None] * recv_stride_replicated
        packed = tl.load(recv_ptr + bases[:, :, None] + (head_dim + tl.arange(0, 4)[None, None, :]) * recv_stride_d,
                         mask[:, None, None] & (ranks[:, :, None] < DCP_SIZE), 0).to(tl.float32)
        even, odd = tl.split(tl.reshape(packed, (BLOCK_ROWS * BLOCK_RANKS, 2, 2)))
        code, mid = tl.split(even)
        hi, lo = tl.split(odd)
        significand = hi * 65536.0 + mid * 256.0 + lo
        lse_bits = (tl.maximum(tl.abs(code).to(tl.int32) - 2, 0) << 23) + significand.to(tl.int32)
        lse_bits = tl.where(significand != 0.0, lse_bits, 0)
        lses = tl.reshape(tl.where(code < 0.0, -1.0, 1.0) * lse_bits.to(tl.float32, bitcast=True), (BLOCK_ROWS, BLOCK_RANKS))
        valid = tl.reshape(code != 0.0, (BLOCK_ROWS, BLOCK_RANKS)) & (ranks < DCP_SIZE) & (tl.abs(lses) < float("inf"))
        max_lse = tl.max(tl.where(valid, lses, -float("inf")), 1)
        if HAS_LOCAL:
            local_lse = tl.load(local_lse_ptr + tokens * local_lse_stride_t + heads * local_lse_stride_h, mask, 0).to(tl.float32)
            local_valid = tl.abs(local_lse) < float("inf")
            max_lse = tl.maximum(max_lse, tl.where(local_valid, local_lse, -float("inf")))
        safe_max = tl.where(max_lse != -float("inf"), max_lse, 0.0)
        weights = tl.where(valid, tl.exp(lses - safe_max[:, None]), 0.0)
        weight_sum = tl.sum(weights, 1)[:, None]
        merged = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)
        d_mask = mask[:, None] & (d_offsets < head_dim)
        for rank in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
            weight = tl.sum(tl.where(ranks == rank, weights, 0.0), 1)[:, None]
            rank_valid = tl.sum((valid & (ranks == rank)).to(tl.int32), 1)[:, None] > 0
            recv_base = rank * recv_stride_rank + heads[:, None] * recv_stride_scatter + tokens[:, None] * recv_stride_replicated
            values = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_d, d_mask, 0).to(tl.float32)
            merged += tl.where(rank_valid, values, 0.0) * weight
        if HAS_LOCAL:
            weight = tl.where(local_valid, tl.exp(local_lse - safe_max), 0.0)[:, None]
            local_base = tokens[:, None] * local_output_stride_t + heads[:, None] * local_output_stride_h
            values = tl.load(local_output_ptr + local_base + d_offsets * local_output_stride_d, d_mask, 0).to(tl.float32)
            merged += tl.where(local_valid[:, None], values, 0.0) * weight
            weight_sum += weight
        merged /= tl.where(weight_sum > 0.0, weight_sum, 1.0)
        out = tokens[:, None] * output_stride_t + heads[:, None] * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + out, merged, d_mask)
