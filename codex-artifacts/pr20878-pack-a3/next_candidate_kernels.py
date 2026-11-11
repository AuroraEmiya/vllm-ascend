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
            if skewed:
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
def _fused_dcp_lse_combine_rowvec(
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
    # Put row coefficients on the inner compute axis to avoid per-row scalar
    # coefficient extraction in broadcast math; retain the rank-streaming math.
    # Each received row therefore ends with four packed LSE values.
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[:, None]

    for row_start in range(program_idx * BLOCK_ROWS, total_rows, num_programs * BLOCK_ROWS):
        linear_idx = row_start + tl.arange(0, BLOCK_ROWS)[None, :]
        row_mask = linear_idx < total_rows
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        # Coalesced four-slot loads use the upstream streaming-rank
        # formulation; only one output accumulator stays live.
        lse_max = tl.full((1, BLOCK_ROWS), -float("inf"), tl.float32)
        if HAS_LOCAL:
            local_lse = tl.load(
                local_lse_ptr + token_idx * local_lse_stride_t + head_idx * local_lse_stride_h,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            local_valid = tl.abs(local_lse) < float("inf")
            lse_max = tl.where(local_valid, local_lse, -float("inf"))
        for rank_idx in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
            recv_base = (
                rank_idx * recv_stride_rank + head_idx * recv_stride_scatter + token_idx * recv_stride_replicated
            )
            packed = tl.load(
                recv_ptr + recv_base + (head_dim + tl.arange(0, 4)[:, None]) * recv_stride_d,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            even, odd = tl.split(tl.reshape(tl.trans(packed), (BLOCK_ROWS, 2, 2)))
            code, mid = tl.split(even)
            hi, lo = tl.split(odd)
            exponent_code = code[None, :]
            significand_hi = hi[None, :]
            significand_mid = mid[None, :]
            significand_lo = lo[None, :]
            packed_valid = exponent_code != 0.0
            sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
            exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
            significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
            # Reconstruct IEEE FP32 bits directly, including subnormals.
            magnitude = exponent_magnitude.to(tl.int32)
            lse_bits = (tl.maximum(magnitude - 2, 0) << 23) + significand.to(tl.int32)
            lse_bits = tl.where(significand != 0.0, lse_bits, 0)
            lse = sign * lse_bits.to(tl.float32, bitcast=True)
            valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))
        any_valid_lse = lse_max != -float("inf")
        safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
        weight_sum = tl.zeros((1, BLOCK_ROWS), tl.float32)
        merged = tl.zeros([BLOCK_D, BLOCK_ROWS], dtype=tl.float32)
        d_mask = (d_offsets < head_dim) & row_mask
        for rank_idx in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
            recv_base = (
                rank_idx * recv_stride_rank + head_idx * recv_stride_scatter + token_idx * recv_stride_replicated
            )
            packed = tl.load(
                recv_ptr + recv_base + (head_dim + tl.arange(0, 4)[:, None]) * recv_stride_d,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            even, odd = tl.split(tl.reshape(tl.trans(packed), (BLOCK_ROWS, 2, 2)))
            code, mid = tl.split(even)
            hi, lo = tl.split(odd)
            exponent_code = code[None, :]
            significand_hi = hi[None, :]
            significand_mid = mid[None, :]
            significand_lo = lo[None, :]
            packed_valid = exponent_code != 0.0
            sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
            exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
            significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
            # Reconstruct IEEE FP32 bits directly, including subnormals.
            magnitude = exponent_magnitude.to(tl.int32)
            lse_bits = (tl.maximum(magnitude - 2, 0) << 23) + significand.to(tl.int32)
            lse_bits = tl.where(significand != 0.0, lse_bits, 0)
            lse = sign * lse_bits.to(tl.float32, bitcast=True)
            valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
            partial_output = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_d, mask=d_mask, other=0.0).to(
                tl.float32
            )
            partial_output = tl.where(valid_lse, partial_output, 0.0)
            merged += partial_output * weight
            weight_sum += weight
        if HAS_LOCAL:
            local_weight = tl.where(local_valid, tl.exp(local_lse - safe_lse_max), 0.0)
            local_offsets = (
                token_idx * local_output_stride_t + head_idx * local_output_stride_h + d_offsets * local_output_stride_d
            )
            local_output = tl.load(local_output_ptr + local_offsets, mask=d_mask, other=0.0).to(tl.float32)
            merged += tl.where(local_valid, local_output, 0.0) * local_weight
            weight_sum += local_weight
        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        merged /= denominator
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + output_offsets, merged, mask=d_mask)
