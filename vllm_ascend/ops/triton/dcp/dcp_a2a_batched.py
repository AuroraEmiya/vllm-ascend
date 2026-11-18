# SPDX-License-Identifier: Apache-2.0
"""Batched DCP packing and row-weighted combine kernels."""

from vllm.triton_utils import tl, triton


@triton.jit
def _encode_lse_fields(lse):
    """Encode the existing four-field wire format without transcendental ops."""
    bits = lse.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    finite = exponent.to(tl.float32) < 255.0
    # Wire exponent bias is 128. Subnormals share code 2 with the smallest
    # normal, but omit its implicit significand bit. Zero keeps code 128.
    significand = fraction | tl.where(exponent == 0, 0, 0x800000)
    significand = tl.where(finite, significand, 0)
    code = tl.maximum(exponent.to(tl.float32), 1.0) + 1.0
    code = tl.where((bits >> 31) == 0, code, -code)
    code = tl.where((bits & 0x7FFFFFFF) == 0, 128.0, code)
    code = tl.where(finite, code, 0.0)
    return (
        code,
        ((significand >> 16) & 255).to(tl.float32),
        ((significand >> 8) & 255).to(tl.float32),
        (significand & 255).to(tl.float32),
    )


@triton.jit
def _pack_dcp_output_lse_batched_kernel(
    output_ptr,
    lse_ptr,
    sum_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
    sum_stride_t,
    sum_stride_h,
    send_stride_rank,
    send_stride_scatter,
    send_stride_replicated,
    send_stride_d,
    local_scatter_size,
    head_dim,
    num_heads,
    total_rows,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    FROM_STATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    OUTPUT_HEAD_MAJOR: tl.constexpr = False,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    if OUTPUT_HEAD_MAJOR:
        num_tokens = total_rows // num_heads
        token_tiles = tl.cdiv(num_tokens, BLOCK_ROWS)
        tiles = token_tiles * num_heads
    else:
        tiles = tl.cdiv(total_rows, BLOCK_ROWS)
    for tile in range(program_idx, tiles, num_programs):
        if OUTPUT_HEAD_MAJOR:
            # Keep each tile inside one head so both copies have affine strides.
            head_idx = (tile // token_tiles).to(tl.int64)
            token_idx = (tile % token_tiles) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
            row_mask = token_idx < num_tokens
            token_idx = token_idx.to(tl.int64)
        else:
            linear_idx = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
            row_mask = linear_idx < total_rows
            token_idx = (linear_idx // num_heads).to(tl.int64)
            head_idx = (linear_idx % num_heads).to(tl.int64)

        if OUTPUT_HEAD_MAJOR:
            send_base = (head_idx * num_tokens + token_idx) * send_stride_replicated
        else:
            if SCATTER_TOKENS:
                rank_idx = token_idx // local_scatter_size
                scatter_idx = token_idx % local_scatter_size
                replicated_idx = head_idx
            else:
                rank_idx = head_idx // local_scatter_size
                scatter_idx = head_idx % local_scatter_size
                replicated_idx = token_idx
            send_base = (
                rank_idx * send_stride_rank
                + scatter_idx * send_stride_scatter
                + replicated_idx * send_stride_replicated
            )
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        d_mask = (d_offsets < head_dim) & row_mask
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h, mask=row_mask, other=0.0).to(
            tl.float32
        )
        if FROM_STATS:
            total = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h, mask=row_mask, other=1.0).to(
                tl.float32
            )
            lse = lse + tl.log(total)
        if LSE_PACK_DIM == 1:
            tl.store(send_ptr + send_base + head_dim * send_stride_d, lse.to(send_ptr.dtype.element_ty), mask=row_mask)
        else:
            exponent_code, significand_hi, significand_mid, significand_lo = _encode_lse_fields(lse)
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                exponent_code.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 1) * send_stride_d,
                significand_hi.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 2) * send_stride_d,
                significand_mid.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 3) * send_stride_d,
                significand_lo.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )


@triton.jit
def _fused_dcp_lse_combine_head_scatter_kernel(
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
    HAS_LOCAL: tl.constexpr = True,
):
    # One rank contributes one vector of row LSEs and one tile of features.
    BLOCK_ROWS: tl.constexpr = 8
    fields = tl.arange(0, 4)
    features = tl.arange(0, BLOCK_D)
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    tile_count = tl.cdiv(total_rows, BLOCK_ROWS)
    tiles_per_program = tile_count // programs
    extra_tiles = tile_count % programs
    first_tile = program * tiles_per_program + tl.minimum(program, extra_tiles)
    stop_tile = first_tile + tiles_per_program + (program < extra_tiles).to(tl.int32)
    for tile in range(first_tile, stop_tile):
        start = tile * BLOCK_ROWS
        rows = start + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < total_rows
        tokens = (rows // num_heads).to(tl.int64)
        heads = (rows % num_heads).to(tl.int64)
        bases = heads * recv_stride_scatter + tokens * recv_stride_replicated
        maximum = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
        if HAS_LOCAL:
            local_lse = tl.load(
                local_lse_ptr + tokens * local_lse_stride_t + heads * local_lse_stride_h, row_mask, 0.0
            ).to(tl.float32)
            local_valid = (local_lse == local_lse) & (local_lse != float("inf")) & (local_lse != -float("inf"))
            maximum = tl.where(local_valid, local_lse, -float("inf"))

        BLOCK_RANKS: tl.constexpr = triton.next_power_of_2(DCP_SIZE)
        rank_lanes = tl.arange(0, BLOCK_RANKS)[:, None]
        decoded = tl.full((BLOCK_RANKS, BLOCK_ROWS), -float("inf"), tl.float32)
        for rank in tl.range(DCP_SIZE, loop_unroll_factor=1):
            offsets = (rank * recv_stride_rank + bases)[:, None] + (head_dim + fields[None, :]) * recv_stride_d
            packed = tl.load(recv_ptr + offsets, row_mask[:, None], 0.0).to(tl.float32)
            even, odd = tl.split(tl.reshape(packed, (BLOCK_ROWS, 2, 2)))
            code, mid = tl.split(even)
            hi, lo = tl.split(odd)
            significand = hi * 65536.0 + mid * 256.0 + lo
            bits = (tl.maximum(tl.abs(code).to(tl.int32) - 2, 0) << 23) + significand.to(tl.int32)
            magnitude = tl.where(significand != 0.0, bits, 0).to(tl.float32, bitcast=True)
            lse = tl.where(code < 0.0, -magnitude, magnitude)
            valid = (code != 0.0) & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            lse = tl.where(valid, lse, -float("inf"))
            decoded = tl.where(rank_lanes == rank, lse[None, :], decoded)
            maximum = tl.maximum(maximum, lse)
        maximum = tl.where(maximum != -float("inf"), maximum, 0.0)
        mask = row_mask[:, None] & (features < head_dim)[None, :]
        numerator = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)
        denominator = tl.zeros((BLOCK_ROWS,), tl.float32)
        for rank in tl.range(DCP_SIZE, loop_unroll_factor=1):
            rank_index = tl.full((1, BLOCK_ROWS), rank, tl.int32)
            lse = tl.reshape(tl.gather(decoded, rank_index, axis=0), (BLOCK_ROWS,))
            weight = tl.exp(lse - maximum)
            offsets = (rank * recv_stride_rank + bases)[:, None] + features[None, :] * recv_stride_d
            values = tl.load(recv_ptr + offsets, mask, 0.0).to(tl.float32)
            numerator += tl.where((lse != -float("inf"))[:, None], values, 0.0) * weight[:, None]
            denominator += weight
        if HAS_LOCAL:
            weight = tl.where(local_valid, tl.exp(local_lse - maximum), 0.0)
            local_bases = tokens * local_output_stride_t + heads * local_output_stride_h
            local_values = tl.load(
                local_output_ptr + local_bases[:, None] + features[None, :] * local_output_stride_d, mask, 0.0
            ).to(tl.float32)
            numerator += tl.where(local_valid[:, None], local_values, 0.0) * weight[:, None]
            denominator += weight
        result = numerator / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
        output_bases = tokens * output_stride_t + heads * output_stride_h
        tl.store(output_ptr + output_bases[:, None] + features[None, :] * output_stride_d, result, mask)
