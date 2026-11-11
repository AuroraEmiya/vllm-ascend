"""DCP iteration 4: loop-carried head-major addresses; original stores and tiles."""
import triton
import triton.language as tl
from baseline_kernels import _encode_lse_fields

@triton.jit
def _pack_dcp_recurrence(
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
        # One division/remainder per program; all tile transitions are affine.
        row_offsets = tl.arange(0, BLOCK_ROWS)[:, None]
        row64 = row_offsets.to(tl.int64)
        head0 = (program_idx // token_tiles).to(tl.int64)
        token_tile = program_idx % token_tiles
        token0 = token_tile * BLOCK_ROWS
        step_head = num_programs // token_tiles
        step_token = num_programs % token_tiles
        out0 = head0 * output_stride_h + token0.to(tl.int64) * output_stride_t
        lse0 = head0 * lse_stride_h + token0.to(tl.int64) * lse_stride_t
        sum0 = head0 * sum_stride_h + token0.to(tl.int64) * sum_stride_t
        send0 = (head0 * num_tokens + token0.to(tl.int64)) * send_stride_replicated
        out_rows = row64 * output_stride_t
        lse_rows = row64 * lse_stride_t
        sum_rows = row64 * sum_stride_t
        send_rows = row64 * send_stride_replicated
    else:
        tiles = tl.cdiv(total_rows, BLOCK_ROWS)
    for tile in range(program_idx, tiles, num_programs):
        if OUTPUT_HEAD_MAJOR:
            row_mask = token0 + row_offsets < num_tokens
            output_offsets = out0 + out_rows + d_offsets * output_stride_d
            lse_offsets = lse0 + lse_rows
            sum_offsets = sum0 + sum_rows
        else:
            linear_idx = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
            row_mask = linear_idx < total_rows
            token_idx = (linear_idx // num_heads).to(tl.int64)
            head_idx = (linear_idx % num_heads).to(tl.int64)

        if OUTPUT_HEAD_MAJOR:
            send_base = send0 + send_rows
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
        if not OUTPUT_HEAD_MAJOR:
            output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
            lse_offsets = token_idx * lse_stride_t + head_idx * lse_stride_h
            sum_offsets = token_idx * sum_stride_t + head_idx * sum_stride_h
        d_mask = (d_offsets < head_dim) & row_mask
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + lse_offsets, mask=row_mask, other=0.0).to(
            tl.float32
        )
        if FROM_STATS:
            total = tl.load(sum_ptr + sum_offsets, mask=row_mask, other=1.0).to(
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
        if OUTPUT_HEAD_MAJOR:
            advanced_tile = token_tile + step_token
            carry = (advanced_tile >= token_tiles).to(tl.int32)
            token_tile = advanced_tile - carry * token_tiles
            token0 = token_tile * BLOCK_ROWS
            head_delta = (step_head + carry).to(tl.int64)
            token_delta = (step_token * BLOCK_ROWS - carry * token_tiles * BLOCK_ROWS).to(tl.int64)
            out0 += head_delta * output_stride_h + token_delta * output_stride_t
            lse0 += head_delta * lse_stride_h + token_delta * lse_stride_t
            sum0 += head_delta * sum_stride_h + token_delta * sum_stride_t
            send0 += (head_delta * num_tokens + token_delta) * send_stride_replicated
