"""DCP iteration 7: rectangular token/head tiles with static tensor metadata."""
import triton
import triton.language as tl
from baseline_kernels import _encode_lse_fields

@triton.jit
def _pack_dcp_rectangular(
    output_ptr,
    lse_ptr,
    sum_ptr,
    send_ptr,
    output_stride_t: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_d: tl.constexpr,
    lse_stride_t: tl.constexpr,
    lse_stride_h: tl.constexpr,
    sum_stride_t: tl.constexpr,
    sum_stride_h: tl.constexpr,
    send_stride_rank: tl.constexpr,
    send_stride_scatter: tl.constexpr,
    send_stride_replicated: tl.constexpr,
    send_stride_d: tl.constexpr,
    local_scatter_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    total_rows: tl.constexpr,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    FROM_STATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    OUTPUT_HEAD_MAJOR: tl.constexpr = False,
):
    TILE_T: tl.constexpr = 4
    TILE_H: tl.constexpr = 8
    T: tl.constexpr = total_rows // num_heads
    token_tiles: tl.constexpr = triton.cdiv(T, TILE_T)
    head_tiles: tl.constexpr = triton.cdiv(num_heads, TILE_H)
    rr = tl.arange(0, TILE_T)
    hh = tl.arange(0, TILE_H)
    dd = tl.arange(0, BLOCK_D)
    for tile in range(tl.program_id(0), token_tiles * head_tiles, tl.num_programs(0)):
        token = (tile % token_tiles * TILE_T + rr).to(tl.int64)
        head = (tile // token_tiles * TILE_H + hh).to(tl.int64)
        row_mask = (token[:, None] < T) & (head[None, :] < num_heads)
        offsets = (token[:, None, None] * output_stride_t
                   + head[None, :, None] * output_stride_h
                   + dd[None, None, :] * output_stride_d)
        mask = row_mask[:, :, None] & (dd[None, None, :] < head_dim)
        values = tl.load(output_ptr + offsets, mask, 0.0)
        send_base = (head[None, :] * T + token[:, None]) * send_stride_replicated
        tl.store(send_ptr + send_base[:, :, None] + dd[None, None, :] * send_stride_d, values, mask)
        lse = tl.load(lse_ptr + token[:, None] * lse_stride_t + head[None, :] * lse_stride_h,
                      row_mask, 0.0).to(tl.float32)
        if FROM_STATS:
            total = tl.load(sum_ptr + token[:, None] * sum_stride_t + head[None, :] * sum_stride_h,
                            row_mask, 1.0).to(tl.float32)
            lse = lse + tl.log(total)
        if LSE_PACK_DIM == 1:
            tl.store(send_ptr + send_base + head_dim * send_stride_d,
                     lse.to(send_ptr.dtype.element_ty), row_mask)
        else:
            code, hi, mid, lo = _encode_lse_fields(lse)
            tl.store(send_ptr + send_base + head_dim * send_stride_d,
                     code.to(send_ptr.dtype.element_ty), row_mask)
            tl.store(send_ptr + send_base + (head_dim + 1) * send_stride_d,
                     hi.to(send_ptr.dtype.element_ty), row_mask)
            tl.store(send_ptr + send_base + (head_dim + 2) * send_stride_d,
                     mid.to(send_ptr.dtype.element_ty), row_mask)
            tl.store(send_ptr + send_base + (head_dim + 3) * send_stride_d,
                     lo.to(send_ptr.dtype.element_ty), row_mask)
