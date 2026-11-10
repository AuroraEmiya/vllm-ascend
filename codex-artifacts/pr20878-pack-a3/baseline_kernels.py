# Exact decorated function definitions extracted from the pinned PR source.
import torch
from vllm.triton_utils import triton, tl
try:
    import triton.language.extra.cann.extension as _ext
except ImportError:
    _ext = None
_sort = getattr(_ext, 'sort', None) if _ext is not None else None
if _sort is None:
    _sort = getattr(tl, 'sort')

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
def _pack_dcp_output_lse_kernel(
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
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)

    for linear_idx in range(program_idx, total_rows, num_programs):
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            rank_idx = token_idx // local_scatter_size
            scatter_idx = token_idx % local_scatter_size
            replicated_idx = head_idx
        else:
            rank_idx = head_idx // local_scatter_size
            scatter_idx = head_idx % local_scatter_size
            replicated_idx = token_idx

        send_base = (
            rank_idx * send_stride_rank + scatter_idx * send_stride_scatter + replicated_idx * send_stride_replicated
        )
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        d_mask = d_offsets < head_dim
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h).to(tl.float32)
        if FROM_STATS:
            total = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h).to(tl.float32)
            lse = lse + tl.log(total)
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                lse.to(send_ptr.dtype.element_ty),
            )
        else:
            exponent_code, significand_hi, significand_mid, significand_lo = _encode_lse_fields(lse)
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                exponent_code.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 1) * send_stride_d,
                significand_hi.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 2) * send_stride_d,
                significand_mid.to(send_ptr.dtype.element_ty),
            )
            tl.store(
                send_ptr + send_base + (head_dim + 3) * send_stride_d,
                significand_lo.to(send_ptr.dtype.element_ty),
            )

def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    if output_dtype in (torch.bfloat16, torch.float16):
        return 4
    if output_dtype == torch.float32:
        return 1
    raise TypeError(f"DCP output/LSE pack supports bfloat16, float16, or float32 output, got {output_dtype}.")

def _launch_pack_dcp_output_lse(partial_output, softmax_lse, softmax_sum, send, scatter_dim, vector_cores, launch):
    """Shared runtime/warmup launch construction; no allocation or collective."""
    num_tokens, num_heads, head_dim = partial_output.shape
    local_scatter_size = send.shape[1]
    sum_input = softmax_lse if softmax_sum is None else softmax_sum
    lse_pack_dim = _lse_pack_dim(partial_output.dtype)
    total_rows = num_tokens * num_heads
    grid_size = total_rows if total_rows < vector_cores else vector_cores
    # Keep enough row tiles to occupy the vector cores. Feature-strided
    # loads can regress with batching, especially for wide heads.
    batched = (
        partial_output.dtype == torch.bfloat16
        and scatter_dim == 1
        and head_dim <= 2048
        and total_rows >= 8 * vector_cores
        and (partial_output.stride(-1) == 1 or (head_dim == 512 and num_tokens >= 128))
    )
    if batched:
        launch(
            _pack_dcp_output_lse_batched_kernel,
            (grid_size,),
            partial_output,
            softmax_lse,
            sum_input,
            send,
            *partial_output.stride(),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            sum_input.stride(0),
            sum_input.stride(1),
            *send.stride(),
            local_scatter_size,
            head_dim,
            num_heads,
            total_rows,
            SCATTER_TOKENS=scatter_dim == 0,
            LSE_PACK_DIM=lse_pack_dim,
            FROM_STATS=softmax_sum is not None,
            BLOCK_D=triton.next_power_of_2(head_dim),
            BLOCK_ROWS=32 if head_dim == 512 and num_tokens >= 128 and partial_output.stride(-1) == 1 else 8,
            OUTPUT_HEAD_MAJOR=head_dim == 512,
        )
        return
    launch(
        _pack_dcp_output_lse_kernel,
        (grid_size,),
        partial_output,
        softmax_lse,
        sum_input,
        send,
        *partial_output.stride(),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        sum_input.stride(0),
        sum_input.stride(1),
        *send.stride(),
        local_scatter_size,
        head_dim,
        num_heads,
        total_rows,
        SCATTER_TOKENS=scatter_dim == 0,
        LSE_PACK_DIM=lse_pack_dim,
        FROM_STATS=softmax_sum is not None,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return
