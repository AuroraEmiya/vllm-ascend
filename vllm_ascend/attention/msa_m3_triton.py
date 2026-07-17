# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention on Ascend.

Migrated from reference/vllm_cp/vllm/models/minimax_m3/common/ops/sparse_attn.py.
The Python wrappers adapt vLLM Ascend's KV cache layout to the paged layout
expected by the migrated kernels.
"""

from __future__ import annotations

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

def _as_triton_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if isinstance(kv_cache, (tuple, list)):
        kv_cache = torch.stack((kv_cache[0], kv_cache[1]), dim=1)
    if kv_cache.ndim != 5:
        raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
    if kv_cache.shape[0] == 2:
        return kv_cache.permute(1, 0, 2, 3, 4)
    if kv_cache.shape[1] == 2:
        return kv_cache
    raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Normalize Ascend indexer cache to ``[num_blocks, 128, head_dim]``."""
    if isinstance(index_kv_cache, (tuple, list)):
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(
                f"Unexpected index cache head dim: {tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(f"Unexpected index cache ndim: {index_kv_cache.ndim}")
    return index_kv_cache


def _is_arch_support_pdl() -> bool:
    if current_platform.device_name == "npu":
        return False
    is_supported = getattr(current_platform, "is_arch_support_pdl", None)
    return bool(is_supported()) if callable(is_supported) else False


_SPARSE_ATTN_NUM_STAGES_KWARG: dict | None = None


def _sparse_attn_num_stages_kwarg() -> dict:
    """Triton ``num_stages`` override for the sparse-attn GEMM kernels."""
    global _SPARSE_ATTN_NUM_STAGES_KWARG
    if _SPARSE_ATTN_NUM_STAGES_KWARG is None:
        kwarg: dict = {}
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx942

            if on_gfx942():
                kwarg = {"num_stages": 1}
        _SPARSE_ATTN_NUM_STAGES_KWARG = kwarg
    return _SPARSE_ATTN_NUM_STAGES_KWARG

# Indexer data-layout and launch policy.
SCORE_BLOCK_STRIDE_ALIGNMENT = 16
PREFILL_SCORE_BLOCK_Q = 128
PREFILL_PREPARE_BLOCK_Q_CANDIDATES = (8, 16, 32, 64)
PREFILL_PREPARE_TARGET_PROGRAMS = 128
PREFILL_PREPARE_TAIL_TILE = 16
PREFILL_TOPK_MASK_BLOCK_Q_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
PREFILL_TOPK_MASK_TARGET_PROGRAMS = 128
PREFILL_TOPK_MASK_MAX_TILE_ELEMENTS = 2048
DECODE_TARGET_PROGRAMS = 512
DECODE_MAX_CHUNKS = 256

def _score_block_stride(max_seq_len: int) -> int:
    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    return round_up(max_block_count, SCORE_BLOCK_STRIDE_ALIGNMENT)

def _select_prefill_prepare_block_q(
    max_query_len: int,
    batch_size: int,
    index_head_count: int,
) -> int:
    """Select the query tile used to prepare scores for top-k."""
    total_rows = max(1, max_query_len * batch_size * index_head_count)
    required = triton.cdiv(total_rows, PREFILL_PREPARE_TARGET_PROGRAMS)
    for block_q in PREFILL_PREPARE_BLOCK_Q_CANDIDATES:
        if required <= block_q:
            return block_q
    return PREFILL_PREPARE_BLOCK_Q_CANDIDATES[-1]

def _select_prefill_topk_mask_block_q(
    max_query_len: int,
    batch_size: int,
    index_head_count: int,
    topk: int,
) -> int:
    """Select the query tile used to mask invalid top-k indices."""
    total_rows = max(1, max_query_len * batch_size * index_head_count)
    required = triton.cdiv(total_rows, PREFILL_TOPK_MASK_TARGET_PROGRAMS)

    desired = PREFILL_TOPK_MASK_BLOCK_Q_CANDIDATES[-1]
    for block_q in PREFILL_TOPK_MASK_BLOCK_Q_CANDIDATES:
        if required <= block_q:
            desired = block_q
            break

    topk_tile = triton.next_power_of_2(max(1, topk))
    max_block_q = max(1, PREFILL_TOPK_MASK_MAX_TILE_ELEMENTS // topk_tile)
    bounded = PREFILL_TOPK_MASK_BLOCK_Q_CANDIDATES[0]
    for block_q in PREFILL_TOPK_MASK_BLOCK_Q_CANDIDATES:
        if block_q > max_block_q:
            break
        bounded = block_q
    return min(desired, bounded)

def _select_decode_chunk_count(
    request_count: int,
    decode_query_len: int,
    max_block_count: int,
) -> int:
    """Keep the decode grid near the tuned target program count."""
    total_query_tokens = max(1, request_count * decode_query_len)
    target = max(
        1,
        min(DECODE_MAX_CHUNKS, DECODE_TARGET_PROGRAMS // total_query_tokens),
    )
    chunk_count = 1 << (target.bit_length() - 1)
    while chunk_count > max_block_count and chunk_count > 1:
        chunk_count //= 2
    return chunk_count

@triton.jit(
    do_not_specialize_on_alignment=[
        "sequence_lengths_ptr",
        "prefix_lengths_ptr",
    ]
)
def _prefill_index_score_kernel(
    query_ptr,
    index_key_cache_ptr,
    score_ptr,
    block_table_ptr,
    query_start_offsets_ptr,
    sequence_lengths_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    head_dim: tl.constexpr,
    query_token_stride,
    query_head_stride,
    query_dim_stride,
    key_block_stride,
    key_position_stride,
    key_dim_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute max dot-product score for each causally visible KV block."""
    tl.static_assert(BLOCK_SIZE_Q <= BLOCK_SIZE_K)

    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    sequence_length = tl.load(sequence_lengths_ptr + batch_id)
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lanes = tl.arange(0, BLOCK_SIZE_Q)
    key_lanes = tl.arange(0, BLOCK_SIZE_K)
    dim_lanes = tl.arange(0, head_dim)
    query_offsets = query_tile_start + query_lanes
    query_mask = query_offsets < query_length
    query_positions = prefix_length + query_offsets

    query = tl.load(
        query_ptr
        + (sequence_start + query_offsets[:, None]) * query_token_stride
        + head_id * query_head_stride
        + dim_lanes[None, :] * query_dim_stride,
        mask=query_mask[:, None].broadcast_to((BLOCK_SIZE_Q, head_dim)),
        other=0.0,
    )

    block_table_row = block_table_ptr + batch_id * block_table_batch_stride
    score_rows = (
        score_ptr
        + head_id * score_head_stride
        + (sequence_start + query_offsets) * score_token_stride
    )

    valid_query_end = tl.minimum(query_length, query_tile_start + BLOCK_SIZE_Q)
    visible_key_end = tl.minimum(
        sequence_length,
        prefix_length + valid_query_end,
    )
    earliest_query_position = prefix_length + query_tile_start
    causally_full_blocks = (earliest_query_position + 1) // BLOCK_SIZE_K
    complete_sequence_blocks = sequence_length // BLOCK_SIZE_K
    full_block_count = tl.minimum(
        causally_full_blocks,
        complete_sequence_blocks,
    )

    key_inner_offsets = (
        dim_lanes[:, None] * key_dim_stride
        + key_lanes[None, :] * key_position_stride
    )

    for block_id in tl.range(0, full_block_count):
        page_id = tl.load(block_table_row + block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

    boundary_start = full_block_count * BLOCK_SIZE_K
    for key_start in tl.range(boundary_start, visible_key_end, BLOCK_SIZE_K):
        block_id = key_start // BLOCK_SIZE_K
        page_id = tl.load(block_table_row + block_id).to(tl.int64)
        key_positions = key_start + key_lanes
        key_mask = key_positions < sequence_length
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
            mask=key_mask[None, :].broadcast_to((head_dim, BLOCK_SIZE_K)),
            other=0.0,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        query_key = tl.where(
            query_mask[:, None]
            & key_mask[None, :]
            & (query_positions[:, None] >= key_positions[None, :]),
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    query_ptr,
    index_key_cache_ptr,
    score_ptr,
    block_table_ptr,
    sequence_lengths_ptr,
    index_head_count: tl.constexpr,
    head_dim: tl.constexpr,
    init_block_count,
    local_block_count,
    decode_query_len,
    query_token_stride,
    query_head_stride,
    query_dim_stride,
    key_block_stride,
    key_position_stride,
    key_dim_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_K: tl.constexpr,
    num_kv_chunks,
):
    """Score one decode query per program while fusing index heads."""
    query_id = tl.program_id(0)
    chunk_id = tl.program_id(1)
    request_id = query_id // decode_query_len
    query_offset = query_id - request_id * decode_query_len

    sequence_length = tl.maximum(tl.load(sequence_lengths_ptr + request_id), 0)
    visible_token_count = tl.maximum(
        sequence_length - decode_query_len + query_offset + 1,
        0,
    )
    valid_block_count = (
        visible_token_count + BLOCK_SIZE_K - 1
    ) // BLOCK_SIZE_K
    chunk_block_count = (
        valid_block_count + num_kv_chunks - 1
    ) // num_kv_chunks
    chunk_start_block = chunk_id * chunk_block_count
    chunk_end_block = tl.minimum(
        chunk_start_block + chunk_block_count,
        valid_block_count,
    )
    if chunk_start_block >= chunk_end_block:
        return

    full_block_count = visible_token_count // BLOCK_SIZE_K
    local_block_start = tl.maximum(
        0,
        valid_block_count - local_block_count,
    )

    # Local priority overrides init priority when the ranges overlap.
    init_only_end = tl.minimum(
        tl.minimum(init_block_count, local_block_start),
        valid_block_count,
    )
    init_chunk_end = tl.minimum(chunk_end_block, init_only_end)
    normal_chunk_start = tl.maximum(chunk_start_block, init_block_count)
    normal_chunk_end = tl.minimum(chunk_end_block, local_block_start)
    normal_full_end = tl.minimum(normal_chunk_end, full_block_count)
    local_chunk_start = tl.maximum(chunk_start_block, local_block_start)

    head_lanes = tl.arange(0, index_head_count)
    score_rows = (
        score_ptr
        + head_lanes * score_head_stride
        + query_id * score_token_stride
    )

    for block_id in tl.range(chunk_start_block, init_chunk_end):
        tl.store(score_rows + block_id * score_block_stride, 1e30)

    key_lanes = tl.arange(0, BLOCK_SIZE_K)
    dim_lanes = tl.arange(0, head_dim)
    block_table_row = block_table_ptr + request_id * block_table_batch_stride
    query = tl.load(
        query_ptr
        + query_id * query_token_stride
        + head_lanes[:, None] * query_head_stride
        + dim_lanes[None, :] * query_dim_stride,
    )
    key_inner_offsets = (
        dim_lanes[:, None] * key_dim_stride
        + key_lanes[None, :] * key_position_stride
    )

    for block_id in tl.range(normal_chunk_start, normal_full_end):
        page_id = tl.load(block_table_row + block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + block_id * score_block_stride,
            block_score,
        )

    tail_block_id = full_block_count
    tail_is_normal = (
        (full_block_count < valid_block_count)
        & (normal_chunk_start <= tail_block_id)
        & (tail_block_id < normal_chunk_end)
    )
    if tail_is_normal:
        page_id = tl.load(block_table_row + tail_block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        key_positions = tail_block_id * BLOCK_SIZE_K + key_lanes
        query_key = tl.where(
            key_positions[None, :] < visible_token_count,
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + tail_block_id * score_block_stride,
            block_score,
        )

    for block_id in tl.range(local_chunk_start, chunk_end_block):
        tl.store(score_rows + block_id * score_block_stride, 1e29)

@triton.jit(do_not_specialize=["num_kv_chunks"])
def _decode_index_score_q1_kernel(
    query_ptr,
    index_key_cache_ptr,
    score_ptr,
    block_table_ptr,
    sequence_lengths_ptr,
    index_head_count: tl.constexpr,
    head_dim: tl.constexpr,
    init_block_count,
    local_block_count,
    query_token_stride,
    query_head_stride,
    query_dim_stride,
    key_block_stride,
    key_position_stride,
    key_dim_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_K: tl.constexpr,
    num_kv_chunks,
):
    """Dedicated q=1 fast path with forced ranges removed from dot computation."""
    request_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    sequence_length = tl.maximum(tl.load(sequence_lengths_ptr + request_id), 0)
    valid_block_count = (
        sequence_length + BLOCK_SIZE_K - 1
    ) // BLOCK_SIZE_K
    chunk_block_count = (
        valid_block_count + num_kv_chunks - 1
    ) // num_kv_chunks
    chunk_start_block = chunk_id * chunk_block_count
    chunk_end_block = tl.minimum(
        chunk_start_block + chunk_block_count,
        valid_block_count,
    )
    if chunk_start_block >= chunk_end_block:
        return

    full_block_count = sequence_length // BLOCK_SIZE_K
    local_block_start = tl.maximum(
        0,
        valid_block_count - local_block_count,
    )
    init_only_end = tl.minimum(
        tl.minimum(init_block_count, local_block_start),
        valid_block_count,
    )
    init_chunk_end = tl.minimum(chunk_end_block, init_only_end)
    normal_chunk_start = tl.maximum(chunk_start_block, init_block_count)
    normal_chunk_end = tl.minimum(chunk_end_block, local_block_start)
    normal_full_end = tl.minimum(normal_chunk_end, full_block_count)
    local_chunk_start = tl.maximum(chunk_start_block, local_block_start)

    head_lanes = tl.arange(0, index_head_count)
    score_rows = (
        score_ptr
        + head_lanes * score_head_stride
        + request_id * score_token_stride
    )

    for block_id in tl.range(chunk_start_block, init_chunk_end):
        tl.store(score_rows + block_id * score_block_stride, 1e30)

    key_lanes = tl.arange(0, BLOCK_SIZE_K)
    dim_lanes = tl.arange(0, head_dim)
    block_table_row = block_table_ptr + request_id * block_table_batch_stride
    query = tl.load(
        query_ptr
        + request_id * query_token_stride
        + head_lanes[:, None] * query_head_stride
        + dim_lanes[None, :] * query_dim_stride,
    )
    key_inner_offsets = (
        dim_lanes[:, None] * key_dim_stride
        + key_lanes[None, :] * key_position_stride
    )

    for block_id in tl.range(normal_chunk_start, normal_full_end):
        page_id = tl.load(block_table_row + block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + block_id * score_block_stride,
            block_score,
        )

    tail_block_id = full_block_count
    tail_is_normal = (
        (full_block_count < valid_block_count)
        & (normal_chunk_start <= tail_block_id)
        & (tail_block_id < normal_chunk_end)
    )
    if tail_is_normal:
        page_id = tl.load(block_table_row + tail_block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_inner_offsets,
        )
        query_key = tl.dot(query, key, out_dtype=tl.float32)
        key_positions = tail_block_id * BLOCK_SIZE_K + key_lanes
        query_key = tl.where(
            key_positions[None, :] < sequence_length,
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_rows + tail_block_id * score_block_stride,
            block_score,
        )

    for block_id in tl.range(local_chunk_start, chunk_end_block):
        tl.store(score_rows + block_id * score_block_stride, 1e29)

@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_mask_invalid_topk_indices_kernel(
    ti_ptr,  # [num_idx_heads, total_q, topk] int32 in/out
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = (
        ti_ptr + pid_h * stride_ti_h + pid_b * stride_ti_b + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    idx = tl.load(ti_ptrs, mask=store_mask, other=0)
    valid_slot = off_t < tl.minimum(topk, num_blocks)
    valid_idx = (idx >= 0) & (idx < num_blocks)
    masked_idx = tl.where(valid_slot & valid_idx, idx, -1)
    tl.store(ti_ptrs, masked_idx.to(ti_ptr.dtype.element_ty), mask=store_mask)

@triton.jit(do_not_specialize=["score_block_count"])
def _prefill_prepare_topk_scores_kernel(
    score_ptr,
    query_start_offsets_ptr,
    prefix_lengths_ptr,
    index_head_count,
    init_block_count: tl.constexpr,
    local_block_count: tl.constexpr,
    score_block_count,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    sparse_block_size: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_FORCE: tl.constexpr,
    BLOCK_SIZE_TAIL: tl.constexpr,
):
    """Apply forced block priorities and clear scores outside the valid range."""
    tl.static_assert(BLOCK_SIZE_Q <= sparse_block_size)

    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    token_indices = sequence_start + query_offsets
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)

    valid_block_counts = (
        prefix_length + query_offsets + sparse_block_size
    ) // sparse_block_size
    valid_block_counts = tl.minimum(
        valid_block_counts,
        score_block_count,
    )

    # Because BLOCK_SIZE_Q <= sparse_block_size, one query tile contains at
    # most two consecutive valid-block counts.
    min_valid_block_count = (
        prefix_length + query_tile_start + sparse_block_size
    ) // sparse_block_size
    min_valid_block_count = tl.minimum(
        min_valid_block_count,
        score_block_count,
    )
    max_valid_block_count = tl.minimum(
        min_valid_block_count + 1,
        score_block_count,
    )

    score_row_ptrs = (
        score_ptr
        + head_id * score_head_stride
        + token_indices[:, None] * score_token_stride
    )
    forced_block_offsets = tl.arange(0, BLOCK_SIZE_FORCE)

    if init_block_count > 0:
        init_mask = (
            query_mask[:, None]
            & (forced_block_offsets[None, :] < init_block_count)
            & (
                forced_block_offsets[None, :]
                < valid_block_counts[:, None]
            )
        )
        tl.store(
            score_row_ptrs
            + forced_block_offsets[None, :] * score_block_stride,
            1e30,
            mask=init_mask,
        )

    rows_with_min_count = query_mask & (
        valid_block_counts == min_valid_block_count
    )
    rows_with_max_count = query_mask & (
        valid_block_counts > min_valid_block_count
    )

    if local_block_count > 0:
        min_local_start = tl.maximum(
            0,
            min_valid_block_count - local_block_count,
        )
        min_local_count = min_valid_block_count - min_local_start
        min_local_blocks = min_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs
            + min_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(
                rows_with_min_count[:, None]
                & (forced_block_offsets[None, :] < min_local_count)
            ),
        )

        max_local_start = tl.maximum(
            0,
            max_valid_block_count - local_block_count,
        )
        max_local_count = max_valid_block_count - max_local_start
        max_local_blocks = max_local_start + forced_block_offsets
        tl.store(
            score_row_ptrs
            + max_local_blocks[None, :] * score_block_stride,
            1e29,
            mask=(
                rows_with_max_count[:, None]
                & (forced_block_offsets[None, :] < max_local_count)
            ),
        )

    tail_lane_offsets = tl.arange(0, BLOCK_SIZE_TAIL)
    tail_block_count = score_block_count - min_valid_block_count
    for tail_offset in tl.range(
        0,
        tail_block_count,
        BLOCK_SIZE_TAIL,
    ):
        block_ids = (
            min_valid_block_count + tail_offset + tail_lane_offsets
        )
        block_mask = block_ids < score_block_count
        row_mask = (
            rows_with_min_count[:, None]
            | (
                rows_with_max_count[:, None]
                & (block_ids[None, :] >= max_valid_block_count)
            )
        )
        tl.store(
            score_row_ptrs + block_ids[None, :] * score_block_stride,
            float("-inf"),
            mask=row_mask & block_mask[None, :],
        )

@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])}
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lengths_ptr"])
def _prefill_mask_invalid_topk_indices_kernel(
    topk_indices_ptr,
    query_start_offsets_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    sparse_block_size: tl.constexpr,
    topk: tl.constexpr,
    index_head_stride,
    index_token_stride,
    index_topk_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Replace invalid prefill top-k block IDs with ``-1``."""
    query_tile_id = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    batch_id = batch_head_id // index_head_count
    head_id = batch_head_id % index_head_count

    sequence_start = tl.load(query_start_offsets_ptr + batch_id)
    sequence_end = tl.load(query_start_offsets_ptr + batch_id + 1)
    query_length = sequence_end - sequence_start
    query_tile_start = query_tile_id * BLOCK_SIZE_Q
    if query_tile_start >= query_length:
        return

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    topk_lane_offsets = tl.arange(0, BLOCK_SIZE_T)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    token_indices = sequence_start + query_offsets
    prefix_length = tl.load(prefix_lengths_ptr + batch_id)
    valid_block_counts = (
        prefix_length + query_offsets + sparse_block_size
    ) // sparse_block_size

    index_ptrs = (
        topk_indices_ptr
        + head_id * index_head_stride
        + token_indices[:, None] * index_token_stride
        + topk_lane_offsets[None, :] * index_topk_stride
    )
    access_mask = (
        query_mask[:, None]
        & (topk_lane_offsets[None, :] < topk)
    )
    block_ids = tl.load(index_ptrs, mask=access_mask, other=0)

    valid_rank_mask = topk_lane_offsets[None, :] < tl.minimum(
        topk,
        valid_block_counts[:, None],
    )
    valid_block_mask = (
        (block_ids >= 0)
        & (block_ids < valid_block_counts[:, None])
    )
    output_block_ids = tl.where(
        valid_rank_mask & valid_block_mask,
        block_ids,
        -1,
    )
    tl.store(
        index_ptrs,
        output_block_ids.to(topk_indices_ptr.dtype.element_ty),
        mask=access_mask,
    )

def _copy_topk_indices(
    raw_indices: torch.Tensor,
    requested_topk: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Copies top-k indices into an int32 result and pads missing slots."""
    head_count, total_query_tokens, selected_count = raw_indices.shape
    if output is None and selected_count == requested_topk:
        return raw_indices.to(torch.int32)

    if output is None:
        result = torch.empty(
            (head_count, total_query_tokens, requested_topk),
            dtype=torch.int32,
            device=raw_indices.device,
        )
    else:
        result = output[:, :total_query_tokens, :requested_topk]

    if selected_count < requested_topk:
        result.fill_(-1)
    result[..., :selected_count].copy_(raw_indices)
    return result

@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    sm_scale = None,
) -> torch.Tensor:
    """Compute one prefill score for every causally visible KV block."""
    del sm_scale
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, (
        "M3 requires num_idx_heads == num_kv_heads"
    )
    assert index_head_count > 0 and not (
        index_head_count & (index_head_count - 1)
    ), "index head count must be a power of two"
    assert head_dim > 0 and not (
        head_dim & (head_dim - 1)
    ), "index head dimension must be a power of two"

    batch_size = cu_seqlens_q.shape[0] - 1
    score = torch.empty(
        (
            index_head_count,
            total_query_tokens,
            _score_block_stride(max_seq_len),
        ),
        dtype=torch.float32,
        device=idx_q.device,
    )
    score_grid = (
        triton.cdiv(max_query_len, PREFILL_SCORE_BLOCK_Q),
        batch_size * index_head_count,
    )
    _prefill_index_score_kernel[score_grid](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        index_head_count,
        head_dim,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=PREFILL_SCORE_BLOCK_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )
    return score

@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply prefill block-selection policy and return zero-based block IDs."""
    assert topk > 0
    index_head_count, total_query_tokens, score_block_count = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1

    force_tile_size = triton.next_power_of_2(
        max(1, init_blocks, local_blocks)
    )
    prepare_query_tile_size = _select_prefill_prepare_block_q(
        max_query_len,
        batch_size,
        index_head_count,
    )
    prepare_grid = (
        triton.cdiv(max_query_len, prepare_query_tile_size),
        batch_size * index_head_count,
    )
    _prefill_prepare_topk_scores_kernel[prepare_grid](
        score,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        init_blocks,
        local_blocks,
        score_block_count,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        sparse_block_size=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=prepare_query_tile_size,
        BLOCK_SIZE_FORCE=force_tile_size,
        BLOCK_SIZE_TAIL=PREFILL_PREPARE_TAIL_TILE,
    )

    selected_count = min(topk, score_block_count)
    score_rows = score[:, :total_query_tokens, :score_block_count]
    raw_indices = torch.topk(
        score_rows,
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    topk_mask_query_tile_size = _select_prefill_topk_mask_block_q(
        max_query_len,
        batch_size,
        index_head_count,
        topk,
    )
    topk_mask_grid = (
        triton.cdiv(max_query_len, topk_mask_query_tile_size),
        batch_size * index_head_count,
    )
    _prefill_mask_invalid_topk_indices_kernel[topk_mask_grid](
        topk_indices,
        cu_seqlens_q,
        prefix_lens,
        index_head_count,
        SPARSE_BLOCK_SIZE,
        topk,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
        BLOCK_SIZE_Q=topk_mask_query_tile_size,
    )
    return topk_indices

@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int = None,
    out: torch.Tensor | None = None,
    sm_scale = None,
) -> torch.Tensor:
    """Compute decode scores and return zero-based top-k block IDs."""
    del sm_scale
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    assert topk > 0
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    assert index_head_count == num_kv_heads, (
        "M3 requires num_idx_heads == num_kv_heads"
    )
    assert index_head_count > 0 and not (
        index_head_count & (index_head_count - 1)
    ), "index head count must be a power of two"
    assert head_dim > 0 and not (
        head_dim & (head_dim - 1)
    ), "index head dimension must be a power of two"

    if max_decode_query_len is None:
        max_decode_query_len = decode_query_len
    assert decode_query_len <= max_decode_query_len

    request_count = seq_lens.shape[0]
    assert total_query_tokens == request_count * decode_query_len

    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score = torch.full(
        (
            index_head_count,
            total_query_tokens,
            _score_block_stride(max_seq_len),
        ),
        float("-inf"),
        dtype=torch.float32,
        device=idx_q.device,
    )
    chunk_count = _select_decode_chunk_count(
        request_count,
        decode_query_len,
        max_block_count,
    )

    if decode_query_len == 1:
        _decode_index_score_q1_kernel[(request_count, chunk_count)](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            index_head_count,
            head_dim,
            init_blocks,
            local_blocks,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            num_kv_chunks=chunk_count,
        )
    else:
        _decode_index_score_kernel[(total_query_tokens, chunk_count)](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            seq_lens,
            index_head_count,
            head_dim,
            init_blocks,
            local_blocks,
            decode_query_len,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            num_kv_chunks=chunk_count,
        )

    selected_count = min(topk, max_block_count)
    raw_indices = torch.topk(
        score[:, :total_query_tokens, :max_block_count],
        k=selected_count,
        dim=-1,
    ).indices
    topk_indices = _copy_topk_indices(raw_indices, topk, out)

    _decode_mask_invalid_topk_indices_kernel[
        (total_query_tokens, index_head_count)
    ](
        topk_indices,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_indices.stride(0),
        topk_indices.stride(1),
        topk_indices.stride(2),
    )
    return topk_indices

# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + 0 * stride_kv_kv
                + off_n[None, :] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + 1 * stride_kv_kv
                + off_n[:, None] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[None, :] * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # number of valid (non-padded) selected blocks for this query token
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 0 * stride_kv_kv
            + off_n[None, :] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 1 * stride_kv_kv
            + off_n[:, None] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[None, :] * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    kv_cache = _as_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_FP8=use_fp8,
        **_sparse_attn_num_stages_kwarg(),
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    kv_cache = _as_triton_main_kv_cache(kv_cache)
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    use_pdl = _is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        USE_PDL=use_pdl,
        **_sparse_attn_num_stages_kwarg(),
        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
