# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone MiniMax-M3 prefill/decode index-score baseline for Ascend NPU.

This file is extracted from the supplied ``Pasted text.txt`` baseline.  It
contains only the index-score computation path and its minimal Python wrappers:

* prefill score:
  ``_prefill_index_score_kernel`` plus the ``head_dim == 1`` specialization;
* decode score:
  mask preparation, autotuned split-K score, and invalid-tail filling.

Top-k selection, sparse attention, vLLM custom-op registration, platform
abstractions, and non-Ascend backend branches are intentionally excluded.

The extraction preserves the baseline score semantics and launch policy.  It
is not an A3 optimization and makes no claim that A5 tuning choices are optimal
on A3.
"""

from __future__ import annotations

from typing import MutableMapping

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128
SCORE_BLOCK_STRIDE_ALIGNMENT = 16
PREFILL_SCORE_QUERY_TILE_SIZE = 96
PREFILL_SCALAR_SCORE_BLOCK_TILE_SIZE = 32


def _round_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def _as_triton_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> torch.Tensor:
    """Normalize the index-key cache to ``[num_pages, 128, head_dim]``."""
    if isinstance(index_kv_cache, (tuple, list)):
        if not index_kv_cache:
            raise ValueError("index_kv_cache tuple/list must not be empty")
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 5 and index_kv_cache.shape[0] == 2:
        index_kv_cache = index_kv_cache[0]
    if index_kv_cache.ndim == 4:
        if index_kv_cache.shape[2] != 1:
            raise ValueError(
                "Unexpected index cache head axis: "
                f"shape={tuple(index_kv_cache.shape)}"
            )
        index_kv_cache = index_kv_cache.squeeze(2)
    if index_kv_cache.ndim != 3:
        raise ValueError(
            f"Expected index cache rank 3, got rank {index_kv_cache.ndim}"
        )
    if index_kv_cache.shape[1] != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"Expected page size {SPARSE_BLOCK_SIZE}, got "
            f"shape={tuple(index_kv_cache.shape)}"
        )
    return index_kv_cache


def _score_shape(
    *,
    index_head_count: int,
    total_query_tokens: int,
    max_seq_len: int,
) -> tuple[int, int, int, int]:
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be nonnegative, got {max_seq_len}")
    max_block_count = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = _round_up(
        max(1, max_block_count),
        SCORE_BLOCK_STRIDE_ALIGNMENT,
    )
    return (
        index_head_count,
        total_query_tokens,
        score_block_stride,
        max_block_count,
    )


def _prepare_output(
    out: torch.Tensor | None,
    *,
    shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, dtype=torch.float32, device=device)
    if out.dtype != torch.float32:
        raise TypeError(f"out must be float32, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if tuple(out.shape) != shape:
        raise ValueError(f"out shape must be {shape}, got {tuple(out.shape)}")
    return out


def make_prefill_workspace(
    index_kv_cache: torch.Tensor,
    *,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Allocate reusable host-visible workspace for ``prefill_score``."""
    if head_dim != 1:
        return {}
    cache = _as_triton_index_kv_cache(index_kv_cache)
    return {
        "page_extrema": torch.empty(
            (cache.shape[0], 2),
            dtype=torch.float32,
            device=cache.device,
        )
    }


def make_decode_workspace(
    *,
    total_query_tokens: int,
    score_block_stride: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Allocate reusable mask buffers for ``decode_score``."""
    init_mask = torch.empty(
        (total_query_tokens, score_block_stride),
        dtype=torch.bool,
        device=device,
    )
    local_mask = torch.empty_like(init_mask)
    return {"init_mask": init_mask, "local_mask": local_mask}


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
    """Compute one max QK score per visible sparse block."""
    tl.static_assert(
        BLOCK_SIZE_Q <= BLOCK_SIZE_K,
        "BLOCK_SIZE_Q must not exceed BLOCK_SIZE_K",
    )

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

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    dim_offsets = tl.arange(0, head_dim)

    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    query_positions = prefix_length + query_offsets

    query = tl.load(
        query_ptr
        + (sequence_start + query_offsets[:, None]) * query_token_stride
        + head_id * query_head_stride
        + dim_offsets[None, :] * query_dim_stride,
        mask=query_mask[:, None].broadcast_to((BLOCK_SIZE_Q, head_dim)),
        other=0.0,
    )

    block_table_row_ptr = (
        block_table_ptr + batch_id * block_table_batch_stride
    )
    score_row_ptrs = (
        score_ptr
        + head_id * score_head_stride
        + (sequence_start + query_offsets) * score_token_stride
    )

    query_tile_valid_end = tl.minimum(
        query_length,
        query_tile_start + BLOCK_SIZE_Q,
    )
    visible_key_end = tl.minimum(
        sequence_length,
        prefix_length + query_tile_valid_end,
    )

    earliest_query_position = prefix_length + query_tile_start
    causally_full_block_count = (
        earliest_query_position + 1
    ) // BLOCK_SIZE_K
    complete_sequence_block_count = sequence_length // BLOCK_SIZE_K
    full_block_count = tl.minimum(
        causally_full_block_count,
        complete_sequence_block_count,
    )

    key_position_offsets = (
        key_lane_offsets[None, :] * key_position_stride
    )
    key_dim_offsets = dim_offsets[:, None] * key_dim_stride

    for block_id in tl.range(0, full_block_count):
        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_position_offsets
            + key_dim_offsets,
        )
        query_key = tl.dot(query, key)
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_row_ptrs + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )

    boundary_key_start = full_block_count * BLOCK_SIZE_K
    for key_block_start in tl.range(
        boundary_key_start,
        visible_key_end,
        BLOCK_SIZE_K,
    ):
        block_id = key_block_start // BLOCK_SIZE_K
        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)

        key_positions = key_block_start + key_lane_offsets
        key_mask = key_positions < sequence_length
        key = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_position_offsets
            + key_dim_offsets,
            mask=key_mask[None, :].broadcast_to(
                (head_dim, BLOCK_SIZE_K)
            ),
            other=0.0,
        )

        query_key = tl.dot(query, key)
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        query_key = tl.where(
            causal_mask & key_mask[None, :],
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_row_ptrs + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )


@triton.jit
def _prefill_scalar_key_extrema_kernel(
    index_key_cache_ptr,
    page_extrema_ptr,
    page_count,
    key_block_stride,
    key_position_stride,
    extrema_page_stride,
    extrema_value_stride,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute one minimum and maximum for each scalar index-key page."""
    page_id = tl.program_id(0)
    if page_id >= page_count:
        return

    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    key_values = tl.load(
        index_key_cache_ptr
        + page_id * key_block_stride
        + key_lane_offsets * key_position_stride,
    ).to(tl.float32)

    page_minimum = tl.min(key_values, axis=0)
    page_maximum = tl.max(key_values, axis=0)
    page_extrema_base = page_extrema_ptr + page_id * extrema_page_stride
    tl.store(page_extrema_base, page_minimum)
    tl.store(
        page_extrema_base + extrema_value_stride,
        page_maximum,
    )


@triton.jit(
    do_not_specialize_on_alignment=[
        "sequence_lengths_ptr",
        "prefix_lengths_ptr",
    ]
)
def _prefill_scalar_index_score_kernel(
    query_ptr,
    index_key_cache_ptr,
    page_extrema_ptr,
    score_ptr,
    block_table_ptr,
    query_start_offsets_ptr,
    sequence_lengths_ptr,
    prefix_lengths_ptr,
    index_head_count: tl.constexpr,
    query_token_stride,
    query_head_stride,
    key_block_stride,
    key_position_stride,
    extrema_page_stride,
    extrema_value_stride,
    score_head_stride,
    score_token_stride,
    score_block_stride,
    block_table_batch_stride,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """Compute scalar prefill scores using reusable page extrema."""
    tl.static_assert(
        BLOCK_SIZE_Q <= BLOCK_SIZE_K,
        "BLOCK_SIZE_Q must not exceed BLOCK_SIZE_K",
    )

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

    query_lane_offsets = tl.arange(0, BLOCK_SIZE_Q)
    query_offsets = query_tile_start + query_lane_offsets
    query_mask = query_offsets < query_length
    query_positions = prefix_length + query_offsets
    query_values = tl.load(
        query_ptr
        + (sequence_start + query_offsets) * query_token_stride
        + head_id * query_head_stride,
        mask=query_mask,
        other=0.0,
    ).to(tl.float32)

    block_table_row_ptr = (
        block_table_ptr + batch_id * block_table_batch_stride
    )
    score_row_ptrs = (
        score_ptr
        + head_id * score_head_stride
        + (sequence_start + query_offsets) * score_token_stride
    )

    query_tile_valid_end = tl.minimum(
        query_length,
        query_tile_start + BLOCK_SIZE_Q,
    )
    visible_key_end = tl.minimum(
        sequence_length,
        prefix_length + query_tile_valid_end,
    )

    earliest_query_position = prefix_length + query_tile_start
    causally_full_block_count = (
        earliest_query_position + 1
    ) // BLOCK_SIZE_K
    complete_sequence_block_count = sequence_length // BLOCK_SIZE_K
    full_block_count = tl.minimum(
        causally_full_block_count,
        complete_sequence_block_count,
    )

    block_lane_offsets = tl.arange(0, BLOCK_SIZE_B)
    for block_tile_start in tl.range(
        0,
        full_block_count,
        BLOCK_SIZE_B,
    ):
        block_ids = block_tile_start + block_lane_offsets
        block_mask = block_ids < full_block_count
        page_ids = tl.load(
            block_table_row_ptr + block_ids,
            mask=block_mask,
            other=0,
        ).to(tl.int64)

        extrema_base_ptrs = (
            page_extrema_ptr + page_ids * extrema_page_stride
        )
        page_minimums = tl.load(
            extrema_base_ptrs,
            mask=block_mask,
            other=0.0,
        )
        page_maximums = tl.load(
            extrema_base_ptrs + extrema_value_stride,
            mask=block_mask,
            other=0.0,
        )

        selected_extrema = tl.where(
            query_values[:, None] >= 0.0,
            page_maximums[None, :],
            page_minimums[None, :],
        )
        block_scores = query_values[:, None] * selected_extrema
        tl.store(
            score_row_ptrs[:, None]
            + block_ids[None, :] * score_block_stride,
            block_scores,
            mask=query_mask[:, None] & block_mask[None, :],
        )

    key_lane_offsets = tl.arange(0, BLOCK_SIZE_K)
    key_position_offsets = key_lane_offsets * key_position_stride
    boundary_key_start = full_block_count * BLOCK_SIZE_K
    for key_block_start in tl.range(
        boundary_key_start,
        visible_key_end,
        BLOCK_SIZE_K,
    ):
        block_id = key_block_start // BLOCK_SIZE_K
        page_id = tl.load(block_table_row_ptr + block_id).to(tl.int64)

        key_positions = key_block_start + key_lane_offsets
        key_mask = key_positions < sequence_length
        key_values = tl.load(
            index_key_cache_ptr
            + page_id * key_block_stride
            + key_position_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)

        query_key = query_values[:, None] * key_values[None, :]
        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (query_positions[:, None] >= key_positions[None, :])
        )
        query_key = tl.where(
            causal_mask,
            query_key,
            float("-inf"),
        )
        block_score = tl.max(query_key, axis=1)
        tl.store(
            score_row_ptrs + block_id * score_block_stride,
            block_score,
            mask=query_mask,
        )


def _prune_decode_score_configs(configs, named_args, **_):
    """Keep decode split-K launches within the baseline 512-program budget."""
    request_count = max(1, named_args["num_reqs"])
    chunk_limit = max(1, 512 // request_count)
    chunk_limit = 1 << (chunk_limit.bit_length() - 1)
    valid_configs = [
        config
        for config in configs
        if config.kwargs["num_kv_chunks"] <= chunk_limit
    ]
    return valid_configs or configs[:1]


@triton.autotune(
    configs=[
        triton.Config(
            {"num_kv_chunks": chunk_count},
            num_stages=stage_count,
        )
        for chunk_count in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        for stage_count in (1, 2)
    ],
    key=["num_idx_heads", "BLOCK_SIZE_Q", "head_dim", "num_reqs"],
    prune_configs_by={"early_config_prune": _prune_decode_score_configs},
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    init_mask_ptr,
    local_mask_ptr,
    block_table_ptr,
    seq_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_reqs: tl.constexpr,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_mask_q,
    stride_mask_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
):
    """Baseline split-K decode score kernel."""
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = q_offsets < decode_query_len
    q_ids = pid_r * decode_query_len + q_offsets

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    kv_len = tl.maximum(query_pos + 1, 0)
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (
        kv_len_max + BLOCK_SIZE_K - 1
    ) // BLOCK_SIZE_K

    chunk_size_blocks = (
        num_blocks + num_kv_chunks - 1
    ) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(
        chunk_start_block + chunk_size_blocks,
        num_blocks,
    )
    if chunk_start_block >= chunk_end_block:
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b

    q = tl.load(
        q_ptr
        + q_ids[:, None] * stride_q_n
        + h_offsets[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None],
        other=0.0,
    )

    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[None, :] < kv_len[:, None]
        key = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        query_key = tl.dot(q, key, out_dtype=tl.float32)
        query_key = tl.where(
            pos_mask & q_mask[:, None],
            query_key,
            float("-inf"),
        )
        score = tl.max(query_key, axis=1)

        mask_offset = (
            q_ids * stride_mask_q + blk * stride_mask_k
        )
        is_init = tl.load(init_mask_ptr + mask_offset) != 0
        is_local = tl.load(local_mask_ptr + mask_offset) != 0
        score = tl.where(
            is_local,
            1e29,
            tl.where(is_init, 1e30, score),
        )
        tl.store(
            score_ptr
            + h_offsets * stride_s_h
            + q_ids * stride_s_n
            + blk * stride_s_k,
            score,
            mask=q_mask,
        )


@triton.jit(
    do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"]
)
def _fill_decode_score_tail_kernel(
    score_ptr,
    seq_lens,
    block_size: tl.constexpr,
    max_block,
    decode_query_len,
    chunk_blocks,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fill the logical score tail ``[row_num_blocks, max_block)`` with -inf."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (
        kv_len + block_size - 1
    ) // block_size

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    fill_start = tl.maximum(chunk_start, num_blocks)
    if fill_start >= chunk_end:
        return

    num_to_fill = chunk_end - fill_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for offset in tl.range(0, num_to_fill, BLOCK_SIZE_K):
        block_id = fill_start + offset + off_k
        store_mask = (offset + off_k) < num_to_fill
        score_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + pid_b * stride_s_b
            + block_id * stride_s_k
        )
        tl.store(score_ptrs, float("-inf"), mask=store_mask)


@triton.jit(
    do_not_specialize=["decode_query_len", "max_block", "chunk_blocks"]
)
def _prepare_decode_score_masks_kernel(
    init_mask_ptr,
    local_mask_ptr,
    seq_lens,
    block_size: tl.constexpr,
    max_block,
    decode_query_len,
    chunk_blocks,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_mask_q,
    stride_mask_k,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Prepare baseline decode init/local priority masks."""
    pid_q = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    req_id = pid_q // decode_query_len
    q_offset = pid_q - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id).to(tl.float32)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1.0, 0.0)
    valid_blocks = tl.floor(
        (query_pos + block_size * 1.0) / (block_size * 1.0)
    )
    local_start = tl.maximum(
        tl.floor(
            (kv_len + (block_size - 1) * 1.0)
            / (block_size * 1.0)
        )
        - local_blocks * 1.0,
        0.0,
    )

    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, max_block)
    if chunk_start >= chunk_end:
        return

    num_blocks = chunk_end - chunk_start
    off_k = tl.arange(0, BLOCK_SIZE_K)
    for offset in tl.range(0, num_blocks, BLOCK_SIZE_K):
        block_id = chunk_start + offset + off_k
        store_mask = (offset + off_k) < num_blocks
        block_float = block_id * 1.0
        block_valid = block_float < valid_blocks
        is_init = (
            (block_float < init_blocks * 1.0) & block_valid
        )
        is_local = (
            (block_float >= local_start) & block_valid
        )
        mask_ptrs = (
            init_mask_ptr
            + pid_q * stride_mask_q
            + block_id * stride_mask_k
        )
        tl.store(mask_ptrs, is_init, mask=store_mask)
        tl.store(
            local_mask_ptr
            + pid_q * stride_mask_q
            + block_id * stride_mask_k,
            is_local,
            mask=store_mask,
        )


@torch.no_grad()
def prefill_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int | None = None,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
    workspace: MutableMapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute prefill index scores without top-k finalization.

    ``sm_scale`` is accepted for interface compatibility.  The supplied
    baseline intentionally omits a positive global scale because only score
    ordering is consumed downstream.

    Only each query's causally visible block domain is semantically defined.
    Score-stride padding and blocks beyond that domain are not initialized by
    this baseline stage.
    """
    del sm_scale
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    if idx_q.ndim != 3:
        raise ValueError(f"idx_q must be rank 3, got {idx_q.ndim}")
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    if num_kv_heads is None:
        num_kv_heads = index_head_count
    if index_head_count != num_kv_heads:
        raise ValueError(
            "M3 requires num_idx_heads == num_kv_heads, got "
            f"{index_head_count} and {num_kv_heads}"
        )
    if index_kv_cache.shape[2] != head_dim:
        raise ValueError(
            f"head_dim mismatch: q={head_dim}, cache={index_kv_cache.shape[2]}"
        )

    batch_size = cu_seqlens_q.shape[0] - 1
    if batch_size <= 0:
        raise ValueError("cu_seqlens_q must describe at least one request")
    if seq_lens.numel() != batch_size or prefix_lens.numel() != batch_size:
        raise ValueError("seq_lens/prefix_lens must match cu_seqlens_q batch")

    shape_h, shape_q, score_stride, _ = _score_shape(
        index_head_count=index_head_count,
        total_query_tokens=total_query_tokens,
        max_seq_len=max_seq_len,
    )
    score = _prepare_output(
        out,
        shape=(shape_h, shape_q, score_stride),
        device=idx_q.device,
    )

    score_grid = (
        triton.cdiv(max_query_len, PREFILL_SCORE_QUERY_TILE_SIZE),
        batch_size * index_head_count,
    )
    if head_dim == 1:
        if workspace is None:
            workspace = make_prefill_workspace(
                index_kv_cache,
                head_dim=head_dim,
            )
        page_extrema = workspace.get("page_extrema")
        expected_extrema_shape = (index_kv_cache.shape[0], 2)
        if page_extrema is None:
            raise ValueError("workspace must contain page_extrema for head_dim=1")
        if tuple(page_extrema.shape) != expected_extrema_shape:
            raise ValueError(
                "page_extrema shape must be "
                f"{expected_extrema_shape}, got {tuple(page_extrema.shape)}"
            )
        if page_extrema.dtype != torch.float32:
            raise TypeError("page_extrema must be float32")

        page_count = index_kv_cache.shape[0]
        _prefill_scalar_key_extrema_kernel[(page_count,)](
            index_kv_cache,
            page_extrema,
            page_count,
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            page_extrema.stride(0),
            page_extrema.stride(1),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
        _prefill_scalar_index_score_kernel[score_grid](
            idx_q,
            index_kv_cache,
            page_extrema,
            score,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            index_head_count,
            idx_q.stride(0),
            idx_q.stride(1),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            page_extrema.stride(0),
            page_extrema.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=PREFILL_SCORE_QUERY_TILE_SIZE,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_B=PREFILL_SCALAR_SCORE_BLOCK_TILE_SIZE,
        )
    else:
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
            BLOCK_SIZE_Q=PREFILL_SCORE_QUERY_TILE_SIZE,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
    return score


@torch.no_grad()
def decode_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
    num_kv_heads: int | None = None,
    max_decode_query_len: int | None = None,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
    workspace: MutableMapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute decode index scores without top-k selection.

    This wrapper preserves the supplied baseline's three-stage score path:
    init/local mask preparation, autotuned split-K score, and logical tail
    filling.  A3-specific grid or fusion changes are deliberately absent.

    The logical range ``[..., :ceil(max_seq_len / 128)]`` is defined.  Extra
    score-stride padding added for alignment is not initialized.
    """
    del sm_scale
    index_kv_cache = _as_triton_index_kv_cache(index_kv_cache)
    if idx_q.ndim != 3:
        raise ValueError(f"idx_q must be rank 3, got {idx_q.ndim}")
    total_query_tokens, index_head_count, head_dim = idx_q.shape
    if num_kv_heads is None:
        num_kv_heads = index_head_count
    if index_head_count != num_kv_heads:
        raise ValueError(
            "M3 requires num_idx_heads == num_kv_heads, got "
            f"{index_head_count} and {num_kv_heads}"
        )
    if index_kv_cache.shape[2] != head_dim:
        raise ValueError(
            f"head_dim mismatch: q={head_dim}, cache={index_kv_cache.shape[2]}"
        )
    if decode_query_len <= 0:
        raise ValueError(
            f"decode_query_len must be positive, got {decode_query_len}"
        )
    if max_decode_query_len is None:
        max_decode_query_len = decode_query_len
    if decode_query_len > max_decode_query_len:
        raise ValueError(
            "decode_query_len must not exceed max_decode_query_len"
        )

    request_count = seq_lens.shape[0]
    if total_query_tokens != request_count * decode_query_len:
        raise ValueError(
            "idx_q total_query_tokens must equal "
            "request_count * decode_query_len"
        )

    shape_h, shape_q, score_stride, max_block_count = _score_shape(
        index_head_count=index_head_count,
        total_query_tokens=total_query_tokens,
        max_seq_len=max_seq_len,
    )
    score = _prepare_output(
        out,
        shape=(shape_h, shape_q, score_stride),
        device=idx_q.device,
    )

    if workspace is None:
        workspace = make_decode_workspace(
            total_query_tokens=total_query_tokens,
            score_block_stride=score_stride,
            device=seq_lens.device,
        )
    init_mask = workspace.get("init_mask")
    local_mask = workspace.get("local_mask")
    expected_mask_shape = (total_query_tokens, score_stride)
    for name, tensor in (
        ("init_mask", init_mask),
        ("local_mask", local_mask),
    ):
        if tensor is None:
            raise ValueError(f"workspace must contain {name}")
        if tuple(tensor.shape) != expected_mask_shape:
            raise ValueError(
                f"{name} shape must be {expected_mask_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")

    mask_chunk_count = max(
        1,
        min(16, 64 // max(1, total_query_tokens)),
    )
    mask_chunk_blocks = triton.cdiv(
        max_block_count,
        mask_chunk_count,
    )
    _prepare_decode_score_masks_kernel[
        (total_query_tokens, mask_chunk_count)
    ](
        init_mask,
        local_mask,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        mask_chunk_blocks,
        init_blocks,
        local_blocks,
        init_mask.stride(0),
        init_mask.stride(1),
        BLOCK_SIZE_K=2048,
    )

    decode_query_tile_size = triton.next_power_of_2(
        max_decode_query_len
    )
    decode_score_grid = lambda metadata: (
        request_count,
        metadata["num_kv_chunks"],
    )
    _decode_index_score_kernel[decode_score_grid](
        idx_q,
        index_kv_cache,
        score,
        init_mask,
        local_mask,
        block_table,
        seq_lens,
        index_head_count,
        head_dim,
        request_count,
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
        init_mask.stride(0),
        init_mask.stride(1),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=decode_query_tile_size,
    )

    tail_chunk_count = max(
        1,
        min(
            16,
            64 // max(1, total_query_tokens * index_head_count),
        ),
    )
    tail_chunk_blocks = triton.cdiv(
        max_block_count,
        tail_chunk_count,
    )
    _fill_decode_score_tail_kernel[
        (total_query_tokens, index_head_count, tail_chunk_count)
    ](
        score,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        max_block_count,
        decode_query_len,
        tail_chunk_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_K=2048,
    )
    return score


# Compatibility aliases for tooling that searches for the original wrapper name.
minimax_m3_index_score = prefill_score
minimax_m3_index_decode_score = decode_score

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "SCORE_BLOCK_STRIDE_ALIGNMENT",
    "decode_score",
    "make_decode_workspace",
    "make_prefill_workspace",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "prefill_score",
]
