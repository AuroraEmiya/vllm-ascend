# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element


_POOL_CHUNK_SIZE = 2048


@triton.jit
def _glm5_next_lightning_indexer_kernel(
    query_ptr,
    indexer_cache_ptr,
    weights_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    output_ptr,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_h: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    NUM_REQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    POOL_TOPK: tl.constexpr,
    MAX_POOL_SEQ_LEN: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
):
    token_idx = tl.program_id(0)

    req_id = 0
    for req in tl.range(NUM_REQS):
        query_end = tl.load(cum_query_lens_ptr + req).to(tl.int32)
        req_id += tl.where(token_idx >= query_end, 1, 0)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for head_idx in tl.range(NUM_HEADS):
        weight = tl.load(weights_ptr + token_idx * weights_stride_t + head_idx * weights_stride_h).to(tl.float32)
        q = tl.load(
            query_ptr + token_idx * query_stride_t + head_idx * query_stride_h + dim_offsets * query_stride_d
        ).to(tl.float32)
        qbar += q * weight

    pool_offsets = tl.arange(0, BLOCK_POOL)
    valid_pool = (pool_offsets < visible_pool_len) & (pool_offsets < MAX_POOL_SEQ_LEN)
    page_offsets = pool_offsets % pool_block_size
    logical_pages = pool_offsets // pool_block_size
    physical_blocks = tl.load(
        indexer_block_table_ptr + req_id * block_table_stride_req + logical_pages * block_table_stride_page,
        mask=pool_offsets < MAX_POOL_SEQ_LEN,
        other=0,
    ).to(tl.int64)
    physical_blocks = tl.maximum(physical_blocks, 0)

    scores = tl.zeros((BLOCK_POOL,), dtype=tl.float32)
    for dim_idx in tl.range(HEAD_DIM):
        q_value = get_element(qbar, (dim_idx,))
        k = tl.load(
            indexer_cache_ptr
            + physical_blocks * cache_stride_block
            + page_offsets * cache_stride_offset
            + dim_idx * cache_stride_d,
            mask=valid_pool,
            other=0.0,
        ).to(tl.float32)
        scores += q_value * k
    scores = tl.where(valid_pool, scores, float("-inf"))

    for topk_idx in tl.range(POOL_TOPK):
        best_value = tl.max(scores, axis=0)
        best_idx = tl.argmax(scores, axis=0).to(tl.int32)
        pool_id = tl.where(best_value == float("-inf"), -1, best_idx)
        for offset in range(INDEX_KPOOL):
            out_col = topk_idx * INDEX_KPOOL + offset
            token_id = pool_id * INDEX_KPOOL + offset
            token_id = tl.where(pool_id >= 0, token_id, -1)
            tl.store(
                output_ptr + token_idx * OUTPUT_WIDTH + out_col,
                token_id,
                mask=out_col < INDEX_TOPK,
            )
        scores = tl.where(pool_offsets == best_idx, float("-inf"), scores)

    tail_start = ((pos + 1) // INDEX_KPOOL) * INDEX_KPOOL
    tail_count = pos + 1 - tail_start
    for tail_idx in range(INDEX_KPOOL - 1):
        out_col = INDEX_TOPK + tail_idx
        tail_value = tl.where(tail_idx < tail_count, tail_start + tail_idx, -1)
        tl.store(output_ptr + token_idx * OUTPUT_WIDTH + out_col, tail_value)


@triton.jit
def _glm5_next_lightning_indexer_local_topk_kernel(
    query_ptr,
    indexer_cache_ptr,
    weights_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    query_stride_t: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_d: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    weights_stride_t: tl.constexpr,
    weights_stride_h: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    NUM_REQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    POOL_TOPK: tl.constexpr,
    MAX_POOL_SEQ_LEN: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
):
    token_idx = tl.program_id(0)

    req_id = 0
    for req in tl.range(NUM_REQS):
        query_end = tl.load(cum_query_lens_ptr + req).to(tl.int32)
        req_id += tl.where(token_idx >= query_end, 1, 0)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for head_idx in tl.range(NUM_HEADS):
        weight = tl.load(weights_ptr + token_idx * weights_stride_t + head_idx * weights_stride_h).to(tl.float32)
        q = tl.load(
            query_ptr + token_idx * query_stride_t + head_idx * query_stride_h + dim_offsets * query_stride_d
        ).to(tl.float32)
        qbar += q * weight

    pool_offsets = tl.arange(0, BLOCK_POOL)
    partial_base = token_idx * NUM_CHUNKS * POOL_TOPK

    for chunk_idx in tl.range(NUM_CHUNKS):
        pool_start = chunk_idx * BLOCK_POOL
        absolute_pool_offsets = pool_start + pool_offsets
        valid_pool = (absolute_pool_offsets < visible_pool_len) & (absolute_pool_offsets < MAX_POOL_SEQ_LEN)
        page_offsets = absolute_pool_offsets % pool_block_size
        logical_pages = absolute_pool_offsets // pool_block_size
        physical_blocks = tl.load(
            indexer_block_table_ptr
            + req_id * block_table_stride_req
            + logical_pages * block_table_stride_page,
            mask=absolute_pool_offsets < MAX_POOL_SEQ_LEN,
            other=0,
        ).to(tl.int64)
        physical_blocks = tl.maximum(physical_blocks, 0)

        scores = tl.zeros((BLOCK_POOL,), dtype=tl.float32)
        for dim_idx in tl.range(HEAD_DIM):
            q_value = get_element(qbar, (dim_idx,))
            k = tl.load(
                indexer_cache_ptr
                + physical_blocks * cache_stride_block
                + page_offsets * cache_stride_offset
                + dim_idx * cache_stride_d,
                mask=valid_pool,
                other=0.0,
            ).to(tl.float32)
            scores += q_value * k
        scores = tl.where(valid_pool, scores, float("-inf"))

        for topk_idx in tl.range(POOL_TOPK):
            best_value = tl.max(scores, axis=0)
            best_idx = tl.argmax(scores, axis=0).to(tl.int32)
            pool_id = tl.where(best_value == float("-inf"), -1, pool_start + best_idx)
            partial_offset = partial_base + chunk_idx * POOL_TOPK + topk_idx
            tl.store(partial_values_ptr + partial_offset, best_value)
            tl.store(partial_indices_ptr + partial_offset, pool_id)
            scores = tl.where(pool_offsets == best_idx, float("-inf"), scores)


@triton.jit
def _glm5_next_lightning_indexer_merge_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    positions_ptr,
    output_ptr,
    INDEX_TOPK: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    POOL_TOPK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
):
    token_idx = tl.program_id(0)

    candidate_count = NUM_CHUNKS * POOL_TOPK
    candidate_offsets = tl.arange(0, MERGE_BLOCK)
    candidate_mask = candidate_offsets < candidate_count
    partial_base = token_idx * candidate_count

    values = tl.load(
        partial_values_ptr + partial_base + candidate_offsets,
        mask=candidate_mask,
        other=float("-inf"),
    ).to(tl.float32)
    indices = tl.load(
        partial_indices_ptr + partial_base + candidate_offsets,
        mask=candidate_mask,
        other=-1,
    ).to(tl.int32)

    for topk_idx in tl.range(POOL_TOPK):
        best_value = tl.max(values, axis=0)
        best_idx = tl.argmax(values, axis=0).to(tl.int32)
        pool_id = get_element(indices, (best_idx,))
        pool_id = tl.where(best_value == float("-inf"), -1, pool_id)

        for offset in range(INDEX_KPOOL):
            out_col = topk_idx * INDEX_KPOOL + offset
            token_id = pool_id * INDEX_KPOOL + offset
            token_id = tl.where(pool_id >= 0, token_id, -1)
            tl.store(
                output_ptr + token_idx * OUTPUT_WIDTH + out_col,
                token_id,
                mask=out_col < INDEX_TOPK,
            )

        values = tl.where(candidate_offsets == best_idx, float("-inf"), values)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    tail_start = ((pos + 1) // INDEX_KPOOL) * INDEX_KPOOL
    tail_count = pos + 1 - tail_start
    for tail_idx in range(INDEX_KPOOL - 1):
        out_col = INDEX_TOPK + tail_idx
        tail_value = tl.where(tail_idx < tail_count, tail_start + tail_idx, -1)
        tl.store(output_ptr + token_idx * OUTPUT_WIDTH + out_col, tail_value)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def glm5_next_lightning_indexer_triton(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    pool_topk = index_topk // index_kpool
    output_width = index_topk + index_kpool - 1
    output = torch.empty(
        (query.shape[0], 1, output_width),
        dtype=torch.int32,
        device=query.device,
    )
    if query.shape[0] == 0:
        return output

    if max_pool_seq_len <= _POOL_CHUNK_SIZE:
        _glm5_next_lightning_indexer_kernel[(query.shape[0],)](
            query,
            indexer_cache,
            weights,
            cum_query_lens,
            indexer_seq_lens,
            indexer_block_table,
            positions,
            output,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            indexer_cache.stride(0),
            indexer_cache.stride(1),
            indexer_cache.stride(3),
            weights.stride(0),
            weights.stride(1),
            indexer_block_table.stride(0),
            indexer_block_table.stride(1),
            indexer_cache.shape[1],
            cum_query_lens.shape[0],
            query.shape[1],
            query.shape[2],
            index_topk,
            index_kpool,
            pool_topk,
            max_pool_seq_len,
            _POOL_CHUNK_SIZE,
            output_width,
        )
        return output

    num_chunks = triton.cdiv(max_pool_seq_len, _POOL_CHUNK_SIZE)
    partial_values = torch.empty(
        (query.shape[0], num_chunks, pool_topk),
        dtype=torch.float32,
        device=query.device,
    )
    partial_indices = torch.empty(
        (query.shape[0], num_chunks, pool_topk),
        dtype=torch.int32,
        device=query.device,
    )
    _glm5_next_lightning_indexer_local_topk_kernel[(query.shape[0],)](
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        partial_values,
        partial_indices,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        indexer_cache.stride(0),
        indexer_cache.stride(1),
        indexer_cache.stride(3),
        weights.stride(0),
        weights.stride(1),
        indexer_block_table.stride(0),
        indexer_block_table.stride(1),
        indexer_cache.shape[1],
        cum_query_lens.shape[0],
        query.shape[1],
        query.shape[2],
        index_kpool,
        pool_topk,
        max_pool_seq_len,
        num_chunks,
        _POOL_CHUNK_SIZE,
    )

    merge_block = _next_power_of_2(num_chunks * pool_topk)
    _glm5_next_lightning_indexer_merge_kernel[(query.shape[0],)](
        partial_values,
        partial_indices,
        positions,
        output,
        index_topk,
        index_kpool,
        pool_topk,
        num_chunks,
        merge_block,
        output_width,
    )
    return output
