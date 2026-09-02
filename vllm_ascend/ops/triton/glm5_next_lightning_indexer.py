# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element


_POOL_CHUNK_SIZE = 2048
_FUSED_MAX_POOL_TOPK = 128



@triton.jit
def _glm5_next_lightning_indexer_score_kernel(
    query_ptr,
    indexer_cache_ptr,
    weights_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
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
    scores_stride_t: tl.constexpr,
    pool_block_size: tl.constexpr,
    NUM_REQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
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

        tl.store(
            scores_ptr + token_idx * scores_stride_t + absolute_pool_offsets,
            scores,
            mask=absolute_pool_offsets < MAX_POOL_SEQ_LEN,
        )


def _topk_pool_ids(scores: torch.Tensor, pool_topk: int) -> torch.Tensor:
    if scores.shape[1] < pool_topk:
        scores = torch.nn.functional.pad(
            scores,
            (0, pool_topk - scores.shape[1]),
            value=float("-inf"),
        )

    values, pool_ids = torch.topk(
        scores,
        k=pool_topk,
        dim=1,
        largest=True,
        sorted=False,
    )
    return torch.where(
        values != float("-inf"),
        pool_ids,
        torch.full_like(pool_ids, -1),
    ).to(torch.int32)


def _expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    positions: torch.Tensor,
    index_topk: int,
    index_kpool: int,
) -> torch.Tensor:
    offsets = torch.arange(
        index_kpool,
        dtype=torch.int64,
        device=pool_ids.device,
    )
    token_ids = pool_ids.to(torch.int64).unsqueeze(-1) * index_kpool + offsets
    token_ids = torch.where(
        (pool_ids >= 0).unsqueeze(-1),
        token_ids,
        torch.full_like(token_ids, -1),
    )
    expanded = token_ids.reshape(pool_ids.shape[0], index_topk).to(torch.int32)

    tail_width = index_kpool - 1
    if tail_width == 0:
        return expanded.unsqueeze(1)

    seq_lens = positions.to(torch.int32) + 1
    tail_start = torch.div(
        seq_lens,
        index_kpool,
        rounding_mode="floor",
    ) * index_kpool
    tail_count = seq_lens - tail_start
    tail_offsets = torch.arange(
        tail_width,
        dtype=torch.int32,
        device=pool_ids.device,
    )
    tail_values = tail_start[:, None] + tail_offsets[None, :]
    tail_values = torch.where(
        tail_offsets[None, :] < tail_count[:, None],
        tail_values,
        torch.full_like(tail_values, -1),
    )
    return torch.cat([expanded, tail_values], dim=1).unsqueeze(1)


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
    if query.shape[0] == 0:
        return torch.empty(
            (0, 1, output_width),
            dtype=torch.int32,
            device=query.device,
        )

    if max_pool_seq_len == 0:
        pool_ids = torch.full(
            (query.shape[0], pool_topk),
            -1,
            dtype=torch.int32,
            device=query.device,
        )
        return _expand_pools_and_append_tail(
            pool_ids,
            positions,
            index_topk,
            index_kpool,
        )
    scores = torch.empty(
        (query.shape[0], max_pool_seq_len),
        dtype=torch.float32,
        device=query.device,
    )
    num_chunks = triton.cdiv(max_pool_seq_len, _POOL_CHUNK_SIZE)
    _glm5_next_lightning_indexer_score_kernel[(query.shape[0],)](
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        scores,
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
        scores.stride(0),
        indexer_cache.shape[1],
        cum_query_lens.shape[0],
        query.shape[1],
        query.shape[2],
        index_kpool,
        max_pool_seq_len,
        num_chunks,
        _POOL_CHUNK_SIZE,
    )

    pool_ids = _topk_pool_ids(scores, pool_topk)
    return _expand_pools_and_append_tail(
        pool_ids,
        positions,
        index_topk,
        index_kpool,
    )
