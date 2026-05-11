#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Thin Python wrapper for store_kv_block and store_kv_block_pre operators."""

from typing import Tuple

import torch


def store_kv_block_pre(
    slot_mapping: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess slot_mapping into grouped metadata for store_kv_block.

    Each group represents a contiguous segment where both source token
    indices and destination cache addresses are consecutive. Invalid
    (negative) slot entries are skipped.

    Args:
        slot_mapping: int32 tensor [num_tokens] on NPU.
        block_size: Maximum group length (also KV cache block size).

    Returns:
        group_len:   int32 tensor [num_groups] on NPU.
        group_key_idx:       int32 tensor [num_groups] on NPU.
        group_key_cache_idx: int32 tensor [num_groups] on NPU.
    """
    if slot_mapping.numel() == 0:
        device = slot_mapping.device
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty

    # Move to CPU for grouping logic
    return torch.ops._C_ascend.npu_store_kv_block_pre(
        slot_mapping.int(), block_size)


def store_kv_block(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Copy KV data into KV cache using precomputed grouped metadata.

    Each group g copies group_len[g] elements from key[group_key_idx[g]]
    into key_cache[group_key_cache_idx[g]].

    Args:
        key:                KV input tensor   [num_tokens, head_dim] on NPU.
        key_cache:          KV cache tensor   [...] on NPU.
        group_len:          Group lengths     [num_groups] int32 on NPU.
        group_key_idx:      Source starts     [num_groups] int32 on NPU.
        group_key_cache_idx: Destination starts [num_groups] int32 on NPU.
        block_size:         KV cache block size.

    Returns:
        Updated key cache tensor (same shape/dtype as key_cache).
    """
    return torch.ops._C_ascend.npu_store_kv_block(
        key, key_cache, group_len, group_key_idx, group_key_cache_idx, block_size)
