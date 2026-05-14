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
"""Unified KV cache store — auto-routes between prefill (P) and decode (D) stages.

Provides a single entry point that handles both:
  - P-stage (numTokens > 1): host-side slot grouping → multi-core grouped copy
  - D-stage (numTokens == 1): single-token, single-core direct write
"""

import torch


def store_kv_cache(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write KV data into KV cache in-place, auto-detecting prefill vs decode.

    Args:
        key:          KV input tensor [numTokens, headDim] on NPU.
        key_cache:    KV cache tensor [...] on NPU (mutated in-place).
        slot_mapping: int32 tensor [numTokens] on NPU.
        block_size:   KV cache block size (used for group-splitting in prefill).
    """
    if key.numel() == 0:
        return
    torch.ops._C_ascend.npu_store_kv_cache(
        key, key_cache, slot_mapping.int(), block_size)
