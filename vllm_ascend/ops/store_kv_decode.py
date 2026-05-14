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
"""Thin Python wrapper for the store_kv_decode (D-stage) operator."""

import torch


def store_kv_decode(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    slot: int,
) -> None:
    """Write a single token into the KV cache in-place (decode stage).

    Args:
        key:      KV input tensor [1, headDim] on NPU (exactly 1 token).
        key_cache: KV cache tensor [...] on NPU (mutated in-place).
        slot:     Destination cache row index.
    """
    slot_tensor = torch.tensor([slot], dtype=torch.int32, device=key.device)
    torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_tensor)
