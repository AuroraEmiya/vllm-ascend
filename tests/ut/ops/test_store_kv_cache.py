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
"""Tests for the unified store_kv_cache operator (P + D stage routing)."""

import itertools
import random

import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C


# ============================================================================
# Helpers
# ============================================================================


def _make_rand(shape, dtype):
    if dtype in (torch.uint8,):
        return torch.randint(0, 128, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def _slot_ref(key, key_cache, slot_mapping):
    """Reference: token-by-token copy from slot_mapping."""
    expected = key_cache.clone()
    for t, s in enumerate(slot_mapping):
        if s >= 0:
            expected[s] = key[t]
    return expected


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.uint8])
class TestStoreKVCacheDecode:
    """D-stage routing: single token → decode path."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_single_token(self, dtype, head_dim):
        nr = 32
        key = _make_rand((1, head_dim), dtype).npu()
        key_cache = torch.zeros(nr, head_dim, dtype=dtype).npu()
        slot_mapping = torch.tensor([5], dtype=torch.int32).npu()
        expected = _slot_ref(key.cpu(), key_cache.cpu(), [5])

        torch.ops._C_ascend.npu_store_kv_cache(
            key, key_cache, slot_mapping, 16)
        assert torch.equal(key_cache.cpu(), expected)

    def test_negative_slot(self, dtype):
        head_dim = 64
        key = _make_rand((1, head_dim), dtype).npu()
        key_cache = torch.zeros(16, head_dim, dtype=dtype).npu()
        original = key_cache.clone()
        slot_mapping = torch.tensor([-1], dtype=torch.int32).npu()

        torch.ops._C_ascend.npu_store_kv_cache(
            key, key_cache, slot_mapping, 16)
        assert torch.equal(key_cache.cpu(), original.cpu())


@pytest.mark.parametrize("dtype", [torch.float16, torch.uint8])
class TestStoreKVCachePrefill:
    """P-stage routing: multi-token → group + multi-core path."""

    @pytest.mark.parametrize("num_tokens,block_size", [(4, 4), (16, 4), (8, 16)])
    def test_continuous(self, dtype, num_tokens, block_size):
        head_dim = 64
        key = _make_rand((num_tokens, head_dim), dtype).npu()
        nc = num_tokens + 16
        key_cache = torch.zeros(nc, head_dim, dtype=dtype).npu()
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32).npu()
        expected = _slot_ref(key.cpu(), key_cache.cpu(), range(num_tokens))

        torch.ops._C_ascend.npu_store_kv_cache(
            key, key_cache, slot_mapping, block_size)
        assert torch.equal(key_cache.cpu(), expected)

    def test_gapped(self, dtype):
        num_tokens = 8
        head_dim = 32
        key = _make_rand((num_tokens, head_dim), dtype).npu()
        key_cache = torch.zeros(64, head_dim, dtype=dtype).npu()
        slots = list(range(0, num_tokens * 3, 3))
        slot_mapping = torch.tensor(slots, dtype=torch.int32).npu()
        expected = _slot_ref(key.cpu(), key_cache.cpu(), slots)

        torch.ops._C_ascend.npu_store_kv_cache(
            key, key_cache, slot_mapping, 32)
        assert torch.equal(key_cache.cpu(), expected)

    def test_negative_mixed(self, dtype):
        head_dim = 32
        key = _make_rand((4, head_dim), dtype).npu()
        key_cache = torch.zeros(16, head_dim, dtype=dtype).npu()
        slots = [0, -1, 3, -1]
        slot_mapping = torch.tensor(slots, dtype=torch.int32).npu()
        expected = _slot_ref(key.cpu(), key_cache.cpu(), slots)

        torch.ops._C_ascend.npu_store_kv_cache(
            key, key_cache, slot_mapping, 16)
        assert torch.equal(key_cache.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16])
class TestStoreKVCachePerf:
    """Unified profiling — auto routing tested at both extremes."""

    WARMUP = 5
    REPEAT = 100

    @pytest.mark.parametrize("num_tokens", [1, 4096])
    @pytest.mark.parametrize("head_dim", [128])
    @pytest.mark.parametrize("block_size", [128])
    def test_latency(self, dtype, num_tokens, head_dim, block_size):
        total_slots = max(4096, num_tokens + 256)
        key = _make_rand((num_tokens, head_dim), dtype).npu()
        key_cache = torch.zeros(total_slots, head_dim, dtype=dtype).npu()
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32).npu()

        # warmup
        for _ in range(self.WARMUP):
            torch.ops._C_ascend.npu_store_kv_cache(
                key, key_cache, slot_mapping, block_size)
        torch.npu.synchronize()

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(self.REPEAT):
            torch.ops._C_ascend.npu_store_kv_cache(
                key, key_cache, slot_mapping, block_size)
        end.record()
        torch.npu.synchronize()
        elapsed_ms = start.elapsed_time(end) / self.REPEAT

        label = "D-stage" if num_tokens == 1 else "P-stage"
        print(f"[{label}] tokens={num_tokens:5d}  head_dim={head_dim:3d}  "
              f"avg={elapsed_ms:.3f} ms")
