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
"""Tests for the store_kv_decode (D-stage) operator."""

import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C


def reference_decode(key, key_cache, slot):
    expected = key_cache.clone()
    expected[slot] = key[0]
    return expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.uint8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
class TestStoreKVDecode:
    """Decode-stage: single token, single slot writes."""

    def test_basic(self, dtype, head_dim):
        """Write 1 token to the first slot."""
        key_cache = torch.zeros(20, head_dim, dtype=dtype).npu()
        key = torch.randn(1, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (1, head_dim), dtype=dtype).npu()

        slot_tensor = torch.tensor([0], dtype=torch.int32).npu()
        expected = reference_decode(key.cpu(), key_cache.cpu(), 0)

        torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_tensor)
        assert torch.equal(key_cache.cpu(), expected)

    def test_mid_cache(self, dtype, head_dim):
        """Write to a slot in the middle of the cache."""
        key_cache = torch.randn(50, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (50, head_dim), dtype=dtype).npu()
        original = key_cache.clone()
        key = torch.randn(1, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (1, head_dim), dtype=dtype).npu()

        slot = 25
        slot_tensor = torch.tensor([slot], dtype=torch.int32).npu()
        expected = reference_decode(key.cpu(), key_cache.cpu(), slot)

        torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_tensor)

        # Only slot 25 should change
        assert torch.equal(key_cache.cpu()[slot], key.cpu()[0])
        assert torch.equal(key_cache.cpu()[:slot], original.cpu()[:slot])
        assert torch.equal(key_cache.cpu()[slot + 1:], original.cpu()[slot + 1:])

    def test_negative_skip(self, dtype, head_dim):
        """Negative slot: operator should be a no-op."""
        key_cache = torch.randn(10, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (10, head_dim), dtype=dtype).npu()
        original = key_cache.clone()
        key = torch.randn(1, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (1, head_dim), dtype=dtype).npu()

        slot_tensor = torch.tensor([-1], dtype=torch.int32).npu()
        torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_tensor)
        assert torch.equal(key_cache.cpu(), original.cpu())

    def test_multi_call(self, dtype, head_dim):
        """Sequential decode calls to populate multiple slots."""
        num_tokens = 8
        key_cache = torch.zeros(32, head_dim, dtype=dtype).npu()
        tokens = torch.randn(num_tokens, head_dim, dtype=dtype).npu() if dtype != torch.uint8 \
            else torch.randint(0, 128, (num_tokens, head_dim), dtype=dtype).npu()

        expected = key_cache.cpu().clone()
        slots = [3, 7, 1, 15, 0, 20, 31, 12]
        for i, slot in enumerate(slots):
            key_i = tokens[i:i + 1]  # [1, headDim]
            slot_tensor = torch.tensor([slot], dtype=torch.int32).npu()
            torch.ops._C_ascend.npu_store_kv_decode(key_i, key_cache, slot_tensor)
            expected[slot] = tokens[i].cpu()

        assert torch.equal(key_cache.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("head_dim", [128])
class TestStoreKVDecodePerf:
    """Decode performance tests for msprof profiling."""

    WARMUP = 10
    REPEAT = 1000

    def test_decode_latency(self, dtype, head_dim):
        """Measure single decode call latency."""
        key_cache = torch.zeros(1024, head_dim, dtype=dtype).npu()
        key = torch.randn(1, head_dim, dtype=dtype).npu()

        # warmup
        for _ in range(self.WARMUP):
            slot_t = torch.tensor([0], dtype=torch.int32).npu()
            torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_t)
        torch.npu.synchronize()

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for i in range(self.REPEAT):
            slot_t = torch.tensor([i % 1024], dtype=torch.int32).npu()
            torch.ops._C_ascend.npu_store_kv_decode(key, key_cache, slot_t)
        end.record()
        torch.npu.synchronize()
        elapsed_ms = start.elapsed_time(end) / self.REPEAT
        print(f"[decode] head_dim={head_dim}  avg={elapsed_ms*1000:.1f} us")
