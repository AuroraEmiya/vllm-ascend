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
"""Tests for store_kv_block and store_kv_block_pre operators."""

import itertools

import pytest
import torch


# ============================================================================
# Reference Implementations
# ============================================================================


def reference_store_kv_block_pre(
    slot_mapping: torch.Tensor, block_size: int
):
    """Reference grouping logic on CPU using plain Python."""
    num_tokens = slot_mapping.numel()
    group_len = []
    group_key_idx = []
    group_key_cache_idx = []

    t = 0
    while t < num_tokens:
        s = int(slot_mapping[t].item())
        if s < 0:
            t += 1
            continue
        src_start = t
        dst_start = s
        length = 1
        while t + 1 < num_tokens:
            next_s = int(slot_mapping[t + 1].item())
            if next_s < 0:
                break
            if next_s == s + 1 and length < block_size:
                length += 1
                s = next_s
                t += 1
            else:
                break
        group_len.append(length)
        group_key_idx.append(src_start)
        group_key_cache_idx.append(dst_start)
        t += 1

    return (
        torch.tensor(group_len, dtype=torch.int32),
        torch.tensor(group_key_idx, dtype=torch.int32),
        torch.tensor(group_key_cache_idx, dtype=torch.int32),
    )


def reference_store_kv_block(
    key: torch.Tensor,
    key_cache: torch.Tensor,
    group_len: torch.Tensor,
    group_key_idx: torch.Tensor,
    group_key_cache_idx: torch.Tensor,
):
    """Reference grouped copy on CPU."""
    expected = key_cache.clone()
    for g in range(group_len.numel()):
        length = int(group_len[g].item())
        src_start = int(group_key_idx[g].item())
        dst_start = int(group_key_cache_idx[g].item())
        for offset in range(length):
            expected[dst_start + offset] = key[src_start + offset]
    return expected


# ============================================================================
# store_kv_block_pre Tests
# ============================================================================


class TestStoreKVBlockPre:
    """Test grouping logic of store_kv_block_pre."""

    @pytest.mark.parametrize("num_tokens", [1, 8, 16, 64, 128])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_fully_continuous(self, num_tokens, block_size):
        """Fully continuous slot_mapping produces a single group."""
        slot = torch.arange(num_tokens, dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, block_size)

        expected_gl = []
        expected_gki = []
        expected_gkci = []
        rem = num_tokens
        pos = 0
        while rem > 0:
            g_len = min(rem, block_size)
            expected_gl.append(g_len)
            expected_gki.append(pos)
            expected_gkci.append(pos)
            pos += g_len
            rem -= g_len

        assert gl.cpu().tolist() == expected_gl
        assert gki.cpu().tolist() == expected_gki
        assert gkci.cpu().tolist() == expected_gkci
        for val in gl.cpu().tolist():
            assert 1 <= val <= block_size

    @pytest.mark.parametrize("block_size", [16, 32])
    def test_non_continuous(self, block_size):
        """Non-continuous slot_mapping: every slot jumps by 2."""
        num_tokens = 20
        slot = torch.arange(0, num_tokens * 2, 2, dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, block_size)

        # Each token is its own group since dst is not consecutive
        assert gl.numel() == num_tokens
        assert all(v == 1 for v in gl.cpu().tolist())
        assert gki.cpu().tolist() == list(range(num_tokens))
        assert gkci.cpu().tolist() == list(range(0, num_tokens * 2, 2))

    def test_invalid_negative_slots(self):
        """Negative slots are skipped and do not form groups."""
        slot = torch.tensor([0, -1, 1, 2, -1, 3], dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, 16)

        expected_gl = [1, 2, 1]
        expected_gki = [0, 2, 5]
        expected_gkci = [0, 1, 3]

        assert gl.cpu().tolist() == expected_gl
        assert gki.cpu().tolist() == expected_gki
        assert gkci.cpu().tolist() == expected_gkci

    def test_all_negative_slots(self):
        """All invalid slots produce empty outputs."""
        slot = torch.tensor([-1, -1, -1], dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, 16)

        assert gl.numel() == 0
        assert gki.numel() == 0
        assert gkci.numel() == 0

    def test_block_boundary_split(self):
        """Group splits at block boundary."""
        block_size = 4
        slot = torch.arange(10, dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, block_size)

        # 10 tokens, block_size=4 -> groups: 4, 4, 2
        assert gl.cpu().tolist() == [4, 4, 2]
        assert gki.cpu().tolist() == [0, 4, 8]
        assert gkci.cpu().tolist() == [0, 4, 8]

    def test_single_token(self):
        slot = torch.tensor([5], dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, 16)

        assert gl.numel() == 1
        assert int(gl.item()) == 1
        assert int(gki.item()) == 0
        assert int(gkci.item()) == 5

    def test_continuity_reset_on_discontiguous_src(self):
        """Split when source tokens are not consecutive (due to skip)."""
        slot = torch.tensor([0, 1, -1, 2, 3], dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, 16)

        # Groups: [0,1], [2,3]
        assert gl.cpu().tolist() == [2, 2]
        assert gki.cpu().tolist() == [0, 3]
        assert gkci.cpu().tolist() == [0, 2]

    @pytest.mark.parametrize(
        "block_size,num_tokens,expected_num_groups",
        [(16, 16, 1), (16, 32, 2), (8, 8, 1), (8, 20, 3)],
    )
    def test_continuity_vs_block_split(
        self, block_size, num_tokens, expected_num_groups
    ):
        slot = torch.arange(num_tokens, dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot, block_size)

        assert gl.numel() == expected_num_groups
        assert gl.sum().item() == num_tokens
        for val in gl.cpu().tolist():
            assert 1 <= val <= block_size


# ============================================================================
# store_kv_block Tests
# ============================================================================


class TestStoreKVBlock:
    """Test the main KV cache store operator."""

    @pytest.mark.parametrize(
        "dtype,block_size,num_tokens",
        list(
            itertools.product(
                [torch.float16, torch.int8],
                [16, 32],
                [1, 8, 32, 64],
            )
        ),
    )
    def test_correctness_single_group(self, dtype, block_size, num_tokens):
        """Single group: all tokens in one contiguous copy."""
        if dtype == torch.int8:
            key = torch.randint(-128, 127, (num_tokens, 128),
                                dtype=torch.int8).npu()
        else:
            key = torch.randn(num_tokens, 128, dtype=dtype).npu()
        key_cache = torch.zeros(256, 128, dtype=dtype).npu()

        group_len = torch.tensor([num_tokens], dtype=torch.int32).npu()
        group_key_idx = torch.tensor([0], dtype=torch.int32).npu()
        group_key_cache_idx = torch.tensor([0], dtype=torch.int32).npu()

        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, group_len, group_key_idx, group_key_cache_idx,
            block_size)

        expected = reference_store_kv_block(
            key.cpu(), key_cache.cpu(), group_len.cpu(),
            group_key_idx.cpu(), group_key_cache_idx.cpu())

        assert out.dtype == key_cache.dtype
        assert out.shape == key_cache.shape
        assert torch.equal(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
    def test_mixed_group_lengths(self, dtype):
        """Multiple groups with different lengths."""
        block_size = 16
        num_tokens = 30
        head_dim = 64
        if dtype == torch.int8:
            key = torch.randint(-128, 127, (num_tokens, head_dim),
                                dtype=torch.int8).npu()
        else:
            key = torch.randn(num_tokens, head_dim, dtype=dtype).npu()
        key_cache = torch.zeros(200, head_dim, dtype=dtype).npu()

        # Groups: 10, 5, 15
        group_len = torch.tensor([10, 5, 15], dtype=torch.int32).npu()
        group_key_idx = torch.tensor([0, 10, 15], dtype=torch.int32).npu()
        group_key_cache_idx = torch.tensor(
            [0, 30, 60], dtype=torch.int32).npu()

        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, group_len, group_key_idx, group_key_cache_idx,
            block_size)

        expected = reference_store_kv_block(
            key.cpu(), key_cache.cpu(), group_len.cpu(),
            group_key_idx.cpu(), group_key_cache_idx.cpu())

        assert torch.equal(out.cpu(), expected)

    def test_group_len_one(self):
        """Each token is its own group (group_len=1)."""
        num_tokens = 16
        head_dim = 32
        key = torch.randn(num_tokens, head_dim, dtype=torch.float16).npu()
        key_cache = torch.zeros(100, head_dim, dtype=torch.float16).npu()

        group_len = torch.ones(num_tokens, dtype=torch.int32).npu()
        group_key_idx = torch.arange(num_tokens, dtype=torch.int32).npu()
        group_key_cache_idx = torch.arange(
            0, num_tokens * 3, 3, dtype=torch.int32).npu()

        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, group_len, group_key_idx, group_key_cache_idx, 16)

        expected = reference_store_kv_block(
            key.cpu(), key_cache.cpu(), group_len.cpu(),
            group_key_idx.cpu(), group_key_cache_idx.cpu())

        assert torch.equal(out.cpu(), expected)

    def test_unchanged_cache_positions(self):
        """Positions not in any group remain unchanged."""
        head_dim = 16
        key = torch.randn(4, head_dim, dtype=torch.float16).npu()
        key_cache = torch.randn(20, head_dim, dtype=torch.float16).npu()
        original = key_cache.clone()

        group_len = torch.tensor([2, 2], dtype=torch.int32).npu()
        group_key_idx = torch.tensor([0, 2], dtype=torch.int32).npu()
        group_key_cache_idx = torch.tensor([0, 5], dtype=torch.int32).npu()

        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, group_len, group_key_idx, group_key_cache_idx, 16)

        # Positions 0-1 and 5-6 should be from key
        assert torch.equal(
            out.cpu()[:2], key.cpu()[:2])
        assert torch.equal(
            out.cpu()[5:7], key.cpu()[2:4])
        # Positions 2-4, 7-19 should be unchanged
        assert torch.equal(out.cpu()[2:5], original.cpu()[2:5])
        assert torch.equal(out.cpu()[7:], original.cpu()[7:])


# ============================================================================
# Integration Tests
# ============================================================================


class TestStoreKVBlockIntegration:
    """End-to-end: slot_mapping -> pre -> main -> cache."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
    @pytest.mark.parametrize("num_tokens", [1, 16, 32, 64])
    @pytest.mark.parametrize("block_size", [16])
    def test_full_pipeline_continuous(self, dtype, num_tokens, block_size):
        """Fully continuous case: pre + main pass through."""
        head_dim = 64
        if dtype == torch.int8:
            key = torch.randint(-128, 127, (num_tokens, head_dim),
                                dtype=torch.int8).npu()
        else:
            key = torch.randn(num_tokens, head_dim, dtype=dtype).npu()
        key_cache = torch.zeros(200, head_dim, dtype=dtype).npu()

        slot_mapping = torch.arange(num_tokens, dtype=torch.int32).npu()

        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot_mapping, block_size)
        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, gl, gki, gkci, block_size)

        # Reference: token-by-token from slot_mapping
        expected = key_cache.cpu().clone()
        for t in range(num_tokens):
            s = int(slot_mapping[t].item())
            if s >= 0:
                expected[s] = key.cpu()[t]

        assert torch.equal(out.cpu(), expected)

    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("block_size", [16])
    def test_full_pipeline_with_negatives(self, dtype, block_size):
        """Pipeline with negative (invalid) slots."""
        head_dim = 32
        key = torch.randn(6, head_dim, dtype=dtype).npu()
        key_cache = torch.zeros(20, head_dim, dtype=dtype).npu()

        # Tokens 0,1 valid; 2 invalid; 3,4 valid; 5 invalid
        slot_mapping = torch.tensor(
            [0, 1, -1, 2, 3, -1], dtype=torch.int32).npu()

        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot_mapping, block_size)
        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, gl, gki, gkci, block_size)

        expected = key_cache.cpu().clone()
        for t in range(6):
            s = int(slot_mapping[t].item())
            if s >= 0:
                expected[s] = key.cpu()[t]

        assert torch.equal(out.cpu(), expected)

    @pytest.mark.parametrize("block_size", [16])
    def test_full_pipeline_non_contiguous(self, block_size):
        """Pipeline with non-contiguous slot_mapping."""
        head_dim = 16
        key = torch.randn(5, head_dim, dtype=torch.float16).npu()
        key_cache = torch.zeros(50, head_dim, dtype=torch.float16).npu()

        slot_mapping = torch.tensor(
            [0, 10, 20, 30, 40], dtype=torch.int32).npu()

        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot_mapping, block_size)
        out = torch.ops._C_ascend.npu_store_kv_block(
            key, key_cache, gl, gki, gkci, block_size)

        # Each token is its own group
        assert gl.cpu().tolist() == [1, 1, 1, 1, 1]
        assert gkci.cpu().tolist() == [0, 10, 20, 30, 40]

        expected = key_cache.cpu().clone()
        for t in range(5):
            s = int(slot_mapping[t].item())
            expected[s] = key.cpu()[t]

        assert torch.equal(out.cpu(), expected)

    def test_empty_slot_mapping(self):
        """Empty slot_mapping produces empty groups."""
        slot_mapping = torch.empty(0, dtype=torch.int32).npu()
        gl, gki, gkci = torch.ops._C_ascend.npu_store_kv_block_pre(
            slot_mapping, 16)

        assert gl.numel() == 0
        assert gki.numel() == 0
        assert gkci.numel() == 0
