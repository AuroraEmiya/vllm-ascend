import gc
import os
import random
import time

import numpy as np
import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C

torch.set_printoptions(threshold=np.inf)


def random_with_zero_prob(zero_prob, index):
    if not 0 <= zero_prob <= 1:
        raise ValueError("the probablity must be in [0, 1]")

    if index <= 0:
        raise ValueError("index must be > 0")

    if random.random() < zero_prob:
        return 0

    return random.randint(1, index)


def assert_read_kv_cache_registered():
    assert hasattr(torch.ops, "_C_ascend"), "torch.ops._C_ascend is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "read_kv_cache_pre",
    ), "torch.ops._C_ascend.read_kv_cache_pre is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "read_kv_cache",
    ), "torch.ops._C_ascend.read_kv_cache is not registered"


def golden_read_kv_cache(key_cache_list, key_table, slotmap, block_size):
    expected_key = key_table.clone()

    for token_idx, slot in enumerate(slotmap):
        if slot < 0:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        expected_key[token_idx, :, :] = key_cache_list[block_idx, block_offset, :, :]

    return expected_key


def golden_store_kv_cache(key_cache_list, key_table, slotmap, block_size):
    expected_cache = key_cache_list.clone()

    for token_idx, slot in enumerate(slotmap):
        if slot < 0:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        expected_cache[block_idx, block_offset, :, :] = key_table[token_idx, :, :]

    return expected_cache


def make_slot_mapping_tensors(slotmap):
    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()

    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()
    return slot_mapping_npu, slot_mapping_list


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [128])
def test_readKVCache_with_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_read_kv_cache_registered()

    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = []
    for i in range(0, num_tokens):
        slotmap.append(i)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    expected_key = cache_table.reshape(
        num_blocks * block_size,
        num_head,
        head_size,
    )[:num_tokens].contiguous()

    slot_mapping_npu, slot_mapping_list = make_slot_mapping_tensors(slotmap)

    keylist_npu = torch.empty_like(keylist).npu()
    cache_table_npu = cache_table.clone().npu()

    epoch = 100
    for _ in range(epoch):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.read_kv_cache_pre(
            slot_mapping_npu,
            slot_mapping_list,
            block_size,
        )

        torch.ops._C_ascend.read_kv_cache(
            cache_table_npu,
            keylist_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )

    torch.npu.synchronize()

    print("group_len:", group_len.cpu()[:10])
    print("group_key_idx:", group_key_idx.cpu()[:10])
    print("group_key_cache_idx:", group_key_cache_idx.cpu()[:10])

    torch.testing.assert_close(
        keylist_npu.cpu(),
        expected_key,
        rtol=0,
        atol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [512])
def test_readKVCache_without_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_read_kv_cache_registered()

    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = []
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.992, 5)
        slotmap.append(i + r)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    expected_key = golden_read_kv_cache(
        key_cache_list=cache_table,
        key_table=keylist,
        slotmap=slotmap,
        block_size=block_size,
    )

    slot_mapping_npu, slot_mapping_list = make_slot_mapping_tensors(slotmap)

    keylist_npu = torch.empty_like(keylist).npu()
    cache_table_npu = cache_table.clone().npu()

    epoch = 100
    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(epoch):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.read_kv_cache_pre(
            slot_mapping_npu,
            slot_mapping_list,
            block_size,
        )

        torch.ops._C_ascend.read_kv_cache(
            cache_table_npu,
            keylist_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )

    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    torch.testing.assert_close(
        keylist_npu.cpu(),
        expected_key,
        rtol=0,
        atol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [64])
def test_readKVCache_without_ascending(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_read_kv_cache_registered()

    keylist = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size),
        dtype=torch.int8,
    )

    slotmap = list(range(num_tokens))
    random.shuffle(slotmap)

    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    cache_table = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )

    expected_key = golden_read_kv_cache(
        key_cache_list=cache_table,
        key_table=keylist,
        slotmap=slotmap,
        block_size=block_size,
    )

    slot_mapping_npu, slot_mapping_list = make_slot_mapping_tensors(slotmap)

    keylist_npu = torch.empty_like(keylist).npu()
    cache_table_npu = cache_table.clone().npu()

    group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.read_kv_cache_pre(
        slot_mapping_npu,
        slot_mapping_list,
        block_size,
    )

    torch.ops._C_ascend.read_kv_cache(
        cache_table_npu,
        keylist_npu,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )

    torch.npu.synchronize()

    torch.testing.assert_close(
        keylist_npu.cpu(),
        expected_key,
        rtol=0,
        atol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [512])
def test_siso_with_continuous(
    num_tokens,
    num_head,
    block_size,
    num_blocks,
    head_size,
):
    key_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size),
        dtype=torch.int8,
    )

    key_cache_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )

    slot_list = []
    for i in range(0, num_tokens):
        slot_list.append(i)

    assert num_tokens == len(slot_list)

    max_slot = max(slot_list)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    expected_cache = golden_store_kv_cache(
        key_cache_list=key_cache_cpu,
        key_table=key_cpu,
        slotmap=slot_list,
        block_size=block_size,
    )

    slot_list_np = np.array(slot_list, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key = key_cpu.npu()
    key_cache = key_cache_cpu.clone().npu()

    epoch = 100
    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(epoch):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)

    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    torch.testing.assert_close(
        key_cache.cpu(),
        expected_cache,
        atol=0,
        rtol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("num_tokens", [32 * 1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [64])
def test_siso_without_continuous(
    num_tokens,
    num_head,
    block_size,
    num_blocks,
    head_size,
):
    key_cpu = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    key_cache_cpu = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    slot_list = []
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.85, 5)
        slot_list.append(i + r)

    assert num_tokens == len(slot_list)

    max_slot = max(slot_list)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    expected_cache = golden_store_kv_cache(
        key_cache_list=key_cache_cpu,
        key_table=key_cpu,
        slotmap=slot_list,
        block_size=block_size,
    )

    slot_list_np = np.array(slot_list, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key = key_cpu.npu()
    key_cache = key_cache_cpu.clone().npu()

    epoch = 100
    torch.npu.synchronize()
    start = time.perf_counter()

    for _ in range(epoch):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)

    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")

    torch.testing.assert_close(
        key_cache.cpu(),
        expected_cache,
        atol=0,
        rtol=0,
    )

    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()