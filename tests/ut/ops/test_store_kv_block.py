import numpy as np
import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C
import gc
import time
torch.set_printoptions(threshold=np.inf)
import os
import random

def random_with_zero_prob(zero_prob, index):
    if not 0 <= zero_prob <= 1:
        raise ValueError("the probablity must be in [0, 1]")

    if index <= 0:
        raise ValueError("index must be > 0")

    if random.random() < zero_prob:
        return 0

    return random.randint(1, index)

def assert_store_kv_block_registered():
    assert hasattr(torch.ops, "_C_ascend"), "torch.ops._C_ascend is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "store_kv_block_pre",
    ), "torch.ops._C_ascend.store_kv_block_pre is not registered"
    assert hasattr(
        torch.ops._C_ascend,
        "store_kv_block",
    ), "torch.ops._C_ascend.store_kv_block is not registered"
    
def golden_store_kv_cache(keylist, cache_table, slotmap, block_size):
    expected_cache = cache_table.clone()

    for token_idx, slot in enumerate(slotmap):
        if slot < 0:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        expected_cache[block_idx, block_offset, :, :] = keylist[token_idx, :, :]

    return expected_cache


@pytest.mark.parametrize("num_tokens", [32*1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [128])
def test_storeKvCacheS_with_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    # keylist = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_tokens,num_head ,head_size),
    #     dtype=torch.int8,
    # )
    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )
    slotmap = []
    for i in range(0, num_tokens):
        slotmap.append(i)
    # print(1)
    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    # cache_table = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_blocks, block_size, num_head, head_size),
    #     dtype=torch.int8,
    # )
    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    expected_cache = golden_store_kv_cache(
        keylist=keylist,
        cache_table=cache_table,
        slotmap=slotmap,
        block_size=block_size,
    )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()
    # print(2)
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()
    # print(3)
    epoch = 100
    for _ in range(epoch):
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
        slot_mapping_npu,
        slot_mapping_list,
        block_size,
    )
        # group_len_cpu = group_len.cpu()
        # group_key_idx_cpu = group_key_idx.cpu()
        # group_key_cache_idx_cpu = group_key_cache_idx.cpu()
        torch.ops._C_ascend.store_kv_block(
            keylist_npu,
            cache_table_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )

    torch.testing.assert_close(
        cache_table_npu.cpu(),
        expected_cache,
        rtol=0,
        atol=0,
    )
    
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@pytest.mark.parametrize("num_tokens", [32*1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [512])
def test_storeKvCacheS_without_continuous(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    # keylist = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_tokens,num_head ,head_size),
    #     dtype=torch.int8,
    # )
    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )
    slotmap = []
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.992,5)
        slotmap.append(i+r)
    # print(1)
    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    # cache_table = torch.randint(
    #     low=0,
    #     high=128,
    #     size=(num_blocks, block_size, num_head, head_size),
    #     dtype=torch.int8,
    # )
    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )
    # expected_cache = golden_store_kv_cache(
    #     keylist=keylist,
    #     cache_table=cache_table,
    #     slotmap=slotmap,
    #     block_size=block_size,
    # )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()
    # print(2)
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()
    # print(3)
    epoch = 100
    start = time.perf_counter()
    for _ in range(epoch):
        
        group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
        slot_mapping_npu,
        slot_mapping_list,
        block_size,
        )



        torch.ops._C_ascend.store_kv_block(
            keylist_npu,
            cache_table_npu,
            group_len,
            group_key_idx,
            group_key_cache_idx,
            block_size,
        )
    #     torch.testing.assert_close(
    #     cache_table_npu.cpu(),
    #     expected_cache,
    #     rtol=0,
    #     atol=0,
    # )
    end = time.perf_counter()
    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

@pytest.mark.parametrize("num_tokens", [32*1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [64])
def test_storeKvCacheS_without_ascending(
    num_tokens,
    num_head,
    num_blocks,
    head_size,
    block_size,
):
    assert_store_kv_block_registered()

    keylist = torch.randint(
        low=0,
        high=128,
        size=(num_tokens,num_head ,head_size),
        dtype=torch.int8,
    )

    slotmap = list(range(num_tokens))
    random.shuffle(slotmap)
    # print(1)
    max_slot = max(slotmap)
    total_cache_slots = num_blocks * block_size
    assert max_slot < total_cache_slots

    cache_table = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )

    expected_cache = golden_store_kv_cache(
        keylist=keylist,
        cache_table=cache_table,
        slotmap=slotmap,
        block_size=block_size,
    )

    slotmap_np = np.array(slotmap, dtype=np.int32)
    slot_mapping_npu = torch.from_numpy(slotmap_np).to(torch.int32).npu()
    slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
    slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)
    torch.npu.synchronize()

    slot_mapping_list = slot_mapping_cpu.tolist()

    keylist_npu = keylist.npu()
    cache_table_npu = cache_table.clone().npu()
    
        
    group_len, group_key_idx, group_key_cache_idx = torch.ops._C_ascend.store_kv_block_pre(
    slot_mapping_npu,
    slot_mapping_list,
    block_size,
    )


    # group_len_cpu = group_len.cpu()
    # group_key_idx_cpu = group_key_idx.cpu()
    # group_key_cache_idx_cpu = group_key_cache_idx.cpu()


    torch.ops._C_ascend.store_kv_block(
        keylist_npu,
        cache_table_npu,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size,
    )
    torch.testing.assert_close(
        cache_table_npu.cpu(),
        expected_cache,
        rtol=0,
        atol=0,
    )
        
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
   
   
@pytest.mark.parametrize('num_tokens', [32*1024])#6398
@pytest.mark.parametrize('num_head', [1])#512
@pytest.mark.parametrize('block_size', [128])#128
@pytest.mark.parametrize('num_blocks', [1773])#1599
@pytest.mark.parametrize('head_size', [512])
def test_siso_with_continuous(num_tokens, num_head, block_size, num_blocks, head_size):
    key_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_tokens, num_head, head_size),
        dtype=torch.int8,
    )
    
    # key_cpu = torch.rand((num_tokens, num_head, head_size), dtype=torch.float16)
    key = key_cpu.npu() 
    key_cache_cpu = torch.randint(
        low=0,
        high=128,
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.int8,
    )
    # key_cache_cpu = torch.rand((num_blocks, block_size, num_head, head_size), dtype=torch.float16)
    key_cache = key_cache_cpu.clone().npu()

    # slot_list = []
    slot_list=[]
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.992,5)
        slot_list.append(i+r)

    assert num_tokens == len(slot_list)
    slot_list_np = np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()

    key_expect = golden_store_kv_cache(
        keylist=key_cpu,
        cache_table=key_cache_cpu,
        slotmap=slot_list,
        block_size=block_size,
    )

    epoch = 100
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(epoch):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    torch.npu.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")
    # prof.stop()
    # end = time.perf_counter()
    # avg_ms = (end - start) / N * 1000
    # print(f"python 耗时: {avg_ms:.4f} ms")
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache.cpu(), atol=0, rtol=0)
    # torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )
   
@pytest.mark.parametrize("num_tokens", [32*1024])
@pytest.mark.parametrize("num_head", [1])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_blocks", [1773])
@pytest.mark.parametrize("head_size", [64])
def test_siso_without_continuous(num_tokens, num_head, block_size, num_blocks,head_size):
    key = torch.rand((num_tokens, num_head, head_size), dtype=torch.float16).npu()
    key_cache = torch.rand((num_blocks, block_size, num_head, head_size), dtype=torch.float16).npu()
    slot_list=[]
    r = 0
    for i in range(0, num_tokens):
        r = r + random_with_zero_prob(0.85,5)
        slot_list.append(i+r)
    
    assert num_tokens==len(slot_list)
    slot_list_np= np.array(slot_list)
    slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
    key_expect = golden_store_kv_cache(key, key_cache, slot_mapping_npu,block_size)
    epoch = 100
    start = time.perf_counter()
    for _ in range(epoch):
        torch_npu._npu_reshape_and_cache_siso(key, key_cache, slot_mapping_npu)
    end = time.perf_counter()
    avg_ms = (end - start) / epoch * 1000
    print(f"python 耗时: {avg_ms:.4f} ms")
    # prof.stop()
    # end = time.perf_counter()
    # avg_ms = (end - start) / N * 1000
    # print(f"python 耗时: {avg_ms:.4f} ms")
    # torch.ops._C_ascend.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.testing.assert_close(key_expect, key_cache, atol=0.001, rtol=0.1 )
        
# @pytest.mark.parametrize('num_tokens', [32*1024])#6398
# @pytest.mark.parametrize('num_head', [1])#512
# @pytest.mark.parametrize('block_size', [128])#128
# @pytest.mark.parametrize('num_blocks', [1773])#1599
# @pytest.mark.parametrize('count', [1])
# def test_myops(num_tokens, num_head, block_size, num_blocks,count):
#     print("PYTEST ASCEND_CUSTOM_OPP_PATH =", os.environ.get("ASCEND_CUSTOM_OPP_PATH"), flush=True)
#     print("PYTEST ASCEND_OPP_PATH =", os.environ.get("ASCEND_OPP_PATH"), flush=True)
#     print("PYTEST LD_LIBRARY_PATH has custom opapi:",
#         "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib"
#         in os.environ.get("LD_LIBRARY_PATH", ""),
#         flush=True)

#     # experimental_config = torch_npu.profiler._ExperimentalConfig(
#     #     export_type=[
#     #         torch_npu.profiler.ExportType.Text,
#     #         torch_npu.profiler.ExportType.Db
#     #         ],
#     #     profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
#     #     msprof_tx=False,
#     #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
#     #     l2_cache=False,
#     #     op_attr=False,
#     #     data_simplification=False,
#     #     record_op_args=False,
#     #     gc_detect_threshold=None,
#     #     host_sys=[
#     #         torch_npu.profiler.HostSystem.CPU,
#     #         torch_npu.profiler.HostSystem.MEM],
#     #     sys_io=False,
#     #     sys_interconnection=False
#     # )

#     # prof = torch_npu.profiler.profile(
#     #     activities=[
#     #         torch_npu.profiler.ProfilerActivity.CPU,
#     #         torch_npu.profiler.ProfilerActivity.NPU
#     #         ],
#     #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
#     #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/home/z00893411/ops/script/ls"),
#     #     record_shapes=True,
#     #     profile_memory=False,
#     #     with_stack=True,
#     #     with_modules=False,
#     #     with_flops=False,
#     #     experimental_config=experimental_config)

#     head_size =64
#     # key_cache = torch.rand((num_blocks, block_size, num_head,head_size), dtype=torch.float16)
#     key_cache = torch.randint(low=0,high=128,size=(num_blocks, block_size, num_head,head_size), dtype=torch.int8 )
#     key_cache_npu = key_cache.npu()

#     # key = torch.rand((num_tokens, num_head, head_size), dtype=torch.float16)
#     # while count >0: 
#     # num_tokens=num_tokens+200*zt_i
#     slot_list=[]
#     for i in range(0,num_tokens):
#         # slot_list.append([6+i])
#         slot_list.append(2+i)
#     # slot_list=[29572, 29236, 29406, 29542, 28872, 28023, 29098, 29184]
#     # slot_list=[29572 ,29236 , 29184,  29406, 29542, 28872, 28023, 29098]
#     # slot_list.append(-1)
#     # slot_list.append(-1)
#     #assert num_tokens==len(slot_list)
    
#     slot_list_np= np.array(slot_list)
#     slot_mapping_npu = torch.from_numpy(slot_list_np).to(torch.int32).npu()
#     #slot_mapping_cpu = slot_mapping_npu.to("cpu",non_blocking=True)
#     # num_draft_tensor = slot_mapping_npu.to("cpu", non_blocking=True)
#     slot_mapping_cpu = torch.empty_like(slot_mapping_npu, device="cpu").pin_memory()
#     slot_mapping_cpu.copy_(slot_mapping_npu, non_blocking=True)

#     # key = torch.rand((num_tokens, num_head,head_size), dtype=torch.float16)
#     key = torch.randint(low=0,high=128,size=(num_tokens,head_size), dtype=torch.int8 )
#     key_npu = key.npu()
#     # key_expect = cal_slot(key_npu, key_cache_npu, slot_list_np, block_size)



#     time.sleep(0.1)
        
#     # print(f"    test_myops            ")
#     # for zt_i in range(100):
#     # 调用你的算子


#     slot_mapping_list = slot_mapping_cpu.tolist() 
#     warm_up=0
#     for _ in range(warm_up):

#         group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.store_kv_block_pre( slot_mapping_npu, slot_mapping_list, block_size)
#         torch.ops._C_ascend.store_kv_block(key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size)
#     N = 100
#     #group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.cache_by_group_pre( slot_mapping_npu, slot_mapping_list, block_size)
#     start = time.perf_counter()
#     for _ in range(N):

#         # group_list_np= np.array(group_list)
#         # prof.start()
#         group_len,group_key_idx,group_key_cache_idx=torch.ops._C_ascend.store_kv_block_pre( slot_mapping_npu, slot_mapping_list, block_size)
#         torch.ops._C_ascend.store_kv_block(key_npu, key_cache_npu, group_len, group_key_idx, group_key_cache_idx, block_size)
#         # prof.stop()
#     end = time.perf_counter()
#     avg_ms = (end - start) / N * 1000
#     print(f"python 耗时: {avg_ms:.4f} ms")

#     # torch.equal(key_expect, key_cache_npu)   
#     # torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1)
#     #torch.testing.assert_close(key_expect, key_cache_npu, atol=0.001, rtol=0.1 )
#     # # 转 MB 方便看
#     # mb_size = byte_size / 1024 / 1024
#     # print(f"    groupInfo Tensor 显存大小：{mb_size:.4f} MB")
#     # # del groupInfo
#     # # torch.npu.empty_cache()
#     # # # 同步，保证算子执行完
#     # torch.npu.synchronize()

#     # # 每 100 次打印一次显存
#     # if zt_i % 10 == 0:
#     #     allocated = torch.npu.memory_allocated() / 1024 / 1024
#     #     reserved = torch.npu.memory_reserved() / 1024 / 1024
#     #         print(f"【第 {zt_i} 次】 allocated: {allocated:.4f} MB | reserved: {reserved:.4f} MB")
        
#     gc.collect()
#     torch.npu.empty_cache()
#     torch.npu.reset_peak_memory_stats()
#         # count=count-1