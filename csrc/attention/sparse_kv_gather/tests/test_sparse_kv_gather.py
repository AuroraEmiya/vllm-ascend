import os
import time

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op
```

msprof op --output=./result_dir --kernel-name="$2" --aic-metrics=MemoryDetail,Occupancy,PipeUtilization,Roofline python3 $1
```

enabled = enable_custom_op()
assert enabled, "vllm-ascend custom ops are not enabled"

torch.npu.set_device(0)


BLOCK_SIZE = 128
CTKV_DIM = 512
KPE_DIM = 64


def ref_gather(
    paged_ctkv,
    paged_kpe,
    block_table,
    topk_indices,
    cur_pos,
):
    num_actual, topk_n = topk_indices.shape

    out_ctkv = torch.zeros(
        (num_actual, topk_n, CTKV_DIM),
        dtype=paged_ctkv.dtype,
    )
    out_kpe = torch.zeros(
        (num_actual, topk_n, KPE_DIM),
        dtype=paged_kpe.dtype,
    )

    num_blocks = paged_ctkv.shape[0]
    max_blocks = block_table.shape[1]

    for query_idx in range(num_actual):
        current_pos = int(cur_pos[query_idx].item())

        for topk_idx in range(topk_n):
            logical_pos = int(
                topk_indices[query_idx, topk_idx].item()
            )

            if logical_pos < 0:
                continue

            if logical_pos == current_pos:
                continue

            logical_block = logical_pos // BLOCK_SIZE
            block_offset = logical_pos % BLOCK_SIZE

            if logical_block >= max_blocks:
                continue

            physical_block = int(
                block_table[
                    query_idx,
                    logical_block,
                ].item()
            )

            if physical_block < 0:
                continue

            if physical_block >= num_blocks:
                continue

            out_ctkv[query_idx, topk_idx] = paged_ctkv[
                physical_block,
                block_offset,
                0,
                :,
            ]

            out_kpe[query_idx, topk_idx] = paged_kpe[
                physical_block,
                block_offset,
                0,
                :,
            ]

    return out_ctkv, out_kpe


def make_input(
    dtype=torch.float16,
    index_dtype=torch.int32,
):
    paged_ctkv = torch.randn(
        (16, BLOCK_SIZE, 1, CTKV_DIM),
        dtype=dtype,
        device="npu",
    )

    paged_kpe = torch.randn(
        (16, BLOCK_SIZE, 1, KPE_DIM),
        dtype=dtype,
        device="npu",
    )

    block_table = torch.tensor(
        [
            [3, 1, 8, 5],
            [6, 2, 9, 4],
        ],
        dtype=index_dtype,
        device="npu",
    )

    topk_indices = torch.tensor(
        [
            [0, 1, 2, 3, 127, 128, 129, 130],
            [5, 6, 126, 127, 128, 129, 130, 131],
        ],
        dtype=index_dtype,
        device="npu",
    )

    cur_pos = torch.tensor(
        [130, 255],
        dtype=index_dtype,
        device="npu",
    )

    return (
        paged_ctkv,
        paged_kpe,
        block_table,
        topk_indices,
        cur_pos,
    )


def call_op(
    paged_ctkv,
    paged_kpe,
    block_table,
    topk_indices,
    cur_pos,
):
    return torch.ops._C_ascend.npu_sparse_kv_gather.default(
        paged_ctkv,
        paged_kpe,
        block_table,
        topk_indices,
        cur_pos,
        BLOCK_SIZE,
    )


@pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="NPU unavailable",
)
class TestSparseKvGather:

    def test_basic_shape(self):
        args = make_input()

        out_ctkv, out_kpe = call_op(*args)

        assert out_ctkv.shape == (
            2,
            8,
            CTKV_DIM,
        )

        assert out_kpe.shape == (
            2,
            8,
            KPE_DIM,
        )

    def test_match_reference(self):
        args = make_input()

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        ref_ctkv, ref_kpe = ref_gather(
            args[0].cpu(),
            args[1].cpu(),
            args[2].cpu(),
            args[3].cpu(),
            args[4].cpu(),
        )

        assert torch.equal(
            out_ctkv.cpu(),
            ref_ctkv,
        )

        assert torch.equal(
            out_kpe.cpu(),
            ref_kpe,
        )

    def test_block_boundary(self):
        paged_ctkv = torch.zeros(
            (2, BLOCK_SIZE, 1, CTKV_DIM),
            dtype=torch.float16,
            device="npu",
        )

        paged_kpe = torch.zeros(
            (2, BLOCK_SIZE, 1, KPE_DIM),
            dtype=torch.float16,
            device="npu",
        )

        for block_idx in range(2):
            for block_offset in range(BLOCK_SIZE):
                physical_token = (
                    block_idx * BLOCK_SIZE
                    + block_offset
                )

                paged_ctkv[
                    block_idx,
                    block_offset,
                    0,
                    0,
                ] = physical_token

                paged_kpe[
                    block_idx,
                    block_offset,
                    0,
                    0,
                ] = physical_token

        block_table = torch.tensor(
            [[0, 1]],
            dtype=torch.int32,
            device="npu",
        )

        topk_indices = torch.tensor(
            [[127, 128]],
            dtype=torch.int32,
            device="npu",
        )

        cur_pos = torch.tensor(
            [-1],
            dtype=torch.int32,
            device="npu",
        )

        out_ctkv, out_kpe = call_op(
            paged_ctkv,
            paged_kpe,
            block_table,
            topk_indices,
            cur_pos,
        )

        torch.npu.synchronize()

        assert out_ctkv[0, 0, 0].item() == 127
        assert out_ctkv[0, 1, 0].item() == 128
        assert out_kpe[0, 0, 0].item() == 127
        assert out_kpe[0, 1, 0].item() == 128

    def test_cur_pos_mask(self):
        args = make_input()

        args[4][0] = args[3][0, 1]

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert torch.equal(
            out_ctkv[0, 1],
            torch.zeros_like(out_ctkv[0, 1]),
        )

        assert torch.equal(
            out_kpe[0, 1],
            torch.zeros_like(out_kpe[0, 1]),
        )

        assert not torch.equal(
            out_ctkv[0, 0],
            torch.zeros_like(out_ctkv[0, 0]),
        )

    def test_negative_index(self):
        args = make_input()

        args[3][0, 0] = -1

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert torch.equal(
            out_ctkv[0, 0],
            torch.zeros_like(out_ctkv[0, 0]),
        )

        assert torch.equal(
            out_kpe[0, 0],
            torch.zeros_like(out_kpe[0, 0]),
        )

    def test_all_invalid(self):
        args = make_input()

        args[3][:] = -1

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert torch.equal(
            out_ctkv,
            torch.zeros_like(out_ctkv),
        )

        assert torch.equal(
            out_kpe,
            torch.zeros_like(out_kpe),
        )

    def test_dtype_fp16_int32(self):
        args = make_input(
            dtype=torch.float16,
            index_dtype=torch.int32,
        )

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert args[2].dtype == torch.int32
        assert args[3].dtype == torch.int32
        assert args[4].dtype == torch.int32

        assert out_ctkv.dtype == torch.float16
        assert out_kpe.dtype == torch.float16

    def test_dtype_bf16_int32(self):
        args = make_input(
            dtype=torch.bfloat16,
            index_dtype=torch.int32,
        )

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert args[2].dtype == torch.int32
        assert args[3].dtype == torch.int32
        assert args[4].dtype == torch.int32

        assert out_ctkv.dtype == torch.bfloat16
        assert out_kpe.dtype == torch.bfloat16

    def test_dtype_bf16_int64(self):
        args = make_input(
            dtype=torch.bfloat16,
            index_dtype=torch.int64,
        )

        out_ctkv, out_kpe = call_op(*args)

        torch.npu.synchronize()

        assert args[2].dtype == torch.int64
        assert args[3].dtype == torch.int64
        assert args[4].dtype == torch.int64

        assert out_ctkv.dtype == torch.bfloat16
        assert out_kpe.dtype == torch.bfloat16

    def test_perf_real_shape_bf16_int32(self):
        num_blocks = 9840
        batch = 16
        max_blocks = 547
        topk_n = 2048

        warmup = int(
            os.getenv(
                "SKG_WARMUP",
                "20",
            )
        )

        repeat = int(
            os.getenv(
                "SKG_REPEAT",
                "100",
            )
        )

        paged_ctkv = torch.empty(
            (
                num_blocks,
                BLOCK_SIZE,
                1,
                CTKV_DIM,
            ),
            dtype=torch.bfloat16,
            device="npu",
        )

        paged_kpe = torch.empty(
            (
                num_blocks,
                BLOCK_SIZE,
                1,
                KPE_DIM,
            ),
            dtype=torch.bfloat16,
            device="npu",
        )

        cpu_generator = torch.Generator(
            device="cpu"
        )
        cpu_generator.manual_seed(20260723)

        block_table_cpu = torch.randint(
            low=0,
            high=num_blocks,
            size=(
                batch,
                max_blocks,
            ),
            dtype=torch.int32,
            generator=cpu_generator,
        )

        topk_indices_cpu = torch.randint(
            low=0,
            high=max_blocks * BLOCK_SIZE,
            size=(
                batch,
                topk_n,
            ),
            dtype=torch.int32,
            generator=cpu_generator,
        )

        block_table = block_table_cpu.to(
            device="npu"
        )

        topk_indices = topk_indices_cpu.to(
            device="npu"
        )

        cur_pos = torch.full(
            (batch,),
            -1,
            dtype=torch.int32,
            device="npu",
        )

        assert paged_ctkv.shape == (
            9840,
            128,
            1,
            512,
        )

        assert paged_kpe.shape == (
            9840,
            128,
            1,
            64,
        )

        assert block_table.shape == (
            16,
            547,
        )

        assert topk_indices.shape == (
            16,
            2048,
        )

        assert cur_pos.shape == (
            16,
        )

        assert paged_ctkv.dtype == torch.bfloat16
        assert paged_kpe.dtype == torch.bfloat16
        assert block_table.dtype == torch.int32
        assert topk_indices.dtype == torch.int32
        assert cur_pos.dtype == torch.int32

        for _ in range(warmup):
            out_ctkv, out_kpe = call_op(
                paged_ctkv,
                paged_kpe,
                block_table,
                topk_indices,
                cur_pos,
            )

        torch.npu.synchronize()

        start_time = time.perf_counter()

        for _ in range(repeat):
            out_ctkv, out_kpe = call_op(
                paged_ctkv,
                paged_kpe,
                block_table,
                topk_indices,
                cur_pos,
            )

        torch.npu.synchronize()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        average_us = (
            elapsed_seconds
            * 1_000_000.0
            / repeat
        )

        assert out_ctkv.shape == (
            batch,
            topk_n,
            CTKV_DIM,
        )

        assert out_kpe.shape == (
            batch,
            topk_n,
            KPE_DIM,
        )

        assert out_ctkv.dtype == torch.bfloat16
        assert out_kpe.dtype == torch.bfloat16

        print()
        print(
            "========== SparseKvGather real-shape benchmark =========="
        )
        print(
            f"paged_ctkv   : {tuple(paged_ctkv.shape)}"
        )
        print(
            f"paged_kpe    : {tuple(paged_kpe.shape)}"
        )
        print(
            f"block_table  : {tuple(block_table.shape)}"
        )
        print(
            f"topk_indices : {tuple(topk_indices.shape)}"
        )
        print(
            f"cur_pos      : {tuple(cur_pos.shape)}"
        )
        print(
            f"data dtype   : {paged_ctkv.dtype}"
        )
        print(
            f"index dtype  : {block_table.dtype}"
        )
        print(
            f"warmup       : {warmup}"
        )
        print(
            f"repeat       : {repeat}"
        )
        print(
            f"average      : {average_us:.3f} us"
        )


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
        ]
    )