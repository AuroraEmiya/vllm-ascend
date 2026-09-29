# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce DFlash split-block DCP slots on an Ascend NPU."""

import argparse
from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

PHYSICAL_BLOCK_SIZE = 384
KERNEL_BLOCK_SIZE = 128
CP_SIZE = 16
CP_INTERLEAVE = 384
PHYSICAL_BLOCK_ID = 7
POSITIONS = (2047, 2048)


def expected_slot(position: int, rank: int) -> int:
    virtual_block, virtual_offset = divmod(position, PHYSICAL_BLOCK_SIZE * CP_SIZE)
    stripe, remainder = divmod(virtual_offset, CP_INTERLEAVE)
    if stripe % CP_SIZE != rank:
        return PAD_SLOT_ID
    local_position = virtual_block * PHYSICAL_BLOCK_SIZE + (stripe // CP_SIZE) * CP_INTERLEAVE + remainder
    kernel_index, kernel_offset = divmod(local_position, KERNEL_BLOCK_SIZE)
    kernel_id = PHYSICAL_BLOCK_ID * (PHYSICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE) + kernel_index
    return kernel_id * KERNEL_BLOCK_SIZE + kernel_offset


def make_inputs(rank: int):
    device = "npu"
    max_num_reqs = 2
    max_num_tokens = 8
    num_speculative_steps = 1
    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        positions=torch.tensor(POSITIONS, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0, 0], dtype=torch.int32, device=device),
    )
    kernel_ids = [
        PHYSICAL_BLOCK_ID * (PHYSICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE) + i
        for i in range(PHYSICAL_BLOCK_SIZE // KERNEL_BLOCK_SIZE)
    ]
    block_table = torch.zeros((max_num_reqs, 64), dtype=torch.int32, device=device)
    block_table[0, : len(kernel_ids)] = torch.tensor(kernel_ids, dtype=torch.int32, device=device)
    query_slots = torch.full((max_num_tokens,), PAD_SLOT_ID, dtype=torch.int32, device=device)
    context_slots = torch.full((max_num_tokens,), PAD_SLOT_ID, dtype=torch.int32, device=device)
    args = (
        input_buffers,
        query_slots,
        torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        context_slots,
        torch.zeros(max_num_reqs * num_speculative_steps, dtype=torch.int64, device=device),
        torch.zeros(max_num_reqs * num_speculative_steps, dtype=torch.int64, device=device),
        torch.zeros(max_num_reqs * num_speculative_steps, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.float32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int64, device=device),
        input_batch,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([101, 102], dtype=torch.int64, device=device),
        torch.tensor([201, 202], dtype=torch.int32, device=device),
        torch.tensor([0.7, 0.7], dtype=torch.float32, device=device),
        torch.tensor([11, 12], dtype=torch.int64, device=device),
        block_table,
        KERNEL_BLOCK_SIZE,
        rank,
        CP_SIZE,
        CP_INTERLEAVE,
        999,
        2,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        8192,
        False,
    )
    return args, context_slots, query_slots, kernel_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", required=True, choices=("ascend", "upstream"))
    implementation = parser.parse_args().implementation
    if implementation == "ascend":
        from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton as prepare
    else:
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs as prepare

    assert [expected_slot(p, 5) for p in (2047, 2048, 2049, 2050)] == [2815, 2816, 2817, 2818]
    assert [expected_slot(p, 0) for p in (2047, 2048, 2049, 2050)] == [PAD_SLOT_ID] * 4

    passed = True
    for rank in (5, 0):
        args, context_slots, query_slots, kernel_ids = make_inputs(rank)
        if implementation == "ascend":
            prepare(*args, physical_block_size=PHYSICAL_BLOCK_SIZE)
        else:
            prepare(*args)
        torch.npu.synchronize()
        actual = (context_slots[:2].cpu().tolist(), query_slots[:2].cpu().tolist())
        expected = (
            [expected_slot(p, rank) for p in POSITIONS],
            [expected_slot(p, rank) for p in (2049, 2050)],
        )
        rank_passed = actual == expected
        passed &= rank_passed
        print(
            f"rank={rank} kernel_ids={kernel_ids} actual={actual} "
            f"expected={expected} {'PASS' if rank_passed else 'FAIL'}"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
