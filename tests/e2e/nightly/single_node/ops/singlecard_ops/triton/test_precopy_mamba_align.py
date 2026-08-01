# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker import mamba_utils
from vllm.v1.worker.mamba_utils import precopy_mamba_align_fused_kernel

import vllm_ascend.patch.worker.patch_mamba_utils  # noqa: F401
from vllm_ascend.ops.triton.mamba.copy import _copy_mamba_state_block
from vllm_ascend.utils import is_310p


NUM_LAYERS = 3
CONV_WIDTH = 4
CONV_DIM = 96
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or is_310p(),
    reason="requires a non-310P NPU Triton path",
)


def _build_sd_states(num_blocks: int, device: torch.device):
    convs, ssms = [], []
    for _ in range(NUM_LAYERS):
        convs.append(
            torch.randn(
                num_blocks,
                CONV_WIDTH,
                CONV_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
        )
        ssms.append(
            torch.randn(
                num_blocks,
                *SSM_SHAPE,
                dtype=torch.float32,
                device=device,
            )
        )
    return convs, ssms


def _build_sd_meta(
    convs: list[torch.Tensor],
    ssms: list[torch.Tensor],
    device: torch.device,
):
    num_states = NUM_LAYERS * 2
    base = torch.zeros(num_states, dtype=torch.int64, device=device)
    block_stride = torch.zeros(num_states, dtype=torch.int64, device=device)
    elem_size = torch.zeros(num_states, dtype=torch.int32, device=device)
    inner_size = torch.zeros(num_states, dtype=torch.int64, device=device)
    conv_width = torch.zeros(num_states, dtype=torch.int32, device=device)
    group_idx = torch.zeros(num_states, dtype=torch.int32, device=device)
    dim_row_count = torch.zeros(num_states, dtype=torch.int32, device=device)
    dim_row_stride = torch.zeros(num_states, dtype=torch.int64, device=device)

    state_idx = 0
    for conv, ssm in zip(convs, ssms):
        base[state_idx] = conv.data_ptr()
        block_stride[state_idx] = conv.stride(0) * conv.element_size()
        elem_size[state_idx] = conv.element_size()
        inner_size[state_idx] = conv.stride(1)
        conv_width[state_idx] = conv.size(1)
        state_idx += 1

        base[state_idx] = ssm.data_ptr()
        block_stride[state_idx] = ssm.stride(0) * ssm.element_size()
        elem_size[state_idx] = ssm.element_size()
        inner_size[state_idx] = ssm[0].numel()
        state_idx += 1

    return (
        base,
        block_stride,
        elem_size,
        inner_size,
        conv_width,
        group_idx,
        dim_row_count,
        dim_row_stride,
    )


def _sd_reference(
    convs: list[torch.Tensor],
    ssms: list[torch.Tensor],
    block_table: torch.Tensor,
    src_col: torch.Tensor,
    dst_col: torch.Tensor,
    token_bias: torch.Tensor,
):
    conv_ref = [conv.clone() for conv in convs]
    ssm_ref = [ssm.clone() for ssm in ssms]
    conv_before = [conv.clone() for conv in convs]
    ssm_before = [ssm.clone() for ssm in ssms]

    for req_idx in range(block_table.shape[0]):
        source = int(src_col[req_idx])
        dest = int(dst_col[req_idx])
        bias = int(token_bias[req_idx])
        if source < 0 or source == dest:
            continue

        source_block = int(block_table[req_idx, source])
        dest_block = int(block_table[req_idx, dest])
        temporal_source_block = int(block_table[req_idx, source + bias])
        for layer_idx in range(NUM_LAYERS):
            conv_ref[layer_idx][dest_block, : CONV_WIDTH - bias] = conv_before[
                layer_idx
            ][source_block, bias:]
            ssm_ref[layer_idx][dest_block] = ssm_before[layer_idx][
                temporal_source_block
            ]

    return conv_ref, ssm_ref


@pytest.mark.parametrize("num_reqs", [1, 4, 16])
@pytest.mark.parametrize("token_bias", [0, 1, 2])
def test_precopy_matches_v1_copy_specs(num_reqs: int, token_bias: int):
    assert mamba_utils._copy_mamba_state_block is _copy_mamba_state_block

    device = torch.device("npu:0")
    torch.manual_seed(0)
    num_blocks = num_reqs * MAX_COLS + 1
    block_table = torch.empty(num_reqs, MAX_COLS, dtype=torch.int32, device=device)
    for req_idx in range(num_reqs):
        block_table[req_idx] = torch.arange(
            1 + req_idx * MAX_COLS,
            1 + (req_idx + 1) * MAX_COLS,
            dtype=torch.int32,
            device=device,
        )

    src_col = torch.full((num_reqs,), 1, dtype=torch.int32, device=device)
    dst_col = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    bias = torch.full((num_reqs,), token_bias, dtype=torch.int32, device=device)
    src_col[0] = -1
    if num_reqs > 1:
        dst_col[1] = 1

    convs, ssms = _build_sd_states(num_blocks, device)
    conv_ref, ssm_ref = _sd_reference(
        convs,
        ssms,
        block_table.cpu(),
        src_col.cpu(),
        dst_col.cpu(),
        bias.cpu(),
    )
    (
        base,
        block_stride,
        elem_size,
        inner_size,
        conv_width,
        group_idx,
        dim_row_count,
        dim_row_stride,
    ) = _build_sd_meta(convs, ssms, device)

    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.int64, device=device
    )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    precopy_mamba_align_fused_kernel[(num_reqs, NUM_LAYERS * 2)](
        dst_col,
        src_col,
        bias,
        block_table_ptrs,
        block_table.stride(0),
        base,
        block_stride,
        elem_size,
        inner_size,
        conv_width,
        group_idx,
        dim_row_count,
        dim_row_stride,
        idx_mapping,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=False,
    )
    torch.accelerator.synchronize()

    for conv, expected in zip(convs, conv_ref):
        torch.testing.assert_close(conv, expected, rtol=0, atol=0)
    for ssm, expected in zip(ssms, ssm_ref):
        torch.testing.assert_close(ssm, expected, rtol=0, atol=0)
