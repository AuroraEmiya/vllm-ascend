# Adapt from:
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_mamba_state_block(
    state_idx,
    bt_row_idx,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_ptr = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_ptr + bt_row_idx * block_table_stride_req

    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride
    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        per_row_bytes = (conv_width - token_bias).to(tl.int64) * state_elem_size
        bias_bytes = token_bias.to(tl.int64) * state_elem_size
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)

        for d in range(0, dim_rows):
            row_src_addr = src_block_addr + d * row_stride + bias_bytes
            row_dst_addr = dst_addr + d * row_stride

            # Keep the address-to-pointer cast outside the copy loop. Triton
            # Ascend's pointer-offset analysis cannot handle the cast after
            # adding a loop induction variable and vector offsets.
            row_src_ptr = row_src_addr.to(tl.pointer_type(tl.uint8))
            row_dst_ptr = row_dst_addr.to(tl.pointer_type(tl.uint8))

            for i in range(0, per_row_bytes, COPY_BLOCK_SIZE):
                mask = i + offsets < per_row_bytes
                src_ptr = row_src_ptr + i + offsets
                dst_ptr = row_dst_ptr + i + offsets
                data = tl.load(src_ptr, mask=mask)
                tl.store(dst_ptr, data, mask=mask)

        return

    if is_conv_state:
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_offset = token_bias.to(tl.int64) * state_inner_size * state_elem_size
        src_addr = state_base_addr + src_block_id * state_block_stride + src_offset
        copy_size = (
            (conv_width - token_bias).to(tl.int64)
            * state_inner_size
            * state_elem_size
        )
    else:
        src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
        src_addr = state_base_addr + src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size

    # Keep the address-to-pointer cast outside the copy loop for the same
    # Triton Ascend pointer-offset analysis limitation as the DS-conv path.
    src_ptr = src_addr.to(tl.pointer_type(tl.uint8))
    dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))
    offsets = tl.arange(0, COPY_BLOCK_SIZE)

    for i in range(0, copy_size, COPY_BLOCK_SIZE):
        mask = i + offsets < copy_size
        curr_src = src_ptr + i + offsets
        curr_dst = dst_ptr + i + offsets
        data = tl.load(curr_src, mask=mask)
        tl.store(curr_dst, data, mask=mask)
