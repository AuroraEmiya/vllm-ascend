/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "../aclnn_torch_adapter/op_api_common.h"

#ifndef STORE_KV_BLOCK_TORCH_ADPT_H
#define STORE_KV_BLOCK_TORCH_ADPT_H

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> store_kv_block_pre(
    const at::Tensor &slot_mapping_npu,
    at::IntArrayRef slot_mapping_list,
    int64_t block_size)
{
    int64_t slot_mapping_len = slot_mapping_list.size();

    std::vector<int32_t> length;
    std::vector<int32_t> key_idx;
    std::vector<int32_t> key_cache_idx;

    length.reserve(16);
    key_idx.reserve(16);
    key_cache_idx.reserve(16);

    int64_t idx_slotmap = 0;

    while (idx_slotmap < slot_mapping_len) {
        int64_t current_idx = slot_mapping_list[idx_slotmap];

        if (current_idx < 0) {
            idx_slotmap++;
            continue;
        }

        int64_t block_offset = current_idx % block_size;

        int64_t max_group_len = std::min(
            block_size - block_offset,
            slot_mapping_len - idx_slotmap
        );

        int64_t group_len = 1;

        for (; group_len < max_group_len; ++group_len) {
            int64_t prev_idx = slot_mapping_list[idx_slotmap + group_len - 1];
            int64_t next_idx = slot_mapping_list[idx_slotmap + group_len];

            if (next_idx < 0 || next_idx != prev_idx + 1) {
                break;
            }
        }

        length.emplace_back(static_cast<int32_t>(group_len));
        key_idx.emplace_back(static_cast<int32_t>(idx_slotmap));
        key_cache_idx.emplace_back(static_cast<int32_t>(current_idx));

        idx_slotmap += group_len;
    }

    int64_t idx_groups = static_cast<int64_t>(length.size());

    at::Tensor group_len = at::empty(
        {idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
    );

    at::Tensor group_key_idx = at::empty(
        {idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
    );

    at::Tensor group_key_cache_idx = at::empty(
        {idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
    );

    if (idx_groups > 0) {
        uint32_t device_size = static_cast<uint32_t>(
            idx_groups * static_cast<int64_t>(sizeof(int32_t))
        );

        aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

        aclrtMemcpyAsync(
            group_len.data_ptr(),
            device_size,
            length.data(),
            device_size,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream
        );

        aclrtMemcpyAsync(
            group_key_idx.data_ptr(),
            device_size,
            key_idx.data(),
            device_size,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream
        );

        aclrtMemcpyAsync(
            group_key_cache_idx.data_ptr(),
            device_size,
            key_cache_idx.data(),
            device_size,
            ACL_MEMCPY_HOST_TO_DEVICE,
            stream
        );
    }

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        group_len,
        group_key_idx,
        group_key_cache_idx
    );
}

void store_kv_block(
    const at::Tensor &key_in,
    const at::Tensor &key_cache_in,
    const at::Tensor &group_len,
    const at::Tensor &group_key_idx,
    const at::Tensor &group_key_cache_idx,
    int64_t block_size)
{
    EXEC_NPU_CMD(
        aclnnStoreKVBlock,
        key_in,
        key_cache_in,
        group_len,
        group_key_idx,
        group_key_cache_idx,
        block_size
    );
}

}  // namespace vllm_ascend

#endif  // STORE_KV_BLOCK_TORCH_ADPT_H