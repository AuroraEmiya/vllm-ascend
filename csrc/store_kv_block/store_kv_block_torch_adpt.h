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
//  #include "../aclnn_torch_adapter/op_api_common.h"

#ifndef STORE_KV_BLOCK_TORCH_ADPT_H
#define STORE_KV_BLOCK_TORCH_ADPT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> store_kv_block_pre(const at::Tensor& slot_mapping_npu,
                                                                         at::IntArrayRef slot_mapping_list,
                                                                         int64_t block_size) {
  // slot_mapping_list #slotmap[i]存放key的位置信息
  const int64_t slot_mapping_len = slot_mapping_list.size();

  std::vector<int32_t> length(16, 0);
  std::vector<int32_t> key_idx(16, 0);
  std::vector<int32_t> key_cache_idx(16, 0);
  int64_t key_idx_slotmap = 0;
  int64_t idx_groups = 0;

  // 检查slot map内地址递增
  bool addr_not_increased = false;
  for (int64_t i = 0; i + 1 < slot_mapping_len; i++) {
    if (slot_mapping_list[i] >= slot_mapping_list[i + 1]) {
      addr_not_increased = true;
      // printf("We've found wrong key distribution which is not ascending, but we try to store.\n");
      break;
    }
  }

  //-------------------修改位置----------------------------------------------------------------------------------------------------------------
  while (key_idx_slotmap < slot_mapping_len) {
    const int64_t cur_key_addr = slot_mapping_list[key_idx_slotmap];
    // cur_key_index = block_id * block_size + block_offset

    if (cur_key_addr < 0) {
      key_idx_slotmap++;
      continue;
    }

    const int64_t block_id = cur_key_addr / block_size;  // 所属 block 编号
    int64_t next = key_idx_slotmap + 1;

    key_idx[idx_groups] = static_cast<int32_t>(key_idx_slotmap);  // 该组起始位置
    key_cache_idx[idx_groups] = static_cast<int32_t>(cur_key_addr);

    const bool can_use_fast_path = !addr_not_increased && key_idx_slotmap + 1 < slot_mapping_len &&
                                   cur_key_addr + 1 == slot_mapping_list[key_idx_slotmap + 1];

    if (can_use_fast_path) {
      const int64_t block_offset = cur_key_addr % block_size;

      const int64_t block_remaining = block_size - block_offset;
      const int64_t token_remaining = slot_mapping_len - key_idx_slotmap;
      const int64_t idx_stride = std::min(block_remaining, token_remaining) - 1;

      const int64_t expected_last = cur_key_addr + idx_stride;
      const int64_t expected_last_idx = key_idx_slotmap + idx_stride;

      if (expected_last == slot_mapping_list[expected_last_idx]) {
        next = expected_last_idx + 1;
      } else {
        for (; next < slot_mapping_len && slot_mapping_list[next] == slot_mapping_list[next - 1] + 1 &&
               slot_mapping_list[next] / block_size == block_id;
             next++);
      }
    } else {
      for (; next < slot_mapping_len && slot_mapping_list[next] == slot_mapping_list[next - 1] + 1 &&
             slot_mapping_list[next] / block_size == block_id;
           next++);
    }

    length[idx_groups] = static_cast<int32_t>(next - key_idx_slotmap);

    key_idx_slotmap = next;
    idx_groups++;

    if (idx_groups >= length.size()) {
      const int64_t new_size = length.size() * 2;
      length.resize(new_size, 0);
      key_idx.resize(new_size, 0);
      key_cache_idx.resize(new_size, 0);
    }
  }

  at::Tensor group_len =
      at::empty({idx_groups}, at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32));
  void* group_len_addr = group_len.data_ptr();

  at::Tensor group_key_idx =
      at::empty({idx_groups}, at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32));
  void* group_key_idx_addr = group_key_idx.data_ptr();

  at::Tensor group_key_cache_idx =
      at::empty({idx_groups}, at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32));
  void* group_key_cache_idx_addr = group_key_cache_idx.data_ptr();

  const size_t device_size = idx_groups * sizeof(length[0]);
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  aclrtMemcpyKind memcpy_type = ACL_MEMCPY_HOST_TO_DEVICE;

  aclError ret = aclrtMemcpyAsync(group_len_addr, device_size, &length[0], device_size, memcpy_type, stream);
  TORCH_CHECK(ret == ACL_SUCCESS, "store_kv_block_pre copy group_len failed, ret=", ret);

  ret = aclrtMemcpyAsync(group_key_idx_addr, device_size, &key_idx[0], device_size, memcpy_type, stream);
  TORCH_CHECK(ret == ACL_SUCCESS, "store_kv_block_pre copy group_key_idx failed, ret=", ret);

  ret = aclrtMemcpyAsync(group_key_cache_idx_addr, device_size, &key_cache_idx[0], device_size, memcpy_type, stream);
  TORCH_CHECK(ret == ACL_SUCCESS, "store_kv_block_pre copy group_key_cache_idx failed, ret=", ret);

  // ret = aclrtSynchronizeStream(stream);
  // TORCH_CHECK(ret == ACL_SUCCESS, "store_kv_block_pre synchronize stream failed, ret=", ret);

  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(group_len, group_key_idx, group_key_cache_idx);
}

inline void store_kv_block(const at::Tensor& key_in, const at::Tensor& key_cache_in, const at::Tensor& group_len,
                           const at::Tensor& group_key_idx, const at::Tensor& group_key_cache_idx, int64_t block_size) {
  EXEC_NPU_CMD(aclnnStoreKVBlock, key_in, key_cache_in, group_len, group_key_idx, group_key_cache_idx, block_size);
}

}  // namespace vllm_ascend
#endif
