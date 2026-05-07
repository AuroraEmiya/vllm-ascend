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

#ifndef STORE_KV_DECODE_TORCH_ADPT_H
#define STORE_KV_DECODE_TORCH_ADPT_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace vllm_ascend {

inline void store_kv_decode(const at::Tensor& key_in, at::Tensor& key_cache_in, const at::Tensor slot_mapping_list) {
  // TORCH_CHECK(slot_mapping_list.scalar_type() == at::ScalarType::Long, "slot_mapping_list must be int64.");
  // TORCH_CHECK(slot_mapping_list.device() == key_in.device(), "slot_mapping_list must be on the same device as key_in.");
  // TORCH_CHECK(slot_mapping_list.numel() == key_in.size(0), "slot_mapping_list length must equal key_in.shape[0].");
  // safty check
  //
  EXEC_NPU_CMD(aclnnStoreKVDecode, key_in, key_cache_in, slot_mapping_list);
}

}  // namespace vllm_ascend
#endif
