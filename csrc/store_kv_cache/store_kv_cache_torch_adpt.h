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
#ifndef STORE_KV_CACHE_TORCH_ADPT_H
#define STORE_KV_CACHE_TORCH_ADPT_H

#include <cstdint>
#include <vector>

namespace vllm_ascend {

inline void npu_store_kv_cache(const at::Tensor& keyIn, const at::Tensor& keyCacheIn,
                                const at::Tensor& slotMapping, int64_t blockSize) {
    TORCH_CHECK(keyIn.dim() >= 1, "keyIn must have at least 1 dimension");
    TORCH_CHECK(keyCacheIn.dim() >= 1, "keyCacheIn must have at least 1 dimension");
    TORCH_CHECK(slotMapping.dim() == 1, "slotMapping must be 1D");
    TORCH_CHECK(slotMapping.size(0) == keyIn.size(0),
                "slotMapping and keyIn must have same numTokens");
    TORCH_CHECK(blockSize > 0, "blockSize must be positive");

    int64_t numTokens = keyIn.size(0);
    if (numTokens == 0) {
        return;
    }

    // ── D-stage fast path: single token, single core, no grouping ──
    if (numTokens == 1) {
        at::Tensor slotTensor = slotMapping.to(at::kInt).contiguous();
        EXEC_NPU_CMD(aclnnStoreKVDecode, keyIn, keyCacheIn, slotTensor, keyCacheIn);
        return;
    }

    // ── P-stage: host-side grouping → multi-core grouped copy ──
    at::Tensor slotCpu = slotMapping.to(at::kCPU).contiguous();
    const int32_t* slotPtr = slotCpu.data_ptr<int32_t>();

    std::vector<int32_t> groupLenVec;
    std::vector<int32_t> groupSrcIdxVec;
    std::vector<int32_t> groupDstIdxVec;

    int64_t t = 0;
    while (t < numTokens) {
        int32_t slot = slotPtr[t];
        if (slot < 0) {
            t++;
            continue;
        }

        int32_t srcStart = static_cast<int32_t>(t);
        int32_t dstStart = slot;
        int32_t groupLen = 1;

        while (t + 1 < numTokens) {
            int32_t nextSlot = slotPtr[t + 1];
            if (nextSlot < 0) {
                break;
            }
            if (nextSlot == slot + 1 && groupLen < static_cast<int32_t>(blockSize)) {
                groupLen++;
                slot = nextSlot;
                t++;
            } else {
                break;
            }
        }

        groupLenVec.push_back(groupLen);
        groupSrcIdxVec.push_back(srcStart);
        groupDstIdxVec.push_back(dstStart);
        t++;
    }

    int64_t numGroups = static_cast<int64_t>(groupLenVec.size());
    if (numGroups == 0) {
        return;
    }

    at::Tensor groupLenOut = at::from_blob(
        const_cast<int32_t*>(groupLenVec.data()), {numGroups},
        at::TensorOptions().dtype(at::kInt).device(at::kCPU)).to(slotMapping.device());
    at::Tensor groupSrcIdxOut = at::from_blob(
        const_cast<int32_t*>(groupSrcIdxVec.data()), {numGroups},
        at::TensorOptions().dtype(at::kInt).device(at::kCPU)).to(slotMapping.device());
    at::Tensor groupDstIdxOut = at::from_blob(
        const_cast<int32_t*>(groupDstIdxVec.data()), {numGroups},
        at::TensorOptions().dtype(at::kInt).device(at::kCPU)).to(slotMapping.device());

    EXEC_NPU_CMD(aclnnStoreKVBlock, keyIn, keyCacheIn, groupLenOut, groupSrcIdxOut,
                 groupDstIdxOut, blockSize, keyCacheIn);
}

}  // namespace vllm_ascend

#endif  // STORE_KV_CACHE_TORCH_ADPT_H
