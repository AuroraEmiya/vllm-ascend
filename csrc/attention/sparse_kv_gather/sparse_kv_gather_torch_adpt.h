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

#ifndef SPARSE_KV_GATHER_TORCH_ADPT_H
#define SPARSE_KV_GATHER_TORCH_ADPT_H

#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

namespace {

void construct_sparse_kv_gather_output_tensors(
    const at::Tensor &sparse_indices,
    const at::Tensor &key_nope,
    const at::Tensor &key_rope,
    at::Tensor &out_ctkv,
    at::Tensor &out_kpe)
{
    // sparse_indices: [num_actual, topk_n]  (2D) or [B, S1, K] (3D)
    int64_t N, K;
    if (sparse_indices.dim() == 2) {
        N = sparse_indices.size(0);
        K = sparse_indices.size(1);
    } else {
        TORCH_CHECK(sparse_indices.dim() == 3,
                    "sparse_indices must be 2D [N, K] or 3D [B, S1, K]");
        N = sparse_indices.size(0) * sparse_indices.size(1);
        K = sparse_indices.size(2);
    }
    TORCH_CHECK(N > 0 && K > 0, "sparse_indices has zero dimension");

    out_ctkv = at::zeros({N, K, 512}, key_nope.options().dtype(key_nope.dtype()));
    out_kpe  = at::zeros({N, K,  64}, key_rope.options().dtype(key_rope.dtype()));
}

}  // namespace

/*!
 * \brief Gather non-contiguous KV-cache entries into two contiguous 3D tensors.
 *
 * Matches the semantics of gather_kv_triton:
 *   out_ctkv [num_actual, topk_n, 512]
 *   out_kpe  [num_actual, topk_n,  64]
 */
std::tuple<at::Tensor, at::Tensor> npu_sparse_kv_gather(
    const at::Tensor &sparse_indices,
    const at::Tensor &key_nope,
    const at::Tensor &key_rope,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_q,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &cur_pos,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv)
{
    TORCH_CHECK(sparse_indices.numel() > 0, "sparse_indices is empty.");
    TORCH_CHECK(key_nope.numel() > 0,      "key_nope is empty.");
    TORCH_CHECK(key_rope.numel() > 0,      "key_rope is empty.");
    TORCH_CHECK(sparse_block_size == 1,
                "sparse_block_size only supports 1, got ", sparse_block_size);

    std::string lq = std::string(layout_query);
    std::string lk = std::string(layout_kv);

    at::Tensor out_ctkv, out_kpe;
    construct_sparse_kv_gather_output_tensors(
        sparse_indices, key_nope, key_rope, out_ctkv, out_kpe);

    char *lq_p = const_cast<char *>(lq.c_str());
    char *lk_p = const_cast<char *>(lk.c_str());

    EXEC_NPU_CMD(aclnnSparseKvGather,
                 sparse_indices,
                 key_nope,
                 key_rope,
                 block_table,
                 actual_seq_lengths_q,
                 actual_seq_lengths_kv,
                 cur_pos,
                 sparse_block_size,
                 lq_p,
                 lk_p,
                 out_ctkv,
                 out_kpe);

    return std::make_tuple(out_ctkv, out_kpe);
}

}  // namespace vllm_ascend

#endif  // SPARSE_KV_GATHER_TORCH_ADPT_H
