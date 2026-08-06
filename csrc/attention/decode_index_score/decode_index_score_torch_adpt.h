/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#ifndef DECODE_INDEX_SCORE_TORCH_ADPT_H
#define DECODE_INDEX_SCORE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_decode_index_score(
    const at::Tensor &query, const at::Tensor &index_kv_cache, const at::Tensor &block_table,
    const at::Tensor &seq_lens, const at::Tensor &global_seq_lens,
    int64_t decode_query_len, int64_t block_offset, int64_t init_blocks,
    int64_t local_blocks, int64_t score_block_stride, int64_t num_chunks)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(index_kv_cache.numel() > 0, "Tensor index_kv_cache is empty.");
    TORCH_CHECK(block_table.numel() > 0, "Tensor block_table is empty.");
    TORCH_CHECK(seq_lens.numel() > 0, "Tensor seq_lens is empty.");
    TORCH_CHECK(global_seq_lens.numel() > 0, "Tensor global_seq_lens is empty.");
    TORCH_CHECK(decode_query_len > 0, "decode_query_len must be positive.");
    TORCH_CHECK(score_block_stride > 0, "score_block_stride must be positive.");
    TORCH_CHECK(num_chunks > 0, "num_chunks must be positive.");

    constexpr int64_t DIM_Q = 0; // total_q
    constexpr int64_t DIM_H = 1; // num_idx_heads
    int64_t total_q = query.size(DIM_Q);
    int64_t num_idx_heads = query.size(DIM_H);
    at::Tensor score_out =
        at::empty({num_idx_heads, total_q, score_block_stride}, query.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnDecodeIndexScore, query, index_kv_cache, block_table, seq_lens, global_seq_lens,
                 decode_query_len, block_offset, init_blocks, local_blocks, score_block_stride, num_chunks, score_out);
    return score_out;
}

}  // namespace vllm_ascend

#endif  // DECODE_INDEX_SCORE_TORCH_ADPT_H
