/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file decode_index_score_def.cpp
 * \brief OpDef for the MiniMax-M3 decode indexer block-score computation.
 *        Mirrors the score-only part of `minimax_m3_index_decode` in
 *        vllm_ascend/models/minimax_m3/ops/msa_m3_triton.py. The op computes
 *        one block-score per candidate KV block (max over the 128 positions of
 *        idx_q . index_k) plus the init/local sentinel priorities. topk/mask
 *        are NOT fused; they stay on the torch side.
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class DecodeIndexScore : public OpDef {
public:
    explicit DecodeIndexScore(const char *name) : OpDef(name)
    {
        // idx_q: [total_q, num_idx_heads, head_dim]
        this->Input("query")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // index_kv_cache: [num_blocks, 128, head_dim]
        this->Input("index_kv_cache")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // block_table: [num_reqs, local_max_blocks], int32
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // seq_lens (shard-local): [num_reqs], int32
        this->Input("seq_lens")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // global_seq_lens: [num_reqs], int32
        this->Input("global_seq_lens")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // score: [num_idx_heads, total_q, score_block_stride], fp32
        this->Output("score")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("decode_query_len").AttrType(REQUIRED).Int(0);
        this->Attr("block_offset").AttrType(REQUIRED).Int(0);
        this->Attr("init_blocks").AttrType(REQUIRED).Int(0);
        this->Attr("local_blocks").AttrType(REQUIRED).Int(0);
        this->Attr("score_block_stride").AttrType(REQUIRED).Int(0);
        this->Attr("num_chunks").AttrType(REQUIRED).Int(1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(DecodeIndexScore);
} // namespace ops
