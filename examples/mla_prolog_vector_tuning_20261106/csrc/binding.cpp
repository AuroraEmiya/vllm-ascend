// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include "attention/mla_prolog_serial/torch_adpt.h"
#include "aclnn_torch_adapter/serial_matmul_v3/torch_adpt.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset=0;
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("mla_prolog_vector_out(Tensor x, Tensor aux, Tensor gamma_q, Tensor gamma_kv, "
            "Tensor sine, Tensor cosine, Tensor index, int stage_mode, int rows, int heads, int cache_slots, "
            "float eps_q, float eps_kv, Tensor(a!) y, Tensor(b!) query_rope, "
            "Tensor(c!) kv_cache, Tensor(d!) kr_cache) -> ()");
    ops.impl("mla_prolog_vector_out", torch::kPrivateUse1, &vllm_ascend::mla_prolog_vector_out);
    ops.def("mla_norm_cache_out(Tensor x, Tensor aux, Tensor gamma_q, Tensor gamma_kv, "
            "Tensor sine, Tensor cosine, Tensor index, int stage_mode, int rows, int heads, int cache_slots, "
            "float eps_q, float eps_kv, Tensor(a!) y, Tensor(b!) query_rope, "
            "Tensor(c!) kv_cache, Tensor(d!) kr_cache) -> ()");
    ops.impl("mla_norm_cache_out", torch::kPrivateUse1, &vllm_ascend::mla_norm_cache_out);
    ops.def("mla_split_rope_out(Tensor x, Tensor aux, Tensor gamma_q, Tensor gamma_kv, "
            "Tensor sine, Tensor cosine, Tensor index, int stage_mode, int rows, int heads, int cache_slots, "
            "float eps_q, float eps_kv, Tensor(a!) y, Tensor(b!) query_rope, "
            "Tensor(c!) kv_cache, Tensor(d!) kr_cache) -> ()");
    ops.impl("mla_split_rope_out", torch::kPrivateUse1, &vllm_ascend::mla_split_rope_out);
    ops.def("mla_layout_out(Tensor x, Tensor aux, Tensor gamma_q, Tensor gamma_kv, "
            "Tensor sine, Tensor cosine, Tensor index, int stage_mode, int rows, int heads, int cache_slots, "
            "float eps_q, float eps_kv, Tensor(a!) y, Tensor(b!) query_rope, "
            "Tensor(c!) kv_cache, Tensor(d!) kr_cache) -> ()");
    ops.impl("mla_layout_out", torch::kPrivateUse1, &vllm_ascend::mla_layout_out);
    ops.def("serial_matmul_v3_out(Tensor a, Tensor b, bool transpose_b, bool batched, Tensor(a!) out) -> ()");
    ops.impl("serial_matmul_v3_out", torch::kPrivateUse1, &vllm_ascend::serial_matmul_v3_out);
}
