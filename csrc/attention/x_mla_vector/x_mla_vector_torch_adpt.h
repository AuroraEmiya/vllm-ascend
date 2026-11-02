// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_X_MLA_VECTOR_TORCH_ADPT_H
#define VLLM_ASCEND_X_MLA_VECTOR_TORCH_ADPT_H
#include <torch/extension.h>
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
inline void mla_prolog_vector_out(const at::Tensor& x, const at::Tensor& aux,
    const at::Tensor& gammaQ, const at::Tensor& gammaKv, const at::Tensor& sine,
    const at::Tensor& cosine, const at::Tensor& index, int64_t mode, int64_t rows,
    int64_t heads, int64_t cacheSlots, double epsQ, double epsKv,
    at::Tensor& y, at::Tensor& queryRope, at::Tensor& cacheKv, at::Tensor& cacheKr)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "XMlaVector requires NPU tensors");
    const at::Tensor* tensors[] = {&x, &aux, &gammaQ, &gammaKv, &sine, &cosine,
        &y, &queryRope, &cacheKv, &cacheKr};
    for (const at::Tensor* tensor : tensors) {
        TORCH_CHECK(tensor->scalar_type() == at::kBFloat16 && tensor->is_contiguous() &&
            tensor->device() == x.device(), "XMlaVector expects contiguous BF16 tensors on one NPU");
    }
    TORCH_CHECK(index.scalar_type() == at::kLong && index.is_contiguous() &&
        index.device() == x.device(), "XMlaVector index must be contiguous int64 on the same NPU");
    EXEC_NPU_CMD(aclnnXMlaVector, x, aux, gammaQ, gammaKv, sine, cosine, index,
        mode, rows, heads, cacheSlots, epsQ, epsKv, y, queryRope, cacheKv, cacheKr);
}
}  // namespace vllm_ascend
#endif
