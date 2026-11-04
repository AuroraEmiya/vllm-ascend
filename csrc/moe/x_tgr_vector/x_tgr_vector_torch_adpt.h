/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0. */
#ifndef VLLM_ASCEND_X_TGR_VECTOR_TORCH_ADPT_H
#define VLLM_ASCEND_X_TGR_VECTOR_TORCH_ADPT_H
namespace vllm_ascend {
void npu_tgr_vector_out(const at::Tensor& x, const at::Tensor& gamma,
                        const at::Tensor& aux, const at::Tensor& gate,
                        int64_t mode, int64_t channels, double eps,
                        at::Tensor y, at::Tensor inv, at::Tensor inject)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && aux.scalar_type() == at::kBFloat16 &&
                gate.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kBFloat16 &&
                inject.scalar_type() == at::kBFloat16, "TGR vector tensors must be BF16");
    TORCH_CHECK(gamma.scalar_type() == at::kFloat && inv.scalar_type() == at::kFloat,
                "TGR gamma and inv_rms must be FP32");
    for (const auto& tensor : {x, gamma, aux, gate, y, inv, inject}) {
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == x.device(),
                    "TGR tensors must be contiguous and on the same NPU");
    }
    EXEC_NPU_CMD(aclnnXTgrVector, x, gamma, aux, gate, mode, channels, eps, y, inv, inject);
}
}  // namespace vllm_ascend
#endif
