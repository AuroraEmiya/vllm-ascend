// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <initializer_list>
#include <torch/extension.h>
#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
inline void CheckBf16Tensors(const at::Tensor& input, std::initializer_list<const at::Tensor*> tensors) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1, "MLA Vector requires NPU tensors");
  for (const auto* tensor : tensors) {
    TORCH_CHECK(tensor->scalar_type() == at::kBFloat16 && tensor->is_contiguous() &&
                    tensor->device() == input.device(),
                "MLA Vector expects contiguous BF16 tensors on one NPU");
  }
}

inline void mla_norm_cache_out(const at::Tensor& x, const at::Tensor& gammaQ, const at::Tensor& gammaKv,
                               const at::Tensor& sine, const at::Tensor& cosine, const at::Tensor& index,
                               int64_t rows, int64_t cacheSlots, double epsQ, double epsKv,
                               at::Tensor& normalizedQ, at::Tensor& cacheKv, at::Tensor& cacheKr) {
  CheckBf16Tensors(x, {&x, &gammaQ, &gammaKv, &sine, &cosine, &normalizedQ, &cacheKv, &cacheKr});
  TORCH_CHECK(index.scalar_type() == at::kLong && index.is_contiguous() && index.device() == x.device(),
              "MLA cache index must be contiguous int64 on the same NPU");
  EXEC_NPU_CMD(aclnnXMlaNormCache, x, gammaQ, gammaKv, sine, cosine, index,
               rows, cacheSlots, epsQ, epsKv, normalizedQ, cacheKv, cacheKr);
}

inline void mla_split_rope_out(const at::Tensor& x, const at::Tensor& sine, const at::Tensor& cosine,
                               int64_t rows, int64_t heads, at::Tensor& headQ, at::Tensor& queryRope) {
  CheckBf16Tensors(x, {&x, &sine, &cosine, &headQ, &queryRope});
  EXEC_NPU_CMD(aclnnXMlaSplitRope, x, sine, cosine, rows, heads, headQ, queryRope);
}

inline void mla_layout_out(const at::Tensor& x, int64_t rows, int64_t heads, at::Tensor& query) {
  CheckBf16Tensors(x, {&x, &query});
  EXEC_NPU_CMD(aclnnXMlaLayout, x, rows, heads, query);
}
}  // namespace vllm_ascend
