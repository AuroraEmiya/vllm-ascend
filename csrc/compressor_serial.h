// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/extension.h>
namespace vllm_ascend {
at::Tensor compressor_serial_pool(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                  const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
                                  int64_t, int64_t, bool);
at::Tensor compressor_serial_finalize(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      at::Tensor&, int64_t, int64_t, bool, double);
}  // namespace vllm_ascend
