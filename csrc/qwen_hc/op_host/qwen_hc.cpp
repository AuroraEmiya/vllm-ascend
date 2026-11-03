// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#if defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS) || defined(VLLM_ASCEND_HC_STANDALONE)
  #include <acl/acl.h>
  #include <torch/extension.h>
  #include <torch/library.h>
  #include <torch_npu/csrc/core/npu/NPUGuard.h>
  #include <torch_npu/csrc/core/npu/NPUStream.h>
  #include <torch_npu/csrc/framework/OpCommand.h>
  #include <cmath>
  #include <limits>
  #include "../qwen_hc_launch.h"
  #include "qwen_hc_tiling.h"

namespace vllm_ascend::hc {
void checkMatrix(const at::Tensor& input) {
  TORCH_CHECK(input.dim() == 2 && input.stride(1) == 1, "HC requires a 2D matrix with unit inner stride");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "HC Ascend C currently supports BF16");
  TORCH_CHECK(input.device().is_privateuseone() || input.device().is_meta(), "HC requires NPU or Meta");
  constexpr auto maxIndex = std::numeric_limits<uint32_t>::max();
  TORCH_CHECK(input.stride(0) >= 0 && input.stride(0) <= maxIndex && input.size(0) <= maxIndex &&
                  input.size(1) <= maxIndex && input.numel() <= maxIndex,
              "HC matrix exceeds uint32 indexing");
  TORCH_CHECK(input.size(0) == 0 || (input.size(0) - 1) * input.stride(0) + input.size(1) <= maxIndex,
              "HC row strides exceed uint32 indexing");
}
void checkSameTypeDevice(const at::Tensor& input, const at::Tensor& operand) {
  TORCH_CHECK(input.device() == operand.device() && input.scalar_type() == operand.scalar_type(),
              "HC operands must have the same dtype and device");
}
HcTilingData makeParams(const at::Tensor& input, int64_t groupCount) {
  checkMatrix(input);
  TORCH_CHECK(groupCount > 0 && groupCount <= std::numeric_limits<uint32_t>::max() && input.size(1) > 0 &&
                  input.size(1) % groupCount == 0,
              "HC width must be positive and divisible by the stream count");
  const int64_t hiddenSize = input.size(1) / groupCount;
  TORCH_CHECK(hiddenSize <= kMaxSupportedHiddenSize, "HC per-stream width exceeds the validated range");
  HcTilingData tilingData{};
  tilingData.rows = input.size(0);
  tilingData.hiddenSize = hiddenSize;
  tilingData.groupCount = groupCount;
  tilingData.residualStride = input.stride(0);
  return tilingData;
}
std::tuple<at::Tensor, at::Tensor> combineMeta(const at::Tensor& residual, const at::Tensor& block,
                                               const c10::optional<at::Tensor>& injection, const at::Tensor& weight,
                                               double eps, int64_t groupCount) {
  const auto tilingData = makeParams(residual, groupCount);
  checkMatrix(block);
  checkSameTypeDevice(residual, block);
  checkSameTypeDevice(residual, weight);
  TORCH_CHECK(block.size(0) == tilingData.rows && block.size(1) == tilingData.hiddenSize,
              "HC block output shape mismatch");
  TORCH_CHECK(weight.is_contiguous() && (weight.numel() == tilingData.hiddenSize || weight.numel() == residual.size(1)),
              "HC norm weight must contain H or K*H contiguous values");
  TORCH_CHECK(std::isfinite(eps) && eps > 0, "HC requires finite positive eps");
  if (injection.has_value()) {
    checkMatrix(*injection);
    checkSameTypeDevice(residual, *injection);
    TORCH_CHECK(injection->size(0) == tilingData.rows && injection->size(1) == groupCount,
                "HC injection shape mismatch");
  }
  return {at::empty(residual.sizes(), residual.options()), at::empty(residual.sizes(), residual.options())};
}
std::tuple<at::Tensor, at::Tensor> combine(const at::Tensor& residual, const at::Tensor& block,
                                           const c10::optional<at::Tensor>& injection, const at::Tensor& weight,
                                           double eps, int64_t groupCount) {
  auto outputs = combineMeta(residual, block, injection, weight, eps, groupCount);
  auto tilingData = makeParams(residual, groupCount);
  if (!tilingData.rows) return outputs;
  tilingData.blockStride = block.stride(0);
  tilingData.injectionStride = injection ? injection->stride(0) : 0;
  tilingData.hasInjection = injection.has_value();
  tilingData.sharedWeight = weight.numel() == tilingData.hiddenSize;
  tilingData.eps = eps;
  c10_npu::NPUGuard deviceGuard(residual.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  tilingData = makeCombineNormTiling(tilingData, queryDeviceResources(residual.device().index())).data;
  auto residualData = residual.data_ptr();
  auto blockData = block.data_ptr();
  auto weightData = weight.data_ptr();
  auto injectionData = injection ? injection->data_ptr() : nullptr;
  auto combinedData = std::get<0>(outputs).data_ptr();
  auto normalizedData = std::get<1>(outputs).data_ptr();
  at_npu::native::OpCommand command;
  command.Name("qwen_hc_combine_norm");
  command.SetCustomHandler([=]() {
    launchCombineNorm(tilingData.blockDim, stream, residualData, blockData, injectionData, weightData, combinedData,
                      normalizedData, tilingData);
    return 0;
  });
  command.Run();
  return outputs;
}
at::Tensor siluMeta(const at::Tensor& input, int64_t groupCount) {
  checkMatrix(input);
  TORCH_CHECK(groupCount > 0 && groupCount <= std::numeric_limits<uint32_t>::max() && input.size(1) > 0 &&
                  input.size(1) <= kMaxSupportedHiddenSize,
              "HC SiLU group count or rank unsupported");
  return at::empty(input.sizes(), input.options());
}
at::Tensor silu(const at::Tensor& input, int64_t groupCount) {
  auto output = siluMeta(input, groupCount);
  if (!input.size(0)) return output;
  HcTilingData tilingData{};
  tilingData.rows = input.size(0);
  tilingData.hiddenSize = input.size(1);
  tilingData.groupCount = groupCount;
  tilingData.residualStride = input.stride(0);
  c10_npu::NPUGuard deviceGuard(input.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  tilingData = makeSiluTiling(tilingData, queryDeviceResources(input.device().index())).data;
  auto inputData = input.data_ptr();
  auto outputData = output.data_ptr();
  at_npu::native::OpCommand command;
  command.Name("qwen_hc_silu");
  command.SetCustomHandler([=]() {
    launchSilu(tilingData.blockDim, stream, inputData, outputData, tilingData);
    return 0;
  });
  command.Run();
  return output;
}
at::Tensor gateMeta(const at::Tensor& input, const at::Tensor& gate, int64_t groupCount) {
  const auto tilingData = makeParams(input, groupCount);
  checkMatrix(gate);
  checkSameTypeDevice(input, gate);
  TORCH_CHECK(input.sizes() == gate.sizes(), "HC gate shape mismatch");
  return at::empty({input.size(0), static_cast<int64_t>(tilingData.hiddenSize)}, input.options());
}
at::Tensor gateMix(const at::Tensor& input, const at::Tensor& gate, int64_t groupCount) {
  auto output = gateMeta(input, gate, groupCount);
  auto tilingData = makeParams(input, groupCount);
  if (!tilingData.rows) return output;
  tilingData.blockStride = gate.stride(0);
  c10_npu::NPUGuard deviceGuard(input.device());
  auto stream = c10_npu::getCurrentNPUStream().stream();
  tilingData = makeGateMixTiling(tilingData, queryDeviceResources(input.device().index())).data;
  auto inputData = input.data_ptr();
  auto gateData = gate.data_ptr();
  auto outputData = output.data_ptr();
  at_npu::native::OpCommand command;
  command.Name("qwen_hc_gate_mix");
  command.SetCustomHandler([=]() {
    launchGateMix(tilingData.blockDim, stream, inputData, gateData, outputData, tilingData);
    return 0;
  });
  command.Run();
  return output;
}
}  // namespace vllm_ascend::hc
TORCH_LIBRARY_FRAGMENT(_C_ascend, registry) {
  registry.def(
      "qwen_hc_combine_norm(Tensor residual, Tensor block_output, Tensor? injection, Tensor weight, float eps, int "
      "groups) -> (Tensor, Tensor)");
  registry.def("qwen_hc_silu(Tensor x, int groups) -> Tensor");
  registry.def("qwen_hc_gate_mix(Tensor x, Tensor gate, int groups) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, registry) {
  registry.impl("qwen_hc_combine_norm", &vllm_ascend::hc::combine);
  registry.impl("qwen_hc_silu", &vllm_ascend::hc::silu);
  registry.impl("qwen_hc_gate_mix", &vllm_ascend::hc::gateMix);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, registry) {
  registry.impl("qwen_hc_combine_norm", &vllm_ascend::hc::combineMeta);
  registry.impl("qwen_hc_silu", &vllm_ascend::hc::siluMeta);
  registry.impl("qwen_hc_gate_mix", &vllm_ascend::hc::gateMeta);
}
  #ifdef VLLM_ASCEND_HC_STANDALONE
PYBIND11_MODULE(vllm_ascend_hc_C, module) {
  module.def("deviceResources", []() {
    int32_t deviceIndex = -1;
    TORCH_CHECK(aclrtGetDevice(&deviceIndex) == ACL_SUCCESS, "HC current-device query failed");
    const auto resources = vllm_ascend::hc::queryDeviceResources(deviceIndex);
    return std::make_tuple(resources.vectorCoreCount, resources.ubSizeBytes);
  });
}
  #endif
#endif
