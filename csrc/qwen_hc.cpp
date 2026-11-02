// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#if defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS) || defined(VLLM_ASCEND_HC_STANDALONE)
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include "qwen_hc_launch.h"

namespace vllm_ascend::hc {
constexpr int64_t MAX_HIDDEN = 8192;
constexpr uint32_t A3_VECTOR_CORES = 40;
constexpr uint32_t GATE_TILE = 512;
void check_matrix(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "HC requires a 2D matrix with unit inner stride");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "HC Ascend C currently supports BF16");
    TORCH_CHECK(x.device().is_privateuseone() || x.device().is_meta(), "HC requires NPU or Meta");
    TORCH_CHECK(x.stride(0) >= 0 && x.numel() <= std::numeric_limits<uint32_t>::max(),
                "HC matrix exceeds uint32 indexing");
    TORCH_CHECK(x.size(0) == 0 ||
        (x.size(0) - 1) * x.stride(0) + x.size(1) <= std::numeric_limits<uint32_t>::max(),
        "HC row strides exceed uint32 indexing");
}
void same_type_device(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device() == b.device() && a.scalar_type() == b.scalar_type(),
                "HC operands must have the same dtype and device");
}
HcParams params(const at::Tensor& x, int64_t groups) {
    check_matrix(x);
    TORCH_CHECK(groups > 0 && x.size(1) > 0 && x.size(1) % groups == 0,
                "HC width must be positive and divisible by the stream count");
    const int64_t hidden = x.size(1) / groups;
    TORCH_CHECK(hidden <= MAX_HIDDEN, "HC per-stream width exceeds the current UB bound");
    HcParams p{};
    p.rows=x.size(0);p.hidden=hidden;p.groups=groups;p.residual_stride=x.stride(0);
    return p;
}
std::tuple<at::Tensor,at::Tensor> combine_meta(
    const at::Tensor& residual,const at::Tensor& block,const c10::optional<at::Tensor>& injection,
    const at::Tensor& weight,double eps,int64_t groups) {
    auto p=params(residual,groups);
    check_matrix(block);same_type_device(residual,block);same_type_device(residual,weight);
    TORCH_CHECK(block.size(0)==p.rows && block.size(1)==p.hidden,"HC block output shape mismatch");
    TORCH_CHECK(weight.is_contiguous() && (weight.numel()==p.hidden ||
                weight.numel()==residual.size(1)),"HC norm weight must contain H or K*H contiguous values");
    TORCH_CHECK(std::isfinite(eps) && eps > 0,"HC requires finite positive eps");
    if(injection.has_value()) {
        check_matrix(*injection);same_type_device(residual,*injection);
        TORCH_CHECK(injection->size(0)==p.rows && injection->size(1)==groups,"HC injection shape mismatch");
    }
    return {at::empty(residual.sizes(),residual.options()),at::empty(residual.sizes(),residual.options())};
}
std::tuple<at::Tensor,at::Tensor> combine(
    const at::Tensor& residual,const at::Tensor& block,const c10::optional<at::Tensor>& injection,
    const at::Tensor& weight,double eps,int64_t groups) {
    auto result=combine_meta(residual,block,injection,weight,eps,groups);
    auto p=params(residual,groups);if(!p.rows)return result;
    p.block_stride=block.stride(0);p.injection_stride=injection ? injection->stride(0):0;
    p.has_injection=injection.has_value();p.shared_weight=weight.numel()==p.hidden;p.eps=eps;
    c10_npu::NPUGuard guard(residual.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto r=residual.data_ptr(),b=block.data_ptr(),w=weight.data_ptr();
    auto i=injection ? injection->data_ptr() : nullptr;
    auto s=std::get<0>(result).data_ptr(),n=std::get<1>(result).data_ptr();
    const uint32_t cores=std::min(A3_VECTOR_CORES,p.rows*p.groups);
    at_npu::native::OpCommand cmd;cmd.Name("qwen_hc_combine_norm");
    cmd.SetCustomHandler([=](){qwen_hc_combine_norm_launch(cores,stream,r,b,i,w,s,n,p);return 0;});
    cmd.Run();return result;
}
at::Tensor silu_meta(const at::Tensor& x,int64_t groups) {
    check_matrix(x);
    TORCH_CHECK(groups>0 && groups<=std::numeric_limits<uint32_t>::max() &&
                x.size(1)>0 && x.size(1)<=MAX_HIDDEN,"HC SiLU group count or rank unsupported");
    return at::empty(x.sizes(),x.options());
}
at::Tensor silu(const at::Tensor& x,int64_t groups) {
    auto result=silu_meta(x,groups);if(!x.size(0))return result;
    HcParams p{};p.rows=x.size(0);p.hidden=x.size(1);p.groups=groups;p.residual_stride=x.stride(0);
    c10_npu::NPUGuard guard(x.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto a=x.data_ptr(),b=result.data_ptr();const uint32_t cores=std::min(A3_VECTOR_CORES,p.rows);
    at_npu::native::OpCommand cmd;cmd.Name("qwen_hc_silu");
    cmd.SetCustomHandler([=](){qwen_hc_silu_launch(cores,stream,a,b,p);return 0;});
    cmd.Run();return result;
}
at::Tensor gate_meta(const at::Tensor& x,const at::Tensor& gate,int64_t groups) {
    auto p=params(x,groups);check_matrix(gate);same_type_device(x,gate);
    TORCH_CHECK(x.sizes()==gate.sizes(),"HC gate shape mismatch");
    return at::empty({x.size(0),static_cast<int64_t>(p.hidden)},x.options());
}
at::Tensor gate_mix(const at::Tensor& x,const at::Tensor& gate,int64_t groups) {
    auto result=gate_meta(x,gate,groups);auto p=params(x,groups);if(!p.rows)return result;
    p.block_stride=gate.stride(0);
    c10_npu::NPUGuard guard(x.device());auto stream=c10_npu::getCurrentNPUStream().stream();
    auto a=x.data_ptr(),b=gate.data_ptr(),c=result.data_ptr();
    const uint32_t cores=std::min(A3_VECTOR_CORES,p.rows*((p.hidden+GATE_TILE-1)/GATE_TILE));
    at_npu::native::OpCommand cmd;cmd.Name("qwen_hc_gate_mix");
    cmd.SetCustomHandler([=](){qwen_hc_gate_mix_launch(cores,stream,a,b,c,p);return 0;});
    cmd.Run();return result;
}
} // namespace vllm_ascend::hc
TORCH_LIBRARY_FRAGMENT(_C_ascend,m) {
    m.def("qwen_hc_combine_norm(Tensor residual, Tensor block_output, Tensor? injection, Tensor weight, float eps, int groups) -> (Tensor, Tensor)");
    m.def("qwen_hc_silu(Tensor x, int groups) -> Tensor");
    m.def("qwen_hc_gate_mix(Tensor x, Tensor gate, int groups) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend,PrivateUse1,m) {
    m.impl("qwen_hc_combine_norm",&vllm_ascend::hc::combine);
    m.impl("qwen_hc_silu",&vllm_ascend::hc::silu);
    m.impl("qwen_hc_gate_mix",&vllm_ascend::hc::gate_mix);
}
TORCH_LIBRARY_IMPL(_C_ascend,Meta,m) {
    m.impl("qwen_hc_combine_norm",&vllm_ascend::hc::combine_meta);
    m.impl("qwen_hc_silu",&vllm_ascend::hc::silu_meta);
    m.impl("qwen_hc_gate_mix",&vllm_ascend::hc::gate_meta);
}
#ifdef VLLM_ASCEND_HC_STANDALONE
PYBIND11_MODULE(vllm_ascend_hc_C,m) {}
#endif
#endif
