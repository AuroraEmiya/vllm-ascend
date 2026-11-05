// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch/library.h>
#include "../../aclnn_torch_adapter/op_api_common.h"
namespace {
void Check(const at::Tensor& x) {
  TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),
      "Contact Vector requires contiguous BF16 NPU storage");
}
void Norm(const at::Tensor& dq,const at::Tensor& dkv,const at::Tensor& gq,const at::Tensor& gkv,
    const at::Tensor& sine,const at::Tensor& cosine,const at::Tensor& index,
    at::Tensor& kv,at::Tensor& kr,at::Tensor& cq) {
  const at::Tensor* tensors[]={&dq,&dkv,&gq,&gkv,&sine,&cosine,&kv,&kr,&cq};
  for(const auto* x:tensors)Check(*x);
  TORCH_CHECK(index.scalar_type()==at::kLong && index.is_contiguous() && index.device()==dq.device(),"int64 cache index required");
  EXEC_NPU_CMD(aclnnContactNorm,dq,dkv,gq,gkv,sine,cosine,index,kv,kr,cq,kv,kr);
}
void Split(const at::Tensor& x,const at::Tensor& sine,const at::Tensor& cosine,
    int64_t heads,int64_t nope,int64_t rope,at::Tensor& qc,at::Tensor& qr) {
  const at::Tensor* tensors[]={&x,&sine,&cosine,&qc,&qr};for(const auto* z:tensors)Check(*z);
  EXEC_NPU_CMD(aclnnContactSplit,x,sine,cosine,heads,nope,rope,qc,qr);
}
void Layout(const at::Tensor& x,at::Tensor& y) {
  Check(x);Check(y);EXEC_NPU_CMD(aclnnContactLayout,x,y);
}
void MetaNorm(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,at::Tensor&,at::Tensor&,at::Tensor&) {}
void MetaSplit(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,at::Tensor&,at::Tensor&) {}
void MetaLayout(const at::Tensor&,at::Tensor&) {}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend,m) {
  m.def("contact_norm_out(Tensor dq, Tensor dkv, Tensor gq, Tensor gkv, Tensor sine, Tensor cosine, Tensor index, Tensor(a!) kv, Tensor(b!) kr, Tensor(c!) cq) -> ()");
  m.def("contact_split_out(Tensor x, Tensor sine, Tensor cosine, int heads, int nope, int rope, Tensor(a!) qc, Tensor(b!) qr) -> ()");
  m.def("contact_layout_out(Tensor x, Tensor(a!) y) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend,PrivateUse1,m) {
  m.impl("contact_norm_out",&Norm);m.impl("contact_split_out",&Split);m.impl("contact_layout_out",&Layout);
}
TORCH_LIBRARY_IMPL(_C_ascend,Meta,m) {
  m.impl("contact_norm_out",&MetaNorm);m.impl("contact_split_out",&MetaSplit);m.impl("contact_layout_out",&MetaLayout);
}
#ifdef CONTACT_JIT_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}
#endif
