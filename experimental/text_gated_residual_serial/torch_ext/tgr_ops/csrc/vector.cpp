#include <torch/extension.h>
#include <torch/library.h>
#include "ops_common.h"

namespace tgr {
void vector_out(const at::Tensor& x,const at::Tensor& gamma,const at::Tensor& aux,
                const at::Tensor& gate,int64_t mode,int64_t channels,double eps,
                at::Tensor& y,at::Tensor& inv,at::Tensor& inject) {
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && gamma.scalar_type()==at::kFloat,
                "XTgrVector requires BF16 x and FP32 gamma");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && aux.is_contiguous() && gate.is_contiguous(),
                "XTgrVector requires contiguous tensors");
    EXEC_NPU_CMD_V1(aclnnXTgrVector,x,gamma,aux,gate,mode,channels,eps,y,inv,inject);
}
void vector_meta(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                 int64_t,int64_t,double,at::Tensor&,at::Tensor&,at::Tensor&) {}
}
TORCH_LIBRARY(custom,m) {
    m.def("tgr_vector_out(Tensor x, Tensor gamma, Tensor aux, Tensor gate, int mode, int channels, float eps, Tensor(a!) y, Tensor(b!) inv, Tensor(c!) inject) -> ()");
}
TORCH_LIBRARY_IMPL(custom,PrivateUse1,m) {m.impl("tgr_vector_out",&tgr::vector_out);}
TORCH_LIBRARY_IMPL(custom,Meta,m) {m.impl("tgr_vector_out",&tgr::vector_meta);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {}
