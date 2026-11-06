// SPDX-License-Identifier: Apache-2.0
// Benchmark adapter only. Uses the installed public API; no fused implementation.
#include <torch/extension.h>
#include <optional>
#include <tuple>
#include "../aclnn_torch_adapter/op_api_common.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_public(
    const at::Tensor& x, const at::Tensor& wdq, const at::Tensor& wuq,
    const at::Tensor& wuk, const at::Tensor& wdkv, const at::Tensor& gammaQ,
    const at::Tensor& gammaKv, const at::Tensor& sine, const at::Tensor& cosine,
    const at::Tensor& index, at::Tensor& cacheKv, at::Tensor& cacheKr,
    double epsQ, double epsKv) {
  TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(1) > 0 && wuk.dim() == 3,
              "MLA public baseline requires nonempty [B,S,H]");
  auto query = at::empty({x.size(0), x.size(1), wuk.size(0), 512}, x.options());
  auto qr = at::empty({x.size(0), x.size(1), wuk.size(0), 64}, x.options());
  const std::optional<at::Tensor> none = std::nullopt;
  char cacheMode[] = "PA_BSND";
  EXEC_NPU_CMD(aclnnMlaProlog, x, wdq, wuq, wuk, wdkv, gammaQ, gammaKv,
               sine, cosine, index, cacheKv, cacheKr, none, none, none, none,
               none, none, none, epsQ, epsKv, cacheMode, query, qr);
  return {query, qr, cacheKv, cacheKr};
}
}
TORCH_LIBRARY_FRAGMENT(mla_benchmark, m) {
  m.def("fused_public(Tensor x, Tensor wdq, Tensor wuq, Tensor wuk, Tensor wdkv, "
        "Tensor gamma_q, Tensor gamma_kv, Tensor sine, Tensor cosine, Tensor index, "
        "Tensor(a!) kv, Tensor(b!) kr, float eps_q=1e-5, float eps_kv=1e-5) "
        "-> (Tensor, Tensor, Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(mla_benchmark, PrivateUse1, m) {
  m.impl("fused_public", &fused_public);
}
