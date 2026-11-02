// SPDX-License-Identifier: Apache-2.0
// Benchmark-only wrapper of the installed public ACLNN declaration. No fused
// implementation is included or inspected. This avoids Torch-NPU aclop fallback.
#include <torch/extension.h>
#include <optional>
#include <tuple>
#include "aclnn_torch_adapter/op_api_common.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mla_prolog_fused_public(
    const at::Tensor& x, const at::Tensor& wdq, const at::Tensor& wuq,
    const at::Tensor& wuk, const at::Tensor& wdkv, const at::Tensor& gammaQ,
    const at::Tensor& gammaKv, const at::Tensor& sine, const at::Tensor& cosine,
    const at::Tensor& index, at::Tensor& cacheKv, at::Tensor& cacheKr,
    double epsQ, double epsKv)
{
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && wuk.dim() == 3,
        "MLA benchmark public baseline requires decode [B,1,He] and [N,128,512] WUK");
    auto query = at::empty({x.size(0), x.size(1), wuk.size(0), 512}, x.options());
    auto queryRope = at::empty({x.size(0), x.size(1), wuk.size(0), 64}, x.options());
    const std::optional<at::Tensor> none = std::nullopt;
    char cacheMode[] = "PA_BSND";
    EXEC_NPU_CMD(aclnnMlaProlog, x, wdq, wuq, wuk, wdkv, gammaQ, gammaKv,
        sine, cosine, index, cacheKv, cacheKr, none, none, none, none, none,
        none, none, epsQ, epsKv, cacheMode, query, queryRope);
    return {query, queryRope, cacheKv, cacheKr};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, m)
{
    m.def("mla_prolog_fused_public(Tensor x, Tensor wdq, Tensor wuq, Tensor wuk, "
        "Tensor wdkv, Tensor gamma_q, Tensor gamma_kv, Tensor sine, Tensor cosine, "
        "Tensor index, Tensor(a!) cache_kv, Tensor(b!) cache_kr, float eps_q=1e-5, "
        "float eps_kv=1e-5) -> (Tensor, Tensor, Tensor(a!), Tensor(b!))");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m)
{
    m.impl("mla_prolog_fused_public", &mla_prolog_fused_public);
}
