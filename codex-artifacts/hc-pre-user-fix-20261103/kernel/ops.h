#ifndef HC_PRE_OPS_H
#define HC_PRE_OPS_H

#include <tuple>

#include <torch/extension.h>

namespace ascend_kernel {

// HcPre (npu_hc_pre_v3 semantics): CV-serial implementation.
// Returns (y, post, comb_frag, pre).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hc_pre(
    const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
    const at::Tensor &hc_base, const at::Tensor &pre_mix, int64_t hc_mult,
    int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_prepare(
    const at::Tensor &x, const at::Tensor &hc_fn);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hc_pre_run_prepared(
    const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
    const at::Tensor &hc_base, const at::Tensor &pre_mix, int64_t hc_mult,
    int64_t hc_sinkhorn_iters, double norm_eps, double hc_eps, const at::Tensor &tiling,
    const at::Tensor &weight, const at::Tensor &mask, int64_t prepared_b);

}  // namespace ascend_kernel

#endif  // HC_PRE_OPS_H
