#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, Tensor pre_mix, "
          "int hc_mult, int hc_sinkhorn_iters, float norm_eps, float hc_eps) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("hc_pre", TORCH_FN(ascend_kernel::hc_pre));
}
