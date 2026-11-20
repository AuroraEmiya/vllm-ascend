"""HC_Pre graph preparation and capture entry for the staged extension."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedHCPre:
    input_shape: tuple
    weight: torch.Tensor
    tiling: torch.Tensor
    packed_weight: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def prepare(cls, x, weight):
        """Prepare and synchronize before capture on the calling device."""
        if x.device.type != "npu" or weight.device != x.device:
            raise ValueError("x and weight must share one NPU")
        tiling, packed_weight, mask = torch.ops.npu.hc_pre_prepare(x, weight)
        torch.npu.synchronize(x.device)
        return cls(tuple(x.shape), weight, tiling, packed_weight, mask)

    def __call__(self, x, hc_scale, hc_base, pre_mix, *, hc_mult=4,
                 sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6):
        if tuple(x.shape) != self.input_shape:
            raise ValueError("prepare again for a different x shape")
        # The graph owner keeps this object alive until it discards the graph.
        # An updated weight requires new preparation and graph capture.
        return torch.ops.npu.hc_pre_run_prepared(
            x, self.weight, hc_scale, hc_base, pre_mix,
            hc_mult, sinkhorn_iters, norm_eps, hc_eps,
            self.tiling, self.packed_weight, self.mask, self.input_shape[0],
        )
