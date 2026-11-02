#!/usr/bin/env python3
# HcPre 算子 PyTorch 参考实现（npu_hc_pre_v3 语义，pre_mix 为外部输入）
#
# 语义来源（逐行转录，无设计改动）:
#   /home/t00886357/dsv4_hcpre/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py
#   中的 _hc_pre_cpu（v3 分支: pre_mix 非 None）与 _make_hc_pre_inputs
#
# 目标输入输出形状来源（仅测试目标形状）:
#   /home/t00886357/dsv4_hcpre/docs/dsv4_1_flash_decode_shape_aggregation.csv HcPre 行:
#     in =[[B, 4, 5120], [24, 20480], [3], [24], [B, 4]]
#     out=[[B, 5120], [B, 4], [B, 4, 4], [B, 4]]
#     dt  =[bf16, fp32, fp32, fp32, fp32] -> [bf16, fp32, fp32, fp32]
#     B ∈ {128, 32}
import torch
import torch.nn as nn
import torch.nn.functional as F

HC_MULT = 4
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HIDDEN_SIZE = 5120  # CSV 目标形状的 hidden_size（fan_in = HC_MULT * HIDDEN_SIZE = 20480）

HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


class Model(nn.Module):
    def __init__(
        self,
        hc_mult: int = HC_MULT,
        hc_sinkhorn_iters: int = HC_SINKHORN_ITERS,
        norm_eps: float = NORM_EPS,
        hc_eps: float = HC_EPS,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

    def forward(self, x, hc_fn, hc_scale, hc_base, pre_mix):
        hc_mult = self.hc_mult
        hc_eps = self.hc_eps

        x_float = x.float()
        x_flat = x_float.flatten(-2)
        inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)

        # HcPre performs this matmul in HF32 round-toward-zero mode.
        mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
        pre, post, comb_frag = mixes.split(
            [hc_mult, hc_mult, hc_mult * hc_mult],
            dim=-1,
        )
        comb_frag = comb_frag.unflatten(-1, (hc_mult, hc_mult))

        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
        post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
        comb_frag = comb_frag * hc_scale[2] + hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

        comb_frag = comb_frag.softmax(-1) + hc_eps
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + hc_eps)
            comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + hc_eps)

        mix_for_y = pre_mix.float()
        y = (mix_for_y.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
        return y, post, comb_frag, pre


def _make_case(batch: int, hidden_size: int = HIDDEN_SIZE):
    # 模仿 test_npu_hc_pre.py::_make_hc_pre_inputs 的输入构造方式
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(batch, HC_MULT, hidden_size, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    pre_mix = torch.rand(batch, HC_MULT, dtype=torch.float32)
    return [x, hc_fn, hc_scale, hc_base, pre_mix]


def get_input_groups():
    # 仅测试目标输入输出形状（CSV HcPre 行）: B=128 与 B=32, hidden=5120
    torch.manual_seed(1024)
    return [_make_case(128), _make_case(32)]


def get_init_inputs():
    return []
