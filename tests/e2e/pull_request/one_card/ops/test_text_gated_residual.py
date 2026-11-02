# SPDX-License-Identifier: Apache-2.0
"""Device checks against public equations, independent of the opaque baseline."""

import math

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
import vllm_ascend.vllm_ascend_C  # noqa: F401 - registers native operators

from vllm_ascend.ops.text_gated_residual import TextGatedResidualSerial


@pytest.mark.parametrize(
    "tokens,gate,gamma,aux",
    (
        (1, True, True, True),
        (4, False, False, True),
        (16, True, False, False),
        (64, True, True, False),
        (256, True, True, False),
        (2048, True, True, False),
    ),
)
def test_text_gated_residual_device(tokens, gate, gamma, aux):
    rng = torch.Generator().manual_seed(20261102 + tokens)
    x = torch.randn(1, tokens, 10240, generator=rng).to(torch.bfloat16)
    down_weight = (torch.randn(320, 10240, generator=rng) / math.sqrt(10240)).to(torch.bfloat16)
    up_weight = (torch.randn(10240, 320, generator=rng) / math.sqrt(320)).to(torch.bfloat16)
    gate_weight = (torch.randn(4, 10240, generator=rng) / math.sqrt(10240)).to(torch.bfloat16) if gate else None
    raw_gamma = torch.randn(10240, generator=rng) * 0.1 if gamma else None
    branches = x.double().reshape(1, tokens, 4, 2560)
    inv = torch.rsqrt(branches.square().mean(-1) + 1e-6)
    normalized = (branches * inv.unsqueeze(-1)).reshape(1, tokens, 10240)
    if gamma:
        normalized = normalized * (1 + raw_gamma.double())
    down = normalized @ down_weight.double().t()
    scaled = down / 4
    up = (scaled * torch.sigmoid(scaled)) @ up_weight.double().t()
    mixed = (normalized.reshape(1, tokens, 4, 2560) * torch.sigmoid(up).reshape(1, tokens, 4, 2560)).mean(2)
    inject = 2 * torch.sigmoid(normalized @ gate_weight.double().t() / 4) if gate else torch.empty(0)
    refs = (
        mixed,
        inject,
        inv if aux else torch.empty(0),
        down if aux else torch.empty(0),
        up if aux else torch.empty(0),
    )
    op = TextGatedResidualSerial(down_weight.npu(), up_weight.npu(), gate_weight.npu() if gate else None)
    actual = op(x.npu(), raw_gamma.npu() if gamma else None, need_backward=aux)
    torch.npu.synchronize()
    for index, (value, reference) in enumerate(zip(actual, refs)):
        assert value.shape == reference.shape
        assert value.dtype == (torch.float32 if index == 2 else torch.bfloat16)
        if not value.numel():
            continue
        error = (value.cpu().double() - reference).abs()
        atol, rtol, maximum = (2**-16, 2**-10, 0.01) if index == 2 else (2**-6, 2**-6, 1.0)
        assert torch.isfinite(value).all()
        assert (error <= atol + rtol * reference.abs()).double().mean() >= 0.99
        assert error.max() <= maximum
