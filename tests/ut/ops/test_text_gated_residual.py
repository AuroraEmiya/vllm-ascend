# SPDX-License-Identifier: Apache-2.0
"""Metadata guards run on CPU without loading an NPU extension."""

import pytest
import torch

from vllm_ascend.ops.text_gated_residual import TextGatedResidualSerial


@pytest.fixture
def weights():
    return torch.empty(320, 10240, dtype=torch.bfloat16), torch.empty(10240, 320, dtype=torch.bfloat16)


@pytest.mark.parametrize("eps", (0, -1, float("nan"), float("inf")))
def test_rejects_invalid_epsilon(weights, eps):
    with pytest.raises(ValueError, match="eps"):
        TextGatedResidualSerial(*weights, eps=eps)


def test_rejects_invalid_weight_metadata(weights):
    down, up = weights
    with pytest.raises(ValueError, match="channels"):
        TextGatedResidualSerial(down, up, channels=2)
    with pytest.raises(ValueError, match="w_down"):
        TextGatedResidualSerial(down.float(), up)
    with pytest.raises(ValueError, match="w_up"):
        TextGatedResidualSerial(down, up.t().contiguous())
    with pytest.raises(ValueError, match="w_gate"):
        TextGatedResidualSerial(down, up, torch.empty(3, 10240, dtype=torch.bfloat16))


@pytest.mark.parametrize("shape", ((1, 0, 10240), (0, 1, 10240), (1, 2049, 10240), (1, 1, 2560), (1, 10240)))
def test_rejects_invalid_input_shapes(weights, shape):
    op = TextGatedResidualSerial(*weights)
    with pytest.raises(ValueError, match="shape"):
        op(torch.empty(shape, dtype=torch.bfloat16))


def test_rejects_input_dtype_and_gamma(weights):
    op = TextGatedResidualSerial(*weights)
    with pytest.raises(ValueError, match="BF16"):
        op(torch.empty(1, 1, 10240))
    with pytest.raises(ValueError, match="gamma"):
        op(torch.empty(1, 1, 10240, dtype=torch.bfloat16), gamma=torch.empty(10240, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="need_backward"):
        op(torch.empty(1, 1, 10240, dtype=torch.bfloat16), need_backward=1)


def test_prepared_weight_views_follow_same_address_updates(weights):
    down, up = weights
    op = TextGatedResidualSerial(down, up)
    down.fill_(1)
    up.fill_(2)
    assert op.w_down.data_ptr() == down.data_ptr()
    assert op.w_up.data_ptr() == up.data_ptr()
    assert torch.equal(op.w_down, down.t())
    assert torch.equal(op.w_up, up.t())


def test_prepared_gate_weights_refresh_without_changing_address(weights):
    down, up = weights
    down.fill_(1)
    gate = torch.ones(4, 10240, dtype=torch.bfloat16)
    op = TextGatedResidualSerial(down, up, gate)
    pointer = op.packed_weight.data_ptr()
    down.fill_(2)
    gate.fill_(3)
    op.refresh_weights_()
    assert op.packed_weight.data_ptr() == pointer
    assert torch.equal(op.packed_weight[:320], down)
    assert torch.equal(op.packed_weight[320:324], gate)
    assert torch.count_nonzero(op.packed_weight[324:]) == 0
