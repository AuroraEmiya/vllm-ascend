# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton.dcp import dcp_a2a


def _npu_metadata(tensor):
    """Exercise host validation with real CPU tensor layouts and no NPU work."""
    return Mock(
        spec=torch.Tensor,
        shape=tensor.shape,
        ndim=tensor.ndim,
        dtype=tensor.dtype,
        device=SimpleNamespace(type="npu"),
        stride=tensor.stride,
        is_contiguous=tensor.is_contiguous,
    )


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("dtype,packed_extra", [(torch.float32, 1), (torch.bfloat16, 4)])
def test_deferred_fake_layout(scatter_dim, dtype, packed_extra):
    output = torch.empty(8, 12, 96, dtype=dtype, device="meta")
    lse = torch.empty(8, 12, 1, device="meta")
    actual = dcp_a2a.dcp_a2a_fused_fake(output, lse, 2, scatter_dim, "dcp", defer_combine=True)
    assert actual.shape == (2, output.shape[scatter_dim] // 2, output.shape[1 - scatter_dim], 96 + packed_extra)
    assert actual.dtype == dtype


@pytest.mark.parametrize("scatter_size", [1, 2])
def test_deferred_exchange_skips_combine(monkeypatch, scatter_size):
    send = torch.empty(scatter_size, 2, 3, 5)
    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(return_value=send))
    collective = Mock()
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    combine = Mock(side_effect=AssertionError("deferred exchange must not combine"))
    monkeypatch.setattr(dcp_a2a, "fused_dcp_lse_combine", combine)
    result = dcp_a2a.dcp_a2a_fused_combine(
        torch.empty(2, 3, 4), torch.empty(2, 3, 1), scatter_size, 1, object(), defer_combine=True
    )
    assert result.shape == send.shape
    assert collective.call_count == int(scatter_size > 1)
    combine.assert_not_called()
    if scatter_size == 1:
        assert result is send


@pytest.mark.parametrize("feature_stride", [0, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("from_stats", [False, True])
@pytest.mark.parametrize("entrypoint", ["pack", "fused"])
def test_pack_rejects_non_unit_feature_stride(monkeypatch, feature_stride, dtype, scatter_dim, from_stats, entrypoint):
    storage = torch.empty(4, 8, 16 * max(feature_stride, 1), dtype=dtype)
    view = storage[..., ::feature_stride] if feature_stride else storage[..., :1].expand(4, 8, 16)
    output = _npu_metadata(view)
    lse = _npu_metadata(torch.empty(4, 8, 1))
    total = _npu_metadata(torch.ones(4, 8, 1)) if from_stats else None
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setitem(dcp_a2a._groups, "feature-stride-contract", lambda: group)
    allocate = Mock(side_effect=AssertionError("invalid layout must fail before allocation"))
    launch = Mock(side_effect=AssertionError("invalid layout must fail before a kernel launch"))
    collective = Mock(side_effect=AssertionError("invalid layout must fail before communication"))
    fast_exchange = Mock(side_effect=AssertionError("invalid layout must fail before fast exchange"))
    monkeypatch.setattr(dcp_a2a.torch, "empty", allocate)
    monkeypatch.setattr(dcp_a2a, "_launch_kernel", launch)
    monkeypatch.setattr(dcp_a2a.dist, "all_to_all_single", collective)
    monkeypatch.setattr(dcp_a2a, "exchange", fast_exchange)

    with pytest.raises(ValueError, match=rf"partial_output.stride\(-1\) == 1, got {feature_stride}\."):
        if entrypoint == "pack":
            dcp_a2a.pack_dcp_output_lse(output, lse, 2, scatter_dim, softmax_sum=total)
        else:
            dcp_a2a.dcp_a2a_fused(output, lse, 2, scatter_dim, "feature-stride-contract", softmax_sum=total)
    allocate.assert_not_called()
    launch.assert_not_called()
    collective.assert_not_called()
    fast_exchange.assert_not_called()


@pytest.mark.parametrize("head_major", [False, True])
@pytest.mark.parametrize("from_stats", [False, True])
def test_pack_accepts_unit_feature_stride_with_strided_stats(monkeypatch, head_major, from_stats):
    view = torch.empty(8, 4, 16).transpose(0, 1) if head_major else torch.empty(4, 8, 16)
    output = _npu_metadata(view)
    lse = _npu_metadata(torch.empty(4, 8, 2)[..., :1])
    total = _npu_metadata(torch.ones(4, 8, 3)[..., :1]) if from_stats else None
    send = torch.empty(2, 4, 4, 17)
    monkeypatch.setattr(dcp_a2a.torch, "empty", Mock(return_value=send))
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", Mock())
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", Mock(return_value=1))
    launch = Mock()
    monkeypatch.setattr(dcp_a2a, "_launch_pack_dcp_output_lse", launch)

    assert dcp_a2a.pack_dcp_output_lse(output, lse, 2, 1, softmax_sum=total) is send
    launch.assert_called_once_with(output, lse, total, send, 1, 1, dcp_a2a._launch_kernel)


@pytest.mark.parametrize("from_stats", [False, True])
def test_fast_exchange_accepts_head_major_and_strided_stats(monkeypatch, from_stats):
    output = _npu_metadata(torch.empty(8, 4, 16, dtype=torch.bfloat16).transpose(0, 1))
    lse = _npu_metadata(torch.empty(4, 8, 2)[..., :1])
    total = _npu_metadata(torch.ones(4, 8, 3)[..., :1]) if from_stats else None
    result = torch.empty(4, 4, 16, dtype=torch.bfloat16)
    group = SimpleNamespace(world_size=2, device_group=object())
    monkeypatch.setitem(dcp_a2a._groups, "feature-stride-contract", lambda: group)
    fast_exchange = Mock(return_value=result)
    monkeypatch.setattr(dcp_a2a, "exchange", fast_exchange)
    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", Mock(side_effect=AssertionError("expected fast exchange")))

    assert dcp_a2a.dcp_a2a_fused(output, lse, 2, 1, "feature-stride-contract", softmax_sum=total) is result
    fast_exchange.assert_called_once_with(output, lse, group.device_group, softmax_sum=total)


def test_combine_retains_feature_strided_local_input(monkeypatch):
    recv = _npu_metadata(torch.empty(2, 4, 4, 20, dtype=torch.bfloat16))
    local = _npu_metadata(torch.empty(4, 4, 32, dtype=torch.bfloat16)[..., ::2])
    lse = _npu_metadata(torch.empty(4, 4, 3)[..., :1])
    result = torch.empty(4, 4, 16, dtype=torch.bfloat16)
    monkeypatch.setattr(dcp_a2a.torch, "empty", Mock(return_value=result))
    monkeypatch.setattr(dcp_a2a, "init_device_properties_triton", Mock())
    monkeypatch.setattr(dcp_a2a, "get_vectorcore_num", Mock(return_value=1))
    launch = Mock()
    monkeypatch.setattr(dcp_a2a, "_launch_fused_dcp_lse_combine", launch)

    assert dcp_a2a.fused_dcp_lse_combine(recv, 16, 1, local_output=local, local_lse=lse) is result
    launch.assert_called_once_with(recv, result, 16, 1, False, local, lse, 1, dcp_a2a._launch_kernel)
