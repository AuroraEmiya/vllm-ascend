from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def test_glm5_sparse_attention_device_contract_is_a5_only():
    assert not BaseDeviceAdaptor.supports_glm5_kv_quant_sparse_attn()
    assert BaseDeviceAdaptor.get_glm5_mla_cache_layout(512, 0) == (
        torch.bfloat16,
        512,
    )

    assert A5DeviceAdaptor.supports_glm5_kv_quant_sparse_attn()
    assert A5DeviceAdaptor.get_glm5_mla_cache_layout(512, 0) == (
        torch.float8_e4m3fn,
        544,
    )
    assert A5DeviceAdaptor.get_glm5_sparse_attn_metadata_kwargs(torch.device("cpu")) == {"kv_quant_mode": 1}
    assert A5DeviceAdaptor.get_glm5_sparse_attn_base_kwargs() == {
        "kv_quant_mode": 1,
        "tile_size": 64,
        "rope_head_dim": 0,
    }
    # The existing DSA contract must retain its 64-dimensional RoPE.
    assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs()["rope_head_dim"] == 64


def test_a5_glm5_cache_writer_uses_quantized_epilog():
    cache = torch.empty((2, 4, 1, 544), dtype=torch.float8_e4m3fn)
    values = torch.randn((3, 512), dtype=torch.bfloat16)
    slots = torch.tensor([0, 3, 5], dtype=torch.int64)

    with mock.patch.object(
        torch.ops._C_ascend,
        "kv_compress_epilog",
        create=True,
    ) as mock_epilog:
        A5DeviceAdaptor.store_glm5_mla_cache(cache, values, slots)

    mock_epilog.assert_called_once()
    kwargs = mock_epilog.call_args.kwargs
    assert kwargs["kv_compress_cache"].shape == (8, 1, 544)
    assert kwargs["x"].shape == (3, 512)
    assert kwargs["slot_mapping"] is slots
    assert kwargs["quant_group_size"] == 64
    assert kwargs["quant_mode"] == 2
    assert kwargs["layout"] == 1


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
