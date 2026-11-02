# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial, _check_tensor, validate_cache_index


@pytest.mark.parametrize("indices,slots", [([0], 1), ([0, 15, 16, 31], 32), ([], 16)])
def test_valid_index_metadata(indices, slots):
    validate_cache_index(torch.tensor(indices, dtype=torch.int64), slots)


@pytest.mark.parametrize("indices,slots", [([-1], 16), ([16], 16), ([0, 0], 16)])
def test_reject_invalid_index_metadata(indices, slots):
    with pytest.raises(ValueError, match="in range and distinct"):
        validate_cache_index(torch.tensor(indices, dtype=torch.int64), slots)


def test_index_checker_requires_host_int64():
    with pytest.raises(ValueError, match="CPU int64"):
        validate_cache_index(torch.ones(2, dtype=torch.float32), 16)


@pytest.mark.parametrize("epsilon", [0, -1, math.inf, math.nan])
def test_reject_invalid_epsilon(epsilon):
    dq = torch.empty((7168, 1536), device="meta")
    uk = torch.empty((32, 128, 512), device="meta")
    with pytest.raises(ValueError, match="finite and positive"):
        MlaPrologSerial(dq, dq, uk, dq, dq, dq, eps_q=epsilon)


def test_check_tensor_shape_and_dtype():
    value = torch.empty((2, 4), dtype=torch.bfloat16)
    _check_tensor(value, (2, 4), value.device, "value")
    with pytest.raises(ValueError, match="shape"):
        _check_tensor(value, (4, 2), value.device, "value")
    with pytest.raises(ValueError, match="dtype"):
        _check_tensor(value.float(), (2, 4), value.device, "value")


def test_check_tensor_stride():
    value = torch.empty((2, 4), dtype=torch.bfloat16).T
    with pytest.raises(ValueError, match="contiguous"):
        _check_tensor(value, value.shape, value.device, "value")
