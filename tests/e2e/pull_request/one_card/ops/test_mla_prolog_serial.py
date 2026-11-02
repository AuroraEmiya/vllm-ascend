# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401


@pytest.mark.parametrize("batch", [1, 3, 16, 31, 33, 128])
def test_mla_prolog_decode_complete_graph(batch):
    root = Path(__file__).resolve().parents[5]
    benchmark = root / "benchmarks/kernels/mla_prolog"
    sys.path.insert(0, str(benchmark))
    spec = importlib.util.spec_from_file_location("mla_serial_validation", benchmark / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    result = module.run_case(
        {"id": f"st-b{batch}", "batch": batch, "hidden": 7168, "heads": 32, "seed": 20261102 + batch}, timing=False
    )
    assert result["status"] == "PASS", result
