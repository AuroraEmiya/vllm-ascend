# SPDX-License-Identifier: Apache-2.0
"""One-call SK Graph guard, including same-address input/weight refresh."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel


@pytest.mark.parametrize("case_id", ["decode-b2", "aux-five"])
def test_text_gated_residual_superkernel_single_call(case_id, monkeypatch):
    root = Path(__file__).resolve().parents[5]
    benchmark = root / "benchmarks/kernels/text_gated_residual"
    monkeypatch.syspath_prepend(str(benchmark))
    spec = importlib.util.spec_from_file_location("tgr_sk_validation", benchmark / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    monkeypatch.setattr(static_kernel, "_compute_opc_compile_jobs", lambda: "4")
    original = module.TextGatedResidualSerial
    monkeypatch.setattr(module, "TextGatedResidualSerial", lambda *a, **kw: original(*a, **kw).compile_superkernel())
    capture_original = module.capture
    monkeypatch.setattr(module, "capture", lambda call, stream, calls=1: capture_original(call, stream, calls=1))
    monkeypatch.setattr(module, "CALLS", 1)
    cases = json.loads((benchmark / "cases.json").read_text())
    result = module.check_case(next(c for c in cases if c["id"] == case_id), False)
    assert result["eager"]["serial"]["status"] == "PASS", result
    assert all(v["status"] == "PASS" for v in result["graph"]["serial"].values()), result
    assert result["input_immutability_eager"] and result["input_immutability_graph"], result
