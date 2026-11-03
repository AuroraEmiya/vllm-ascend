# SPDX-License-Identifier: Apache-2.0
"""One-call SK Graph guard, including cache and same-address data refresh."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel


@pytest.mark.parametrize("case_id", ["decode-b1", "decode-b32"])
def test_mla_prolog_superkernel_single_call(case_id, monkeypatch):
    root = Path(__file__).resolve().parents[5]
    benchmark = root / "benchmarks/kernels/mla_prolog"
    monkeypatch.syspath_prepend(str(benchmark))
    spec = importlib.util.spec_from_file_location("mla_sk_validation", benchmark / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    monkeypatch.setattr(static_kernel, "_compute_opc_compile_jobs", lambda: "4")
    original = module.MlaPrologSerial
    monkeypatch.setattr(module, "MlaPrologSerial", lambda *a, **kw: original(*a, **kw).compile_superkernel())
    monkeypatch.setattr(module, "CALLS_PER_GRAPH", 1)
    cases = json.loads((benchmark / "cases.json").read_text())["cases"]
    result = module.run_case(next(c for c in cases if c["id"] == case_id), timing=False)
    assert result["status"] == "PASS", result
    assert all(check["pass"] for check in result["checks"].values()), result
