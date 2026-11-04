# SPDX-License-Identifier: Apache-2.0
"""CPU wiring guards; actual static compilation is covered by the NPU STs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.compilation.serial_superkernel import CompiledSerialPipeline
from vllm_ascend.ops.serial_matmul import serial_matmul_v3


@pytest.fixture
def compiler(monkeypatch):
    events = []
    captured = {}
    npu = SimpleNamespace(
        NPUGraph=SimpleNamespace(super_kernel_optimize=True),
        synchronize=lambda: events.append("sync"),
        super_kernel_scope_begin=lambda scope: events.append(("begin", scope)),
        super_kernel_scope_end=lambda scope: events.append(("end", scope)),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "serial_matmul_v3_out", lambda *args: None, raising=False)

    def compile_forward(forward, **kwargs):
        captured.update(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", compile_forward)
    return events, captured


class PreparedPipeline:
    def __init__(self):
        self._matrix_backend = "torch"
        self.calls = []
        self.refresh_count = 0

    def __call__(self, value):
        self.calls.append((self._matrix_backend, value))
        return value

    def refresh_weights(self):
        self.refresh_count += 1


def test_execution_copy_preserves_owner_and_scopes_calls(compiler):
    events, captured = compiler
    owner = PreparedPipeline()
    compiled = CompiledSerialPipeline(owner, "mla_prolog")
    value = torch.ones(2)
    assert compiled(value) is value
    assert compiled(value + 1).equal(value + 1)
    assert owner._matrix_backend == "torch"
    assert [backend for backend, _ in owner.calls] == ["v3_nz", "v3_nz"]
    assert events.count(("begin", "mla_prolog")) == 2
    assert events.count(("end", "mla_prolog")) == 2
    assert captured["backend"] == "npugraph_ex"
    assert captured["options"]["static_kernel_compile"] is True
    assert captured["fullgraph"] is True
    compiled.refresh_weights()
    assert owner.refresh_count == 1


@pytest.mark.parametrize("optimize", [False, True])
def test_static_control_only_disables_outer_grouping(compiler, optimize):
    _, captured = compiler
    options = {"constant_codegen": 1, "preload_code": 5}
    compiled = CompiledSerialPipeline(PreparedPipeline(), "tgr", optimize=optimize, optimize_options=options)
    options["preload_code"] = 1
    groups = []
    graph = SimpleNamespace(super_kernel_optimize=lambda **kwargs: groups.append(kwargs))
    assert compiled.optimize_graph(graph) is graph
    assert captured["options"]["super_kernel_optimize"] is True
    assert len(groups) == int(optimize)
    if optimize:
        assert groups[0]["optimize_options"]["preload_code"] == 5


@pytest.mark.parametrize(
    "shape,weight_shape,transpose", [((2, 4), (4, 3), False), ((2, 4), (3, 4), True), ((2, 3, 4), (2, 4, 5), False)]
)
def test_matrix_wrapper_preserves_output_and_schema(monkeypatch, shape, weight_shape, transpose):
    calls = []

    def matrix_out(a, b, transpose_b, batched, out):
        calls.append((transpose_b, batched))
        out.copy_(a @ (b.transpose(-1, -2) if transpose_b else b))

    monkeypatch.setattr(torch.ops._C_ascend, "serial_matmul_v3_out", matrix_out, raising=False)
    a = torch.ones(shape, dtype=torch.bfloat16)
    b = torch.ones(weight_shape, dtype=torch.bfloat16)
    out = serial_matmul_v3(a, b, transpose_b=transpose)
    assert calls == [(transpose, len(shape) == 3)]
    assert out.dtype == a.dtype and out.device == a.device
    assert torch.equal(out, a @ (b.transpose(-1, -2) if transpose else b))


def test_original_import_is_a_compatibility_alias():
    from vllm_ascend.ops import serial_superkernel as old

    assert old.CompiledSerialPipeline is CompiledSerialPipeline
    assert old.serial_matmul_v3 is serial_matmul_v3
