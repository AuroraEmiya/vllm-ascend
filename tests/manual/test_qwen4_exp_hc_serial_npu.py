# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Device UT and bounded full-path Graph benchmark for the serial HC operator.

The independent reference is CPU float64 with the specified BF16 materialization
boundaries. The performance baseline imports the original pinned Triton source.
No model weights, server configuration, or profiler outputs belong in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SHA256 = "28c6c8c6f26ff6c92023334eaed86d3c8606200dd27a67446cef1d624a0c1f91"
NAMES = ("combined_residual", "block_input", "next_injection")
BOUNDARIES = [
    {"id": "tail-strided", "m": 3, "h": 65, "k": 3, "r": 17, "strided": True},
    {"id": "no-injection-shared", "m": 2, "h": 63, "k": 1, "r": 15, "injection": False, "shared_norm": True},
    {"id": "final-mixer", "m": 3, "h": 127, "k": 4, "r": 33, "use_combine": False},
    {"id": "empty", "m": 0, "h": 65, "k": 3, "r": 17},
    {"id": "saturated-injection", "m": 2, "h": 511, "k": 4, "r": 31, "saturated": True},
]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def implementations(extension=None, weights="original"):
    if weights != "original":
        raise ValueError("Experimental prepared-weight paths are not part of the final implementation")
    if extension:
        torch.ops.load_library(str(Path(extension).resolve()))
    elif not hasattr(torch.ops._C_ascend, "qwen_hc_combine_norm"):
        import vllm_ascend_C  # noqa: F401
    serial = load_file("_hc_serial_candidate", ROOT / "vllm_ascend/ops/qwen_hc_serial.py")
    path = ROOT / "vllm_ascend/ops/triton/qwen4_exp/hc.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise RuntimeError("The original Triton baseline source changed")
    original = load_file("_hc_original_baseline", path)

    def triton_serial(residual, block, injection, weight, down, up, **kw):
        k = kw["hc_count"]
        combined, normalized = original.hc_combine_norm(residual, block, injection, weight, kw["eps"], k)
        projected = F.linear(normalized, down)
        rank = up.shape[1]
        lowrank = original.hc_silu(projected[:, :rank], k)
        next_injection = projected[:, rank : rank + k] if kw["use_combine"] else None
        gate = F.linear(lowrank, up)
        return combined, original.hc_gate_mix(normalized, gate, k), next_injection

    def candidate(*inputs, **kw):
        return serial.combine_and_mix(*inputs, **kw)

    return {"ascendc_serial": candidate, "triton_serial": triton_serial}


def bind_inputs(funcs, inputs):
    return {
        label: (function.bind_inputs(inputs) if hasattr(function, "bind_inputs") else function)
        for label, function in funcs.items()
    }


def make_inputs(case):
    m, h, k, r = (case[x] for x in ("m", "h", "k", "r"))
    d = h * k
    combine = case.get("use_combine", True)
    p = (-(r + k)) % 16 if combine else 0
    width = r + k + p if combine else r
    generator = torch.Generator().manual_seed(case.get("seed", 1701))

    def sample(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    cpu = [
        sample((m, d)),
        sample((m, h)),
        sample((m, k)),
        sample((h if case.get("shared_norm") else d,), 0.1),
        sample((width, d), 1 / math.sqrt(d)),
        sample((d, r), 1 / math.sqrt(r)),
    ]
    if not case.get("injection", True):
        cpu[2] = None
    elif case.get("saturated") and m:
        cpu[2][0, :] = 80
        cpu[2][1, :] = -80
        cpu[0][0, :] = 0
        cpu[1][0, :] = 0
    device = []
    for index, x in enumerate(cpu):
        if x is None:
            device.append(None)
        elif case.get("strided") and index < 3:
            storage = torch.empty((m, x.shape[1] + 7), dtype=x.dtype, device="npu")
            view = storage[:, : x.shape[1]]
            view.copy_(x)
            device.append(view)
        else:
            device.append(x.to("npu"))
    kw = {"eps": case.get("eps", 1e-6), "hc_count": k, "use_combine": combine}
    return cpu, device, kw


def reference(cpu, kw):
    residual, block, injection, weight, down, up = cpu
    m, d = residual.shape
    k = kw["hc_count"]
    h = d // k

    def materialize(value):
        return value.to(torch.bfloat16).double()

    scale = 1 if injection is None else (2 * torch.sigmoid(injection.double() / k)).reshape(m, k, 1)
    combined = materialize(residual.double().reshape(m, k, h) + block.double().reshape(m, 1, h) * scale)
    inverse_rms = torch.rsqrt((combined * combined).mean(-1, keepdim=True) + kw["eps"])
    normalized = combined * inverse_rms
    affine = weight.double().reshape(1, 1 if weight.numel() == h else k, h)
    normalized = materialize(normalized + normalized * affine).reshape(m, d)
    projected = materialize(normalized @ down.double().t())
    r = up.shape[1]
    z = projected[:, :r] / k
    activated = materialize(z * torch.sigmoid(z))
    gate = materialize(activated @ up.double().t()).reshape(m, k, h)
    mixed = (normalized.reshape(m, k, h) * torch.sigmoid(gate)).sum(1) / k
    injection_out = projected[:, r : r + k].to(torch.bfloat16) if kw["use_combine"] else None
    return combined.reshape(m, d).to(torch.bfloat16), mixed.to(torch.bfloat16), injection_out


def compare(actual, expected):
    if actual is None or expected is None:
        return {"pass": actual is None and expected is None, "none": True}
    if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
        return {"pass": False, "reason": "shape or dtype mismatch"}
    a, b = actual.cpu().double(), expected.double()
    if not a.numel():
        return {"pass": True, "elements": 0, "matched_ratio": 1, "max_abs_error": 0}
    finite = torch.isfinite(a) & torch.isfinite(b)
    same_special = torch.isnan(a) & torch.isnan(b) | (torch.isinf(a) & (a == b))
    delta = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    matched = (finite & (delta <= 2**-6 + 2**-6 * b.abs())) | same_special
    ratio, max_abs = matched.double().mean().item(), delta.max().item()
    special_ok = bool((finite | same_special).all())
    return {
        "pass": special_ok and ratio >= 0.99 and max_abs <= 1.0,
        "elements": a.numel(),
        "matched_ratio": ratio,
        "max_abs_error": max_abs,
        "special_values_match": special_ok,
        "atol": 2**-6,
        "rtol": 2**-6,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": 1.0,
    }


def metadata(outputs, inputs, case):
    s, y, injection = outputs
    m, h, k, r = (case[x] for x in ("m", "h", "k", "r"))
    ok = s.shape == (m, k * h) and y.shape == (m, h)
    if m:
        ok = ok and s.data_ptr() != y.data_ptr()
        ok = ok and all(out.data_ptr() != x.data_ptr() for out in (s, y) for x in inputs if x is not None)
    if case.get("use_combine", True):
        width = r + k + (-(r + k)) % 16
        ok = ok and injection.shape == (m, k) and injection.stride() == (width, 1)
        ok = ok and injection.storage_offset() == r
    else:
        ok = ok and injection is None
    return bool(ok)


def capture(function, inputs, kw, calls=1):
    for _ in range(4):
        function(*inputs, **kw)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            outputs = function(*inputs, **kw)
    torch.npu.synchronize()
    return graph, outputs


def validate_case(case, funcs):
    cpu, inputs, kw = make_inputs(case)
    funcs = bind_inputs(funcs, inputs)
    expected = reference(cpu, kw)
    before = [None if x is None else x.cpu().clone() for x in inputs]
    results = {"id": case["id"], "case": case, "implementations": {}}
    for label, function in funcs.items():
        outputs = function(*inputs, **kw)
        torch.npu.synchronize()
        eager = {name: compare(a, b) for name, a, b in zip(NAMES, outputs, expected)}
        abi_ok = metadata(outputs, inputs, case)
        graph, graph_outputs = capture(function, inputs, kw)
        graph.replay()
        torch.npu.synchronize()
        replay = {name: compare(a, b) for name, a, b in zip(NAMES, graph_outputs, expected)}
        mutated = any(not torch.equal(x.cpu(), old) for x, old in zip(inputs, before) if x is not None)
        fresh = None
        if case["id"] == "tail-strided":
            inputs[1].add_(0.125)
            changed = [None if x is None else x.cpu() for x in inputs]
            changed_expected = reference(changed, kw)
            graph.replay()
            torch.npu.synchronize()
            fresh = {name: compare(a, b) for name, a, b in zip(NAMES, graph_outputs, changed_expected)}
            inputs[1].copy_(before[1])
        passed = (
            abi_ok
            and not mutated
            and all(x["pass"] for x in list(eager.values()) + list(replay.values()))
            and (fresh is None or all(x["pass"] for x in fresh.values()))
        )
        results["implementations"][label] = {
            "pass": passed,
            "eager": eager,
            "graph": replay,
            "fresh_input_replay": fresh,
            "abi_pass": abi_ok,
            "inputs_mutated": mutated,
        }
        print(
            json.dumps({"case": case["id"], "implementation": label, "validation": results["implementations"][label]}),
            flush=True,
        )
        del outputs, graph_outputs, graph
    return results


def timed_replays(graph, calls, repeats):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * repeats)


def benchmark(case, funcs, samples=25, calls=8, repeats=4):
    _, inputs, kw = make_inputs(case)
    funcs = bind_inputs(funcs, inputs)
    graphs = {}
    kept_outputs = {}
    for label, function in funcs.items():
        for _ in range(20):
            function(*inputs, **kw)
        graphs[label], kept_outputs[label] = capture(function, inputs, kw, calls)
        for _ in range(10):
            graphs[label].replay()
    torch.npu.synchronize()
    raw = {label: [] for label in funcs}
    for index in range(samples):
        labels = list(funcs) if index % 2 == 0 else list(reversed(funcs))
        for label in labels:
            raw[label].append(timed_replays(graphs[label], calls, repeats))
    medians = {label: statistics.median(values) for label, values in raw.items()}
    return {
        "id": case["id"],
        "case": case,
        "weight_preparation": getattr(funcs["ascendc_serial"], "preparation", None),
        "graph_calls": calls,
        "replays_per_sample": repeats,
        "samples_us": raw,
        "medians_us": medians,
        "speedup": medians["triton_serial"] / medians["ascendc_serial"],
        "target": 1.5,
        "pass": medians["triton_serial"] / medians["ascendc_serial"] >= 1.5,
    }


def test_qwen_hc_serial_boundary_cases():
    """Run with the main repository registered extension on an Ascend device."""
    import pytest

    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    torch.npu.set_device(0)
    funcs = implementations()
    for case in BOUNDARIES:
        result = validate_case(case, funcs)
        assert all(x["pass"] for x in result["implementations"].values()), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "validate", "bench", "profile"), default="validate")
    parser.add_argument("--weights", choices=("original",), default="original")
    parser.add_argument("--case")
    parser.add_argument("--impl", choices=("ascendc_serial", "triton_serial"), default="ascendc_serial")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    funcs = implementations(args.extension, args.weights)
    cases = BOUNDARIES
    if args.cases:
        cases = json.loads(args.cases.read_text())["cases"]
    if args.case:
        cases = [x for x in cases if x["id"] == args.case]
        if not cases:
            raise ValueError("Unknown case")
    if args.mode == "smoke":
        cases = [BOUNDARIES[0]]
    result = {
        "mode": args.mode,
        "weight_mode": args.weights,
        "started": time.time(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "soc": torch.npu.get_device_name(0),
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if args.mode in ("smoke", "validate"):
            row = validate_case(case, funcs)
            row["pass"] = all(x["pass"] for x in row["implementations"].values())
        elif args.mode == "bench":
            row = benchmark(case, funcs)
        else:
            _, inputs, kw = make_inputs(case)
            function = bind_inputs({args.impl: funcs[args.impl]}, inputs)[args.impl]
            graph, outputs = capture(function, inputs, kw)
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            print("HC_PROFILE_READY", case["id"], args.impl, flush=True)
            for _ in range(16):
                graph.replay()
            torch.npu.synchronize()
            row = {
                "id": case["id"],
                "implementation": args.impl,
                "profile_replays": 16,
                "outputs": [None if x is None else list(x.shape) for x in outputs],
            }
        result["cases"].append(row)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
    result["completed"] = time.time()
    args.out.write_text(json.dumps(result, indent=2))
    if any(not x.get("pass", True) for x in result["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
