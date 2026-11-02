"""Independent public-math oracle, eager/Graph checks and paired Graph timing.

No fused implementation is inspected. The baseline is invoked as a black box.
Each captured Graph contains ten complete forward calls, including all V/C work.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

import cann_ops_transformer_custom  # noqa: F401 - registers the opaque baseline
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as native_extension

from vllm_ascend.ops.text_gated_residual import TextGatedResidualSerial

NAMES = ("mixed", "inject", "inv_rms", "down_pre_act", "up_pre_act")
CALLS = 10


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def make_inputs(case):
    g = torch.Generator().manual_seed(case["seed"])
    b, s = case["B"], case["S"]
    x = torch.randn(b, s, 10240, generator=g).to(torch.bfloat16)
    wd = (torch.randn(320, 10240, generator=g) / math.sqrt(10240)).to(torch.bfloat16)
    wu = (torch.randn(10240, 320, generator=g) / math.sqrt(320)).to(torch.bfloat16)
    wg = (torch.randn(4, 10240, generator=g) / math.sqrt(10240)).to(torch.bfloat16) if case["gate"] else None
    gamma = torch.randn(10240, generator=g) * 0.1 if case["gamma"] else None
    if case["input_mode"] == "zero":
        x.zero_()
    if case["input_mode"] == "gamma-minus-one":
        gamma.fill_(-1)
    return [x, wd, wu, wg, gamma]


def oracle(cpu, case):
    """FP64 equation implementation from the public paper and raw-gamma ABI."""
    x, wd, wu, wg, gamma = cpu
    b, s, d = x.shape
    z = x.double().reshape(b, s, 4, 2560)
    inv = torch.rsqrt(z.square().mean(-1) + 1e-6)
    norm = (z * inv.unsqueeze(-1)).reshape(b, s, d)
    if gamma is not None:
        norm = norm * (1 + gamma.double())
    down = norm @ wd.double().t()
    scaled = down / 4
    act = scaled * torch.sigmoid(scaled)
    up = act @ wu.double().t()
    mixed = (norm.reshape(b, s, 4, 2560) * torch.sigmoid(up).reshape(b, s, 4, 2560)).mean(2)
    inject = 2 * torch.sigmoid((norm @ wg.double().t()) / 4) if wg is not None else torch.empty(0, dtype=torch.bfloat16)
    return (
        mixed,
        inject,
        inv if case["need_backward"] else torch.empty(0, dtype=torch.float32),
        down if case["need_backward"] else torch.empty(0, dtype=torch.bfloat16),
        up if case["need_backward"] else torch.empty(0, dtype=torch.bfloat16),
    )


def check_outputs(outputs, refs):
    rows = []
    if len(outputs) != 5:
        return {"status": "FAIL", "reason": "output arity", "actual": len(outputs)}
    for i, (a, ref) in enumerate(zip(outputs, refs)):
        expected_dtype = torch.float32 if i == 2 else torch.bfloat16
        row = {"name": NAMES[i], "shape": list(a.shape), "dtype": str(a.dtype)}
        structural = a.shape == ref.shape and a.dtype == expected_dtype
        row["structure"] = bool(structural)
        if not structural:
            row["status"] = "FAIL"
        elif not a.numel():
            row["status"] = "PASS"
        else:
            av, rv = a.detach().cpu().double(), ref.double()
            err = (av - rv).abs()
            atol, rtol, limit = (2**-16, 2**-10, 0.01) if i == 2 else (2**-6, 2**-6, 1.0)
            finite = bool(torch.isfinite(av).all() and torch.isfinite(rv).all())
            ratio = float((err <= atol + rtol * rv.abs()).double().mean()) if finite else 0.0
            maximum = float(err.max()) if finite else None
            row.update(
                finite=finite,
                max_abs=maximum,
                matched_ratio=ratio,
                atol=atol,
                rtol=rtol,
                required_matched_ratio=0.99,
                max_abs_limit=limit,
            )
            row["status"] = "PASS" if finite and ratio >= 0.99 and maximum <= limit else "FAIL"
        rows.append(row)
    return {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "outputs": rows}


def identity(work_root):
    import inspect

    paths = [Path(__file__), Path(inspect.getfile(TextGatedResidualSerial)), Path(native_extension.__file__)]
    root = work_root
    for vendor in (
        root / "baseline-opp/vendors/custom_transformer",
        Path(__file__).resolve().parents[3] / "vllm_ascend/_cann_ops_custom/vendors/tgr_serial",
    ):
        # Hash opaque runtime objects only; their implementation is not inspected.
        paths.extend(sorted(vendor.glob("op_api/lib/*.so")))
        paths.extend(sorted(vendor.glob("op_impl/ai_core/tbe/kernel/ascend910_93/**/*.o")))
    return {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_name": torch.npu.get_device_name(0),
        "device_count": torch.npu.device_count(),
        "allow_hf32_matmul": bool(torch.npu.matmul.allow_hf32),
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "loaded_libraries": [],
    }


def make_calls(cpu, case):
    tensors = [a.npu() if a is not None else None for a in cpu]
    x, wd, wu, wg, gamma = tensors
    before = time.perf_counter()
    serial = TextGatedResidualSerial(wd, wu, wg)
    torch.npu.synchronize()
    prep_us = (time.perf_counter() - before) * 1e6

    def candidate():
        return serial(x, gamma, case["need_backward"])

    def fused():
        return torch.ops.cann_ops_transformer.text_gated_residual(
            x, wd, wu, w_gate=wg, channels=4, gamma=gamma, eps=1e-6, need_backward=case["need_backward"]
        )

    return (
        tensors,
        {"fused": fused, "serial": candidate},
        {
            "wall_us": prep_us,
            "description": "caller-owned weight views and dummy tensors; no input-dependent precomputation",
            "persistent_tensor_bytes": 6,
            "rebuild": "new prepared object for changed weight address/metadata",
        },
    )


def capture(call, stream, calls=CALLS):
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(calls):
            outputs = call()
    return graph, outputs


def graph_samples(graphs, stream):
    samples = []
    for pair in range(20):
        order = ("fused", "serial") if pair % 2 == 0 else ("serial", "fused")
        values = {}
        for name in order:
            start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            with torch.npu.stream(stream):
                start.record()
                graphs[name].replay()
                stop.record()
            stop.synchronize()
            values[name + "_graph_ms_raw"] = float(start.elapsed_time(stop))
            values[name + "_us"] = values[name + "_graph_ms_raw"] * 1000 / CALLS
        samples.append({"pair": pair, "order": list(order), **values})
    med = {n + "_us": statistics.median(r[n + "_us"] for r in samples) for n in ("fused", "serial")}
    return {
        "status": "PASS",
        "calls_per_graph": CALLS,
        "pairs": 20,
        "raw_samples": samples,
        **med,
        "fused_min_max_us": [min(r["fused_us"] for r in samples), max(r["fused_us"] for r in samples)],
        "serial_min_max_us": [min(r["serial_us"] for r in samples), max(r["serial_us"] for r in samples)],
        "speedup": med["fused_us"] / med["serial_us"],
        "target_pass": med["fused_us"] / med["serial_us"] >= 1.5,
        "floor_pass": med["fused_us"] / med["serial_us"] >= 1.0,
    }


def check_case(case, timing):
    cpu = make_inputs(case)
    refs = oracle(cpu, case)
    tensors, calls, prep = make_calls(cpu, case)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    result = {
        "id": case["id"],
        "case": case,
        "preparation": prep,
        "eager": {},
        "graph": {},
        "performance": {"status": "NOT_RUN"},
    }
    graphs, gout = {}, {}
    with torch.npu.stream(stream):
        for name, call in calls.items():
            for _ in range(10):
                eager = call()
            stream.synchronize()
            result["eager"][name] = check_outputs(eager, refs)
        result["input_immutability_eager"] = all(t is None or torch.equal(t.cpu(), c) for t, c in zip(tensors, cpu))
        for name, call in calls.items():
            graphs[name], gout[name] = capture(call, stream)
            for _ in range(10):
                graphs[name].replay()
            stream.synchronize()
            result["graph"][name] = {"original": check_outputs(gout[name], refs)}
        # Keep addresses stable while changing data, raw gamma and weights.
        updated = [t.clone() if t is not None else None for t in cpu]
        updated[0] = (updated[0].float() * -0.7 + 0.03125).to(torch.bfloat16)
        updated[1].mul_(-1)
        updated[2].mul_(-1)
        if updated[3] is not None:
            updated[3].mul_(-1)
        if updated[4] is not None:
            updated[4].add_(0.0625)
        for t, c in zip(tensors, updated):
            if t is not None:
                t.copy_(c)
        updated_refs = oracle(updated, case)
        for name in calls:
            graphs[name].replay()
            stream.synchronize()
            result["graph"][name]["updated_same_address"] = check_outputs(gout[name], updated_refs)
        result["input_immutability_graph"] = all(t is None or torch.equal(t.cpu(), c) for t, c in zip(tensors, updated))
        # Restore the identical frozen seeded input outside both timed Graphs.
        for t, c in zip(tensors, cpu):
            if t is not None:
                t.copy_(c)
        for name in calls:
            graphs[name].replay()
        stream.synchronize()
    valid = (
        all(r["status"] == "PASS" for r in result["eager"].values())
        and all(r["status"] == "PASS" for phases in result["graph"].values() for r in phases.values())
        and result["input_immutability_eager"]
        and result["input_immutability_graph"]
    )
    result["comparison_admitted"] = valid
    if timing and case["role"] == "acceptance" and valid:
        result["performance"] = graph_samples(graphs, stream)
    elif not valid:
        result["performance"] = {"status": "NOT_RUN", "reason": "eager or Graph correctness gate failed"}
    return result


def profile(case, out):
    cpu = make_inputs(case)
    tensors, calls, prep = make_calls(cpu, case)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(10):
            calls["serial"]()
        stream.synchronize()
        graph, outputs = capture(calls["serial"], stream)
        for _ in range(10):
            graph.replay()
        stream.synchronize()
        # Warmups and capture precede the two explicit target replay markers.
        markers = []
        for i in range(2):
            a = time.time_ns()
            graph.replay()
            stream.synchronize()
            markers.append({"replay": i, "host_before_ns": a, "host_after_ns": time.time_ns()})
    write_json(
        out,
        {
            "case": case,
            "calls_per_graph": CALLS,
            "target_replays": markers,
            "warmup_eager": 10,
            "warmup_graph": 10,
            "preparation": prep,
            "note": "Full msprof application includes warmup/capture; retain last two complete replays only",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    if args.case_id:
        cases = [c for c in cases if c["id"] in args.case_id]
        assert len(cases) == len(set(args.case_id)), "Unknown case"
    if args.profile:
        assert len(cases) == 1
        profile(cases[0], args.out)
        return
    data = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": args.source_commit,
        "identity": identity(args.work_root),
        "cases": [],
        "protocol": {
            "oracle": "CPU FP64 public equations",
            "calls_per_graph": CALLS,
            "eager_warmup": 10,
            "graph_warmup": 10,
            "pairs": 20,
            "order": "AB/BA alternation",
            "stream": "one explicit NPU stream per case",
            "thresholds_frozen": True,
        },
    }
    write_json(args.out, data)
    for case in cases:
        print("CASE_BEGIN", case["id"], flush=True)
        try:
            row = check_case(case, not args.no_timing)
        except Exception:
            row = {
                "id": case["id"],
                "case": case,
                "status": "INVALID",
                "error": traceback.format_exc(),
                "performance": {"status": "NOT_RUN"},
            }
        data["cases"].append(row)
        maps = Path("/proc/self/maps").read_text().splitlines()
        data["identity"]["loaded_libraries"] = sorted(
            set(
                line.split()[-1]
                for line in maps
                if "/" in line
                and any(
                    s in line for s in ("vllm_ascend_C", "tgr_serial", "custom_transformer", "cann_ops_transformer")
                )
            )
        )
        write_json(args.out, data)
        print(
            "CASE_END",
            row["id"],
            json.dumps(
                {
                    "valid": row.get("comparison_admitted"),
                    "performance": {k: v for k, v in row["performance"].items() if k != "raw_samples"},
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
        if row.get("status") == "INVALID":
            break  # Async device errors may poison a context; preserve the first error.
    data["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(args.out, data)
    print("VALIDATION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
