# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Four-way HC comparison isolating public matmul versus linear entry points.

Keep the original weights and vector stages. Weight transpose is a view; no
prepared weights or hidden cache is introduced. Use the same inputs for all
four implementations and keep full Graph timing separate from profiler sums.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu
from test_qwen4_exp_hc_serial_npu import (
    BOUNDARIES,
    NAMES,
    ROOT,
    bind_inputs,
    capture,
    implementations,
    load_file,
    make_inputs,
    timed_replays,
    validate_case,
)

EAGER_WARMUPS = 20
GRAPH_WARMUPS = 10
TIMING_SAMPLES = 25
GRAPH_CALLS = 8
REPLAYS_PER_SAMPLE = 4
PROFILE_WARMUPS = 5
PROFILE_REPLAYS = 16


def four_implementations(extension):
    controls = implementations(extension)
    candidate = load_file("_hc_matmul_candidate", ROOT / "vllm_ascend/ops/qwen_hc_serial_matmul.py")
    original = sys.modules["_hc_original_baseline"]

    def triton_matmul(residual, block, injection, weight, down, up, **kw):
        k = kw["hc_count"]
        combined, normalized = original.hc_combine_norm(residual, block, injection, weight, kw["eps"], k)
        projected = torch.matmul(normalized, down.transpose(0, 1))
        rank = up.shape[1]
        lowrank = original.hc_silu(projected[:, :rank], k)
        next_injection = projected[:, rank : rank + k] if kw["use_combine"] else None
        gate = torch.matmul(lowrank, up.transpose(0, 1))
        return combined, original.hc_gate_mix(normalized, gate, k), next_injection

    return {
        "ascendc_linear": controls["ascendc_serial"],
        "ascendc_matmul": candidate.combine_and_mix,
        "triton_linear": controls["triton_serial"],
        "triton_matmul": triton_matmul,
    }


def api_pair_equality(case, funcs):
    _, inputs, kw = make_inputs(case)
    outputs = {
        label: tuple(None if value is None else value.cpu() for value in function(*inputs, **kw))
        for label, function in funcs.items()
    }
    return {
        family: {
            name: (a is None and b is None) if a is None or b is None else torch.equal(a, b)
            for name, a, b in zip(NAMES, outputs[family + "_linear"], outputs[family + "_matmul"])
        }
        for family in ("ascendc", "triton")
    }


def benchmark_four(case, funcs):
    _, inputs, kw = make_inputs(case)
    funcs = bind_inputs(funcs, inputs)
    graphs, kept_outputs = {}, {}
    for label, function in funcs.items():
        for _ in range(EAGER_WARMUPS):
            function(*inputs, **kw)
        graphs[label], kept_outputs[label] = capture(function, inputs, kw, GRAPH_CALLS)
        for _ in range(GRAPH_WARMUPS):
            graphs[label].replay()
    torch.npu.synchronize()
    raw = {label: [] for label in funcs}
    orders = []
    for index in range(TIMING_SAMPLES):
        labels = list(funcs)
        shift = index % len(labels)
        labels = labels[shift:] + labels[:shift]
        if index % 2:
            labels.reverse()
        orders.append(labels)
        for label in labels:
            raw[label].append(timed_replays(graphs[label], GRAPH_CALLS, REPLAYS_PER_SAMPLE))
    medians = {label: statistics.median(values) for label, values in raw.items()}
    speedup = medians["triton_matmul"] / medians["ascendc_matmul"]
    return {
        "id": case["id"],
        "case": case,
        "graph_calls": GRAPH_CALLS,
        "replays_per_sample": REPLAYS_PER_SAMPLE,
        "sample_orders": orders,
        "samples_us": raw,
        "medians_us": medians,
        "ascendc_api_speedup": medians["ascendc_linear"] / medians["ascendc_matmul"],
        "triton_api_speedup": medians["triton_linear"] / medians["triton_matmul"],
        "speedup": speedup,
        "target": 1.5,
        "target_met": speedup >= 1.5,
    }


def profile_four(cases, funcs, result, out):
    kept = []
    result.update(warm_graph_replays=PROFILE_WARMUPS, retained_graph_replays=PROFILE_REPLAYS)
    result["ordered_profiles"] = []
    for case in cases:
        _, inputs, kw = make_inputs(case)
        bound = bind_inputs(funcs, inputs)
        for label, function in bound.items():
            graph, outputs = capture(function, inputs, kw)
            for _ in range(PROFILE_WARMUPS):
                graph.replay()
            torch.npu.synchronize()
            for _ in range(PROFILE_REPLAYS):
                graph.replay()
            torch.npu.synchronize()
            kept.append((graph, outputs, inputs, bound))
            result["ordered_profiles"].append(
                {
                    "label": case["id"] + "-" + label,
                    "case": case,
                    "implementation": label,
                    "profile_replays": PROFILE_REPLAYS,
                    "graph_execution_calls": PROFILE_WARMUPS + PROFILE_REPLAYS,
                }
            )
            out.write_text(json.dumps(result, indent=2))
            print("HC_PROFILE_BATCH", result["ordered_profiles"][-1]["label"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "bench", "profile"), default="validate")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    funcs = four_implementations(args.extension)
    cases = json.loads(args.cases.read_text())["cases"] if args.cases else BOUNDARIES
    result = {
        "mode": args.mode,
        "weight_mode": "original",
        "started": time.time(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "soc": torch.npu.get_device_name(0),
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "profile":
        profile_four(cases, funcs, result, args.out)
    else:
        for case in cases:
            if args.mode == "validate":
                row = validate_case(case, funcs)
                row["api_pair_exact"] = api_pair_equality(case, funcs)
                row["pass"] = all(x["pass"] for x in row["implementations"].values())
            else:
                row = benchmark_four(case, funcs)
            result["cases"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
    result["completed"] = time.time()
    args.out.write_text(json.dumps(result, indent=2))
    if any(not x.get("pass", True) for x in result["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
