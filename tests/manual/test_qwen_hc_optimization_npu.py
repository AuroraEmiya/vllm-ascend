# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Fixed public matmul HC validation, full Graph timing and full-path profiling."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from test_qwen4_exp_hc_serial_npu import (
    BOUNDARIES,
    NAMES,
    ROOT,
    benchmark,
    capture,
    compare,
    load_file,
    make_inputs,
    reference,
    validate_case,
)
from test_qwen_hc_matmul_ab_npu import four_implementations, profile_four


class PreparedMatmul:
    """Test adapter binding explicit prepared weights before warmup/capture."""

    def __init__(self):
        self.module = load_file("_hc_prepared_candidate", ROOT / "vllm_ascend/ops/qwen_hc_serial_matmul.py")
        self.preparations = []

    def bind_inputs(self, inputs):
        down, up = inputs[4:6]
        torch.npu.synchronize()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        prepared = self.module.PreparedHCWeights(down, up)
        end.record()
        end.synchronize()
        prep = {
            "stage": "explicit_load_time_before_warmup_and_graph_capture",
            "npu_event_us": start.elapsed_time(end) * 1000,
            "wall_us": (time.perf_counter() - wall) * 1e6,
            "input_formats": [torch_npu.get_npu_format(x) for x in (down, up)],
            "prepared_formats": [torch_npu.get_npu_format(x) for x in (prepared.down_weight, prepared.up_weight)],
            "logical_shapes": [list(x.shape) for x in (down, up)],
            "prepared_storage_bytes": [
                x.untyped_storage().nbytes() for x in (prepared.down_weight, prepared.up_weight)
            ],
            "values_equal": [
                torch.equal(a.cpu(), b.cpu()) for a, b in zip((down, up), (prepared.down_weight, prepared.up_weight))
            ],
        }
        if not all(prep["values_equal"]):
            raise AssertionError("prepared projection weights changed values")
        self.preparations.append(prep)

        def forward(residual, block, injection, weight, original_down, original_up, **kwargs):
            if original_down is not down or original_up is not up:
                raise ValueError("weight replacement requires explicit preparation and recapture")
            return prepared(residual, block, injection, weight, **kwargs)

        forward.preparation = prep
        return forward


def validate_preparation_rebuild(adapter):
    """A new projection snapshot must reflect changed weight values."""
    cpu, inputs, kw = make_inputs(BOUNDARIES[0])
    first = adapter.bind_inputs(inputs)
    first(*inputs, **kw)
    inputs[4].add_(0.0625)
    cpu[4] = inputs[4].cpu()
    rebuilt = adapter.bind_inputs(inputs)
    expected = reference(cpu, kw)
    eager = rebuilt(*inputs, **kw)
    graph, outputs = capture(rebuilt, inputs, kw)
    graph.replay()
    torch.npu.synchronize()
    checks = {
        "eager": {name: compare(a, b) for name, a, b in zip(NAMES, eager, expected)},
        "graph": {name: compare(a, b) for name, a, b in zip(NAMES, outputs, expected)},
    }
    checks["pass"] = all(check["pass"] for group in checks.values() for check in group.values())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("validate", "bench", "profile"), default="validate")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    four = four_implementations(args.extension)
    funcs = {
        "ascendc_serial": PreparedMatmul(),
        "triton_serial": four["triton_linear"],
        "triton_matmul": four["triton_matmul"],
    }
    boundaries = [
        *BOUNDARIES,
        {"id": "aligned-strided", "m": 3, "h": 64, "k": 3, "r": 16, "strided": True},
    ]
    cases = json.loads(args.cases.read_text())["cases"] if args.cases else boundaries
    result = {
        "mode": args.mode,
        "started": time.time(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "soc": torch.npu.get_device_name(0),
        "c_api": {
            "candidate": "torch.matmul",
            "original_baseline": "F.linear",
            "api_control": "torch.matmul",
        },
        "weight_mode": "explicit_load_time_NZ_snapshot_for_candidate_original_for_baselines",
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "profile":
        profile_four(cases, funcs, result, args.out)
    else:
        for case in cases:
            if args.mode == "validate":
                row = validate_case(case, funcs)
                row["pass"] = all(x["pass"] for x in row["implementations"].values())
            else:
                row = benchmark(case, funcs)
                row["target_met"] = row.pop("pass")
            result["cases"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
    if args.mode == "validate":
        result["prepared_weight_rebuild"] = validate_preparation_rebuild(funcs["ascendc_serial"])
    result["weight_preparations"] = funcs["ascendc_serial"].preparations
    result["completed"] = time.time()
    args.out.write_text(json.dumps(result, indent=2))
    if not result.get("prepared_weight_rebuild", {"pass": True})["pass"] or any(
        not x.get("pass", True) for x in result["cases"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
