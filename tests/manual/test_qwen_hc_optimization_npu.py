# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Fixed public matmul HC validation, full Graph timing and full-path profiling."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from test_qwen4_exp_hc_serial_npu import BOUNDARIES, benchmark, validate_case
from test_qwen_hc_matmul_ab_npu import four_implementations, profile_four


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
        "ascendc_serial": four["ascendc_matmul"],
        "triton_serial": four["triton_linear"],
        "triton_matmul": four["triton_matmul"],
    }
    cases = json.loads(args.cases.read_text())["cases"] if args.cases else BOUNDARIES
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
        "weight_mode": "original",
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
    result["completed"] = time.time()
    args.out.write_text(json.dumps(result, indent=2))
    if any(not x.get("pass", True) for x in result["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
