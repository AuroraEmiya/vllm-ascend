# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Profile every requested HC case in one process; keep graphs alive for IDs."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu
from test_qwen4_exp_hc_serial_npu import bind_inputs, capture, implementations, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", default="original")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    funcs = implementations(args.extension, args.weights)
    cases = json.loads(args.cases.read_text())["cases"]
    result = {
        "mode": "profile_batch",
        "started": time.time(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "weight_mode": args.weights,
        "soc": torch.npu.get_device_name(0),
        "warm_graph_replays": 5,
        "retained_graph_replays": 16,
        "ordered_profiles": [],
    }
    kept = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        _, inputs, kw = make_inputs(case)
        bound = bind_inputs(funcs, inputs)
        for label, function in bound.items():
            graph, outputs = capture(function, inputs, kw)
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
            for _ in range(16):
                graph.replay()
            torch.npu.synchronize()
            kept.append((graph, outputs, inputs, bound))
            result["ordered_profiles"].append(
                {
                    "label": case["id"] + "-" + label,
                    "case": case,
                    "implementation": label,
                    "profile_replays": 16,
                    "graph_execution_calls": 21,
                }
            )
            args.out.write_text(json.dumps(result, indent=2))
            print("HC_PROFILE_BATCH", result["ordered_profiles"][-1]["label"], flush=True)
    result["completed"] = time.time()
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
