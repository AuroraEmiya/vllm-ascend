# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Isolate an eager implementation in a fresh process after device errors."""

import argparse
import json
from pathlib import Path

import torch
from validate import check_outputs, make_calls, make_inputs, oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--implementation", choices=("fused", "serial"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    case = next(c for c in json.loads(Path(__file__).with_name("cases.json").read_text()) if c["id"] == args.case_id)
    cpu = make_inputs(case)
    refs = oracle(cpu, case)
    tensors, calls, _ = make_calls(cpu, case)
    print("EAGER_BEGIN", args.implementation, args.case_id, flush=True)
    outputs = calls[args.implementation]()
    torch.npu.synchronize()
    result = {"case": case, "implementation": args.implementation, "outputs": check_outputs(outputs, refs)}
    result["unmodified_inputs"] = all(t is None or torch.equal(t.cpu(), c) for t, c in zip(tensors, cpu))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
