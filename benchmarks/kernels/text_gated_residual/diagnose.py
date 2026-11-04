# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Isolate an eager implementation in a fresh process after device errors."""

import argparse
import json
from pathlib import Path

import torch
from validate import capture, check_outputs, make_calls, make_inputs, oracle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--implementation", choices=("fused", "serial"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    case = next(c for c in json.loads(Path(__file__).with_name("cases.json").read_text()) if c["id"] == args.case_id)
    cpu = make_inputs(case)
    refs = oracle(cpu, case)
    tensors, calls, _, serial = make_calls(cpu, case)
    print("EAGER_BEGIN", args.implementation, args.case_id, flush=True)
    outputs = calls[args.implementation]()
    torch.npu.synchronize()
    result = {"case": case, "implementation": args.implementation, "outputs": check_outputs(outputs, refs)}
    result["unmodified_inputs"] = all(t is None or torch.equal(t.cpu(), c) for t, c in zip(tensors, cpu))
    if args.graph:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            graph, graph_outputs = capture(calls[args.implementation], stream)
            for _ in range(10):
                graph.replay()
            stream.synchronize()
            result["graph"] = check_outputs(graph_outputs, refs)
            updated = [t.clone() if t is not None else None for t in cpu]
            updated[0].mul_(-1)
            updated[1].mul_(-1)
            updated[2].mul_(-1)
            if updated[3] is not None:
                updated[3].mul_(-1)
            if updated[4] is not None:
                updated[4].add_(0.0625)
            for tensor, value in zip(tensors, updated):
                if tensor is not None:
                    tensor.copy_(value)
            serial.refresh_weights_()
            graph.replay()
            stream.synchronize()
            result["graph_updated_same_address"] = check_outputs(graph_outputs, oracle(updated, case))
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
