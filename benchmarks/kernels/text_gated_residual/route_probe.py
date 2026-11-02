# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Same-shape public matrix-route diagnostics, separate from pipeline acceptance."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch_npu

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PROJECTIONS = (("down", 10240, 320), ("down_gate", 10240, 336), ("up", 320, 10240), ("write_gate", 10240, 4))
CALLS = 10
SAMPLES = 20


def routes(a, weight):
    choices = {
        "mm": lambda: torch.mm(a, weight.t()),
        "matmul": lambda: torch.matmul(a, weight.t()),
        "linear": lambda: torch_npu.npu_linear(a, weight),
        "bmm": lambda: torch.bmm(a.unsqueeze(0), weight.t().unsqueeze(0)).squeeze(0),
        "gmm": lambda: torch_npu.npu_grouped_matmul([a], [weight.t()], split_item=0, group_type=-1)[0],
    }
    if a.shape[0] == 1:
        choices["mv"] = lambda: torch.mv(weight, a[0]).unsqueeze(0)
    return choices


def probe(batch, projection, stream, profile_route=None, weight_format="nd"):
    name, k, n = projection
    generator = torch.Generator().manual_seed(20261102 + batch + n)
    a_cpu = torch.randn(batch, k, generator=generator).to(torch.bfloat16)
    w_cpu = (torch.randn(n, k, generator=generator) / math.sqrt(k)).to(torch.bfloat16)
    reference = a_cpu.double() @ w_cpu.double().t()
    a, weight = a_cpu.npu(), w_cpu.npu()
    torch.npu.synchronize()
    memory_before = torch.npu.memory_allocated()
    prep_start = time.perf_counter()
    if weight_format == "nz":
        weight = torch_npu.npu_format_cast(weight, 29)
    torch.npu.synchronize()
    preparation = {
        "wall_us": (time.perf_counter() - prep_start) * 1e6,
        "extra_device_bytes": torch.npu.memory_allocated() - memory_before,
        "actual_weight_format": torch_npu.get_npu_format(weight),
    }
    if weight_format == "nz" and preparation["actual_weight_format"] != 29:
        raise RuntimeError("NZ preparation did not produce format 29")
    stream.wait_stream(torch.npu.current_stream())
    choices = routes(a, weight)
    if profile_route:
        choices = {profile_route: choices[profile_route]}
    graphs, outputs, result = {}, {}, {}
    with torch.npu.stream(stream):
        for route, call in choices.items():
            for _ in range(10):
                call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(CALLS):
                    output = call()
            graphs[route], outputs[route] = graph, output
            for _ in range(10):
                graph.replay()
            stream.synchronize()
            error = (output.cpu().double() - reference).abs()
            valid = (
                bool(torch.isfinite(output).all())
                and float((error <= 2**-6 + 2**-6 * reference.abs()).double().mean()) >= 0.99
            )
            result[route] = {"correct": valid, "max_abs": float(error.max()), "samples_us": []}
        # A prepared route must continue reading both inputs at stable addresses.
        a.neg_()
        weight.mul_(0.5)
        for route, graph in graphs.items():
            graph.replay()
            stream.synchronize()
            error = (outputs[route].cpu().double() + reference * 0.5).abs()
            result[route]["same_address_update_correct"] = (
                float((error <= 2**-6 + 2**-6 * reference.abs() * 0.5).double().mean()) >= 0.99
            )
        a.neg_()
        weight.mul_(2)
        for sample in range(SAMPLES):
            order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
            for route in order:
                start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[route].replay()
                stop.record()
                stop.synchronize()
                result[route]["samples_us"].append(float(start.elapsed_time(stop)) * 1000 / CALLS)
    for row in result.values():
        row["median_us"] = statistics.median(row["samples_us"])
    return {"batch": batch, "projection": name, "mkn": [batch, k, n], "preparation": preparation, "routes": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, action="append", choices=BATCHES)
    parser.add_argument("--projection", choices=tuple(p[0] for p in PROJECTIONS))
    parser.add_argument("--route", choices=("mm", "matmul", "linear", "bmm", "mv", "gmm"))
    parser.add_argument("--weight-format", choices=("nd", "nz"), default="nd")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    if args.weight_format == "nz":
        torch.npu.config.allow_internal_format = True
    stream = torch.npu.Stream()
    data = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cases": [],
        "boundary": "individual projection Graph; diagnostic only, not complete pipeline speedup",
        "calls_per_graph": CALLS,
        "samples": SAMPLES,
        "weight_format_requested": args.weight_format,
    }
    for batch in args.batch or BATCHES:
        for projection in PROJECTIONS:
            if args.projection and projection[0] != args.projection:
                continue
            row = probe(batch, projection, stream, args.route, args.weight_format)
            data["cases"].append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(data, indent=2))
            print(
                json.dumps(
                    {
                        "batch": batch,
                        "projection": projection[0],
                        "medians": {r: v["median_us"] for r, v in row["routes"].items()},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
