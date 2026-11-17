# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Source-identified Graph comparisons for two-stage histogram experiments."""

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import statistics
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

import vllm_ascend.distributed.eplb.tensor_ops as public_module
import vllm_ascend.ops.triton.eplb_count as current_module

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--before", required=True)
parser.add_argument("--store", required=True)
parser.add_argument("--labels", default="main,before,store,current")
parser.add_argument("--shapes", default="10240,8,128;81920,1,128;2561,8,128;1088,8,512")
parser.add_argument("--patterns", default="uniform,hot,mixed")
parser.add_argument("--output", default="results/trial-graph.json")
parser.add_argument("--profile", choices=["main", "before", "store", "current", "tile1024"])
args = parser.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


main_module = load_module("eplb_trial_main", args.baseline)
before_module = load_module("eplb_trial_before", args.before)
store_module = load_module("eplb_trial_store", args.store)
current_function = current_module.count_allgather_tokens_triton
modules = {"main": main_module, "before": before_module, "store": store_module, "current": current_module}
identity = {
    name: {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    for name, module in modules.items()
}
identity["tile1024"] = {**identity["current"], "assignments_per_tile": 1024}
outpath = Path(args.output)
outpath.parent.mkdir(parents=True, exist_ok=True)
outpath.with_suffix(".identity.json").write_text(json.dumps(identity, indent=2))
CALLS, SAMPLES = 20, 10


def invoke(label, ids, mask, experts):
    if label == "main":
        return main_module.count_allgather_tokens(ids, mask, 7, experts)
    function = current_function if label in ("current", "tile1024") else modules[label].count_allgather_tokens_triton
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(current_module, "count_allgather_tokens_triton", function))
        if label == "tile1024":
            stack.enter_context(patch.object(current_module, "ASSIGNMENTS_PER_TILE", 1024))
        return public_module.count_allgather_tokens(ids, mask, 7, experts)


def inputs(shape, pattern):
    tokens, topk, experts = shape
    ids = (torch.arange(tokens * topk).reshape(tokens, topk) % experts + 7).to(torch.int64)
    mask = torch.ones(tokens, dtype=torch.bool)
    if pattern == "hot":
        ids.fill_(7)
    elif pattern == "mixed":
        ids = (torch.arange(tokens * topk).reshape(tokens, topk) % (experts + 4) + 5).to(torch.int64)
        mask = torch.arange(tokens) % 3 != 0
    expected = main_module.count_allgather_tokens(ids, mask, 7, experts)
    return ids.npu(), mask.npu(), expected


def capture(label, ids, mask, experts, expected):
    for _ in range(5):
        output = invoke(label, ids, mask, experts)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CALLS):
            output = invoke(label, ids, mask, experts)
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    return graph, output


def measure(shape, pattern, labels):
    ids, mask, expected = inputs(shape, pattern)
    captures = {label: capture(label, ids, mask, shape[2], expected) for label in labels}
    samples = {label: [] for label in labels}
    for trial in range(SAMPLES):
        order = labels[trial % len(labels) :] + labels[: trial % len(labels)]
        if trial % 2:
            order = list(reversed(order))
        for label in order:
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            captures[label][0].replay()
            end.record()
            end.synchronize()
            samples[label].append(begin.elapsed_time(end) * 1000 / CALLS)
    for label, (_, output) in captures.items():
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0, atol=0, msg=lambda message, label=label: label + message
        )
    medians = {label: statistics.median(values) for label, values in samples.items()}
    return {
        "status": "pass",
        "samples_us": samples,
        "median_us": medians,
        "relative_to_first": {label: medians[labels[0]] / value for label, value in medians.items()},
    }


shapes = [tuple(map(int, text.split(","))) for text in args.shapes.split(";")]
patterns = args.patterns.split(",")
labels = args.labels.split(",")
if args.profile:
    ids, mask, expected = inputs(shapes[0], patterns[0])
    graph, output = capture(args.profile, ids, mask, shapes[0][2], expected)
    # Five eager calls plus five warm Graphs precede 200 measured invocations.
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    print("PROFILE_COMPLETE", args.profile, flush=True)
else:
    rows = []
    for shape in shapes:
        for pattern in patterns:
            row = {"shape": shape, "pattern": pattern, "dtype": "int64", "calls_per_graph": CALLS, "samples": SAMPLES}
            print("START", json.dumps(row), flush=True)
            try:
                row.update(measure(shape, pattern, labels))
            except Exception:
                row.update(status="failed", error=traceback.format_exc(), median_us=None)
            rows.append(row)
            outpath.write_text(json.dumps(rows, indent=2))
            print("RESULT", json.dumps(row), flush=True)
            gc.collect()
    print("COMPLETE", len(rows), sum(row["status"] == "pass" for row in rows), flush=True)
    raise SystemExit(any(row["status"] != "pass" for row in rows))
