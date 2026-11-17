# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Temporary diagnostic: unchanged two-stage wrapper, Graph replay only."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.eplb.tensor_ops import count_allgather_tokens
from vllm_ascend.ops.triton import eplb_count

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, required=True)
parser.add_argument("--experts", type=int, required=True)
args = parser.parse_args()
torch.npu.set_device(0)
topk = 8
ids_cpu = torch.arange(args.tokens * topk, dtype=torch.int64).reshape(args.tokens, topk) % args.experts + 7
ids = ids_cpu.npu()
mask = torch.ones(args.tokens, dtype=torch.bool, device="npu:0")
expected = torch.bincount((ids_cpu - 7).flatten(), minlength=args.experts)
kernel_path = Path(eplb_count.__file__)
print(json.dumps({"tokens": args.tokens, "topk": topk, "experts": args.experts,
                  "kernel_path": str(kernel_path), "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                  "calls_per_graph": 20, "warmup_calls": 5, "warmup_replays": 5, "measured_replays": 10}), flush=True)
for _ in range(5):
    out = count_allgather_tokens(ids, mask, 7, args.experts)
torch.npu.synchronize()
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    for _ in range(20):
        out = count_allgather_tokens(ids, mask, 7, args.experts)
for _ in range(5):
    graph.replay()
torch.npu.synchronize()
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
print("MEASURED_GRAPH_REPLAYS_BEGIN 10x20", flush=True)
samples = []
for _ in range(10):
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record()
    graph.replay()
    end.record()
    end.synchronize()
    samples.append(begin.elapsed_time(end) * 1000 / 20)
print("MEASURED_GRAPH_REPLAYS_END", flush=True)
torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
print(json.dumps({"status": "pass", "graph_median_us": statistics.median(samples), "graph_samples_us": samples}), flush=True)
