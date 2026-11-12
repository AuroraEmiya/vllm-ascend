"""Opt-in task-local diagnostic hook, for profiled runs only.

PYTHONPATH=<this dir>:$PYTHONPATH MODEL_BATCH_PROBE_DIR=<output> vllm serve ...
Logs CPU scheduling metadata without reading model payloads or synchronizing NPU.
Do not enable for official unprofiled throughput measurements.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path

TARGET = "vllm_ascend.worker.v2.model_runner"
ROOT = os.environ.get("MODEL_BATCH_PROBE_DIR")


def install(module):
    cls = module.NPUModelRunner
    original = cls._prepare_inputs_impl

    @functools.wraps(original)
    def wrapped(self, scheduler_output, batch_req_state, batch_desc):
        result = original(self, scheduler_output, batch_req_state, batch_desc)
        # CPU numpy scheduling metadata; no tensor payload or device copies.
        entry = {"unix_ns": time.time_ns(), "pid": os.getpid(), "device": str(self.device),
                 "rank": os.environ.get("RANK", "unknown"),
                 "scheduled_requests": len(batch_req_state.req_ids),
                 "scheduled_tokens": int(batch_req_state.num_tokens),
                 "tokens_per_request": [int(v) for v in batch_req_state.num_scheduled_tokens],
                 "graph_requests": int(batch_desc.num_reqs), "graph_tokens": int(batch_desc.num_tokens),
                 "graph_mode": str(batch_desc.cg_mode),
                 "phase": "unclassified: correlate scheduled tokens, sequence lengths and request phase"}
        entry["sequence_lengths_cpu"] = [int(v) for v in self.input_buffers.seq_lens_np[:entry["scheduled_requests"]]]
        path = Path(ROOT)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"batch-pid{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        return result

    cls._prepare_inputs_impl = wrapped


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def execute(module):
            original_exec(module)
            install(module)

        spec.loader.exec_module = execute
        return spec


if ROOT:
    sys.meta_path.insert(0, Finder())
