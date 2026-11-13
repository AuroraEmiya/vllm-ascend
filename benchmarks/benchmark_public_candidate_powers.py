# SPDX-License-Identifier: Apache-2.0
"""Validation-only benchmark of the unchanged normal eager candidate-pack API.

Primary: per-call host wall time for the public API plus completion synchronize.
Supplement: a separate loop of NPU Event stream intervals around the same API.
No graph capture, direct kernel launch, profiler sum, or production change.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from validate_radix_powers import (
    EXPECTED_SHA,
    INTERLEAVE,
    K,
    RANK,
    ROWS,
    SEED,
    WORLD,
    LaunchRecorder,
    inputs,
    save,
)

DEPENDENCY_SHA = "b9d987df81ae17123d8f96f0f111004a0022ba5896ecaeb4a5422e1d8f40634b"
WARMUP, GROUPS, CALLS_PER_GROUP = 10, 10, 10


def summary(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    median = statistics.median(values)
    return {"count": len(values), "median_us": median, "mean_us": statistics.mean(values),
            "min_us": min(values), "max_us": max(values), "stdev_us": statistics.stdev(values),
            "range_over_median_pct": 100 * (max(values) - min(values)) / median}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(EXPECTED_SHA), required=True)
    parser.add_argument("--rows", type=int, choices=ROWS, required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    import validate_radix_powers as dependency
    from vllm_ascend.ops.triton import sharded_indexer as op

    dependency_sha = hashlib.sha256(Path(dependency.__file__).read_bytes()).hexdigest()
    assert dependency_sha == DEPENDENCY_SHA
    source = Path(op.__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == EXPECTED_SHA[args.variant], (str(source), source_sha)
    torch.npu.set_device(0)
    op.init_device_properties_triton()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    ids, scores, oracle, identity = inputs(torch, args.rows, dtype)

    def public():
        return op.pack_local_candidates(ids, scores, WORLD, RANK, INTERLEAVE,
                                        sort_candidates=True, fuse_sort=True)

    report = {"complete": False, "variant": args.variant, "rows": args.rows, "dtype": args.dtype,
              "source": str(source), "source_sha256": source_sha,
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dependency_sha256": dependency_sha, "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "device_name": torch.npu.get_device_name(0),
              "soc_version": int(torch_npu._C._npu_get_soc_version()),
              "vector_cores": op.get_vectorcore_num(), "K": K, "WORLD": WORLD,
              "RANK": RANK, "INTERLEAVE": INTERLEAVE,
              "seeds": {"ids": SEED, "scores": SEED + 1}, "input_identity": identity,
              "ids_stride": list(ids.stride()), "scores_stride": list(scores.stride()),
              "public_kwargs": {"sort_candidates": True, "fuse_sort": True},
              "protocol": {"primary": "normal eager API plus per-call completion synchronize / host perf_counter_ns",
                           "supplement": "separate normal eager API NPU Event stream interval; includes host submission gaps visible to stream",
                           "graph_capture": False, "direct_kernel_launch": False,
                           "warmup_calls_per_timing_loop": WARMUP, "groups": GROUPS,
                           "calls_per_group": CALLS_PER_GROUP,
                           "included_in_primary": ["Python public call", "cached allocator and output allocation", "dispatch", "all fallback/fused kernels", "per-call completion synchronize"],
                           "outside_timing": ["input generation and H2D", "initialization and compilation warmup", "output D2H and CPU oracle after each group", "statistics and JSON writing"],
                           "lifetime": "normal result reassignment; previous output retirement occurs on assignment; last output retained for post-group oracle",
                           "validation_cadence": "public branch probe, warmup last result, and last result of each ten-call group"},
              "oracle_checkpoints": [], "wall_groups": [], "event_groups": []}
    save(args.output, report)

    def check(result, label):
        torch.npu.synchronize()
        assert torch.equal(result.cpu(), oracle), label + " CPU oracle mismatch"
        report["oracle_checkpoints"].append(label)

    kernel = op._pack_sorted_candidates
    observer = LaunchRecorder(kernel)
    op._pack_sorted_candidates = observer
    try:
        result = public()
        check(result, "public_dispatch_probe")
    finally:
        op._pack_sorted_candidates = kernel
    report["public_fused"] = bool(observer.calls)
    report["public_sorted_kernel_launches"] = observer.calls
    assert op._pack_sorted_candidates is kernel
    save(args.output, report)

    def warm(label):
        result = None
        for _ in range(WARMUP):
            result = public()
        check(result, label)
        return result

    result = warm("wall_warmup")
    for group in range(GROUPS):
        torch.npu.synchronize()
        values = []
        for _ in range(CALLS_PER_GROUP):
            begin = time.perf_counter_ns()
            result = public()
            torch.npu.synchronize()
            values.append((time.perf_counter_ns() - begin) / 1000)
        check(result, f"wall_group_{group}")
        report["wall_groups"].append({"group": group, "samples_us": values, "summary": summary(values)})
        save(args.output, report)

    result = warm("event_warmup")
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    for group in range(GROUPS):
        torch.npu.synchronize()
        event_values, instrumented_wall = [], []
        for _ in range(CALLS_PER_GROUP):
            begin = time.perf_counter_ns()
            start.record()
            result = public()
            end.record()
            end.synchronize()
            instrumented_wall.append((time.perf_counter_ns() - begin) / 1000)
            event_values.append(start.elapsed_time(end) * 1000)
        check(result, f"event_group_{group}")
        report["event_groups"].append({"group": group, "event_samples_us": event_values,
                                       "instrumented_wall_samples_us": instrumented_wall,
                                       "event_summary": summary(event_values)})
        save(args.output, report)

    report["primary_wall"] = summary([v for g in report["wall_groups"] for v in g["samples_us"]])
    report["supplement_event"] = summary([v for g in report["event_groups"] for v in g["event_samples_us"]])
    report["group_medians"] = {"wall_us": [g["summary"]["median_us"] for g in report["wall_groups"]],
                               "event_us": [g["event_summary"]["median_us"] for g in report["event_groups"]]}
    report["checked_public_calls"] = {"dispatch_probe": 1, "warmup": 2 * WARMUP,
                                      "wall_timed": GROUPS * CALLS_PER_GROUP,
                                      "event_timed": GROUPS * CALLS_PER_GROUP,
                                      "oracle_checkpoints": len(report["oracle_checkpoints"])}
    report["cpu_oracle_bit_exact_at_all_checkpoints"] = True
    report["complete"] = True
    save(args.output, report)
    print(json.dumps({k: report[k] for k in ("complete", "variant", "rows", "dtype", "public_fused", "primary_wall", "supplement_event", "checked_public_calls")}), flush=True)


if __name__ == "__main__":
    main()
