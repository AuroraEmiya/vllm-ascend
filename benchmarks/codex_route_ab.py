"""Standalone same-source fused/native preparation A/B; never changes production dispatch."""

import argparse
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path

SOURCE_SHA = "37cc2517c29f9baa562b96453a13b3439eb01ea3671a712f6c388ca1fc36b6d4"


def cases():
    result = []
    for dtype in ("fp32", "bf16"):
        shapes = {(r, w) for r in (32, 64, 128) for w in (2, 6, 7, 8)}
        shapes |= {(r, 8) for r in (1, 29, 30, 31, 35, 36, 37, 39, 40, 41, 47, 48, 49, 144, 152)}
        for rows, world in sorted(shapes):
            result.append(dict(name=f"r{rows}_w{world}_{dtype}_small", rows=rows,
                               world=world, dtype=dtype, ids="small", stride=1))
        # Probe both sides of the short-pass boundary; exact integer IDs only.
        for mode in ("short_max", "short_boundary", "full", "fallback"):
            result.append(dict(name=f"r32_w8_{dtype}_{mode}", rows=32,
                               world=8, dtype=dtype, ids=mode, stride=1))
        result.append(dict(name=f"r32_w8_{dtype}_stride2", rows=32,
                           world=8, dtype=dtype, ids="small", stride=2))
    return result


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def inputs(torch, case):
    g = torch.Generator().manual_seed(20878)
    shape = (case["rows"], 1, 2048)
    ids = torch.randint(0, 131072, shape, generator=g, dtype=torch.int32)
    limit = 1 << (20 if case["dtype"] == "fp32" else 23)
    mode = case["ids"]
    if mode == "short_max":
        ids[:, :, -1] = limit - 1
    elif mode == "short_boundary":
        ids[:, :, -1] = limit
    elif mode == "full":
        ids[:, :, -1] = (1 << 25) - 2
    elif mode == "fallback":
        # 2**25 is exactly representable, unlike arbitrary INT32 values there.
        ids[:, :, -1] = 1 << 25
    dtype = torch.float32 if case["dtype"] == "fp32" else torch.bfloat16
    scores = torch.randn(shape, generator=g).to(dtype)
    if mode != "small":
        scores[:, :, -1] = 1.0
    scores[:, :, :8] = torch.tensor([0., -0., 1., 1., float("nan"),
                                    float("inf"), -float("inf"), -1.], dtype=dtype)
    ids[:, :, 7] = -1
    valid = (ids >= 0) & torch.isfinite(scores)
    values = torch.where(valid, scores.float(), -float("inf"))
    keys = torch.where(valid, ids, 0)
    order = torch.argsort(keys, stable=True, dim=-1)
    keys, values = keys.gather(-1, order), values.gather(-1, order)
    order = torch.argsort(values, descending=True, stable=True, dim=-1)
    keys, values = keys.gather(-1, order), values.gather(-1, order)
    mapped = keys + (keys // 768 * (case["world"] - 1) + case["world"] - 1) * 768
    mapped = torch.where(torch.isfinite(values), mapped, -1)
    oracle = torch.stack((mapped, values.contiguous().view(torch.int32)), dim=-1).squeeze(1)
    def upload(x):
        if case["stride"] == 1:
            return x.to("npu")
        storage = torch.empty((*shape[:-1], shape[-1] * 2), dtype=x.dtype, device="npu")
        view = storage[..., ::2]
        view.copy_(x)
        return view
    return upload(ids), upload(scores), oracle


def binary_record(kernel, folder):
    folder.mkdir(parents=True, exist_ok=True)
    record = {"hash": str(getattr(kernel, "hash", None)), "files": {}}
    for name, value in getattr(kernel, "asm", {}).items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        target = folder / ("compiled." + name.replace("/", "_"))
        target.write_bytes(data)
        record["files"][name] = {"path": str(target), "sha256": hashlib.sha256(data).hexdigest()}
    record["metadata"] = str(getattr(kernel, "metadata", None))
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true")
    p.add_argument("--cases", help="Comma-separated exact case names; default all")
    p.add_argument("--output", type=Path)
    p.add_argument("--mode", choices=("event", "profile"), default="event")
    p.add_argument("--side", choices=("both", "fused", "native"), default="both")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--capture-inner-repeats", type=int, default=32,
                   help="Full preparation calls captured inside each graph; amortizes replay submission")
    p.add_argument("--expected-source-sha", default=SOURCE_SHA)
    args = p.parse_args()
    selected = cases()
    if args.cases:
        names = set(args.cases.split(","))
        selected = [c for c in selected if c["name"] in names]
        if {c["name"] for c in selected} != names:
            p.error("Unknown case name")
    if args.list:
        print(json.dumps(selected, indent=2))
        return
    if not args.output or args.replays < 1 or args.capture_inner_repeats < 1:
        p.error("--output required; --replays and --capture-inner-repeats must be positive")
    import torch
    import torch_npu
    from vllm_ascend.ops.triton import sharded_indexer as op
    source = Path(op.__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != args.expected_source_sha:
        raise RuntimeError(f"Source mismatch: {source} {source_sha}")
    torch.npu.set_device(args.device)
    op.init_device_properties_triton()
    cores = op.get_vectorcore_num()
    report = dict(source=str(source), source_sha256=source_sha, torch=torch.__version__,
                  torch_npu=torch_npu.__version__, device=torch.npu.get_device_name(args.device),
                  cores=cores, mode=args.mode, replays=args.replays,
                  capture_inner_repeats=args.capture_inner_repeats, samples=10, cases=[])
    sides = ("fused", "native") if args.side == "both" else (args.side,)
    save(args.output, report)
    for case in selected:
        row = dict(case=case, status="running", measurements={})
        report["cases"].append(row)
        try:
            ids, scores, oracle = inputs(torch, case)
            rows, world = case["rows"], case["world"]
            compiled = []
            def fused():
                packed = torch.empty((rows, 2048, 2), dtype=torch.int32, device=ids.device)
                binary = op._pack_sorted_candidates[(min(rows, cores),)](
                    ids, scores, packed, rows, ids.stride(0), ids.stride(-1),
                    scores.stride(0), scores.stride(-1), world, world - 1, 768, 11,
                    SCORE_BITS=16 if case["dtype"] == "bf16" else 32, multibuffer=False)
                if not compiled:
                    compiled.append(binary)
                return packed
            def native():
                return op.pack_local_candidates(ids, scores, world, world - 1, 768,
                                                sort_candidates=True, fuse_sort=False)
            funcs = {"fused": fused, "native": native}
            for side in sides:
                actual = funcs[side]()
                if not torch.equal(actual.cpu(), oracle):
                    raise AssertionError(f"{side} differs from independent CPU oracle")
            row["correctness"] = "bitwise CPU oracle passed"
            row["launch"] = {"fused_kernel": "_pack_sorted_candidates", "fused_grid": min(rows, cores),
                             "native": "pack_local_candidates(sort_candidates=True,fuse_sort=False)",
                             "indices_stride": list(ids.stride()), "scores_stride": list(scores.stride()),
                             "score_bits": 16 if case["dtype"] == "bf16" else 32}
            if compiled:
                row["compiled"] = binary_record(compiled[0], args.output.parent / "compiled" / case["name"])
            graphs, kept = {}, []
            for side in sides:
                fn = funcs[side]
                for _ in range(10):
                    fn()
                torch.npu.synchronize()
                if args.mode == "event":
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        for _ in range(args.capture_inner_repeats):
                            kept.append(fn())
                    for _ in range(10):
                        graph.replay()
                    graphs[side] = graph
                row["measurements"][side] = {"samples": []}
            torch.npu.synchronize()
            # Adjacent AB/BA windows keep both paths under similar device state.
            warmup = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="npu") if args.mode == "profile" else None
            if warmup is not None:
                warmup.fill_(1)
                torch.argmax(warmup)
                torch.npu.synchronize()
            for sample in range(10):
                for side in (sides if sample % 2 == 0 else tuple(reversed(sides))):
                    if args.mode == "event":
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.replays):
                            graphs[side].replay()
                        end.record()
                        end.synchronize()
                        result = {"index": sample, "us": start.elapsed_time(end) * 1000
                                  / (args.replays * args.capture_inner_repeats)}
                    else:
                        # External msprof must map the entire target sequence; wall timestamps are NOT latency.
                        torch.argmax(warmup)
                        torch.npu.synchronize()
                        begin = time.time_ns()
                        with torch.autograd.profiler.record_function(f"route_ab/{case['name']}/{side}/{sample}"):
                            output = funcs[side]()
                        torch.npu.synchronize()
                        result = {"index": sample, "host_begin_ns": begin, "host_end_ns": time.time_ns()}
                        del output
                    row["measurements"][side]["samples"].append(result)
                    save(args.output, report)
            for graph in graphs.values():
                graph.reset()
            if args.mode == "event":
                for measurement in row["measurements"].values():
                    values = [s["us"] for s in measurement["samples"]]
                    median = statistics.median(values)
                    measurement.update(median_us=median, spread=(max(values)-min(values))/median,
                                       stable=(max(values)-min(values))/median <= 0.10)
                if args.side == "both":
                    row["native_over_fused"] = row["measurements"]["native"]["median_us"] / row["measurements"]["fused"]["median_us"]
            row["status"] = "complete"
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        save(args.output, report)
    if any(r["status"] != "complete" for r in report["cases"]):
        raise SystemExit(1)
    if args.mode == "event" and any(not m["stable"] for r in report["cases"] for m in r["measurements"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
