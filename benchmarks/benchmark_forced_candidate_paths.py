# SPDX-License-Identifier: Apache-2.0
"""Validation-only full public native versus forced fixed20 fused paths.

Both production source files remain unchanged. Exactly the inner occupancy and
merge-work if-test of each public wrapper is replaced in memory by False/True.
The outer dtype, topk and sort/fuse guards and all allocations/launches remain.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import statistics
import sys
import textwrap
import time
from pathlib import Path

import validate_radix_powers as dependency

DEPENDENCY_SHA = "b9d987df81ae17123d8f96f0f111004a0022ba5896ecaeb4a5422e1d8f40634b"
COMMITS = {"main": "6385299e417741fc576c30c95f7860fd69dbec24",
           "fixed20": "955894179886df79374aef0090a0b0b45b9e9fbe"}
GROUPS, CALLS, WARMUP, GRAPH_INNER = 10, 10, 10, 32


def summary(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    median = statistics.median(values)
    return {"count": len(values), "median_us": median,
            "mean_us": statistics.mean(values), "min_us": min(values),
            "max_us": max(values), "stdev_us": statistics.stdev(values) if len(values) > 1 else 0,
            "range_over_median_pct": 100 * (max(values) - min(values)) / median}


def load_wrapper(root, variant, force):
    source = root / "vllm_ascend/ops/triton/sharded_indexer.py"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == dependency.EXPECTED_SHA[variant], (source, source_sha)
    name = "forced_path_" + variant
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    original = textwrap.dedent(inspect.getsource(module.pack_local_candidates))
    tree = ast.parse(original)
    gates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and {"rows", "cores", "serial_merge_levels", "local_sort_levels"}
             <= {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}]
    assert len(gates) == 1, "Production routing structure changed"
    gate = gates[0]
    gate_expression = ast.unparse(gate.test)
    assert gate_expression == "rows * 100 >= cores * MIN_FUSED_SORT_ROW_PERCENT and serial_merge_levels >= local_sort_levels"
    original_tree = ast.dump(tree, include_attributes=False)
    gate.test = ast.copy_location(ast.Constant(force), gate.test)
    modified_tree = ast.dump(tree, include_attributes=False)
    assert modified_tree != original_tree
    ast.fix_missing_locations(tree)
    local = {}
    exec(compile(tree, str(source) + "::validation_forced_gate", "exec"), module.__dict__, local)
    function = local["pack_local_candidates"]
    return module, function, {"variant": variant, "production_commit": COMMITS[variant],
            "source": str(source), "source_sha256": source_sha,
            "original_public_source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "original_gate": gate_expression, "forced_gate": force,
            "transformed_public_ast_sha256": hashlib.sha256(modified_tree.encode()).hexdigest(),
            "only_transformation": "one inner If.test replaced by ast.Constant; no production file modified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--fixed20-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("time", "profile", "graph"), default="time")
    parser.add_argument("--profile-route", choices=("native", "fused"))
    parser.add_argument("--rows", type=int, choices=(1, 2, 4, 8, 16, 32, 64, 128, 256), required=True)
    parser.add_argument("--world", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve prior samples"
    assert args.mode != "profile" or args.profile_route
    assert hashlib.sha256(Path(dependency.__file__).read_bytes()).hexdigest() == DEPENDENCY_SHA
    import torch
    import torch_npu
    torch.npu.set_device(0)
    dependency.WORLD, dependency.RANK = args.world, args.world - 1
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    ids, scores, oracle, identity = dependency.inputs(torch, args.rows, dtype)
    modules, functions, evidence = {}, {}, {}
    for route, root, variant, force in (("native", args.main_root, "main", False),
                                        ("fused", args.fixed20_root, "fixed20", True)):
        modules[route], functions[route], evidence[route] = load_wrapper(root, variant, force)
        modules[route].init_device_properties_triton()

    def public(route):
        return functions[route](ids, scores, args.world, args.world - 1, dependency.INTERLEAVE,
                                sort_candidates=True, fuse_sort=True)

    report = {"complete": False, "mode": args.mode, "rows": args.rows, "dtype": args.dtype,
              "WORLD": args.world, "RANK": args.world - 1, "K": dependency.K,
              "INTERLEAVE": dependency.INTERLEAVE, "input_identity": identity,
              "ids_stride": list(ids.stride()), "scores_stride": list(scores.stride()),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "device_name": torch.npu.get_device_name(0),
              "soc_version": int(torch_npu._C._npu_get_soc_version()),
              "vector_cores": modules["fused"].get_vectorcore_num(), "sources": evidence,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "dependency_sha256": DEPENDENCY_SHA, "oracle_checkpoints": [],
              "protocol": {"groups": GROUPS, "calls_per_group": CALLS, "warmup_per_path": WARMUP,
                "pairing": "same process/device/inputs; alternate native-fused/fused-native each group",
                "wall": "perf_counter_ns before full public call through completion synchronize",
                "event": "separate eager NPU Event interval around full public call including stream-visible host gaps",
                "wall_includes": ["Python wrapper", "output allocation/cached allocator", "dispatch", "all kernels", "completion synchronize"],
                "outside_timing": ["inputs/H2D", "compilation/warmup", "D2H/oracle", "statistics/JSON"],
                "lifetime": "normal result reassignment; previous output retirement on assignment; last output retained until check"},
              "paths": {r: {"wall_groups": [], "event_groups": []} for r in functions}}
    dependency.save(args.output, report)

    def check(result, label):
        torch.npu.synchronize()
        assert torch.equal(result.cpu(), oracle), label + " CPU oracle mismatch"
        report["oracle_checkpoints"].append(label)

    if args.mode == "profile":
        # No benchmark warmup/probe: msprof op retains exactly ten full entries.
        route = args.profile_route
        for sample in range(CALLS):
            result = public(route)
            check(result, f"profile_{route}_{sample}")
        report["profile_route"] = route
        report["full_api_calls"] = CALLS
        report["timing"] = "external msprof op; all kernel records retained; no kernel sum interpreted as wall"
    else:
        for route in functions:
            module = modules[route]
            originals = {k: getattr(module, k) for k in ("_pack_sorted_candidates", "_pack_candidates")}
            observers = {k: dependency.LaunchRecorder(v) for k, v in originals.items()}
            for k, v in observers.items():
                setattr(module, k, v)
            try:
                result = public(route)
                check(result, route + "_dispatch_probe")
            finally:
                for k, v in originals.items():
                    setattr(module, k, v)
            calls = {k: v.calls for k, v in observers.items()}
            assert bool(calls["_pack_sorted_candidates"]) == (route == "fused")
            assert bool(calls["_pack_candidates"]) == (route == "native")
            report["paths"][route]["launch_probe"] = calls

        graphs, graph_outputs = {}, {}
        if args.mode == "graph":
            for route in functions:
                for _ in range(WARMUP):
                    result = public(route)
                check(result, route + "_graph_warmup")
                graphs[route] = torch.npu.NPUGraph()
                with torch.npu.graph(graphs[route]):
                    for _ in range(GRAPH_INNER):
                        graph_outputs[route] = public(route)
                check(graph_outputs[route], route + "_capture")
            report["graph_protocol"] = "32 full entries captured per graph; fixed captured outputs/intermediates; replay omits Python allocation/dispatch; Event interval around replay divided by32, completion synchronized"
            report["graph_inner_calls"] = GRAPH_INNER
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record(); end.record(); end.synchronize()
            for group in range(GROUPS):
                for route in (("native", "fused") if group % 2 == 0 else ("fused", "native")):
                    values = []
                    for _ in range(CALLS):
                        start.record(); graphs[route].replay(); end.record(); end.synchronize()
                        values.append(start.elapsed_time(end) * 1000 / GRAPH_INNER)
                    check(graph_outputs[route], f"{route}_graph_group_{group}")
                    report["paths"][route].setdefault("graph_groups", []).append({"group": group, "samples_us": values, "summary": summary(values)})
                dependency.save(args.output, report)
            for route in functions:
                report["paths"][route]["graph_event"] = summary([x for g in report["paths"][route]["graph_groups"] for x in g["samples_us"]])
        else:
            for phase in ("wall", "event"):
                for route in functions:
                    for _ in range(WARMUP):
                        result = public(route)
                    check(result, route + "_" + phase + "_warmup")
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record(); end.record(); end.synchronize()
                for group in range(GROUPS):
                    order = ("native", "fused") if group % 2 == 0 else ("fused", "native")
                    for route in order:
                        torch.npu.synchronize()
                        values, instrumented = [], []
                        for _ in range(CALLS):
                            begin = time.perf_counter_ns()
                            if phase == "event":
                                start.record()
                            result = public(route)
                            if phase == "event":
                                end.record(); end.synchronize()
                                instrumented.append((time.perf_counter_ns() - begin) / 1000)
                                values.append(start.elapsed_time(end) * 1000)
                            else:
                                torch.npu.synchronize()
                                values.append((time.perf_counter_ns() - begin) / 1000)
                        check(result, f"{route}_{phase}_group_{group}")
                        report["paths"][route][phase + "_groups"].append({"group": group,
                            "order": list(order), "samples_us": values, "summary": summary(values),
                            "instrumented_wall_us": instrumented})
                    dependency.save(args.output, report)
            for route in functions:
                for phase in ("wall", "event"):
                    report["paths"][route][phase] = summary([x for g in report["paths"][route][phase + "_groups"] for x in g["samples_us"]])
    for route, item in evidence.items():
        assert hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest() == item["source_sha256"]
    report["cpu_oracle_bit_exact_at_all_checkpoints"] = True
    report["production_files_unchanged"] = True
    report["complete"] = True
    dependency.save(args.output, report)
    print(json.dumps({"complete": True, "mode": args.mode, "rows": args.rows, "world": args.world,
          "paths": {r: {k: v for k, v in p.items() if k in ("wall", "event", "graph_event")} for r, p in report["paths"].items()},
          "oracle_checkpoints": len(report["oracle_checkpoints"])}), flush=True)


if __name__ == "__main__":
    main()
