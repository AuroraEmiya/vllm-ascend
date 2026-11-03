# SPDX-License-Identifier: Apache-2.0
"""Pinned linear/mm versus matmul trial; retain the original 15-case contract."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path

BASE_COMMIT = "af0d582472abb826d3ea6cbe37c75e7f49f9481c"
BASE_SHA256 = "654bb8fe5761e19edfdaf2e8a595c9d599cf10a7e40f1f410d44008b2107af51"
MANIFEST_SHA256 = "824f1b88598244182516c164dc9a369307ce89a2d1bf20b4887a8092a9528858"
CALLS_PER_GRAPH = 10
EAGER_WARMUP = 10
GRAPH_WARMUP = 10
PAIRED_SAMPLES = 20
PROFILE_REPLAYS = 2


def canonical_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def sha256(path):
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def select_cases(manifest, requested):
    available = {case["id"] for case in manifest["cases"]}
    unknown = set(requested or ()) - available
    if unknown:
        raise ValueError(f"unknown frozen case IDs: {sorted(unknown)}")
    cases = [case for case in manifest["cases"] if not requested or case["id"] in requested]
    if not cases:
        raise ValueError("case selection is empty")
    return cases


def pinned_inputs():
    folder = Path(__file__).resolve().parent
    baseline = folder / "baseline_serial_af0d5824.py"
    manifest = folder / "cases.json"
    if sha256(baseline) != BASE_SHA256 or sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("pinned original wrapper or frozen 15-case manifest changed")
    return {
        "baseline_commit": BASE_COMMIT,
        "baseline_path": str(baseline),
        "baseline_canonical_sha256": sha256(baseline),
        "manifest_path": str(manifest),
        "manifest_canonical_sha256": sha256(manifest),
        "manifest": json.loads(manifest.read_text(encoding="utf-8")),
    }


def runtime():
    # Import only in a device worker; --plan and CPU contract tests need no NPU.
    import baseline_serial_af0d5824 as original
    import torch
    import torch.nn.functional as functional
    import torch_npu
    import validate

    from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial

    return {
        "torch": torch,
        "functional": functional,
        "torch_npu": torch_npu,
        "validate": validate,
        "classes": {"original": original.MlaPrologSerial, "matmul": MlaPrologSerial},
    }


def tensor_metadata(value, rt):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "data_ptr": value.data_ptr(),
        "storage_ptr": value.untyped_storage().data_ptr(),
        "storage_bytes": value.untyped_storage().nbytes(),
        "storage_offset": value.storage_offset(),
        "format": int(rt["torch_npu"].get_npu_format(value)),
    }


def prepared_metadata(candidate, rt):
    return {
        "weights": candidate.prepared_weight_info(),
        "nz_layouts": [tensor_metadata(value, rt) for value in candidate._matrix_nz],
        "nd_layouts": [tensor_metadata(value, rt) for value in candidate._matrix_nd],
    }


def prepare(case, rt):
    torch, validation = rt["torch"], rt["validate"]
    data, initial_kv, initial_kr = validation.build_case(case)
    device = {name: value.npu() for name, value in data.items()}
    candidates, metadata = {}, {}
    for name, cls in rt["classes"].items():
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        start = time.perf_counter_ns()
        candidates[name] = cls(*(device[k] for k in ("wdq", "wuq", "wuk", "wdkv", "gq", "gkv")))
        torch.npu.synchronize()
        metadata[name] = {
            "prepare_host_synchronized_us": (time.perf_counter_ns() - start) / 1000,
            "allocated_bytes_delta": torch.npu.memory_allocated() - before,
            "prepared": prepared_metadata(candidates[name], rt),
        }
    return data, device, initial_kv, initial_kr, candidates, metadata


def paired_timing(graphs, stream, first, second, torch, unit="complete_forward"):
    samples = {first: [], second: []}
    orders = []
    for pair in range(PAIRED_SAMPLES):
        order = (first, second) if pair % 2 == 0 else (second, first)
        orders.append(list(order))
        for name in order:
            start = torch.npu.Event(enable_timing=True)
            stop = torch.npu.Event(enable_timing=True)
            with torch.npu.stream(stream):
                start.record()
                graphs[name].replay()
                stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / CALLS_PER_GRAPH)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "metric": f"{first}_graph_us / {second}_graph_us",
        "medians_per_call_us": medians,
        "ratio": medians[first] / medians[second],
        "raw_per_call_us": samples,
        "work_unit": unit,
        "orders": orders,
        "pairs": PAIRED_SAMPLES,
        "calls_per_graph": CALLS_PER_GRAPH,
        "timing_boundary": "synchronized NPU Event around complete Graph replay",
    }


def profile_replays(graph, stream, case_id, implementation, torch, result):
    marker = {
        "phase": "target_profile_replay",
        "case_id": case_id,
        "implementation": "serial",
        "trial_implementation": implementation,
        "calls_per_graph": CALLS_PER_GRAPH,
        "replays": PROFILE_REPLAYS,
        "host_begin_ns": time.time_ns(),
    }
    with torch.npu.stream(stream):
        for _ in range(PROFILE_REPLAYS):
            graph.replay()
        stream.synchronize()
    marker["host_end_ns"] = time.time_ns()
    result["profile_marker"] = marker


def run_pair(case, mode, implementation, rt, result):
    torch, validation = rt["torch"], rt["validate"]
    data, device, initial_kv, initial_kr, candidates, preparation = prepare(case, rt)
    expected = validation.oracle(data, case["heads"], initial_kv, initial_kr)
    immutable = {name: value.cpu().clone() for name, value in device.items()}
    device_layouts = {name: tensor_metadata(value, rt) for name, value in device.items()}
    caches = {name: (initial_kv.npu(), initial_kr.npu()) for name in candidates}
    cache_layouts = {name: [tensor_metadata(value, rt) for value in values] for name, values in caches.items()}
    result.update(
        checks={},
        preparation=preparation,
        input_metadata=device_layouts,
        cache_metadata=cache_layouts,
        matrix_route=candidates["matmul"].matrix_route(case["batch"]),
        route_label_is_original_layout_selector=True,
        changed_apis="down/up: ND linear(x,wt) to matmul(x,wt.t()); NZ mm to matmul",
        unchanged="weight preparation, row route, head bmm/transpose_batchmatmul, V kernel, thresholds",
        timing=None,
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, outputs = {}, {}

    def reset(kv_value, kr_value):
        for kv, kr in caches.values():
            kv.copy_(kv_value)
            kr.copy_(kr_value)

    def invoke(name):
        return candidates[name](device["x"], device["sin"], device["cos"], device["index"], *caches[name])

    def check(name, phase, reference, host_data, host_kv, host_kr):
        result["checks"][name + "_" + phase] = validation.check_outputs(
            outputs[name], caches[name], reference, host_data, host_kv, host_kr
        )

    with torch.npu.stream(stream):
        for name in candidates:
            reset(initial_kv, initial_kr)
            outputs[name] = invoke(name)
            stream.synchronize()
            check(name, "eager", expected, data, initial_kv, initial_kr)
        if not all(c["pass"] for c in result["checks"].values()):
            result["status"] = "FAIL_EAGER"
            return
        for name in candidates:
            for _ in range(EAGER_WARMUP):
                invoke(name)
            stream.synchronize()
            reset(initial_kv, initial_kr)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(CALLS_PER_GRAPH):
                    outputs[name] = invoke(name)
            graphs[name] = graph
            for _ in range(GRAPH_WARMUP):
                graph.replay()
            stream.synchronize()
            check(name, "graph", expected, data, initial_kv, initial_kr)
        result["inputs_immutable"] = all(torch.equal(value.cpu(), immutable[name]) for name, value in device.items())
        if not result["inputs_immutable"] or not all(c["pass"] for c in result["checks"].values()):
            result["status"] = "FAIL_GRAPH"
            return
        updated = validation.make_data(case["batch"], case["hidden"], case["heads"], case["seed"] + 1)
        slots = initial_kv.numel() // validation.KV_RANK
        updated["index"] = (slots - 1 - updated["index"]).contiguous()
        validation.validate_cache_index(updated["index"], slots)
        changed_kv, changed_kr = -initial_kv, initial_kr + 1
        updated_expected = validation.oracle(updated, case["heads"], changed_kv, changed_kr)
        for phase, host_data, host_kv, host_kr, reference in (
            ("changed_data", updated, changed_kv, changed_kr, updated_expected),
            ("restored_data", data, initial_kv, initial_kr, expected),
        ):
            for name, value in device.items():
                value.copy_(host_data[name])
            stream.synchronize()
            for name, candidate in candidates.items():
                start = time.perf_counter_ns()
                candidate.refresh_weights()
                stream.synchronize()
                actual = prepared_metadata(candidate, rt)
                if actual != preparation[name]["prepared"]:
                    raise AssertionError(f"{name}: refresh changed prepared layouts, storage, addresses or formats")
                result.setdefault("refreshes", []).append(
                    {
                        "implementation": name,
                        "phase": phase,
                        "host_synchronized_us": (time.perf_counter_ns() - start) / 1000,
                        "same_layout_storage_addresses_formats": True,
                    }
                )
            reset(host_kv, host_kr)
            for name, graph in graphs.items():
                graph.replay()
                stream.synchronize()
                check(name, "graph_" + phase, reference, host_data, host_kv, host_kr)
        result["source_addresses_strides_formats_stable"] = device_layouts == {
            name: tensor_metadata(value, rt) for name, value in device.items()
        }
        result["cache_addresses_strides_formats_stable"] = cache_layouts == {
            name: [tensor_metadata(value, rt) for value in values] for name, values in caches.items()
        }
        result["restored_inputs_exact"] = all(
            torch.equal(value.cpu(), immutable[name]) for name, value in device.items()
        )
        if not all(c["pass"] for c in result["checks"].values()) or not all(
            result[key]
            for key in (
                "source_addresses_strides_formats_stable",
                "cache_addresses_strides_formats_stable",
                "restored_inputs_exact",
            )
        ):
            result["status"] = "FAIL_CHANGED_OR_RESTORED_GRAPH"
            return
    if mode == "paired":
        result["timing"] = paired_timing(graphs, stream, "original", "matmul", torch)
    elif mode == "profile":
        profile_replays(graphs[implementation], stream, case["id"], implementation, torch, result)
    result["status"] = "PASS"


def run_micro(case, rt, result):
    torch, functional, validation = rt["torch"], rt["functional"], rt["validate"]
    data, device, _, _, candidates, preparation = prepare(case, rt)
    candidate = candidates["original"]
    result.update(
        stages=[],
        preparation=preparation,
        status="PASS",
        numerical_reference="CPU FP64, original frozen BF16 thresholds",
    )
    query_rank = candidate.weight_uq_qr.shape[0]
    up_cpu = torch.randn((case["batch"], query_rank), generator=torch.Generator().manual_seed(case["seed"] + 200)).to(
        torch.bfloat16
    )
    activations = (device["x"].view(case["batch"], case["hidden"]), up_cpu.npu())
    weights_cpu = (torch.cat((data["wdq"], data["wdkv"]), dim=1), data["wuq"])
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    for stage_index, stage_name in enumerate(("down", "up")):
        activation = activations[stage_index]
        reference = activation.cpu().double() @ weights_cpu[stage_index].double()
        for layout in ("nd", "nz"):
            # Both argument conventions are views of the exact same storage.
            rhs = candidate._matrix_nd[stage_index].t() if layout == "nd" else candidate._matrix_nz[stage_index]
            linear_weight = rhs.t()
            assert rhs.untyped_storage().data_ptr() == linear_weight.untyped_storage().data_ptr()
            methods = {
                "linear": partial(functional.linear, activation, linear_weight),
                "mm": partial(torch.mm, activation, rhs),
                "matmul": partial(torch.matmul, activation, rhs),
            }
            stage = {
                "stage": stage_name,
                "layout": layout,
                "activation_source": "original x"
                if stage_index == 0
                else "seeded BF16 normalized-Q-shaped activation, shared across APIs",
                "activation": tensor_metadata(activation, rt),
                "mm_matmul_rhs": tensor_metadata(rhs, rt),
                "linear_weight_view": tensor_metadata(linear_weight, rt),
                "methods": {},
                "paired": [],
                "kernel_version": None,
                "kernel_version_reason": "Read actual OP Type from full msprof; API names do not fix kernel versions.",
            }
            result["stages"].append(stage)
            graphs, outputs = {}, {}
            for name, function in methods.items():
                record = {"status": "NOT_RUN"}
                stage["methods"][name] = record
                try:
                    with torch.npu.stream(stream):
                        for _ in range(EAGER_WARMUP):
                            function()
                        stream.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, stream=stream):
                            for _ in range(CALLS_PER_GRAPH):
                                outputs[name] = function()
                        for _ in range(GRAPH_WARMUP):
                            graph.replay()
                        stream.synchronize()
                    check = validation.compare(outputs[name], reference)
                    finite = bool(torch.isfinite(outputs[name].cpu().float()).all())
                    record.update(check=check, finite=finite)
                    if check["matched_ratio"] < 0.99 or check["max_abs_error"] > 1 or not finite:
                        record["status"] = "FAIL_PRECISION"
                    else:
                        record["status"] = "PASS"
                        graphs[name] = graph
                except Exception as error:
                    record.update(
                        status="ERROR",
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
            for first, second in (("linear", "mm"), ("linear", "matmul"), ("mm", "matmul")):
                if first in graphs and second in graphs:
                    stage["paired"].append(
                        paired_timing(graphs, stream, first, second, torch, unit="one_projection_matrix_call")
                    )
                else:
                    stage["paired"].append(
                        {"metric": f"{first}/{second}", "status": "NOT_ADMITTED_METHOD_FAILURE", "ratio": None}
                    )
            if any(method["status"] != "PASS" for method in stage["methods"].values()):
                result["status"] = "PARTIAL_METHOD_FAILURE"
    result["prepared_layout_storage_addresses_unchanged"] = preparation["original"]["prepared"] == prepared_metadata(
        candidate, rt
    )
    if not result["prepared_layout_storage_addresses_unchanged"]:
        result["status"] = "FAIL_PREPARED_MUTATION"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--mode", choices=("verify", "paired", "micro", "profile"), default="verify")
    parser.add_argument("--implementation", choices=("original", "matmul"), default="matmul")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="validate pinned contract and print selected cases without torch/NPU imports",
    )
    args = parser.parse_args()
    try:
        pinned = pinned_inputs()
        cases = select_cases(pinned["manifest"], args.case_id)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "profile" and len(cases) != 1:
        parser.error("profile requires exactly one frozen --case-id per process")
    protocol = {
        **pinned["manifest"]["protocol"],
        "purpose": "round1 of the newly authorized ten-round decode budget; historical10/10 unchanged",
        "new_budget_round": 1,
        "new_budget_limit": 10,
        "direct_comparison": "original complete Graph / matmul complete Graph",
        "micro_scope": "two projection stages, identical activation and ND/NZ storage; no layout preparation timed",
        "fusion_comparison": "unchanged validate.py --baseline-route aclnn, current matmul wrapper",
        "no_new_V3_domain": True,
        "kernel_versions": "observed from full msprof; no presumption of V2 or V3",
    }
    report = {
        "protocol": protocol,
        "pinned_baseline": {k: v for k, v in pinned.items() if k != "manifest"},
        "mode": args.mode,
        "cases": [],
        "identity": None,
    }
    if args.plan:
        report["selected_cases"] = cases
        print(json.dumps(report, indent=2))
        return
    if args.out is None or args.source_commit is None:
        parser.error("device execution requires --out and --source-commit")
    if args.out.exists():
        parser.error("refusing to overwrite existing output; choose a new attempt path")
    repo = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.source_commit:
        parser.error(f"--source-commit does not match running checkout HEAD {head}")
    rt = runtime()
    torch = rt["torch"]
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    for case in cases:
        value = {"case": case, "status": "NOT_RUN", "timing": None}
        try:
            if args.mode == "micro":
                run_micro(case, rt, value)
            else:
                run_pair(case, args.mode, args.implementation, rt, value)
        except Exception as error:
            value.update(
                status="ERROR",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
                timing=None,
            )
        report["cases"].append(value)
        try:
            report["identity"] = rt["validate"].identity(args.source_commit, "aclnn", True)
            report["identity"]["candidate_canonical_sha256"] = sha256(repo / "vllm_ascend/ops/mla_prolog_serial.py")
            report["identity"]["source_tree"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
            ).strip()
        except Exception as error:
            value["identity_error"] = {"type": type(error).__name__, "message": str(error)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "mode": args.mode,
                    "status": value["status"],
                    "timing": value.get("timing"),
                    "error": value.get("error"),
                }
            ),
            flush=True,
        )
    raise SystemExit(0 if all(case["status"] == "PASS" for case in report["cases"]) else 1)


if __name__ == "__main__":
    main()
