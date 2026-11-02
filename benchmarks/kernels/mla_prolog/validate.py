# SPDX-License-Identifier: Apache-2.0
"""Complete decode Graph, state and precision checks for serial MLA Prolog."""

import argparse
import hashlib
import json
import math
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C as native_extension
from preflight import KV_RANK, ROPE_DIM, compare, make_data, projections, rotate_half

from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial, validate_cache_index

CALLS_PER_GRAPH = 10
WARMUP_CALLS = 10
GRAPH_WARMUP = 10
PAIRED_SAMPLES = 20


def rope_reference(x, sine, cosine):
    base = x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).flatten(-2)
    return base * cosine + rotate_half(base) * sine


def oracle(data, heads, initial_kv, initial_kr):
    query, raw_qr, kv, raw_kr = projections(data, heads)
    sine, cosine = data["sin"].double(), data["cos"].double()
    query_rope = rope_reference(raw_qr, sine.unsqueeze(-2), cosine.unsqueeze(-2))
    kr = rope_reference(raw_kr, sine, cosine)
    output_kv, output_kr = initial_kv.double().clone(), initial_kr.double().clone()
    index = data["index"].flatten()
    output_kv.view(-1, KV_RANK)[index] = kv.flatten(0, 1)
    output_kr.view(-1, ROPE_DIM)[index] = kr.flatten(0, 1)
    return query, query_rope, output_kv, output_kr


def check_outputs(outputs, caches, expected, data, initial_kv, initial_kr):
    results = {}
    names = ("query", "query_rope", "kv_cache_returned", "kr_cache_returned")
    for name, value, reference in zip(names, outputs, expected):
        if list(value.shape) != list(reference.shape) or value.dtype != torch.bfloat16:
            raise AssertionError(f"{name} ABI differs: {value.shape}, {value.dtype}, expected {reference.shape}")
        result = compare(value, reference)
        result["finite"] = bool(torch.isfinite(value.cpu().float()).all())
        result["pass"] = result["matched_ratio"] >= 0.99 and result["max_abs_error"] <= 1 and result["finite"]
        results[name] = result
    slots = initial_kv.numel() // KV_RANK
    touched = torch.zeros(slots, dtype=torch.bool)
    touched[data["index"].flatten()] = True
    for name, value, returned, reference, initial, width in (
        ("kv_cache", caches[0], outputs[2], expected[2], initial_kv, KV_RANK),
        ("kr_cache", caches[1], outputs[3], expected[3], initial_kr, ROPE_DIM),
    ):
        actual = value.cpu()
        written = compare(actual.reshape(-1, width)[touched], reference.reshape(-1, width)[touched])
        untouched_exact = torch.equal(actual.reshape(-1, width)[~touched], initial.reshape(-1, width)[~touched])
        returned_untouched_exact = torch.equal(
            returned.cpu().reshape(-1, width)[~touched], initial.reshape(-1, width)[~touched]
        )
        results[name + "_state"] = {
            "written": written,
            "untouched_exact": untouched_exact,
            "returned_untouched_exact": returned_untouched_exact,
            "returned_aliases_input": returned.data_ptr() == value.data_ptr(),
            "pass": written["matched_ratio"] >= 0.99
            and written["max_abs_error"] <= 1
            and untouched_exact
            and returned_untouched_exact,
        }
    results["pass"] = all(result["pass"] for result in results.values())
    return results


def build_case(case):
    data = make_data(case["batch"], case["hidden"], case["heads"], case["seed"])
    if case.get("zeros"):
        data["x"].zero_()
    if case.get("zero_gamma"):
        data["gq"].zero_()
        data["gkv"].zero_()
    block_size = case.get("block_size", 16)
    slots = math.ceil(max(block_size, case["batch"] * 3 + 2) / block_size) * block_size
    if case.get("index_mode") == "reverse":
        data["index"] = (slots - 1 - data["index"]).contiguous()
    if case.get("index_mode") == "edges":
        data["index"] = torch.linspace(0, slots - 1, case["batch"], dtype=torch.int64).reshape(-1, 1)
    validate_cache_index(data["index"], slots)
    generator = torch.Generator().manual_seed(case["seed"] + 100)
    initial_kv = torch.randn((slots // block_size, block_size, 1, KV_RANK), generator=generator).to(torch.bfloat16)
    initial_kr = torch.randn((slots // block_size, block_size, 1, ROPE_DIM), generator=generator).to(torch.bfloat16)
    return data, initial_kv, initial_kr


def paired_timing(graphs, stream):
    samples = {name: [] for name in graphs}
    for pair in range(PAIRED_SAMPLES):
        order = ("fused", "serial") if pair % 2 == 0 else ("serial", "fused")
        for name in order:
            start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            with torch.npu.stream(stream):
                start.record()
                graphs[name].replay()
                stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / CALLS_PER_GRAPH)
    fused, serial = (statistics.median(samples[name]) for name in ("fused", "serial"))
    return {
        "fused_graph_us": fused,
        "serial_graph_us": serial,
        "speedup": fused / serial,
        "target_pass": fused / serial >= 1.5,
        "floor_pass": fused / serial >= 1.0,
        "raw_per_call_us": samples,
        "pairs": PAIRED_SAMPLES,
    }


def run_case(case, timing=True, profile=None, baseline_route="aclnn", internal_format=True, progress=None):
    torch.npu.config.allow_internal_format = True
    data, initial_kv, initial_kr = build_case(case)
    expected = oracle(data, case["heads"], initial_kv, initial_kr)
    device = {name: value.npu() for name, value in data.items()}
    # Baseline's documented NZ representation is prepared outside forward.
    fused_weights = {name: torch_npu.npu_format_cast(device[name], 29) for name in ("wdq", "wuq", "wdkv")}
    formats = {name: torch_npu.get_npu_format(value) for name, value in fused_weights.items()}
    if set(formats.values()) != {29}:
        raise AssertionError(f"baseline weights did not use documented NZ: {formats}")
    torch.npu.config.allow_internal_format = internal_format
    torch.npu.synchronize()
    allocated_before_prepare = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    prepare_begin = time.perf_counter_ns()
    candidate = MlaPrologSerial(
        device["wdq"], device["wuq"], device["wuk"], device["wdkv"], device["gq"], device["gkv"]
    )
    torch.npu.synchronize()
    prepare_us = (time.perf_counter_ns() - prepare_begin) / 1000
    prepared_initial = candidate.prepared_weight_info()
    prepared_allocated_bytes = torch.npu.memory_allocated() - allocated_before_prepare
    prepare_peak_extra_bytes = torch.npu.max_memory_allocated() - allocated_before_prepare
    immutable_inputs = {name: value.cpu().clone() for name, value in device.items()}
    immutable_fused_weights = {name: value.cpu().clone() for name, value in fused_weights.items()}
    caches = {name: (initial_kv.npu(), initial_kr.npu()) for name in ("fused", "serial")}

    def serial():
        return candidate(device["x"], device["sin"], device["cos"], device["index"], *caches["serial"])

    def fused():
        baseline = (
            torch.ops._C_ascend.mla_prolog_fused_public if baseline_route == "aclnn" else torch_npu.npu_mla_prolog
        )
        return baseline(
            device["x"],
            fused_weights["wdq"],
            fused_weights["wuq"],
            device["wuk"],
            fused_weights["wdkv"],
            device["gq"],
            device["gkv"],
            device["sin"],
            device["cos"],
            device["index"],
            *caches["fused"],
        )

    functions = {"fused": fused, "serial": serial}

    def reset_caches():
        for kv, kr in caches.values():
            kv.copy_(initial_kv)
            kr.copy_(initial_kr)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, captured = {}, {}
    result = {
        "case": case,
        "baseline_weight_formats": formats,
        "baseline_route": baseline_route,
        "serial_pipeline": "merged-down-adaptive-linear-transpose-bmm",
        "serial_matrix_route": candidate.matrix_route(case["batch"]),
        "serial_weight_preparation": {
            "contract": "Explicit NZ29/ND2 buffers; refresh_weights after source changes outside Graph.",
            "prepare_host_synchronized_us": prepare_us,
            "allocated_bytes_delta": prepared_allocated_bytes,
            "prepare_peak_extra_allocated_bytes": prepare_peak_extra_bytes,
            "initial_weights": prepared_initial,
            "refreshes": [],
        },
        "checks": {},
        "timing": None,
    }
    if progress is not None:
        progress.update(result)
        result = progress
    with torch.npu.stream(stream):
        for name, function in functions.items():
            outputs = function()
            stream.synchronize()
            result["checks"][name + "_eager"] = check_outputs(
                outputs, caches[name], expected, data, initial_kv, initial_kr
            )
        if not all(check["pass"] for check in result["checks"].values()):
            result["status"] = "FAIL_EAGER"
            return result
        for name, function in functions.items():
            for _ in range(WARMUP_CALLS):
                function()
            stream.synchronize()
            reset_caches()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(CALLS_PER_GRAPH):
                    outputs = function()
            graphs[name], captured[name] = graph, outputs
            for _ in range(GRAPH_WARMUP):
                graph.replay()
            stream.synchronize()
            result["checks"][name + "_graph"] = check_outputs(
                outputs, caches[name], expected, data, initial_kv, initial_kr
            )
        result["inputs_immutable"] = all(
            torch.equal(value.cpu(), immutable_inputs[name]) for name, value in device.items()
        ) and all(torch.equal(value.cpu(), immutable_fused_weights[name]) for name, value in fused_weights.items())
        if not result["inputs_immutable"] or not all(check["pass"] for check in result["checks"].values()):
            result["status"] = "FAIL_GRAPH"
            return result

        # Change input, gamma, all four matrix values, RoPE, metadata, and cache
        # contents at stable addresses. NZ copies are updated explicitly too.
        updated = make_data(case["batch"], case["hidden"], case["heads"], case["seed"] + 1)
        slots = initial_kv.numel() // KV_RANK
        updated["index"] = (slots - 1 - updated["index"]).contiguous()
        validate_cache_index(updated["index"], slots)
        for name, value in device.items():
            value.copy_(updated[name])
        for name, value in fused_weights.items():
            value.copy_(torch_npu.npu_format_cast(device[name], 29))
        stream.synchronize()
        refresh_begin = time.perf_counter_ns()
        candidate.refresh_weights()
        stream.synchronize()
        refreshed = candidate.prepared_weight_info()
        if refreshed != prepared_initial:
            raise AssertionError("Prepared refresh changed formats, captured addresses or storage sizes")
        result["serial_weight_preparation"]["refreshes"].append(
            {
                "phase": "changed_data",
                "host_synchronized_us": (time.perf_counter_ns() - refresh_begin) / 1000,
                "stable_format_addresses_storage": True,
                "weights": refreshed,
            }
        )
        updated_expected = oracle(updated, case["heads"], initial_kv, initial_kr)
        for name, graph in graphs.items():
            reset_caches()
            graph.replay()
            stream.synchronize()
            result["checks"][name + "_graph_changed_data"] = check_outputs(
                captured[name], caches[name], updated_expected, updated, initial_kv, initial_kr
            )
        for name, value in device.items():
            value.copy_(data[name])
        for name, value in fused_weights.items():
            value.copy_(torch_npu.npu_format_cast(device[name], 29))
        stream.synchronize()
        refresh_begin = time.perf_counter_ns()
        candidate.refresh_weights()
        stream.synchronize()
        refreshed = candidate.prepared_weight_info()
        if refreshed != prepared_initial:
            raise AssertionError("Prepared restore changed formats, captured addresses or storage sizes")
        result["serial_weight_preparation"]["refreshes"].append(
            {
                "phase": "restored_data",
                "host_synchronized_us": (time.perf_counter_ns() - refresh_begin) / 1000,
                "stable_format_addresses_storage": True,
                "weights": refreshed,
            }
        )
        reset_caches()
        for graph in graphs.values():
            graph.replay()
        stream.synchronize()
        if not all(check["pass"] for check in result["checks"].values()):
            result["status"] = "FAIL_GRAPH_CHANGED_DATA"
            return result
    if profile:
        # One process/case/mode: retain the last two complete captured Graphs.
        marker = {
            "phase": "target_profile_replay",
            "case_id": case["id"],
            "implementation": profile,
            "calls_per_graph": CALLS_PER_GRAPH,
            "replays": 2,
            "host_begin_ns": time.time_ns(),
        }
        with torch.npu.stream(stream):
            for _ in range(2):
                graphs[profile].replay()
            stream.synchronize()
        marker["host_end_ns"] = time.time_ns()
        result["profile_marker"] = marker
    elif timing:
        result["timing"] = paired_timing(graphs, stream)
    result["status"] = "PASS"
    return result


def identity(source_commit, baseline_route, internal_format):
    loaded = sorted(
        {
            row.split()[-1]
            for row in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in row and ("opapi" in row or "vllm_ascend" in row)
        }
    )
    return {
        "source_commit": source_commit,
        "extension_path": native_extension.__file__,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "soc": torch.npu.get_device_name(0),
        "allow_internal_format_requested": internal_format,
        "baseline_route": baseline_route,
        "schema": str(torch.ops._C_ascend.mla_prolog_vector_out.default._schema),
        "loaded_libraries": [
            {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for path in loaded
            if Path(path).is_file()
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--profile", choices=("serial", "fused"))
    parser.add_argument("--baseline-route", choices=("aclnn", "torch"), default="aclnn")
    parser.add_argument("--internal-format", choices=("true", "false"), default="true")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
    cases = [case for case in manifest["cases"] if not args.case_id or case["id"] in args.case_id]
    unknown_ids = set(args.case_id or ()) - {case["id"] for case in manifest["cases"]}
    if not cases or unknown_ids:
        parser.error(f"case selection is empty or unknown: {sorted(unknown_ids)}")
    report = {"protocol": manifest["protocol"], "cases": [], "identity": None}
    for case in cases:
        progress = {}
        try:
            value = run_case(
                case,
                timing=not args.no_timing,
                profile=args.profile,
                baseline_route=args.baseline_route,
                internal_format=args.internal_format == "true",
                progress=progress,
            )
        except Exception as error:
            value = progress
            value.update(
                {
                    "case": case,
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "timing": None,
                }
            )
        report["cases"].append(value)
        print(
            json.dumps(
                {
                    "id": case["id"],
                    "status": value["status"],
                    "timing": value.get("timing"),
                    "error": value.get("error"),
                }
            ),
            flush=True,
        )
        report["identity"] = identity(args.source_commit, args.baseline_route, args.internal_format == "true")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if all(case["status"] == "PASS" for case in report["cases"]) else 1)
