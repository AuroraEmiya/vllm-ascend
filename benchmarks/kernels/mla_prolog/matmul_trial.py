# SPDX-License-Identifier: Apache-2.0
"""Pinned MLA matrix-route trials; retain the original 15-case contract."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path

BASE_COMMIT = "af0d582472abb826d3ea6cbe37c75e7f49f9481c"
BASE_SHA256 = "654bb8fe5761e19edfdaf2e8a595c9d599cf10a7e40f1f410d44008b2107af51"
HEAD_BASE_COMMIT = "592d3963e3ef59fc147ef88ab0f114f35b731776"
HEAD_BASE_SHA256 = "0ee2fe6008ea62b83d1b815249d2517b3d39550e587850858a60ad1f07e7ba8c"
GMM_DOWN_SHA256 = "6ab63adaad223b69eda5e02f2690b8fe052c4c357b13bc19eeafecdf5b2f2cfe"
BROADCAST_BASE_COMMIT = "8b0d4da8d4897b3099736689c63ae7baea1dff3e"
BROADCAST_BASE_SHA256 = "cc2cacc7d07beb06138643d1aa868f0ef0865b5166a574b94be7d95df7e734fc"
GAMMA_CACHE_BASE_COMMIT = "5e35db3be11162a6edcb06e131d8dbaae7a2948e"
GAMMA_CACHE_BASE_SHA256 = "ff46e1fa24ed70794a191aaee14bdb68fa5512bffab0f09eedbe46ea9b254b42"
HEAD_LAYOUT_SHA256 = "7f6e2de290735bce07a27fe7d7f2523352eb7a8a763065048d525566d7a177df"
MANIFEST_SHA256 = "824f1b88598244182516c164dc9a369307ce89a2d1bf20b4887a8092a9528858"
CALLS_PER_GRAPH = 10
EAGER_WARMUP = 10
GRAPH_WARMUP = 10
PAIRED_SAMPLES = 20
PROFILE_REPLAYS = 2
HEAD_ACTIVATION_SEED_OFFSET = 300


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


def comparison_definition(comparison):
    if comparison == "projection":
        return {
            "baseline_commit": BASE_COMMIT,
            "baseline_file": "baseline_serial_af0d5824.py",
            "baseline_sha256": BASE_SHA256,
            "labels": ("original", "matmul"),
            "round": 1,
            "changed": "down/up: ND linear(x,wt) to matmul(x,wt.t()); NZ mm to matmul",
            "unchanged": "weight preparation, row route, head bmm/transpose_batchmatmul, V kernel, thresholds",
        }
    if comparison == "head-layout":
        return {
            "baseline_commit": HEAD_BASE_COMMIT,
            "baseline_file": "baseline_serial_592d3963.py",
            "baseline_sha256": HEAD_BASE_SHA256,
            "labels": ("control", "candidate"),
            "round": 2,
            "changed": "B>1 head: transpose_batchmatmul to bmm + permute + explicit contiguous",
            "unchanged": "BF16 boundaries, down/up matmul, prepared weights, row route, B1 head, V, thresholds",
        }
    if comparison == "gmm-down":
        return {
            "baseline_commit": HEAD_BASE_COMMIT,
            "baseline_file": "baseline_serial_592d3963.py",
            "baseline_sha256": HEAD_BASE_SHA256,
            "labels": ("control", "candidate"),
            "round": 3,
            "changed": "ND-route merged-down only: matmul to single-group npu_grouped_matmul",
            "unchanged": "BF16, prepared weights, static ND/NZ row route, NZ down, all up/head and V versus round1",
        }
    if comparison == "norm-broadcast":
        return {
            "baseline_commit": HEAD_BASE_COMMIT,
            "baseline_file": "baseline_serial_592d3963.py",
            "baseline_sha256": HEAD_BASE_SHA256,
            "labels": ("control", "candidate"),
            "round": 4,
            "changed": "RMS inverse application only: V_S/GetValue/Muls to UB Brcb/broadcast Mul",
            "unchanged": (
                "all public BF16 matrices, preparation, static routes, "
                "V arithmetic/order outside inverse broadcast, cache and thresholds versus round1"
            ),
            "native_comparison": (
                "one rebuilt package retains original mode3 scalar-read control; "
                "candidate mode4 broadcasts the same FP32 inverse in UB"
            ),
        }
    if comparison == "norm-gamma-cache":
        return {
            "baseline_commit": BROADCAST_BASE_COMMIT,
            "baseline_file": "baseline_serial_8b0d4da8.py",
            "baseline_sha256": BROADCAST_BASE_SHA256,
            "labels": ("control", "candidate"),
            "round": 5,
            "changed": "B>=32 RMS gamma DMA/Cast reuse within one core/launch; B<32 keeps mode4",
            "unchanged": "public BF16 matrices, all layouts, FP32 math, UB size, cache and precision versus round4",
            "native_comparison": (
                "mode4 retains round4 load/cast per unit; candidate mode5 caches gamma in f1[64:] "
                "with per-launch width tag, RoPE uses only f1[:64]"
            ),
        }
    if comparison == "split-dma-overlap":
        return {
            "baseline_commit": GAMMA_CACHE_BASE_COMMIT,
            "baseline_file": "baseline_serial_5e35db3b.py",
            "baseline_sha256": GAMMA_CACHE_BASE_SHA256,
            "labels": ("control", "candidate"),
            "round": 6,
            "changed": "B>=32 split: independent b0 Q-output and b3 RoPE-input DMA; B<32 keeps mode1",
            "unchanged": "round5 norm modes5/4, BF16 matrix routes/layouts, FP32 math, UB size, cache and thresholds",
            "native_comparison": (
                "mode1 retains the shared-b0 wait; mode6 loads RoPE into b3 while Q is read from b0, "
                "keeps both next-unit reuse fences and explicitly drains the last MTE3 output"
            ),
        }
    raise ValueError(f"unknown comparison {comparison}")


def pinned_inputs(comparison="projection"):
    definition = comparison_definition(comparison)
    folder = Path(__file__).resolve().parent
    baseline = folder / definition["baseline_file"]
    manifest = folder / "cases.json"
    if sha256(baseline) != definition["baseline_sha256"] or sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("pinned original wrapper or frozen 15-case manifest changed")
    return {
        "baseline_commit": definition["baseline_commit"],
        "baseline_path": str(baseline),
        "baseline_canonical_sha256": sha256(baseline),
        "manifest_path": str(manifest),
        "manifest_canonical_sha256": sha256(manifest),
        "manifest": json.loads(manifest.read_text(encoding="utf-8")),
    }


def runtime(comparison="projection"):
    # Import only in a device worker; --plan and CPU contract tests need no NPU.
    import torch
    import torch.nn.functional as functional
    import torch_npu
    import validate

    from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial

    definition = comparison_definition(comparison)
    spec = importlib.util.spec_from_file_location(
        "mla_trial_baseline", Path(__file__).parent / definition["baseline_file"]
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    control_label, candidate_label = definition["labels"]
    return {
        "torch": torch,
        "functional": functional,
        "torch_npu": torch_npu,
        "validate": validate,
        "classes": {control_label: original.MlaPrologSerial, candidate_label: MlaPrologSerial},
        "comparison": definition,
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
        "is_contiguous": value.is_contiguous(),
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
    control_label, candidate_label = rt["comparison"]["labels"]
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
        matrix_route=candidates[candidate_label].matrix_route(case["batch"]),
        route_label_is_original_layout_selector=True,
        changed_apis=rt["comparison"]["changed"],
        unchanged=rt["comparison"]["unchanged"],
        output_metadata={},
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
        metadata = [tensor_metadata(value, rt) for value in outputs[name]]
        result["output_metadata"][name + "_" + phase] = metadata
        if any(not value.is_contiguous() or value.dtype != torch.bfloat16 for value in outputs[name]):
            raise AssertionError(f"{name}/{phase}: all public outputs must be contiguous BF16")
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
        result["timing"] = paired_timing(graphs, stream, control_label, candidate_label, torch)
    elif mode == "profile":
        profile_replays(graphs[implementation], stream, case["id"], implementation, torch, result)
    result["status"] = "PASS"


def run_micro(case, rt, result):
    torch, functional, validation = rt["torch"], rt["functional"], rt["validate"]
    data, device, _, _, candidates, preparation = prepare(case, rt)
    control_label = rt["comparison"]["labels"][0]
    candidate = candidates[control_label]
    grouped_down = rt["comparison"]["round"] == 3
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
        if grouped_down and stage_name != "down":
            continue
        activation = activations[stage_index]
        reference = activation.cpu().double() @ weights_cpu[stage_index].double()
        actual_layout = "nd" if candidate.matrix_route(case["batch"]) == "nd_transposed_linear" else "nz"
        for layout in (actual_layout,) if grouped_down else ("nd", "nz"):
            # Both argument conventions are views of the exact same storage.
            rhs = candidate._matrix_nd[stage_index].t() if layout == "nd" else candidate._matrix_nz[stage_index]
            linear_weight = rhs.t()
            assert rhs.untyped_storage().data_ptr() == linear_weight.untyped_storage().data_ptr()
            methods = {
                "linear": partial(functional.linear, activation, linear_weight),
                "mm": partial(torch.mm, activation, rhs),
                "matmul": partial(torch.matmul, activation, rhs),
            }
            comparisons = (("linear", "mm"), ("linear", "matmul"), ("mm", "matmul"))
            if grouped_down:
                methods = {"control": partial(torch.matmul, activation, rhs)}
                if layout == "nd":
                    methods["candidate"] = partial(grouped_matmul_down, activation, rhs, rt)
                else:
                    methods["candidate"] = partial(torch.matmul, activation, rhs)
                comparisons = (("control", "candidate"),)
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
                "candidate_is_unchanged_control_route": grouped_down and layout == "nz",
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
                    metadata = tensor_metadata(outputs[name], rt)
                    record.update(check=check, finite=finite, output_metadata=metadata)
                    if (
                        check["matched_ratio"] < 0.99
                        or check["max_abs_error"] > 1
                        or not finite
                        or outputs[name].dtype != torch.bfloat16
                        or not outputs[name].is_contiguous()
                    ):
                        record["status"] = "FAIL_PRECISION_OR_LAYOUT"
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
            for first, second in comparisons:
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
    result["prepared_layout_storage_addresses_unchanged"] = preparation[control_label]["prepared"] == prepared_metadata(
        candidate, rt
    )
    if not result["prepared_layout_storage_addresses_unchanged"]:
        result["status"] = "FAIL_PREPARED_MUTATION"


def grouped_matmul_down(activation, rhs, rt):
    return rt["torch_npu"].npu_grouped_matmul(
        [activation], [rhs], split_item=0, group_type=-1, output_dtype=rt["torch"].bfloat16
    )[0]


def run_head_micro(case, rt, result):
    torch, validation = rt["torch"], rt["validate"]
    torch_npu = rt["torch_npu"]
    rows, heads = case["batch"], case["heads"]
    data, _, _ = validation.build_case(case)
    nope_dim, kv_rank = data["wuk"].shape[1:]
    head_cpu = torch.randn(
        (heads, rows, nope_dim), generator=torch.Generator().manual_seed(case["seed"] + HEAD_ACTIVATION_SEED_OFFSET)
    ).to(torch.bfloat16)
    head, weight = head_cpu.npu(), data["wuk"].npu()
    reference = torch.bmm(head_cpu.double(), data["wuk"].double()).permute(1, 0, 2).unsqueeze(1)

    def control():
        if rows == 1:
            return torch.bmm(head, weight).view(rows, 1, heads, kv_rank)
        return torch_npu.npu_transpose_batchmatmul(
            head, weight, perm_x1=[0, 1, 2], perm_x2=[0, 1, 2], perm_y=[1, 0, 2], batch_split_factor=1
        ).view(rows, 1, heads, kv_rank)

    def candidate():
        if rows == 1:
            return torch.bmm(head, weight).view(rows, 1, heads, kv_rank)
        return torch.bmm(head, weight).permute(1, 0, 2).contiguous().view(rows, 1, heads, kv_rank)

    raw = torch.bmm(head, weight)
    view = raw.permute(1, 0, 2).unsqueeze(1)
    result.update(
        status="PASS",
        input_metadata={"head_q": tensor_metadata(head, rt), "weight_uk": tensor_metadata(weight, rt)},
        raw_bmm_metadata=tensor_metadata(raw, rt),
        permuted_view_metadata=tensor_metadata(view, rt),
        view_alone_satisfies_contiguity=view.is_contiguous(),
        explicit_contiguous_inside_graph=True,
        extra_bmm_intermediate_logical_bytes_per_call=raw.numel() * raw.element_size()
        if not view.is_contiguous()
        else 0,
        numerical_reference="seeded shared BF16 head Q and source weight, CPU FP64 bmm; frozen BF16 thresholds",
        methods={},
        timing=None,
        kernel_version=None,
        kernel_version_reason="Read actual OP Type and layout/copy tasks from full msprof.",
    )
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, outputs = {}, {}
    for name, function in (("control", control), ("candidate", candidate)):
        record = {"status": "NOT_RUN"}
        result["methods"][name] = record
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
            metadata = tensor_metadata(outputs[name], rt)
            record.update(check=check, finite=finite, output_metadata=metadata)
            if (
                check["matched_ratio"] < 0.99
                or check["max_abs_error"] > 1
                or not finite
                or not outputs[name].is_contiguous()
                or outputs[name].dtype != torch.bfloat16
                or tuple(outputs[name].shape) != (rows, 1, heads, kv_rank)
            ):
                record["status"] = "FAIL_PRECISION_OR_LAYOUT"
            else:
                record["status"] = "PASS"
                graphs[name] = graph
        except Exception as error:
            record.update(
                status="ERROR", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()
            )
    if len(graphs) == 2:
        result["timing"] = paired_timing(
            graphs, stream, "control", "candidate", torch, unit="complete_head_matrix_and_output_layout"
        )
    else:
        result["status"] = "PARTIAL_METHOD_FAILURE"
    result["inputs_immutable"] = torch.equal(head.cpu(), head_cpu) and torch.equal(weight.cpu(), data["wuk"])
    if not result["inputs_immutable"]:
        result["status"] = "FAIL_INPUT_MUTATION"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--comparison",
        choices=("projection", "head-layout", "gmm-down", "norm-broadcast", "norm-gamma-cache", "split-dma-overlap"),
        default="projection",
    )
    parser.add_argument("--mode", choices=("verify", "paired", "micro", "head-micro", "profile"), default="verify")
    parser.add_argument("--implementation", choices=("original", "matmul", "control", "candidate"))
    parser.add_argument(
        "--plan",
        action="store_true",
        help="validate pinned contract and print selected cases without torch/NPU imports",
    )
    args = parser.parse_args()
    try:
        definition = comparison_definition(args.comparison)
        pinned = pinned_inputs(args.comparison)
        cases = select_cases(pinned["manifest"], args.case_id)
    except ValueError as error:
        parser.error(str(error))
    if args.mode == "profile" and len(cases) != 1:
        parser.error("profile requires exactly one frozen --case-id per process")
    if (args.mode == "micro" and args.comparison not in {"projection", "gmm-down"}) or (
        args.mode == "head-micro" and args.comparison != "head-layout"
    ):
        parser.error("micro requires projection or gmm-down; head-micro requires head-layout comparison")
    if args.implementation is None:
        args.implementation = definition["labels"][1]
    if args.implementation not in definition["labels"]:
        parser.error("--implementation must belong to the selected comparison")
    protocol = {
        **pinned["manifest"]["protocol"],
        "purpose": "newly authorized ten-round decode budget; historical10/10 unchanged",
        "comparison": args.comparison,
        "new_budget_round": definition["round"],
        "new_budget_limit": 10,
        "direct_comparison": " / ".join(definition["labels"]) + " complete Graph",
        "micro_scope": "identical BF16 inputs/weight storage; explicit output layout operations included",
        "changed": definition["changed"],
        "unchanged": definition["unchanged"],
        "native_comparison": definition.get("native_comparison", "unchanged shared V algorithm"),
        "fusion_comparison": "unchanged validate.py --baseline-route aclnn, current candidate wrapper",
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
    historical_hash = {
        "projection": HEAD_BASE_SHA256,
        "head-layout": HEAD_LAYOUT_SHA256,
        "gmm-down": GMM_DOWN_SHA256,
        "norm-broadcast": BROADCAST_BASE_SHA256,
        "norm-gamma-cache": GAMMA_CACHE_BASE_SHA256,
    }.get(args.comparison)
    if historical_hash and args.mode in {"verify", "paired", "profile"}:
        if sha256(repo / "vllm_ascend/ops/mla_prolog_serial.py") != historical_hash:
            parser.error("historical full trials require their exact candidate wrapper; select the current comparison")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.source_commit:
        parser.error(f"--source-commit does not match running checkout HEAD {head}")
    rt = runtime(args.comparison)
    torch = rt["torch"]
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    for case in cases:
        value = {"case": case, "status": "NOT_RUN", "timing": None}
        try:
            if args.mode == "micro":
                run_micro(case, rt, value)
            elif args.mode == "head-micro":
                run_head_micro(case, rt, value)
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
