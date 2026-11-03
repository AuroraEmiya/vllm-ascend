# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Capture actual public TextGatedResidual call shapes with torch-npu profiler.

Input parameters form an explicit synthetic operator sweep. Profiler traces and
runtime tensor metadata are evidence of operator calls, not full-model requests
or a newly validated continuous domain. No serial implementation is changed.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import traceback
from pathlib import Path

import cann_ops_transformer_custom  # noqa: F401
import torch
import torch_npu

from validate import make_inputs

OP = "cann_ops_transformer::text_gated_residual"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matrix():
    batches = [1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
               255, 256, 257, 511, 512, 1023, 1024, 2047, 2048]
    cases = []
    for b in batches:
        cases.append(dict(id=f"decode-b{b}", B=b, S=1, C=4, H=2560, R=320,
                          gate=True, gamma=True, need_backward=False,
                          seed=20261103 + b, input_mode="random", phase="decode",
                          source="synthetic parameter sweep; actual NPU metadata captured by profiler"))
    for label, gate, gamma, backward in [
        ("optional-none", False, False, False),
        ("aux-five", True, True, True),
        ("aux-no-gate", False, True, True),
    ]:
        cases.append(dict(cases[5], id=f"decode-b17-{label}", gate=gate,
                          gamma=gamma, need_backward=backward))
    return {"kind": "operator_level_runtime_capture_matrix",
            "model_config_dimensions": "Qwen3.8-Flash-Next H2560/C4/R320",
            "batch_selection": "explicit minimum, powers and branch-adjacent discrete samples",
            "not_full_model_capture": True, "not_new_performance_iteration": True,
            "cases": cases}


def tensor_meta(value):
    if value is None:
        return None
    result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                  stride=list(value.stride()), storage_offset=value.storage_offset(),
                  storage_bytes=value.untyped_storage().nbytes(),
                  data_ptr=value.data_ptr(), is_contiguous=value.is_contiguous())
    if value.device.type == "npu":
        result["npu_format"] = int(torch_npu.get_npu_format(value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = matrix()
    if args.write_manifest:
        dump(args.out / "capture-cases.json", manifest)
        return
    case = next(c for c in manifest["cases"] if c["id"] == args.case_id)
    result = {"id": case["id"], "case": case, "status": "STARTED",
              "source_level": "real_A3_operator_harness_not_model_capture",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "tool": "torch_npu.profiler.profile(record_shapes=True,CPU+NPU)",
              "correctness": "NOT_EVALUATED", "performance": "NOT_EVALUATED"}
    dump(args.out / "result.json", result)
    try:
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        result["environment"] = {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "device_name": torch.npu.get_device_name(0),
            "visible_device_count": torch.npu.device_count(),
            "schema": str(torch.ops.cann_ops_transformer.text_gated_residual.default._schema),
        }
        cpu = make_inputs(case)
        tensors = [t.npu() if t is not None else None for t in cpu]
        x, wd, wu, wg, gamma = tensors
        torch.npu.synchronize()
        result["inputs"] = dict(zip(("x", "w_down", "w_up", "w_gate", "gamma"),
                                   map(tensor_meta, tensors)))
        result["scalar_kwargs"] = dict(channels=4, eps=1e-6, need_backward=case["need_backward"])
        handler = torch_npu.profiler.tensorboard_trace_handler(
            str(args.out / "profiler"), analyse_flag=True, async_mode=False)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, profile_memory=False, with_stack=False,
            on_trace_ready=handler,
        ):
            outputs = torch.ops.cann_ops_transformer.text_gated_residual(
                x, wd, wu, w_gate=wg, channels=4, gamma=gamma, eps=1e-6,
                need_backward=case["need_backward"])
            torch.npu.synchronize()
        result["synchronized_call_completed"] = True
        result["outputs"] = dict(zip(("mixed", "inject", "inv_rms", "down_pre_act", "up_pre_act"),
                                    map(tensor_meta, outputs)))
        traces = sorted((args.out / "profiler").rglob("trace_view.json"))
        result["traces"] = [str(p) for p in traces]
        matches, kernels = [], []
        for trace in traces:
            content = json.loads(trace.read_text(encoding="utf-8"))
            events = content.get("traceEvents", []) if isinstance(content, dict) else content
            for event in events:
                name = event.get("name", "")
                if name == OP or name.startswith(OP + "."):
                    matches.append(event)
                if "textgatedresidual" in name.replace("_", "").lower():
                    kernels.append({"name": name, "cat": event.get("cat"), "args": event.get("args")})
        result["exact_dispatch_events"] = matches
        result["related_profiler_events"] = kernels
        result["exact_dispatch_hit"] = bool(matches)
        result["input_shape_recorded_by_profiler"] = any(
            any("shape" in str(k).lower() or "dim" in str(k).lower()
                for k in event.get("args", {})) for event in matches)
        result["status"] = ("CAPTURED" if matches and result["input_shape_recorded_by_profiler"]
                            else "CALL_COMPLETED_PROFILER_METADATA_INCOMPLETE")
        if result["status"] == "CAPTURED":
            # This adapter serializes observed runtime metadata, without calculating shapes.
            columns = ["node_id", "name", "op_name", "ori_name", "source",
                       "inputs_shape", "inputs_dtype", "outputs_shape", "outputs_dtype"]
            ins = [m for m in result["inputs"].values() if m is not None]
            outs = list(result["outputs"].values())
            row = dict(node_id=case["id"], name="TextGatedResidual", op_name=OP, ori_name=OP,
                       source="actual_synchronized_NPU_call_with_profiler_exact_dispatch_hit",
                       inputs_shape=json.dumps([m["shape"] for m in ins]),
                       inputs_dtype=json.dumps([m["dtype"] for m in ins]),
                       outputs_shape=json.dumps([m["shape"] for m in outs]),
                       outputs_dtype=json.dumps([m["dtype"] for m in outs]))
            with (args.out / "output-forward-runtime.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row)
        result["artifact_sha256"] = {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in traces}
    except Exception as exc:
        result.update(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    dump(args.out / "result.json", result)
    print(json.dumps({"id": case["id"], "status": result["status"],
                      "exact_dispatch_hit": result.get("exact_dispatch_hit"),
                      "outputs": result.get("outputs"), "error": result.get("error")}), flush=True)
    if result["status"] != "CAPTURED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
