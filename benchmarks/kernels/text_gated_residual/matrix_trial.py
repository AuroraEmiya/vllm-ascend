# SPDX-License-Identifier: Apache-2.0
"""Controlled old/current wrapper comparison using the loaded own V binary."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import torch
import torch_npu
import validate as validation
from baseline_serial_a0d27176 import TextGatedResidualSerial as Original

from vllm_ascend.ops.text_gated_residual import TextGatedResidualSerial as Candidate

BASE_SHA256 = "46e16e5527147d5d8a64bbabf14c113e68fc363459d03cadd5e2dbfcc899d081"


def metadata(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "format": int(torch_npu.get_npu_format(value)),
        "data_ptr": value.data_ptr(),
        "storage_bytes": value.untyped_storage().nbytes(),
    }


def run_case(case, candidate_only=False, mix_tile_control=False, norm_buffer_control=False, up_format_control=False):
    cpu = validation.make_inputs(case)
    refs = validation.oracle(cpu, case)
    inputs = [v.npu() if v is not None else None for v in cpu]
    x, wd, wu, wg, gamma = inputs
    torch.npu.synchronize()
    prepare_start = time.perf_counter()
    memory_before = torch.npu.memory_allocated()
    objects = {"candidate": Candidate(wd, wu, wg)}
    torch.npu.synchronize()
    preparation = {
        "candidate_wall_us": (time.perf_counter() - prepare_start) * 1e6,
        "candidate_extra_device_bytes": torch.npu.memory_allocated() - memory_before,
        "scope": "outside capture, includes packed down/gate and explicit NZ up",
    }
    if up_format_control:
        control = Candidate(wd, wu, wg)
        control._use_nz_up = False
        objects = {"original": control, **objects}
    elif norm_buffer_control:
        control = Candidate(wd, wu, wg)
        control._norm_mode = 8
        objects = {"original": control, **objects}
    elif mix_tile_control:
        control = Candidate(wd, wu, wg)
        control._mix_mode = 6
        objects = {"original": control, **objects}
    elif not candidate_only:
        objects = {"original": Original(wd, wu, wg), **objects}
    calls = {name: (lambda obj=obj: obj(x, gamma, case["need_backward"])) for name, obj in objects.items()}
    result = {
        "case": case,
        "checks": {},
        "input_metadata": [metadata(v) if v is not None else None for v in inputs],
        "timing": None,
        "preparation": preparation,
        "comparison": "same-binary ND-up / candidate"
        if up_format_control
        else "same-binary legacy-norm / candidate"
        if norm_buffer_control
        else "same-binary legacy-mix-tile / candidate"
        if mix_tile_control
        else "historical-wrapper / candidate",
    }
    result["prepared"] = {
        name: {
            "down": metadata(obj.w_down),
            "up": metadata(obj.w_up),
            "up_nz": None if getattr(obj, "up_nz", None) is None else metadata(obj.up_nz),
            "packed": None if obj.packed_weight is None else metadata(obj.packed_weight),
        }
        for name, obj in objects.items()
    }
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, outputs = {}, {}
    with torch.npu.stream(stream):
        for name, call in calls.items():
            outputs[name] = call()
            stream.synchronize()
            result["checks"][name + "_eager"] = validation.check_outputs(outputs[name], refs)
            for _ in range(10):
                call()
            stream.synchronize()
            graphs[name], outputs[name] = validation.capture(call, stream)
            for _ in range(10):
                graphs[name].replay()
            stream.synchronize()
            result["checks"][name + "_graph"] = validation.check_outputs(outputs[name], refs)
        result["immutable_inputs"] = all(a is None or torch.equal(a.cpu(), b) for a, b in zip(inputs, cpu))
        updated = [v.clone() if v is not None else None for v in cpu]
        for i in (0, 1, 2, 3):
            if updated[i] is not None:
                updated[i].neg_()
        if updated[4] is not None:
            updated[4].add_(0.0625)
        addresses = [v.data_ptr() if v is not None else None for v in inputs]
        for phase, values in (("changed", updated), ("restored", cpu)):
            for tensor, value in zip(inputs, values):
                if tensor is not None:
                    tensor.copy_(value)
            for obj in objects.values():
                obj.refresh_weights_()
            expected = validation.oracle(values, case)
            for name, graph in graphs.items():
                graph.replay()
                stream.synchronize()
                result["checks"][name + "_" + phase] = validation.check_outputs(outputs[name], expected)
        result["prepared_addresses_stable"] = all(
            getattr(obj, "up_nz", None) is None or result["prepared"][name]["up_nz"]["data_ptr"] == obj.up_nz.data_ptr()
            for name, obj in objects.items()
        )
        result["same_addresses"] = addresses == [v.data_ptr() if v is not None else None for v in inputs]
        passed = (
            result["immutable_inputs"]
            and result["same_addresses"]
            and result["prepared_addresses_stable"]
            and all(v["status"] == "PASS" for v in result["checks"].values())
        )
        result["status"] = "PASS" if passed else "FAIL_ACCURACY"
        if passed:
            samples, orders = {name: [] for name in objects}, []
            for pair in range(20):
                order = tuple(objects) if pair % 2 == 0 else tuple(reversed(objects))
                orders.append(list(order))
                for name in order:
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(begin.elapsed_time(end) * 1000 / validation.CALLS)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            result["timing"] = {
                "medians_per_forward_us": medians,
                "original_over_candidate": None if candidate_only else medians["original"] / medians["candidate"],
                "raw_us": samples,
                "orders": orders,
                "pairs": 20,
                "forwards_per_graph": validation.CALLS,
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--mix-tile-control", action="store_true")
    parser.add_argument("--norm-buffer-control", action="store_true")
    parser.add_argument("--up-format-control", action="store_true")
    args = parser.parse_args()
    if sum((args.mix_tile_control, args.norm_buffer_control, args.up_format_control, args.candidate_only)) > 1:
        parser.error("select one control or candidate-only option")
    root = Path(__file__).resolve().parents[3]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != args.source_commit:
        raise ValueError("source commit does not match runtime checkout")
    baseline = Path(__file__).with_name("baseline_serial_a0d27176.py")
    if hashlib.sha256(baseline.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != BASE_SHA256:
        raise ValueError("pinned original wrapper mismatch")
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    if args.case_id:
        cases = [c for c in cases if c["id"] in args.case_id]
        if len(cases) != len(set(args.case_id)):
            raise ValueError("unknown case")
    result = {
        "source_commit": actual,
        "baseline_wrapper": (
            "current Candidate with ND up; only selected up weight format differs"
            if args.up_format_control
            else "current Candidate with legacy-norm mode8; only buffering differs"
            if args.norm_buffer_control
            else "current Candidate with internal legacy-mix-tile mode6; only tile differs"
            if args.mix_tile_control
            else "a0d271767512cce899982182f44ceb4824f8d3b7"
        ),
        "baseline_V": (
            "Both wrappers invoke the currently loaded own V binary; round01 unchanged from original hash. "
            "Later V changes invalidate an old-binary interpretation."
        ),
        "protocol": (
            "Complete NPUGraph10forward/replay;20 AB/BA pairs; CPU FP64 oracle outside capture; "
            "no precision casts before matrix calls"
        ),
        "identity": validation.identity(args.work_root),
        "cases": [],
    }
    validation.write_json(args.out, result)
    for case in cases:
        start = time.time_ns()
        try:
            row = run_case(
                case, args.candidate_only, args.mix_tile_control, args.norm_buffer_control, args.up_format_control
            )
        except Exception:
            row = {"case": case, "status": "INVALID", "timing": None, "error": traceback.format_exc()}
        row["host_begin_ns"] = start
        row["host_end_ns"] = time.time_ns()
        result["cases"].append(row)
        validation.write_json(args.out, result)
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "status": row["status"],
                    "ratio": (row["timing"] or {}).get("original_over_candidate"),
                    "error": row.get("error"),
                }
            ),
            flush=True,
        )
        if row["status"] == "INVALID":
            break


if __name__ == "__main__":
    main()
