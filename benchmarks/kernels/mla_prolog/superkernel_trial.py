# SPDX-License-Identifier: Apache-2.0
"""Run the frozen serial/fused checks with an explicitly selected compile mode."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
import validate as validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("serial", "static", "superkernel"), required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--matrix-backend", choices=("torch", "v3"), default="v3")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--baseline-route", choices=("aclnn", "torch", "v3"), default="aclnn")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    is_tgr = hasattr(validation, "TextGatedResidualSerial")
    name = "TextGatedResidualSerial" if is_tgr else "MlaPrologSerial"
    original = getattr(validation, name)
    preparations = []

    def factory(*a, **kw):
        instance = original(*a, **kw)
        if args.mode == "serial":
            return instance
        begin = time.perf_counter_ns()
        compiled = instance.compile_superkernel(optimize=args.mode == "superkernel", matrix_backend=args.matrix_backend)
        preparations.append({"runtime_preparation": compiled.preparation, "wrapper_creation_us": (time.perf_counter_ns() - begin) / 1000,
                             "options": compiled.options,
                             "note": "lazy compile/installation occurs on first warmup, outside timing"})
        return compiled

    setattr(validation, name, factory)
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
    cases = manifest if is_tgr else manifest["cases"]
    if args.case_id:
        unknown = set(args.case_id) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"unknown cases: {unknown}")
        cases = [c for c in cases if c["id"] in args.case_id]
    report = {"matrix_backend": args.matrix_backend, "mode": args.mode, "baseline_route": args.baseline_route, "source_commit": args.source_commit, "cases": [],
              "compile_preparations": preparations, "grouping_status": "REQUIRES_DEVICE_TRACE",
              "protocol": "frozen validate.py; 10 complete calls/Graph; 20 alternating paired samples"}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["source_files"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(validation.__file__),
            Path(__file__).parents[3] / "vllm_ascend/ops/serial_superkernel.py")}
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for case in cases:
        print("CASE_BEGIN", args.mode, case["id"], flush=True)
        begin = time.perf_counter_ns()
        try:
            if is_tgr:
                row = validation.check_case(case, not args.no_timing)
                if args.profile:
                    validation.profile(case, str(args.out.with_suffix(".profile.json")))
            else:
                row = validation.run_case(case, timing=not args.no_timing,
                                          profile="serial" if args.profile else None, baseline_route=args.baseline_route)
        except Exception:
            row = {"case": case, "status": "ERROR", "traceback": traceback.format_exc()}
        row["active_matrix_backend"] = "torch" if args.mode == "serial" else args.matrix_backend
        row["retained_serial_matrix_route"] = row.pop("serial_matrix_route", None)
        row["validation_host_wall_s_including_compile"] = (time.perf_counter_ns() - begin) / 1e9
        report["cases"].append(row)
        save()
        print("CASE_END", case["id"], row.get("status", row.get("comparison_admitted")), flush=True)
    loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if ".so" in line and any(token in line for token in
                        ("opapi", "nnopbase", "vllm_ascend", "custom_transformer"))})
    report["runtime_identity"] = {
        "torch": torch.__version__, "soc": torch.npu.get_device_name(0),
        "allow_hf32_matmul": bool(torch.npu.matmul.allow_hf32),
        "files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in loaded if Path(p).is_file()}}
    save()
    success = all(r.get("comparison_admitted", r.get("status") == "PASS") for r in report["cases"])
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
