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
    parser.add_argument("--mode", choices=("serial", "native", "static", "superkernel"), required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--matrix-backend", choices=("torch", "v3", "v3_nz"), default="v3_nz")
    parser.add_argument("--pad-down", type=int, choices=(0, 640), default=640)
    parser.add_argument("--split-mode", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--single-call-graph", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--batch-activation", type=int, choices=(0, 1), default=1)
    parser.add_argument("--preload-code", type=int, choices=(1, 5), default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--vector-backend", choices=("legacy", "staged"), default="staged")
    args = parser.parse_args()
    if args.single_call_graph and not args.no_timing:
        parser.error("single-call Graph is a correctness guard; use --no-timing")
    if args.candidate_only and not args.no_timing:
        parser.error("candidate-only checks cannot produce paired performance; use --no-timing")
    if args.single_call_graph:
        if hasattr(validation, "CALLS"):
            validation.CALLS = 1
            capture_original = validation.capture
            validation.capture = lambda call, stream, calls=1: capture_original(call, stream, calls=1)
        else:
            validation.CALLS_PER_GRAPH = 1
    # Installed9.2 ignores container CPU quotas when choosing op_compiler -j.
    # Bound only this benchmark process's compiler workers; no installed edits.
    from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel

    static_kernel._compute_opc_compile_jobs = lambda: "4"
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    is_tgr = hasattr(validation, "TextGatedResidualSerial")
    name = "TextGatedResidualSerial" if is_tgr else "MlaPrologSerial"
    original = getattr(validation, name)
    preparations = []

    def factory(*a, **kw):
        instance = original(*a, **kw)
        if args.mode in ("serial", "native"):
            if args.mode == "native":
                instance._matrix_backend = args.matrix_backend
                instance.prepare_superkernel_packing(args.pad_down if args.matrix_backend == "v3_nz" else 0)
            return instance
        begin = time.perf_counter_ns()
        compiled = instance.compile_superkernel(
            vector_backend=args.vector_backend,
            optimize_options={"constant_codegen": 1, "split_mode": args.split_mode, "preload_code": args.preload_code},
            optimize=args.mode == "superkernel",
            matrix_backend=args.matrix_backend,
            pad_down=args.pad_down,
            batch_activation=bool(args.batch_activation),
        )
        preparations.append(
            {
                "runtime_preparation": compiled.preparation,
                "wrapper_creation_us": (time.perf_counter_ns() - begin) / 1000,
                "options": compiled.options,
                "note": "lazy compile/installation occurs on first warmup, outside timing",
            }
        )
        return compiled

    make_calls_original = validation.make_calls

    def make_calls_with_storage(*a, **kw):
        tensors, calls, prep, serial = make_calls_original(*a, **kw)
        if args.candidate_only:
            calls = {"serial": calls["serial"]}
        owner = getattr(serial, "pipeline", serial)
        storage = {}
        for field in ("dummy", "dummy_gamma", "packed_weight", "up_nz", "_sk_packed_weight"):
            value = getattr(owner, field, None)
            if value is not None:
                storage[field] = {"bytes": value.untyped_storage().nbytes(), "data_ptr": value.data_ptr()}
        prep["persistent_owned_storage"] = storage
        prep["persistent_tensor_bytes"] = sum(v["bytes"] for v in storage.values())
        prep["storage_scope"] = (
            "owned prepared weights and dummy tensors, excluding caller source weights and per-call intermediates"
        )
        return tensors, calls, prep, serial

    validation.make_calls = make_calls_with_storage
    setattr(validation, name, factory)
    manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
    cases = manifest if is_tgr else manifest["cases"]
    if args.case_id:
        unknown = set(args.case_id) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"unknown cases: {unknown}")
        cases = [c for c in cases if c["id"] in args.case_id]
    report = {
        "batch_activation": args.batch_activation,
        "pad_down": args.pad_down,
        "matrix_backend": args.matrix_backend,
        "mode": args.mode,
        "vector_backend": args.vector_backend,
        "source_commit": args.source_commit,
        "cases": [],
        "compile_preparations": preparations,
        "grouping_status": "REQUIRES_DEVICE_TRACE",
        "calls_per_graph": 1 if args.single_call_graph else 10,
        "split_mode": args.split_mode,
        "preload_code": args.preload_code,
        "compiler_workers": 4,
        "candidate_only": args.candidate_only,
        "protocol": "frozen validation thresholds; single-call Graph correctness guard"
        if args.single_call_graph
        else "frozen validate.py; 10 complete calls/Graph; 20 alternating paired samples",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["source_files"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(validation.__file__),
                Path(__file__).parents[3] / "vllm_ascend/ops/serial_superkernel.py",
            )
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for case in cases:
        print("CASE_BEGIN", args.mode, case["id"], flush=True)
        begin = time.perf_counter_ns()
        try:
            if is_tgr:
                row = validation.check_case(case, not args.no_timing)
                row["candidate_correctness"] = (
                    row["eager"]["serial"]["status"] == "PASS"
                    and all(v["status"] == "PASS" for v in row["graph"]["serial"].values())
                    and row["input_immutability_eager"]
                    and row["input_immutability_graph"]
                )
                if args.candidate_only:
                    row["baseline_correctness"] = None
                    row["baseline_status"] = "NOT_RUN_CANDIDATE_ONLY"
                    row["comparison_admitted"] = False
                else:
                    row["baseline_correctness"] = row["eager"]["fused"]["status"] == "PASS" and all(
                        v["status"] == "PASS" for v in row["graph"]["fused"].values()
                    )
                if args.profile:
                    profile_path = args.out.with_name(args.out.stem + "." + case["id"] + ".profile.json")
                    validation.profile(case, str(profile_path))
                    row["profile_marker_file"] = profile_path.name
            else:
                row = validation.run_case(case, timing=not args.no_timing, profile="serial" if args.profile else None)
        except Exception:
            row = {"case": case, "status": "ERROR", "traceback": traceback.format_exc()}
        row["active_matrix_backend"] = "torch" if args.mode == "serial" else args.matrix_backend
        row["retained_serial_matrix_route"] = row.pop("serial_matrix_route", None)
        row["validation_host_wall_s_including_compile"] = (time.perf_counter_ns() - begin) / 1e9
        report["cases"].append(row)
        save()
        print("CASE_END", case["id"], row.get("status", row.get("comparison_admitted")), flush=True)
    loaded = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line
            and any(token in line for token in ("opapi", "nnopbase", "vllm_ascend", "custom_transformer"))
        }
    )
    report["runtime_identity"] = {
        "torch": torch.__version__,
        "soc": torch.npu.get_device_name(0),
        "allow_hf32_matmul": bool(torch.npu.matmul.allow_hf32),
        "files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in loaded if Path(p).is_file()},
    }
    save()
    success = all(r.get("candidate_correctness", r.get("status") == "PASS") for r in report["cases"])
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
