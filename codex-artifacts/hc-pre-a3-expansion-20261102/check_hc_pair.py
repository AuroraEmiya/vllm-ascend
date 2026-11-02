"""Prepared small-T output preflight. Explicit invocation and a free NPU required.

This collects factual output checks, never changes deferred acceptance, and does
not profile performance. It has not been NPU executed in the resource-blocked run.
"""

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from measure_hc_expansion import load, sha
from pair_evidence import LIMITS, build_current_context, case_contract, require, validate_pair_evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/s30083169/perf-serial-graph"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--environment-evidence", type=Path, required=True)
    p.add_argument("--interface-evidence", type=Path, required=True)
    p.add_argument("--source-inventory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    assert out.is_relative_to(root / "outputs")
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
    assert os.environ.get("HC_PRE_STAGES", "123") in ("", "123")
    environment = json.loads(args.environment_evidence.read_text())
    _interface = json.loads(args.interface_evidence.read_text())
    inventory = json.loads(args.source_inventory.read_text())
    context, expected_cases = build_current_context(
        root, args.manifest, args.environment_evidence, args.interface_evidence, args.source_inventory
    )
    cases = json.loads(args.manifest.read_text())["cases"]
    selected = [next(c for c in cases if c["id"] == expected["case_id"]) for expected in expected_cases]
    for rel, expected in inventory["files"].items():
        if rel.startswith("work/hc_pre/"):
            assert sha(root / rel) == expected, "Frozen serial source/binary changed: " + rel
    vendor = Path(os.environ["PERF_BASELINE_VENDOR"]).resolve()
    api_binary = vendor / "op_api/lib/libcust_opapi.so"
    assert sha(api_binary) == inventory["files"][str(api_binary.relative_to(root))]
    serial_binaries = list((root / "work/hc_pre/kernel/build").glob("hc_pre_ext*.so"))
    assert len(serial_binaries) == 1
    out.mkdir(parents=True, exist_ok=False)
    guard = load(Path(__file__).with_name("resource_guard_a3.py"), "preflight_resource_guard")
    guard.gate(out, "before-initialization", set())
    snapshot = guard.read_snapshot(out, "before-initialization")
    assert snapshot["HBM_total_MiB"] - snapshot["HBM_used_MiB"] >= 32768
    # Imports and every NPU action are below the fresh resource gate.
    import torch
    import torch_npu

    require(str(torch.__version__) == environment["torch_version"], "Actual torch version differs")
    require(str(torch_npu.__version__) == environment["torch_npu_version"], "Actual torch_npu version differs")
    torch.set_num_threads(8)
    runtime_selection = guard.verify_runtime_selection(torch, out)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    torch.npu.matmul.allow_hf32 = False
    torch.ops.load_library(str(serial_binaries[0]))
    model = load(root / "work/hc_pre/model.py", "preflight_cpu_model")
    adapter = load(Path(__file__).with_name("hc_pre_aclnn_expansion.py"), "preflight_adapter")
    report = {
        "schema_version": 2,
        "status": "RUNNING",
        "NPU_executed": True,
        "context": context,
        "accepted_correctness": False,
        "precision_acceptance": "DEFERRED_BY_USER",
        "scope": (
            "T4 and T16 sampled synthetic inputs, external-premix v3 function; "
            "no claim of runtime template-v2 equivalence"
        ),
        "environment": environment,
        "runtime_device_selection": runtime_selection,
        "source_inventory_sha256": sha(args.source_inventory),
        "interface_evidence_sha256": sha(args.interface_evidence),
        "cases": [],
        "tolerance_source": context["tolerance"],
        "serial_binary_sha256": sha(serial_binaries[0]),
        "baseline_binary_sha256": sha(api_binary),
    }

    def save():
        (out / "pair-output-check.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for case in selected:
            axes = case["axes"]
            params = {k: axes[k] for k in ("hc_mult", "hc_sinkhorn_iters", "norm_eps", "hc_eps")}
            torch.manual_seed(1024 + axes["T"])
            cpu = model._make_case(axes["T"], hidden_size=axes["H"])
            digest = hashlib.sha256()
            for value in cpu:
                digest.update(memoryview(value.contiguous().view(torch.uint8).numpy()))
            input_shapes = [list(v.shape) for v in cpu]
            input_dtypes = [str(v.dtype).removeprefix("torch.") for v in cpu]
            expected = case_contract(case)
            require(
                input_shapes == expected["input_shapes"] and input_dtypes == expected["input_dtypes"],
                "Actual sampler contract differs",
            )
            require(
                all(v.is_contiguous() and v.storage_offset() == 0 for v in cpu),
                "Sampler must provide contiguous ND inputs",
            )
            reference = model.Model(**params)(*cpu)
            inputs = [value.npu() for value in cpu]
            torch.npu.synchronize()
            own = guard.gate(out, str(axes["T"]) + "-initialized", None)
            assert len(own) == 1
            row = {
                "case_id": case["id"],
                "T": axes["T"],
                "axes": expected["axes"],
                "parameters": params,
                "seed": 1024 + axes["T"],
                "device": context["device"],
                "input_sha256": digest.hexdigest(),
                "implementations": {},
                "input_shapes": input_shapes,
                "input_dtypes": input_dtypes,
            }
            report["cases"].append(row)
            save()
            for name, fn in [
                ("serial", lambda inputs=inputs, params=params: torch.ops.npu.hc_pre(*inputs, **params)),
                ("current-main", adapter.HcPreAclnn(inputs, **params)),
            ]:
                guard.gate(out, str(axes["T"]) + "-" + name + "-before", own)
                eager = fn()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    values = fn()
                graph.replay()
                torch.npu.synchronize()
                binary = serial_binaries[0] if name == "serial" else api_binary
                mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/" in line}
                require(str(binary.resolve()) in mapped, "Expected implementation binary is not mapped")
                checks = []
                for i, (actual, expected) in enumerate(zip(values, reference)):
                    value, ref = actual.cpu().float(), expected.float()
                    diff = (value - ref).abs()
                    magnitude = torch.maximum(value.abs(), ref.abs()).clamp_min(torch.finfo(torch.float32).tiny)
                    threshold = LIMITS["Y_DIFF_THRESHOLD" if i == 0 else "AUX_DIFF_THRESHOLD"]
                    required = LIMITS["Y_REQUIRED_PASS_RATE" if i == 0 else "AUX_REQUIRED_PASS_RATE"]
                    elements = actual.numel()
                    passed = int(((diff <= threshold) | (diff / magnitude <= threshold)).sum())
                    fraction = passed / elements
                    checks.append(
                        {
                            "output_index": i,
                            "shape": list(actual.shape),
                            "dtype": str(actual.dtype).removeprefix("torch."),
                            "finite": bool(torch.isfinite(actual).all()),
                            "max_abs": float(diff.max()),
                            "mean_abs": float(diff.mean()),
                            "threshold": threshold,
                            "required_fraction": required,
                            "observed_fraction": fraction,
                            "elements": elements,
                            "passed_elements": passed,
                            "within_recorded_tolerance": fraction >= required,
                            "graph_matches_eager": bool(torch.equal(actual, eager[i])),
                        }
                    )
                row["implementations"][name] = {
                    "outputs": checks,
                    "input_sha256": digest.hexdigest(),
                    "binary_sha256": sha(binary),
                    "binary_mapped": True,
                    "mapped_binary": {"path": str(binary.resolve()), "sha256": sha(binary)},
                    "eager_structure_matches": len(eager) == 4
                    and [list(v.shape) for v in eager] == case["output_shapes"]
                    and [str(v.dtype).removeprefix("torch.") for v in eager] == case["output_dtypes"],
                    "structure_matches": len(values) == 4
                    and [list(v.shape) for v in values] == case["output_shapes"]
                    and [str(v.dtype).removeprefix("torch.") for v in values] == case["output_dtypes"],
                }
                save()
                guard.gate(out, str(axes["T"]) + "-" + name + "-after", own)
            save()
        # The status is a result of strict validation, never its only evidence.
        report["status"] = "OUTPUT_CHECK_PASS"
        validate_pair_evidence(report, context, expected_cases)
        save()
        return 0
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
