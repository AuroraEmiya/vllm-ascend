"""Collect one pending HC_Pre case and one implementation per process.

Requires a fresh unoccupied-device gate, an existing owned container, and saved
environment/interface evidence. Preparation is outside the profiling interval.
This runner is prepared, not NPU validated. It never promotes pending cases.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from pair_evidence import build_current_context, require, validate_pair_evidence, validate_sample_input


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/s30083169/perf-serial-graph"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--implementation", choices=["serial", "current-main"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--environment-evidence", type=Path, required=True)
    parser.add_argument(
        "--interface-evidence",
        type=Path,
        required=True,
        help="Saved review of the fixed external-pre_mix v3 function; template-v2 label remains unresolved",
    )
    parser.add_argument(
        "--pair-check-evidence",
        type=Path,
        required=True,
        help="Actual T4/T16 eager/graph output checks from check_hc_pair.py",
    )
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--diagnostic-deferred-precision", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    assert out.is_relative_to(root / "outputs"), "Output must stay inside task outputs"
    assert args.diagnostic_deferred_precision, "Precision remains deferred; explicit diagnostic flag required"
    manifest = json.loads(args.manifest.read_text())
    case = next(row for row in manifest["cases"] if row["id"] == args.case)
    assert not case["measured"] and case["axes"]["norm_eps"] == 1e-20
    assert case["axes"]["T"] > 0 and case["axes"]["T"] % case["axes"]["hc_mult"] == 0
    assert case["source_runs"] and all(
        re.fullmatch(r"[a-f0-9]{64}", row["raw_csv_sha256"]) for row in case["source_runs"]
    )
    environment = json.loads(args.environment_evidence.read_text())
    interface = json.loads(args.interface_evidence.read_text())
    assert environment.get("full_soc") and environment.get("npu_arch") and environment.get("cann_version")
    assert interface.get("source_external_pre_mix_contract_verified") is True
    assert interface.get("comparison_contract") == "external_pre_mix_5in_4out_v3_function"
    assert interface.get("norm_eps") == case["axes"]["norm_eps"]
    assert interface.get("hc_eps") == case["axes"]["hc_eps"]
    assert interface.get("hc_sinkhorn_iters") == case["axes"]["hc_sinkhorn_iters"]
    context, expected_cases = build_current_context(
        root, args.manifest, args.environment_evidence, args.interface_evidence, args.source_inventory
    )
    checked = validate_pair_evidence(json.loads(args.pair_check_evidence.read_text()), context, expected_cases)
    inventory = json.loads(args.source_inventory.read_text())
    for rel, expected in inventory["files"].items():
        if rel.startswith("work/hc_pre/"):
            assert sha(root / rel) == expected, "Frozen serial source/binary changed: " + rel
    assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
    assert os.environ.get("PERF_BASELINE_VENDOR")
    assert os.environ.get("HC_PRE_STAGES", "123") in ("", "123"), "Full serial pipeline required"
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    guard = load(Path(__file__).with_name("resource_guard_a2.py"), "hc_expansion_guard")
    physical = int(os.environ["PERF_PHYSICAL_DEVICE"])

    def resources(label, expected):
        pids = guard.gate(out, label, expected)
        snapshot = guard.read_snapshot(out, label)
        used, total = snapshot["HBM_used_MiB"], snapshot["HBM_total_MiB"]
        if not expected:
            assert total - used >= 32 * 1024, "At least 32 GiB HBM headroom required before allocation"
        return pids

    try:
        resources("before-initialization", set())
    except (RuntimeError, AssertionError) as error:
        (out / "resource-blocked.json").write_text(
            json.dumps({"status": "BLOCKED_RESOURCE", "error": str(error)}) + "\n"
        )
        return 75
    # Nothing above allocates an NPU tensor or imports a torch NPU backend.
    import acl
    import torch
    import torch_npu

    require(str(torch.__version__) == environment["torch_version"], "Actual torch version differs")
    require(str(torch_npu.__version__) == environment["torch_npu_version"], "Actual torch_npu version differs")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.environ["PERF_BASELINE_VENDOR"]
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    warm = torch.empty((128, 1024, 1024), dtype=torch.float16, device="npu")
    assert warm.numel() * warm.element_size() == 256 * 1024 * 1024
    torch.argmax(warm)
    torch.npu.synchronize()
    own = resources("after-warmup", None)
    assert len(own) == 1, "Expected exactly one task-owned process"
    axes = case["axes"]
    params = {key: axes[key] for key in ("hc_mult", "hc_sinkhorn_iters", "norm_eps", "hc_eps")}
    base = root / "work/hc_pre"
    inputs_module = load(base / "model.py", "hc_expansion_inputs")
    torch.manual_seed(1024 + axes["T"])
    cpu = inputs_module._make_case(axes["T"], hidden_size=axes["H"])
    digest = hashlib.sha256()
    for value in cpu:
        # Keep a view on the existing storage; avoid concatenating multi-GiB inputs.
        digest.update(memoryview(value.contiguous().view(torch.uint8).numpy()))
    validate_sample_input(
        case,
        [list(v.shape) for v in cpu],
        [str(v.dtype).removeprefix("torch.") for v in cpu],
        digest.hexdigest(),
        checked,
    )
    inputs = [value.npu() for value in cpu]
    del cpu
    if args.implementation == "serial":
        binaries = list(base.glob("kernel/build/hc_pre_ext*.so"))
        assert len(binaries) == 1, "Exactly one frozen serial extension required"
        binary = binaries[0]
        assert sha(binary) == context["files"]["serial_binary"]["sha256"]
        torch.ops.load_library(str(binary))

        def call():
            return torch.ops.npu.hc_pre(*inputs, **params)
    else:
        adapter = load(Path(__file__).with_name("hc_pre_aclnn_expansion.py"), "hc_expansion_baseline")
        binary = adapter.BINARY
        assert sha(binary) == context["files"]["baseline_binary"]["sha256"]
        call = adapter.HcPreAclnn(inputs, **params)
    source_files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix in (".cpp", ".h", ".py")
        and not any(part in ("build", "__pycache__", "bench", "tests") for part in path.parts)
    )
    row = {
        "case_id": case["id"],
        "axes": axes,
        "T": axes["T"],
        "phase_source": "existing official template manifest; runtime request distribution not captured",
        "request_batch": None,
        "input_values_are_model_captured": False,
        "implementation": args.implementation,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "input_sha256": digest.hexdigest(),
        "validation": {"passed": None, "status": "DEFERRED_BY_USER"},
        "input_seed": 1024 + axes["T"],
        "parameters": params,
        "accepted_correctness": False,
        "source_hashes": {str(path): sha(path) for path in source_files},
        "preparation": "sampler, H2D, static weight packing, tiling, warmup and graph capture excluded",
    }
    report = {
        "operator": "hc_pre",
        "protocol": manifest["performance_protocol"],
        "physical_device": physical,
        "logical_device": 0,
        "diagnostic": True,
        "runner_sha256": sha(__file__),
        "resource_guard_sha256": sha(Path(__file__).with_name("resource_guard_a2.py")),
        "case_manifest_sha256": sha(args.manifest),
        "environment": environment,
        "environment_evidence_sha256": sha(args.environment_evidence),
        "preflight_current_context": context,
        "interface_evidence": interface,
        "interface_evidence_sha256": sha(args.interface_evidence),
        "pair_output_preflight": {
            "path": str(args.pair_check_evidence),
            "sha256": sha(args.pair_check_evidence),
            "status": checked["status"],
            "scope": "T4/T16 output checks; precision acceptance remains deferred",
        },
        "started_unix": time.time(),
        "cases": [row],
    }

    def save():
        (out / "execution.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        resources("before-prepare", own)
        _eager = call()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = call()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        row["graph_output_finite"] = all(bool(torch.isfinite(value).all()) for value in output)
        assert len(output) == 4 and row["graph_output_finite"]
        row["output_shapes"] = [list(value.shape) for value in output]
        assert row["output_shapes"] == case["output_shapes"]
        row["output_dtypes"] = [str(value.dtype).removeprefix("torch.") for value in output]
        assert row["output_dtypes"] == case["output_dtypes"]
        if args.implementation != "serial":
            row.update(call.identity())
            row["adapter_sha256"] = sha(Path(__file__).with_name("hc_pre_aclnn_expansion.py"))
            row["baseline_source_commit"] = subprocess.check_output(
                ["git", "-C", str(root / "dependencies/vllm-ascend-current-main"), "rev-parse", "HEAD"], text=True
            ).strip()
            assert all(
                "fla_npu" not in path and "site-packages/vllm_ascend" not in path
                for path in row["mapped_custom_libraries"]
            )
        resources("before-profile", own)
        assert acl.prof.init(str(out)) == 0
        config = acl.prof.create_config([0], 1, 0, 0x0001 | 0x0002 | 0x0004)
        assert config
        roots = {path.name for path in out.glob("PROF_*")}
        try:
            assert acl.prof.start(config) == 0
            try:
                row["profile_start_unix_ns"] = time.time_ns()
                for sample in range(10):
                    resources(f"sample-{sample + 1}-before", own)
                    torch.argmax(warm)
                    torch.npu.synchronize()
                    graph.replay()
                    torch.npu.synchronize()
                    resources(f"sample-{sample + 1}-after", own)
                row["profile_stop_unix_ns"] = time.time_ns()
                row["profiled_groups"] = 10
                row["resource_gate_passed"] = True
            finally:
                assert acl.prof.stop(config) == 0
        finally:
            acl.prof.destroy_config(config)
            acl.prof.finalize()
        row["profile_roots"] = sorted(path.name for path in out.glob("PROF_*") if path.name not in roots)
        save()
        with (out / "export.log").open("w") as log:
            report["export_returncode"] = subprocess.run(
                ["msprof", "--export=on", "--output=" + str(out)], stdout=log, stderr=subprocess.STDOUT
            ).returncode
        report["exported_op_summary_files"] = [str(path.relative_to(out)) for path in out.rglob("op_summary*.csv")]
        assert report["export_returncode"] == 0 and report["exported_op_summary_files"]
        report["finished_unix"] = time.time()
        save()
    except Exception:
        row["error"] = traceback.format_exc()
        row["resource_gate_passed"] = (
            False if "RESOURCE_GATE_FAILED" in row["error"] else row.get("resource_gate_passed")
        )
        save()
        raise
    # Static inputs, outputs, graph, eager outputs and workspace stay alive through export.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
