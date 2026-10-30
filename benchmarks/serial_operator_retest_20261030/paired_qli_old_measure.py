"""Same-frozen-input QLI serial / 20260930 old DSL graph collector.

The frozen DSL is imported only at runtime as a black box. --preflight loads
its public callable and reports inspect.signature, without creating tensors.
The documented old test caller supplies the exact argument mapping below.
"""
from pathlib import Path
from decimal import Decimal
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT_NAME = "qli-old-retest-20261030"
OUTPUT_ROOT = ((ROOT.parents[1] / "outputs") if ROOT.parent.name == "projects" else (ROOT / "outputs")) / OUT_NAME
OLD_DSL_SHA256 = "6222a4732d3b187dc9f6cd3a5b6f868a730772a2f34d7aee587b3f1449e1b308"
PROTOCOL = "10 adjacent 256MiB ArgMax/sync/full graph replay/sync; median complete task sums"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_public_setup(package):
    """Match the package test caller's sibling-helper import context."""
    previous_path = list(sys.path)
    sys.path.insert(0, str(package / "benchmark/full_validation"))
    try:
        return load(package / "benchmark/full_validation/run_full.py", "qli_old_public_test_setup")
    finally:
        sys.path[:] = previous_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def old_call_arguments(data):
    # Verbatim public-call mapping from package profile_b_final.py:25 and
    # benchmark/full_validation/run_full.py:146-153. No implementation read.
    positional = [data[k] for k in ("q", "k", "w", "q_descale", "k_descale")]
    keywords = {k: data[k] for k in ("cu_seqlens_q", "seqused_q", "seqused_k", "block_table",
                                    "max_seqlen_q", "candidate_topk_blocks", "candidate_block_size")}
    keywords.update(topk=512, quant_mode=1, mask_mode=3, cmp_ratio=1,
                    layout_q="TND", layout_kv="PA_BBND", return_value=True)
    return positional, keywords


def resource_snapshot(args, out, label, expected=None):
    physical = args.physical_device
    assert physical >= 0, "Explicit --physical-device is required"
    assert Path(f"/dev/davinci{physical}").exists(), "Selected physical device is not mapped"
    nodes = sorted(int(p.name[len("davinci"):]) for p in Path("/dev").glob("davinci[0-9]*")
                   if p.name[len("davinci"):].isdigit())
    assert nodes == [physical] or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(physical), "Require one mapped device or explicit ASCEND_RT_VISIBLE_DEVICES"
    smi = args.smi_device if args.smi_device is not None else (0 if nodes == [physical] else physical)
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    (out / ("device-" + label + ".txt")).write_text(text, encoding="utf-8")
    assert re.search(r"^\|\s*" + str(smi) + r"\s*\|\s*Ascend950\S*\s*\|\s*OK\s*\|", text, re.M), "Selected A5 health missing"
    pids = set(re.findall(r"^\|\s*" + str(smi) + r"(?:\s+0)?\s*\|\s*(\d+)\s*\|", text, re.M))
    assert pids or f"No running processes found in NPU {smi}" in text, "Selected device process table missing"
    if expected is not None and pids != expected:
        raise RuntimeError("RESOURCE_GATE_FAILED: selected physical NPU" + str(physical) + " process ownership changed: " + str(sorted(pids)))
    return pids


def package_identity(package, row):
    sums = read_json(package / "SHA256SUMS.json")
    wanted = ["sources/baseline_b_dsl.py", "benchmark/inputs.py", "benchmark/case_manifest.json",
              "benchmark/full_validation/run_full.py", "benchmark/full_validation/score_oracle.py",
              "design/qli_cpu_oracle.py", "generation/validation/profile_b_final.py", "provenance.json", "README.md"]
    wanted += [f"generation/validation/full_results/case_{i}.json" for i in range(12)]
    hashes = {name: digest(package / name) for name in wanted}
    assert all(hashes[name] == sums[name] for name in wanted), "Frozen package identity mismatch"
    assert hashes["sources/baseline_b_dsl.py"] == OLD_DSL_SHA256
    manifest = read_json(package / "benchmark/case_manifest.json")
    serial_manifest = ROOT / "src/qli_a5/benchmark/case_manifest.json"
    assert read_json(serial_manifest) == manifest, "Serial manifest differs from frozen q128 historical cases"
    assert len(manifest["cases"]) == 12
    assert all(c["metadata_assumption"]["query_segment"] == 128 for c in manifest["cases"] if c["phase"] == "prefill")
    assert {c["candidate_enabled"] for c in manifest["cases"]} == {False, True}
    row["package_identity"] = {"baseline_sha256": OLD_DSL_SHA256, "files": hashes,
                               "serial_manifest_sha256": digest(serial_manifest), "manifest_status": manifest["status"]}
    return manifest


def runtime_baseline(package, out, row, cann_mode="native92"):
    forbidden = [key for key in os.environ if key.startswith(("CANNBOTDSL_DUMP_", "CANNBOTDSL_PRINT_")) or key in ("CANNBOTDSL_REUSE_ASCENDC", "CANNBOTDSL_PIPE_STAGE")]
    assert not forbidden, "Remove debug or alternate compiler switches: " + str(forbidden)
    versions = {"python": platform.python_version()}
    for package_name in ("torch", "torch_npu", "cannbotdsl", "numpy"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    row["versions"] = versions
    row["historical_dependencies"] = {"python": "3.11.10", "torch": "2.10.0", "torch_npu": "2.10.0.post4", "cannbotdsl": "0.4.dev28", "CANN": "9.3.0"}
    row["dependency_version_matches_history"] = all(versions[k] == v for k, v in row["historical_dependencies"].items() if k != "CANN")
    row["dependency_wheel_identity"] = "Historical wheels were not included; equal version strings do not prove equal wheel bytes"
    cache = out / "dsl-cache"
    cache.mkdir()
    if cann_mode == "compat93":
        row["runtime_setup_stage"] = "historical actual CANN 9.3.0 metadata/ABI check"
        helper = load_public_setup(package)
        row["compat93"] = helper.activate_compat93()
    else:
        row["runtime_setup_stage"] = "user-authorized native CANN 9.2 adaptation"
        os.environ["CANNBOTDSL_NPU_ARCH"] = "dav-3510"
        os.environ["CANNBOTDSL_CACHE_MEM_DISABLE"] = "1"
        os.environ["CANNBOTDSL_CACHE_DISK_DISABLE"] = "1"
        import cannbotdsl.core.compiler.builder as builder
        import cannbotdsl.core.toolchain.ascendc as ascendc
        home = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
        release, sources = ascendc._cann_release_version(str(home))
        row["cann_adaptation"] = {"requested": "native92", "actual_release": release,
            "version_sources": list(sources), "home": str(home),
            "historical_93_shim_applied": False, "frozen_baseline_modified": False,
            "authorization": "User explicitly requested forcing the retest to CANN 9.2"}
        assert release == "9.2.0", "Native92 mode requires actual CANN 9.2.0"
        assert builder.resolve_l0c2l1_abi is ascendc.resolve_l0c2l1_abi, "Unexpected ABI resolver override"
        row["cann_adaptation"]["native_abi"] = str(ascendc.resolve_l0c2l1_abi(str(home), required=True))
    os.environ["CANNBOTDSL_CACHE_DIR"] = str(cache)
    row["dsl_environment"] = {key: value for key, value in os.environ.items() if key.startswith("CANNBOTDSL_")}
    row["runtime_setup_stage"] = "black-box frozen DSL import/signature bind"
    baseline = load(package / "sources/baseline_b_dsl.py", "qli_old_frozen_blackbox")
    fn = baseline.quant_lightning_indexer
    signature = inspect.signature(fn)
    dummy = {name: object() for name in ("q", "k", "w", "q_descale", "k_descale", "cu_seqlens_q", "seqused_q", "seqused_k", "block_table", "max_seqlen_q", "candidate_topk_blocks", "candidate_block_size")}
    args, kwargs = old_call_arguments(dummy)
    signature.bind(*args, **kwargs)
    row["baseline_runtime_signature"] = str(signature)
    row["baseline_argument_binding"] = "PASS: exact preserved public test-call mapping"
    return fn


def summarize(out, row):
    files = list(out.rglob("op_summary*.csv"))
    assert len(files) == 1, "Missing or ambiguous complete raw op_summary"
    raw = files[0]
    with raw.open(encoding="utf-8-sig", newline="") as handle:
        tasks = list(csv.DictReader(handle))
    tasks.sort(key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    markers = [i for i, task in enumerate(tasks) if "argmax" in task["Op Name"].lower()]
    assert len(markers) == 10 and markers[0] == 0, "Expected 10 warmups with no unassigned leading task"
    samples, topology, graph_stream = [], None, None
    for sample, (begin, end) in enumerate(zip(markers, markers[1:] + [len(tasks)]), 1):
        group = tasks[begin + 1:end]
        names = [t["Op Name"] for t in group]
        streams = {t["Stream ID"] for t in group}
        assert group and len(streams) == 1, "Incomplete graph or multiple target streams"
        if topology is None:
            topology, graph_stream = names, streams
        assert names == topology and streams == graph_stream, "Graph task topology changed"
        for previous, current in zip(group, group[1:]):
            assert Decimal(previous["Task Start Time(us)"].strip()) + Decimal(previous["Task Duration(us)"]) <= Decimal(current["Task Start Time(us)"].strip()), "Dependent graph tasks overlap"
        durations = [float(t["Task Duration(us)"]) for t in group]
        assert all(duration > 0 for duration in durations)
        cube = None
        if row["implementation"] == "serial":
            expected = (row["axes"]["T"] + 1023) // 1024
            assert len(group) == 2 * expected and sum(name.startswith("qli_c_") for name in names) == expected, "Incomplete serial C/V graph"
            assert all(names[i].startswith("qli_c_") and not names[i + 1].startswith("qli_c_") for i in range(0, len(names), 2)), "Serial C/V order changed"
            cube = sum(float(t["Task Duration(us)"]) for t in group if t["Op Name"].startswith("qli_c_"))
        total = sum(durations)
        samples.append({"sample": sample, "total_us": total, "C_us": cube, "C_fraction": None if cube is None else cube / total,
                        "task_names": names, "task_ids": [t["Task ID"] for t in group], "task_durations_us": durations})
    med = statistics.median(sample["total_us"] for sample in samples)
    spread = (max(s["total_us"] for s in samples) - min(s["total_us"] for s in samples)) / med
    result = {"total_ms": med / 1000, "sample_count": 10, "samples": samples, "kernel_sequence": topology,
              "C_fraction": statistics.median(s["C_fraction"] for s in samples) if row["implementation"] == "serial" else None,
              "spread_over_median": spread, "stability_passed": spread <= .10, "frequency_gate": "UNVERIFIED",
              "timing_scope": "complete graph device task duration sum", "op_summary": str(raw), "op_summary_sha256": digest(raw),
              "all_raw_device_tasks_accounted_for": len(tasks) == 10 + sum(len(s["task_names"]) for s in samples)}
    assert result["all_raw_device_tasks_accounted_for"]
    write_json(out / "summary.json", result)
    row["measurement_summary"] = result
    row["op_summary_files"] = [str(raw.relative_to(out))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--implementation", choices=("serial", "old-dsl"), default="old-dsl")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--cann-mode", choices=("native92", "compat93"), default="native92",
                        help="Use actual CANN 9.2 native ABI as requested, or preserved historical 9.3 setup")
    parser.add_argument("--out", type=Path, required=True)
    preflight_group = parser.add_mutually_exclusive_group()
    preflight_group.add_argument("--preflight", action="store_true", help="Guard an idle mapped device, then import/signature only")
    preflight_group.add_argument("--cpu-preflight", action="store_true", help="Require no mapped NPU compute nodes; environment/ABI/import/signature only")
    parser.add_argument("--physical-device", type=int, default=int(os.environ.get("PERF_PHYSICAL_DEVICE", "-1")))
    parser.add_argument("--smi-device", type=int)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index inside the isolated runtime")
    parser.add_argument("--cpu-input-limit-bytes", type=int, default=40 << 30)
    args = parser.parse_args()
    signature_only = args.preflight or args.cpu_preflight
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT_ROOT.resolve()), "Output must stay under " + str(OUTPUT_ROOT)
    assert not out.exists(), "Use a fresh output directory; never overwrite prior evidence"
    out.mkdir(parents=True)
    package = args.baseline_root.resolve()
    row = {"operator": "qli_a5", "case_id": args.case, "implementation": args.implementation,
           "physical_device": args.physical_device, "logical_device": args.device, "host": platform.node(),
           "mode": "cpu-preflight" if args.cpu_preflight else "preflight" if args.preflight else "graph", "protocol": PROTOCOL,
           "runner_sha256": digest(__file__), "resource_gate_passed": False, "completed": False,
           "correctness_status": "NOT_REVALIDATED_IN_PERFORMANCE_RUN", "samples": [], "started_unix": time.time(),
           "baseline_contract": "Frozen old DSL SHA6222a473; MXFP4,N1=32,D128,TopK512,mask3,ratio1,TND/PA_BBND,return_value=True; full matching candidate outputs; historical q128 prefill metadata"}
    profiling, config = False, None
    try:
        assert shutil.disk_usage(out).free >= 2 << 30, "Less than 2GiB output space"
        manifest = package_identity(package, row)
        assert 0 <= args.case < len(manifest["cases"])
        cpu_only_preflight = args.cpu_preflight
        row["cpu_only_preflight"] = cpu_only_preflight
        if cpu_only_preflight:
            assert not any(p.name[len("davinci"):].isdigit() for p in Path("/dev").glob("davinci[0-9]*")), "--cpu-preflight requires no NPU compute-device nodes"
            row["resource_check_scope"] = "No NPU compute-device nodes are mapped; CPU import/metadata/signature only"
            row["resource_gate_status"] = "NOT_APPLICABLE_NO_MAPPED_COMPUTE_DEVICE"
        else:
            resource_snapshot(args, out, "initial", set())
        fn = runtime_baseline(package, out, row, args.cann_mode)
        if signature_only:
            if not cpu_only_preflight:
                resource_snapshot(args, out, "after-import", set())
            row.update(completed=True, resource_gate_passed=not cpu_only_preflight, preflight_only=True, tensor_workload_executed=False)
            return 0
        import torch
        import torch_npu
        import acl
        row["torch_runtime_version"] = torch.__version__
        row["torch_npu_runtime_version"] = getattr(torch_npu, "__version__", "unknown")
        torch.npu.set_device(args.device)
        torch.set_num_threads(8)
        torch.npu.matmul.allow_hf32 = False
        torch.set_float32_matmul_precision("highest")
        warm = torch.empty((128, 1024, 1024), dtype=torch.float16, device=f"npu:{args.device}")
        torch.argmax(warm)
        torch.npu.synchronize()
        own = resource_snapshot(args, out, "initialized")
        assert len(own) == 1, "Expected exactly one newly task-owned NPU process"
        builder = load(package / "benchmark/inputs.py", "qli_old_frozen_input_builder")
        case = builder.load_cases()[args.case]
        row["axes"] = dict(case["axes"], candidate=case["candidate_enabled"], seqused_k=case["metadata_assumption"]["seqused_k"])
        os.environ["QLI_CPU_INPUT_LIMIT_BYTES"] = str(args.cpu_input_limit_bytes)
        cpu = builder.make_inputs(case["axes"], "cpu")
        input_hashes = {name: hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest() for name, value in cpu.items() if isinstance(value, torch.Tensor)}
        frozen = read_json(package / f"generation/validation/full_results/case_{args.case}.json")
        assert input_hashes == frozen["input_sha256"], "Input bytes differ from frozen historical q128 case"
        row["input_sha256"] = input_hashes
        row["input_identity"] = {"case": case, "input_sha256": input_hashes, "scalars": {k: v for k, v in cpu.items() if not isinstance(v, torch.Tensor)},
                                 "builder_sha256": digest(package / "benchmark/inputs.py")}
        row["input_identity_sha256"] = hashlib.sha256(json.dumps(row["input_identity"], sort_keys=True).encode()).hexdigest()
        data = {name: value.to(f"npu:{args.device}") if isinstance(value, torch.Tensor) else value for name, value in cpu.items()}
        row["input_hash_scope"] = "Full CPU tensor bytes before H2D; exact match against frozen historical records for both implementations"
        del cpu
        if args.implementation == "serial":
            sys.path.insert(0, str(ROOT / "src/qli_a5"))
            from ascendc_quant_lightning_indexer import prepare_quant_lightning_indexer
            prepared = prepare_quant_lightning_indexer(*[data[k] for k in ("q", "k", "w", "q_descale", "k_descale")],
                **{k: data[k] for k in ("cu_seqlens_q", "seqused_q", "seqused_k", "block_table", "candidate_topk_blocks", "candidate_block_size")})
            call = prepared.run
            row["serial_public_wrapper_sha256"] = digest(ROOT / "src/qli_a5/ascendc_quant_lightning_indexer/api.py")
        else:
            positional, keywords = old_call_arguments(data)
            inspect.signature(fn).bind(*positional, **keywords)
            def call():
                return fn(*positional, **keywords)
        resource_snapshot(args, out, "before-warmup", own)
        start = time.perf_counter()
        output = call()
        torch.npu.synchronize()
        row["first_public_call_wall_ms"] = (time.perf_counter() - start) * 1000
        row["first_call_scope"] = "Public call plus synchronize; may include JIT, not pure compile timing"
        for _ in range(10):
            output = call()
            torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = call()
        graph.replay()
        torch.npu.synchronize()
        row["output_shapes"] = [list(t.shape) for t in output]
        row["output_dtypes"] = [str(t.dtype) for t in output]
        assert row["output_shapes"] == [spec["shape"] for spec in case["expected_outputs"].values()], "Output geometry/candidate contract mismatch"
        row["binaries"] = {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "src/qli_a5").rglob("*.so")}
        row["dsl_cache_files_after_warmup"] = {str(p.relative_to(out)): digest(p) for p in (out / "dsl-cache").rglob("*") if p.is_file()}
        row["mapped_custom_libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcust_op" in line})
        assert acl.prof.init(str(out)) == 0
        profiling = True
        config = acl.prof.create_config([args.device], 1, 0, 0x0001 | 0x0002 | 0x0004)
        assert config
        resource_snapshot(args, out, "before-profile", own)
        assert acl.prof.start(config) == 0
        try:
            for sample in range(10):
                resource_snapshot(args, out, f"before-sample-{sample + 1:02d}", own)
                torch.argmax(warm)
                torch.npu.synchronize()
                begin = time.time_ns()
                graph.replay()
                torch.npu.synchronize()
                row["samples"].append({"sample": sample + 1, "start_unix_ns": begin, "stop_unix_ns": time.time_ns()})
        finally:
            assert acl.prof.stop(config) == 0
        resource_snapshot(args, out, "after-profile", own)
        row["resource_gate_passed"] = True
        acl.prof.destroy_config(config)
        config = None
        acl.prof.finalize()
        profiling = False
        with (out / "export.log").open("w") as log:
            row["export_returncode"] = subprocess.run(["msprof", "--export=on", "--output=" + str(out)], stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
        assert row["export_returncode"] == 0
        summarize(out, row)
        assert digest(package / "sources/baseline_b_dsl.py") == OLD_DSL_SHA256
        row["completed"] = True
    except Exception:
        row["error"] = traceback.format_exc()
    finally:
        if config is not None:
            acl.prof.destroy_config(config)
        if profiling:
            acl.prof.finalize()
        row["finished_unix"] = time.time()
        write_json(out / "execution.json", row)
        print(json.dumps({key: value for key, value in row.items() if key not in ("input_identity", "binaries", "samples", "measurement_summary")}, indent=2), flush=True)
    return 0 if row["completed"] else (75 if "RESOURCE_GATE_FAILED" in row.get("error", "") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
