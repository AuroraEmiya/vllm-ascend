"""Black-box TGR pairing. Local preparation is not deployment or acceptance.

No mathematical reference, fused implementation, torch.compile, graph capture or
replay is used. Runtime imports are intentionally deferred past CPU guards.
"""

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "contract.json").read_text(encoding="utf-8"))
OUTPUTS = tuple(x["name"] for x in CONTRACT["return_positions"])
PHASES = tuple(CONTRACT["precision"]["phases"])
DIGEST_SCOPE = "all_inputs_weights_initial_state_scalars"
SCALAR_KEYS = ("channels", "eps", "gate", "gamma", "need_backward")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize(case):
    if case.get("operator_id") != CONTRACT["operator_id"] or not case.get("case_id"):
        raise ValueError("explicit TGR operator_id and case_id are required")
    shape = dict(case.get("shape", {}))
    for key in ("B", "S"):
        if type(shape.get(key)) is not int or shape[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if shape["B"] > 512 or shape["S"] > 32768 or shape["B"] * shape["S"] > 2048:
        raise ValueError("candidate/operational domain: B<=512, S<=32768, B*S<=2048")
    for key, expected in (("C", 4), ("H", 2560), ("R", 320)):
        if shape.get(key) != expected or type(shape.get(key)) is not int:
            raise ValueError("fixed candidate dimension " + key)
    if shape.get("dtype") != "BF16" or case.get("dtype") != "BF16" or shape.get("layout") != "ND":
        raise ValueError("only BF16 ND source inputs are supported")
    expected_scene = "decode" if shape["S"] == 1 else "prefill"
    if case.get("scenario") != expected_scene:
        raise ValueError("explicit scenario must match TGR per-request sequence semantics")
    scalar = dict(case.get("scalars", {}))
    if set(scalar) != set(SCALAR_KEYS):
        raise ValueError("all scalar/optional choices must be explicit; no implicit defaults")
    if type(scalar["channels"]) is not int or scalar["channels"] != 4:
        raise ValueError("channels must be integer 4")
    for key in ("gate", "gamma", "need_backward"):
        if type(scalar[key]) is not bool:
            raise ValueError(key + " must be bool")
    if type(scalar["eps"]) not in (int, float) or not math.isfinite(scalar["eps"]) or scalar["eps"] <= 0:
        raise ValueError("eps must be finite and positive")
    if "eager" not in case.get("allowed_execution_modes", []):
        raise ValueError("this adapter only implements explicit eager execution")
    return {**shape, **scalar}


def output_spec(case):
    value = normalize(case)
    b, s = value["B"], value["S"]
    widths = (2560, 4, 4, 320, 10240)
    enabled = (True, value["gate"], value["need_backward"], value["need_backward"], value["need_backward"])
    return [{"name": name, "shape": [b, s, width] if active else [0],
             "dtype": "FP32" if name == "inv_rms" else "BF16", "enabled": active}
            for name, width, active in zip(OUTPUTS, widths, enabled)]


def plan(case):
    v = normalize(case)
    # A lower bound for one logical input set, not workspace or peak HBM.
    elements = v["B"] * v["S"] * 10240 + 2 * 320 * 10240 + (4 * 10240 if v["gate"] else 0)
    return {"case_id": case["case_id"], "status": "PLANNED_NOT_RUNTIME_READY", "scenario": case["scenario"],
            "candidate_route": CONTRACT["candidate_route"], "execution_mode": "eager", "outputs": output_spec(case),
            "one_input_set_bytes_lower_bound": elements * 2 + (10240 * 4 if v["gamma"] else 0),
            "estimated_peak_bytes": None, "baseline_domain_verified": False, "accepted": False,
            "pending": ["fixed deployment and actual library identities", "A3/A5 baseline compatibility",
                        "same-input complete five-return precision", "peak/workspace and fresh idle gate",
                        "actual per-case warmup/target kernel topology"]}


def validate_metric(metric, dtype, enabled=True):
    required = ("shape_matches", "dtype_matches", "all_finite", "inputs_disjoint")
    if any(metric.get(key) is not True for key in required):
        return False
    if not enabled:
        return metric.get("disabled_empty") is True
    threshold = CONTRACT["precision"][dtype]
    for key in ("matched_ratio", "max_abs_error"):
        if type(metric.get(key)) not in (int, float) or not math.isfinite(metric[key]):
            return False
    return (0 <= metric["matched_ratio"] <= 1 and metric["matched_ratio"] >= threshold["matched_ratio_min"]
            and 0 <= metric["max_abs_error"] <= threshold["max_abs_error_limit"])


def precision_rules(case):
    result = {name: [] for name in (*OUTPUTS, "inputs_immutable")}
    for phase in PHASES:
        for spec in output_spec(case):
            rules = result[spec["name"]]
            for key in ("shape_matches", "dtype_matches", "all_finite", "inputs_disjoint"):
                rules.append({"path": [phase, key], "comparison": "==", "threshold": True})
            if spec["enabled"]:
                tol = CONTRACT["precision"][spec["dtype"]]
                rules.extend([{"path": [phase, "matched_ratio"], "comparison": ">=", "threshold": tol["matched_ratio_min"]},
                              {"path": [phase, "max_abs_error"], "comparison": "<=", "threshold": tol["max_abs_error_limit"]}])
            else:
                rules.append({"path": [phase, "disabled_empty"], "comparison": "==", "threshold": True})
        result["inputs_immutable"].append({"path": [phase, "exact_equal"], "comparison": "==", "threshold": True})
    return result


def checked_evidence(value, label):
    if not isinstance(value, dict) or not value.get("path") or sha_file(value["path"]) != value.get("sha256"):
        raise RuntimeError(label + " evidence missing or changed")
    return value


def input_contract(case, seed, calibration=False):
    value = {"case": case, "seed": seed, "contract_sha256": digest(CONTRACT)}
    return {**value, "input_contract_sha256": digest(value), "samples": 10,
            "scope": "CALIBRATION_ONLY_NOT_FROZEN_WORKER_INPUT" if calibration else "FROZEN_WORKER_INPUT"}


def source_identity(config):
    root = Path(config["source_root"]).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=20).strip()
    commit, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    if commit != CONTRACT["candidate_public_commit"] or tree != CONTRACT["candidate_tree"]:
        raise RuntimeError("fixed candidate GitHub commit/tree mismatch")
    expected = {str((root / CONTRACT["candidate_module"]).resolve()), str((root / "vllm_ascend/ops/serial_matmul.py").resolve())}
    checked = []
    for item in config.get("source_artifacts", []):
        p = Path(item["path"]).resolve()
        if not p.is_relative_to(root) or sha_file(p) != item["sha256"]:
            raise RuntimeError("candidate source artifact path/hash mismatch")
        checked.append({"path": str(p), "sha256": item["sha256"]})
    if not expected.issubset({x["path"] for x in checked}):
        raise RuntimeError("both TGR and shared V3 serial wrappers must be frozen")
    return {"root": str(root), "commit": commit, "tree": tree, "artifacts": checked,
            "module": str(root / CONTRACT["candidate_module"])}


def mapped_artifacts(config, route):
    requested = config.get("required_loaded_artifacts", {}).get(route, [])
    if not requested:
        raise RuntimeError("root must freeze actual runtime library identities for " + route)
    paths = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
             if line.split()[-1].startswith("/")}
    result = []
    for item in requested:
        path = str(Path(item["path"]).resolve())
        if path not in paths or sha_file(path) != item["sha256"]:
            raise RuntimeError("actually mapped binary mismatch: " + path)
        result.append({"path": path, "sha256": item["sha256"]})
    return result


def make_inputs(torch, case, seed):
    v = normalize(case)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = {}
    for name, shape, scale, dtype in (
        ("x", (v["B"], v["S"], 10240), 1.0, torch.bfloat16),
        ("w_down", (320, 10240), .01, torch.bfloat16),
        ("w_up", (10240, 320), .01, torch.bfloat16),
        ("w_gate", (4, 10240), .01, torch.bfloat16),
        ("gamma", (10240,), .1, torch.float32),
    ):
        if (name == "w_gate" and not v["gate"]) or (name == "gamma" and not v["gamma"]):
            continue
        result[name] = (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(dtype).contiguous()
    return result


def inputs_digest(torch, tensors, case):
    h = hashlib.sha256(canonical({"shape": case["shape"], "scalars": case["scalars"]}))
    for key in sorted(tensors):
        tensor = tensors[key].detach().cpu().contiguous()
        h.update(canonical({"name": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}))
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def baseline_call(torch, tensors, scalar):
    return torch.ops.cann_ops_transformer.text_gated_residual(
        tensors["x"], tensors["w_down"], tensors["w_up"], w_gate=tensors.get("w_gate"),
        channels=scalar["channels"], gamma=tensors.get("gamma"), eps=scalar["eps"],
        need_backward=scalar["need_backward"])


def candidate_settings(owner):
    for key, value in CONTRACT["candidate_settings"].items():
        setattr(owner, key, value)
    return owner


class PairedSession:
    def __init__(self, torch, torch_npu, config, case):
        self.torch, self.torch_npu, self.config, self.case = torch, torch_npu, config, case
        self.scalar = normalize(case)
        self.source = source_identity(config)
        self.handles = []
        # Both routes preload the same explicitly selected libraries before any
        # target. There is no silent discovery or fallback to a global vendor.
        for item in config.get("preload_libraries", []):
            if sha_file(item["path"]) != item["sha256"]:
                raise RuntimeError("preload library hash mismatch")
            self.handles.append(ctypes.CDLL(str(Path(item["path"]).resolve()), mode=ctypes.RTLD_GLOBAL))
        if not config.get("candidate_registration_libraries"):
            raise RuntimeError("candidate torch registration library not frozen")
        for item in config["candidate_registration_libraries"]:
            if sha_file(item["path"]) != item["sha256"]:
                raise RuntimeError("candidate registration binary hash mismatch")
            torch.ops.load_library(str(Path(item["path"]).resolve()))
        baseline_module = importlib.import_module(CONTRACT["baseline_module"])
        baseline_module.text_gated_residual_op_builder.load()  # registration, not a target pre-call
        self.actual_baseline_schema = str(torch.ops.cann_ops_transformer.text_gated_residual.default._schema)
        schema = json.loads(Path(config["baseline_schema_evidence"]["path"]).read_text(encoding="utf-8"))
        if schema.get("schema") != self.actual_baseline_schema:
            raise RuntimeError("installed public schema differs from root-frozen schema evidence")
        spec = importlib.util.spec_from_file_location("scan_tgr_vector_r5", self.source["module"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.host = make_inputs(torch, case, config["seed"])
        self.initial_digest = inputs_digest(torch, self.host, case)
        device = "npu:" + str(config["logical_device"])
        self.d = {route: {key: value.to(device) for key, value in self.host.items()} for route in ("baseline", "candidate")}
        d = self.d["candidate"]
        self.owner = candidate_settings(module.TextGatedResidualSerial(d["w_down"], d["w_up"], d.get("w_gate"),
                                                                     self.scalar["channels"], self.scalar["eps"]))
        torch.npu.synchronize()
        for route in self.d:
            if inputs_digest(torch, self.d[route], case) != self.initial_digest:
                raise RuntimeError("actual prepared device input differs from deterministic host input")
            mapped_artifacts(config, route)

    def call(self, route):
        d = self.d[route]
        if route == "baseline":
            return baseline_call(self.torch, d, self.scalar)
        if route != "candidate":
            raise ValueError("unknown route")
        return self.owner(d["x"], d.get("gamma"), self.scalar["need_backward"])

    def change_values(self):
        self.host = make_inputs(self.torch, self.case, self.config["seed"] + 1)
        for route in self.d:
            for key, value in self.host.items():
                self.d[route][key].copy_(value)
        self.owner.refresh_weights_()
        self.torch.npu.synchronize()

    def snapshot(self, route, outputs):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
            raise RuntimeError("public TGR low-level ABI must return exactly five positions")
        storages = {x.untyped_storage().data_ptr() for x in self.d[route].values()}
        if not all(isinstance(x, self.torch.Tensor) for x in outputs):
            raise RuntimeError("all five positions must be typed tensors, including disabled outputs")
        return {"outputs": [x.detach().cpu().clone() for x in outputs],
                "inputs_disjoint": [x.numel() == 0 or x.untyped_storage().data_ptr() not in storages for x in outputs],
                "input_digest_after_call": inputs_digest(self.torch, self.d[route], self.case)}

    def receipt(self, route, contract):
        result = {"case_id": self.case["case_id"], "platform": self.config["platform"], "route": route,
                  "device_id": str(self.config["device_id"]), "runtime_device_id": str(self.config["logical_device"]),
                  "contract_sha256": digest(CONTRACT), "input_contract_sha256": contract["input_contract_sha256"],
                  "inputs_sha256": self.initial_digest, "input_digest_scope": DIGEST_SCOPE,
                  "source_commit": CONTRACT["candidate_public_commit"] if route == "candidate" else None,
                  "loaded_artifacts": mapped_artifacts(self.config, route), "candidate_checkout_identity": self.source,
                  "baseline_schema": self.actual_baseline_schema, "candidate_route": CONTRACT["candidate_route"],
                  "protocol": "msprof_application", "scenario": self.case["scenario"], "execution_mode": "eager",
                  "timing_scope": "msprof_application_device_task_duration_sum",
                  "calibration_only": self.config.get("calibration_only") is True,
                  "launch_mapping_source": "PENDING_PROFILER_EXIT", "ordered_launches": []}
        if route == "baseline":
            result["source_identity"] = self.config["baseline_source_identity"]
        return result


def output_checks(torch, case, snapshots, expected_input_digest):
    result = []
    for index, spec in enumerate(output_spec(case)):
        ref = snapshots["baseline"]["outputs"][index]
        actual = snapshots["candidate"]["outputs"][index]
        dtype = torch.float32 if spec["dtype"] == "FP32" else torch.bfloat16
        metric = {"shape_matches": list(ref.shape) == list(actual.shape) == spec["shape"],
                  "dtype_matches": ref.dtype == actual.dtype == dtype,
                  "all_finite": bool(torch.isfinite(ref).all() and torch.isfinite(actual).all()),
                  "inputs_disjoint": all(snapshots[r]["inputs_disjoint"][index] for r in ("baseline", "candidate"))}
        if not spec["enabled"]:
            metric["disabled_empty"] = ref.numel() == actual.numel() == 0
        elif metric["shape_matches"] and metric["dtype_matches"] and metric["all_finite"]:
            tol = CONTRACT["precision"][spec["dtype"]]
            reference, value = ref.double(), actual.double()
            error = (value - reference).abs()
            metric.update(matched_ratio=float((error <= tol["atol"] + tol["rtol"] * reference.abs()).double().mean()),
                          max_abs_error=float(error.max()))
        else:
            metric.update(matched_ratio=0.0, max_abs_error=None)
        result.append({"output": spec["name"], "passed": validate_metric(metric, spec["dtype"], spec["enabled"]), "metric": metric})
    immutable = all(s["input_digest_after_call"] == expected_input_digest for s in snapshots.values())
    result.append({"output": "inputs_immutable", "passed": immutable, "metric": {"exact_equal": immutable}})
    return result


def precision(session, contract, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    phases, artifacts, route_receipts = [], [], None
    for phase in PHASES:
        if phase == "changed_values_same_eager":
            session.change_values()
        expected = inputs_digest(session.torch, session.host, session.case)
        snapshots = {}
        for route in ("baseline", "candidate"):
            if inputs_digest(session.torch, session.d[route], session.case) != expected:
                raise RuntimeError("input mutated or did not refresh before paired call")
            value = session.call(route)
            session.torch.npu.synchronize()
            snapshots[route] = session.snapshot(route, value)
        path = out / ("outputs-" + phase + ".pt")
        session.torch.save({"scope": "raw black-box paired outputs and exact generated inputs; no reference equations",
                            "inputs": session.host, "inputs_sha256": expected, "snapshots": snapshots}, path)
        artifacts.append({"path": str(path.resolve()), "sha256": sha_file(path)})
        phases.append({"phase": phase, "inputs_sha256": expected,
                       "checks": output_checks(session.torch, session.case, snapshots, expected)})
        if phase == "repeated_eager_call":
            route_receipts = {route: session.receipt(route, contract) for route in ("baseline", "candidate")}
    rules = precision_rules(session.case)
    checks = [{"output": name, "passed": all(p["checks"][i]["passed"] for p in phases),
               "metric": {p["phase"]: p["checks"][i]["metric"] for p in phases}, "rules": rules[name]}
              for i, name in enumerate((*OUTPUTS, "inputs_immutable"))]
    result = {"case_id": session.case["case_id"], "platform": session.config["platform"],
              "status": "PASS" if all(x["passed"] for x in checks) else "FAIL", "checks": checks, "phases": phases,
              "contract_sha256": digest(CONTRACT), "input_contract_sha256": contract["input_contract_sha256"],
              "inputs_sha256": session.initial_digest, "input_digest_scope": DIGEST_SCOPE,
              "output_equivalence_checked": True, "oracle": CONTRACT["precision"]["oracle"],
              "independent_mathematical_accuracy_verified": False, "raw_artifacts": artifacts,
              "route_receipts": route_receipts, "scenario": session.case["scenario"], "execution_mode": "eager",
              "graph_capture_or_replay_used": False, "calibration_only": session.config.get("calibration_only") is True,
              "environment": {"torch": session.torch.__version__, "torch_npu": session.torch_npu.__version__,
                              "full_soc": session.torch.npu.get_device_name(session.config["logical_device"])},
              "evidence": str((out / "precision.json").resolve()), "accepted": False}
    save(out / "precision.json", result)
    return result


def verify_paired_precision(value, session, contract, route):
    if (value.get("status") != "PASS" or value.get("case_id") != session.case["case_id"]
            or value.get("platform") != session.config["platform"]
            or value.get("contract_sha256") != digest(CONTRACT)
            or value.get("input_contract_sha256") != contract["input_contract_sha256"]
            or value.get("inputs_sha256") != session.initial_digest
            or value.get("execution_mode") != "eager" or value.get("scenario") != session.case["scenario"]
            or value.get("calibration_only") is not (session.config.get("calibration_only") is True)):
        raise RuntimeError("paired precision evidence missing, failed or mismatched")
    checks = {x["output"]: x for x in value.get("checks", [])}
    expected = set((*OUTPUTS, "inputs_immutable"))
    if len(value.get("checks", [])) != len(expected) or set(checks) != expected:
        raise RuntimeError("incomplete or duplicate precision outputs")
    for name, rules in precision_rules(session.case).items():
        for rule in rules:
            observed = checks[name]["metric"]
            for key in rule["path"]:
                observed = observed[key]
            threshold = rule["threshold"]
            if type(threshold) is bool:
                ok = type(observed) is bool and observed == threshold
            else:
                ok = type(observed) in (int, float) and math.isfinite(observed)
                if ok:
                    ok = observed >= threshold if rule["comparison"] == ">=" else observed <= threshold
            if not ok:
                raise RuntimeError("observed precision metric violates actual rules")
    if len(value.get("raw_artifacts", [])) != len(PHASES):
        raise RuntimeError("raw three-phase output/input evidence is missing")
    for artifact in value["raw_artifacts"]:
        checked_evidence(artifact, "raw precision")
    previous, actual = value["route_receipts"][route], session.receipt(route, contract)
    for key in ("source_commit", "source_identity", "candidate_checkout_identity", "loaded_artifacts", "baseline_schema",
                "candidate_route", "device_id", "runtime_device_id", "contract_sha256", "inputs_sha256"):
        if previous.get(key) != actual.get(key):
            raise RuntimeError("precision/profile actual runtime identity mismatch: " + key)


def profile_ready(config, case, route, samples, calibration=False):
    normalize(case)
    if samples != 10 or config.get("execution_mode") != "eager":
        raise RuntimeError("exactly ten samples and matched eager routes are required")
    if route not in ("baseline", "candidate"):
        raise RuntimeError("explicit route required")
    if calibration:
        return []
    protocol = config.get("profile_protocol", {})
    if protocol.get("status") != "VERIFIED" or protocol.get("protocol") != "msprof_application":
        raise RuntimeError("standard application protocol not verified")
    checked_evidence(protocol.get("evidence"), "profile protocol")
    sequence = config.get("profiler_kernel_contract", {}).get(route, [])
    if not sequence or not all(isinstance(x, dict) and x.get("name") for x in sequence):
        raise RuntimeError("actual complete per-case target topology not frozen")
    warmup = config.get("application_warmup", {})
    checked_evidence(warmup.get("probe_evidence"), "warmup probe")
    if not warmup.get("kernel_sequence"):
        raise RuntimeError("actual complete warmup topology not frozen")
    return sequence


def profile(session, contract, route, out, precision_evidence, samples=10):
    calibration = session.config.get("calibration_only") is True
    sequence = profile_ready(session.config, session.case, route, samples, calibration)
    precision_evidence = Path(precision_evidence)
    checked = json.loads(precision_evidence.read_text(encoding="utf-8"))
    verify_paired_precision(checked, session, contract, route)
    before = session.receipt(route, contract)
    cfg = session.config.get("application_warmup", {})
    if cfg.get("mode") != "framework_ops" or cfg.get("logical_op") != "ArgMax" or cfg.get("tensor_bytes") != 268435456:
        raise RuntimeError("this adapter implements only explicit 256MiB FP16 framework ArgMax warmup")
    warm = session.torch.empty((128, 1024, 1024), dtype=session.torch.float16,
                               device="npu:" + str(session.config["logical_device"]))
    session.torch.npu.synchronize()
    output = None
    for _ in range(10):
        session.torch.argmax(warm)
        session.torch.npu.synchronize()
        output = session.call(route)
        session.torch.npu.synchronize()
    if not isinstance(output, (tuple, list)) or len(output) != 5:
        raise RuntimeError("profile returned unexpected five-position TGR ABI")
    receipt = session.receipt(route, contract)
    if receipt["loaded_artifacts"] != before["loaded_artifacts"]:
        raise RuntimeError("runtime mapped binary identity changed across profile")
    if inputs_digest(session.torch, session.d[route], session.case) != session.initial_digest:
        raise RuntimeError("profile mutated an input/weight")
    receipt.update(samples_per_case=10, pair_order="warmup_then_target", pid=os.getpid(), target_precalled=False,
                   graph_capture_or_replay_used=False, accepted=False,
                   precision_evidence={"path": str(precision_evidence.resolve()), "sha256": sha_file(precision_evidence)})
    out = Path(out)
    if calibration:
        receipt.update(formal_performance_generated=False, comparison_admitted=False, speedup=None,
                       exact_kernel_names="DISCOVERY_PENDING_RAW_OP_SUMMARY")
        save(out / "calibration-runtime-receipt.json", receipt)
        save(out / "calibration-execution-contract.json", {"case": session.case, "route": route, "samples": 10,
             "scope": "CALIBRATION_ONLY_NOT_A_FORMAL_PERFORMANCE_CASE", "pair_order": "ArgMax_256MiB_fp16_sync_then_complete_TGR_target_sync",
             "execution_mode": "eager", "graph_capture": False, "graph_replay": False, "accepted": False})
    else:
        names = [x["name"] for x in sequence]
        warmup_identity = {k: v for k, v in cfg.items() if k not in ("kernel_sequence", "probe_evidence", "binary")}
        save(out / "execution-manifest.json", {"schema_version": 5, "samples_per_case": 10, "target_operator": "TextGatedResidual",
             "warmup": warmup_identity, "case_runs": [{"case_id": session.case["case_id"],
                 "warmup_kernel_sequence": cfg["kernel_sequence"], "target_kernel_sequence": names,
                 "iteration_kernel_sequence": cfg["kernel_sequence"] + names}]})
        save(out / "runtime-receipt.json", receipt)


def runtime_guard(config, case, calibration, action, route, samples, precision_evidence):
    normalize(case)
    expected = "CALIBRATION_ONLY" if calibration else "DEPLOYED_VERIFIED"
    if config.get("status") != expected or config.get("execution_mode") != "eager":
        raise RuntimeError("explicit deployed or calibration config and eager required; no NPU initialized")
    if config.get("platform") not in ("A3", "A5") or config.get("device_id") is None or not config.get("expected_full_soc"):
        raise RuntimeError("actual platform, physical device and full_soc must be frozen")
    if type(config.get("seed")) is not int:
        raise RuntimeError("explicit deterministic seed is required")
    checked_evidence(config.get("baseline_deployment_evidence"), "baseline deployment")
    checked_evidence(config.get("baseline_schema_evidence"), "public baseline schema")
    baseline_identity = config.get("baseline_source_identity", {})
    if (config.get("baseline_source_commit") is not None or baseline_identity.get("kind") != "installed_binary_release"
            or baseline_identity.get("version") != CONTRACT["baseline_identity"]["version"]):
        raise RuntimeError("custom binary baseline has no fabricated source commit")
    if action == "profile":
        profile_ready(config, case, route, samples, calibration)
        if precision_evidence is None or not Path(precision_evidence).is_file():
            raise RuntimeError("profile requires actual same-input paired precision evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "inspect", "precision", "profile", "calibration"))
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--input-contract-file", type=Path)
    parser.add_argument("--route", choices=("baseline", "candidate"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--calibration-action", choices=("precision", "profile"))
    parser.add_argument("--calibration-precision-evidence", "--precision-evidence", dest="precision_evidence", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    case = json.loads(args.case_file.read_text(encoding="utf-8"))
    normalize(case)
    if args.mode in ("plan", "inspect"):
        save(args.out / (args.mode + ".json"), {**plan(case), "contract_sha256": digest(CONTRACT),
             "NPU_initialized": False, "extension_built": False, "inspection_scope": "static public ABI only"})
        return 0
    if not args.execute or not args.runtime_config:
        raise RuntimeError("runtime requires explicit --execute and config; no NPU initialized")
    config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    calibration = args.mode == "calibration"
    action = args.calibration_action if calibration else args.mode
    if action not in ("precision", "profile"):
        raise RuntimeError("explicit calibration action required")
    runtime_guard(config, case, calibration, action, args.route, args.samples, args.precision_evidence)
    config["calibration_only"] = calibration
    if calibration:
        paired_contract = input_contract(case, config["seed"], True)
        save(args.out / "calibration-input-contract.json", paired_contract)
    else:
        if not args.input_contract_file:
            raise RuntimeError("frozen worker input contract required")
        paired_contract = json.loads(args.input_contract_file.read_text(encoding="utf-8"))
        expected = input_contract(case, config["seed"])
        for key in ("case", "seed", "contract_sha256", "input_contract_sha256", "samples"):
            if paired_contract.get(key) != expected[key]:
                raise RuntimeError("frozen input contract mismatch: " + key)
    import torch
    import torch_npu
    torch.set_num_threads(4)
    torch.npu.set_device(config["logical_device"])
    torch.npu.config.allow_internal_format = True
    if torch.npu.get_device_name(config["logical_device"]) != config["expected_full_soc"]:
        raise RuntimeError("actual full_soc mismatch")
    session = PairedSession(torch, torch_npu, config, case)
    if action == "precision":
        precision(session, paired_contract, args.out)
    else:
        profile(session, paired_contract, args.route, args.out, args.precision_evidence, args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
