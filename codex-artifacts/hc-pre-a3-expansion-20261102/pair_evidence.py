"""CPU-only identities and strict validation for the T4/T16 output preflight.

No torch imports, device calls, or simulated performance. A factual small-input
check never changes the user's deferred precision-acceptance decision.
"""

import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path

CONTRACT = "external_pre_mix_5in_4out_v3_function"
TOLERANCE_REL = (
    "dependencies/vllm-ascend-current-main/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py"
)
TOLERANCE_SHA = "f2753baa5db4400dbe0ab8fe506f785b7bb12b8d91112dda4ae4498f7c94a247"
LIMITS = {
    "Y_DIFF_THRESHOLD": 4e-3,
    "Y_REQUIRED_PASS_RATE": 0.98,
    "AUX_DIFF_THRESHOLD": 1e-4,
    "AUX_REQUIRED_PASS_RATE": 0.995,
}
AXES = ("T", "H", "hc_mult", "hc_sinkhorn_iters", "norm_eps", "hc_eps")
PARAMS = AXES[2:]
IMPLEMENTATIONS = {"serial", "current-main"}


def require(condition, message):
    if condition is not True:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same(actual, expected, label):
    # JSON encoding distinguishes bool/int and rejects non-finite float values.
    require(
        json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False),
        label,
    )


def digest_string(value, label):
    require(type(value) is str and re.fullmatch("[a-f0-9]{64}", value) is not None, label)


def number(value, label, minimum=0, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), label)
    require(value >= minimum and (maximum is None or value <= maximum), label)


def case_contract(case):
    axes = {key: case["axes"][key] for key in AXES}
    for key in ("T", "H", "hc_mult", "hc_sinkhorn_iters"):
        require(type(axes[key]) is int and axes[key] > 0, "Invalid integer axis: " + key)
    require(
        axes["H"] == 5120 and axes["hc_mult"] == 4 and axes["hc_sinkhorn_iters"] == 20,
        "Sampler/source contract currently supports H5120/mult4/20 iterations",
    )
    same([axes["norm_eps"], axes["hc_eps"]], [1e-20, 1e-6], "Explicit epsilons differ")
    require(axes["T"] % axes["hc_mult"] == 0, "Unsupported T division")
    t, h, m = axes["T"], axes["H"], axes["hc_mult"]
    expected = {
        "input_shapes": [[t, m, h], [m * (m + 2), m * h], [3], [m * (m + 2)], [t, m]],
        "input_dtypes": ["bfloat16", "float32", "float32", "float32", "float32"],
        "output_shapes": [[t, h], [t, m], [t, m, m], [t, m]],
        "output_dtypes": ["bfloat16", "float32", "float32", "float32"],
    }
    for key, value in expected.items():
        same(case[key], value, "Manifest interface differs: " + key)
    require(case.get("measured") is False, "Only pending cases may use this runner")
    return dict(
        case_id=case["id"], T=t, axes=axes, parameters={key: axes[key] for key in PARAMS}, seed=1024 + t, **expected
    )


def expected_small_cases(manifest):
    result = []
    for t in (4, 16):
        found = [case for case in manifest["cases"] if case["axes"]["T"] == t]
        require(len(found) == 1, "Manifest needs exactly one small case at T=" + str(t))
        result.append(case_contract(found[0]))
    return result


def build_current_context(root, manifest_path, environment_path, interface_path, inventory_path):
    """Read actual current bytes; never obtain expected identities from a PASS report.

    The environment evidence must be this run's container/card capture. Explicit
    PERF_* identity variables bind the saved evidence to the selected invocation;
    they supplement, and never replace, fresh guard/ownership checks.
    """
    root = Path(root).resolve()
    here = Path(__file__).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    environment = json.loads(Path(environment_path).read_text())
    interface = json.loads(Path(interface_path).read_text())
    inventory = json.loads(Path(inventory_path).read_text())
    small = expected_small_cases(manifest)
    for case in manifest["cases"]:
        contract = case_contract(case)
        same(
            {k: contract["axes"][k] for k in AXES if k != "T"},
            {k: small[0]["axes"][k] for k in AXES if k != "T"},
            "Manifest mixes shared parameters",
        )
    require(interface.get("source_external_pre_mix_contract_verified") is True, "Source contract unverified")
    require(interface.get("comparison_contract") == CONTRACT, "Wrong functional contract")
    require(interface.get("precision_acceptance") == "DEFERRED_BY_USER", "Precision boundary changed")
    for key in AXES[1:]:
        same(interface[key], small[0]["axes"][key], "Interface parameter differs: " + key)
    require(interface["source_inventory_sha256"] == sha(inventory_path), "Interface inventory identity differs")
    for key in (
        "full_soc",
        "npu_arch",
        "cann_version",
        "host",
        "container_id",
        "collection_epoch",
        "torch_version",
        "torch_npu_version",
    ):
        require(type(environment.get(key)) is str and bool(environment[key].strip()), "Missing environment: " + key)
    require(environment["host"] == os.environ["PERF_HOST"], "Selected host differs")
    require(environment["container_id"] == os.environ["PERF_CONTAINER_ID"], "Selected container differs")
    device = {}
    for key, variable in [
        ("physical_device", "PERF_PHYSICAL_DEVICE"),
        ("npu_id", "PERF_NPU_ID"),
        ("chip_id", "PERF_CHIP_ID"),
    ]:
        require(type(environment.get(key)) is int and environment[key] >= 0, "Invalid device field: " + key)
        require(environment[key] == int(os.environ[variable]), "Selected device differs: " + key)
        device[key] = environment[key]
    require(
        environment.get("logical_device") == 0 and type(environment.get("logical_device")) is int,
        "This runner selects runtime logical device 0; actual visibility is checked before tensor allocation",
    )
    require(
        environment.get("target_platform") == "A3" and environment["host"] == "80.48.33.118",
        "This independent epoch supports only the authorized A3 host",
    )
    require(
        0 <= device["chip_id"] < 2 and 0 <= device["npu_id"] < 8 and 0 <= device["physical_device"] < 16,
        "A3 selected device identity invalid",
    )
    require(
        environment["full_soc"] == "Ascend910_9362" and environment["npu_arch"] == "2201",
        "Actual A3 architecture differs",
    )
    same(
        environment.get("device_mapping"),
        {
            "npu_id": device["npu_id"],
            "chip_id": device["chip_id"],
            "chip_logic_id": device["physical_device"],
            "chip_physic_id": device["physical_device"],
            "device_node": "/dev/davinci" + str(device["physical_device"]),
            "source": "actual npu-smi info -m Chip Logic/Physic IDs",
        },
        "Recorded actual A3 mapping differs",
    )
    device.update(
        logical_device=0,
        host=environment["host"],
        container_id=environment["container_id"],
        collection_epoch=environment["collection_epoch"],
    )
    files = inventory["files"]
    require(type(files) is dict and bool(files), "Empty frozen source inventory")
    for rel, expected in files.items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root), "Inventory path outside project")
        digest_string(expected, "Invalid inventory digest")
        require(sha(path) == expected, "Current frozen source/binary changed: " + rel)
    tolerance = root / TOLERANCE_REL
    require(
        sha(tolerance) == TOLERANCE_SHA and files.get(TOLERANCE_REL) == TOLERANCE_SHA,
        "Actual fixed tolerance test bytes differ",
    )
    require(interface["tolerance_source"]["sha256"] == TOLERANCE_SHA, "Tolerance reference identity differs")
    parsed = {}
    for node in ast.parse(tolerance.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in LIMITS:
                    parsed[target.id] = ast.literal_eval(node.value)
    same(parsed, LIMITS, "Actual fixed test thresholds differ")
    serial = list((root / "work/hc_pre/kernel/build").glob("hc_pre_ext*.so"))
    require(len(serial) == 1, "Exactly one frozen serial extension required")
    vendor = Path(os.environ["PERF_BASELINE_VENDOR"]).resolve()
    require(vendor.is_relative_to(root), "Vendor outside project")
    baseline = vendor / "op_api/lib/libcust_opapi.so"
    for path in (serial[0], baseline):
        require(files.get(path.relative_to(root).as_posix()) == sha(path), "Binary absent from frozen inventory")
    identities = {
        "manifest": Path(manifest_path),
        "environment": Path(environment_path),
        "interface": Path(interface_path),
        "inventory": Path(inventory_path),
        "checker": here / "check_hc_pair.py",
        "collector": here / "measure_hc_expansion.py",
        "adapter": here / "hc_pre_aclnn_expansion.py",
        "validator": Path(__file__),
        "resource_guard": here / "resource_guard_a3.py",
        "sampler_reference": root / "work/hc_pre/model.py",
        "tolerance_test": tolerance,
        "serial_binary": serial[0],
        "baseline_binary": baseline,
    }
    context = {
        "schema_version": 2,
        "comparison_contract": CONTRACT,
        "files": {name: {"path": str(path.resolve()), "sha256": sha(path)} for name, path in identities.items()},
        "device": device,
        "environment": environment,
        "frozen_files": files,
        "source_commit": inventory["source_commit"],
        "vendor_artifacts": {
            path.relative_to(vendor).as_posix(): sha(path)
            for path in sorted(vendor.rglob("*"))
            if path.is_file() and path.suffix in (".so", ".o", ".json")
        },
        "sampler": {
            "function": "_make_case(T, hidden_size=H)",
            "seed_rule": "1024+T",
            "input_digest": "sha256 of ordered contiguous CPU tensor storage bytes; no concatenation",
            "model_captured": False,
        },
        "tolerance": {
            "path": TOLERANCE_REL,
            "sha256": TOLERANCE_SHA,
            "constants": parsed,
            "rule": "absolute <= threshold OR abs/max(abs(actual),abs(reference),float32.tiny) <= threshold",
        },
    }
    return context, small


def validate_pair_evidence(report, expected_context, expected_cases):
    """Reject incomplete/stale/self-contradictory evidence; no status-only bypass."""
    require(report.get("schema_version") == 2 and type(report.get("schema_version")) is int, "Preflight schema differs")
    require(report.get("status") == "OUTPUT_CHECK_PASS", "Output preflight did not pass")
    require(report.get("NPU_executed") is True, "No actual NPU output check recorded")
    require(report.get("accepted_correctness") is False, "Deferred acceptance must remain false")
    require(report.get("precision_acceptance") == "DEFERRED_BY_USER", "Precision acceptance changed")
    same(report["context"], expected_context, "Preflight is stale: current identity/context differs")
    runtime = report["runtime_device_selection"]
    require(runtime.get("status") == "PASS", "Actual framework device selection did not pass")
    same(runtime["framework_device_count"], 1, "Actual framework visibility differs")
    same(runtime["framework_current_device"], 0, "Actual framework current device differs")
    same(runtime["physical_isolation"], False, "Privileged device visibility was misrepresented")
    same(
        runtime["ASCEND_RT_VISIBLE_DEVICES"],
        str(expected_context["device"]["physical_device"]),
        "Actual runtime selection differs from recorded physical identity",
    )
    for key in ("physical_device", "npu_id", "chip_id"):
        same(runtime[key], expected_context["device"][key], "Runtime device identity differs: " + key)
    same(expected_context["tolerance"]["constants"], LIMITS, "Expected tolerance is not the fixed test")
    require(expected_context["tolerance"]["sha256"] == TOLERANCE_SHA, "Expected tolerance source differs")
    rows = report["cases"]
    require(type(rows) is list and len(rows) == 2, "Exactly T4 and T16 are required")
    require(all(type(row.get("T")) is int for row in rows), "T must be an integer")
    require(sorted(row["T"] for row in rows) == [4, 16], "Duplicate or missing small case")
    require(
        sorted(case["T"] for case in expected_cases) == [4, 16] and len(expected_cases) == 2,
        "Caller must supply two current manifest contracts",
    )
    for expected in expected_cases:
        row = next(row for row in rows if row["T"] == expected["T"])
        for key in ("case_id", "T", "axes", "parameters", "seed", "input_shapes", "input_dtypes"):
            same(row[key], expected[key], "Small-case contract differs: " + key)
        digest_string(row["input_sha256"], "Missing actual ordered input digest")
        same(row["device"], expected_context["device"], "Per-case device differs")
        impls = row["implementations"]
        require(type(impls) is dict and set(impls) == IMPLEMENTATIONS, "Both exact implementations are required")
        for name in sorted(IMPLEMENTATIONS):
            impl = impls[name]
            require(impl.get("structure_matches") is True, "Output structure check failed: " + name)
            require(impl.get("eager_structure_matches") is True, "Eager output structure check failed: " + name)
            require(impl.get("input_sha256") == row["input_sha256"], "Implementations did not share identical input")
            require(
                impl.get("binary_sha256")
                == expected_context["files"]["serial_binary" if name == "serial" else "baseline_binary"]["sha256"],
                "Implementation binary differs",
            )
            identity = expected_context["files"]["serial_binary" if name == "serial" else "baseline_binary"]
            same(impl["mapped_binary"], identity, "Actually mapped binary identity differs")
            require(impl.get("binary_mapped") is True, "Expected implementation binary is not mapped")
            outputs = impl["outputs"]
            require(type(outputs) is list and len(outputs) == 4, "Four output checks required")
            require(
                all(type(c.get("output_index")) is int for c in outputs)
                and sorted(c["output_index"] for c in outputs) == list(range(4)),
                "Output indices must be exactly 0..3",
            )
            for i in range(4):
                check = next(c for c in outputs if c["output_index"] == i)
                same(check["shape"], expected["output_shapes"][i], "Actual output shape differs")
                same(check["dtype"], expected["output_dtypes"][i], "Actual output dtype differs")
                threshold = LIMITS["Y_DIFF_THRESHOLD" if i == 0 else "AUX_DIFF_THRESHOLD"]
                required = LIMITS["Y_REQUIRED_PASS_RATE" if i == 0 else "AUX_REQUIRED_PASS_RATE"]
                for key in ("finite", "within_recorded_tolerance", "graph_matches_eager"):
                    require(check.get(key) is True, "Actual output failed: " + key)
                for key in ("max_abs", "mean_abs", "threshold", "required_fraction", "observed_fraction"):
                    number(check[key], "Invalid numeric output statistic: " + key)
                same(check["threshold"], threshold, "Threshold differs from fixed test")
                same(check["required_fraction"], required, "Required fraction differs from fixed test")
                number(check["observed_fraction"], "Invalid observed fraction", maximum=1)
                require(check["observed_fraction"] >= required, "Recorded fraction fails tolerance")
                elements = math.prod(expected["output_shapes"][i])
                require(
                    type(check.get("elements")) is int and check["elements"] == elements, "Output element count differs"
                )
                require(
                    type(check.get("passed_elements")) is int and 0 <= check["passed_elements"] <= elements,
                    "Invalid tolerance pass count",
                )
                same(
                    check["observed_fraction"],
                    check["passed_elements"] / elements,
                    "Observed fraction contradicts actual element/pass counts",
                )
                require(check["mean_abs"] <= check["max_abs"], "Mean error exceeds maximum")
    return report


def validate_sample_input(case, shapes, dtypes, digest, checked):
    """Called on newly sampled CPU bytes before H2D, including a T4/T16 digest tie."""
    expected = case_contract(case)
    same(shapes, expected["input_shapes"], "New sampler input shapes differ")
    same(dtypes, expected["input_dtypes"], "New sampler input dtypes differ")
    digest_string(digest, "New sampler input digest missing")
    if expected["T"] in (4, 16):
        prior = next(row for row in checked["cases"] if row["T"] == expected["T"])
        require(digest == prior["input_sha256"], "New sampled input differs from the checked small case")
