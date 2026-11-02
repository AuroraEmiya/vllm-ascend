"""Independent Docker build from a new, explicitly fixed GitHub tree.

No SSH client, credentials, resource collector, old remote sources or proxy
script access. This program records the actual new build inputs and products.
It never performs NPU work or declares numerical/precision acceptance.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASELINE_COMMIT = "1b8b0c398e96dd3f8f3408aacbb61085ae43fcfe"


def require(condition, message):
    if condition is not True:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path, parent):
    path = Path(path).resolve()
    require(path.is_relative_to(parent), "Path outside the new task tree")
    return path


def run(argv, cwd, log):
    with log.open("w") as stream:
        result = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, "Build command failed; inspect " + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sync-commit", required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "serial", "fusion", "inventory"), required=True)
    parser.add_argument("--architecture-evidence", type=Path)
    parser.add_argument("--vendor", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == Path("/home/s30083169/perf-serial-graph/hc-a3-20261102"), "Use only the new A3 task tree")
    require(re.fullmatch("[a-f0-9]{40}", args.sync_commit) is not None, "Invalid sync commit")
    require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", "Disable automatic torch device backend loading")
    repo = root / "dependencies/vllm-ascend-current-main"
    actual_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(actual_commit == args.sync_commit, "Fixed GitHub commit differs")
    require(
        subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"]).returncode == 0,
        "Tracked new-tree sources are modified",
    )
    config_path = within(args.source_config, repo)
    config = json.loads(config_path.read_text())
    source = within(config_path.parent / config["serial_source"], repo)
    require(
        type(config["serial_files"]) is dict and bool(config["serial_files"]), "Explicit serial file whitelist required"
    )
    frozen = {}
    for relative, expected in config["serial_files"].items():
        path = within(source / relative, source)
        require(
            type(expected) is str and re.fullmatch("[a-f0-9]{64}", expected) is not None,
            "Invalid source whitelist digest",
        )
        require(path.is_file() and sha(path) == expected, "GitHub serial source differs: " + relative)
        frozen[path.relative_to(root).as_posix()] = expected
    out = root / "outputs/build-evidence" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.phase)
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "status": "RUNNING",
        "target_platform": "A3",
        "host": "80.48.33.118",
        "sync_commit": actual_commit,
        "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "baseline_source_commit": BASELINE_COMMIT,
        "source_config": {"path": str(config_path), "sha256": sha(config_path)},
        "new_source_inputs": frozen,
        "NPU_executed": False,
        "accepted_correctness": False,
        "precision_acceptance": "DEFERRED_BY_USER",
    }

    def save():
        (out / "build-record.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        serial = root / "work/hc_pre"
        if args.phase == "prepare":
            require(not serial.exists(), "Refuse to overwrite an existing serial tree")
            for relative in config["serial_files"]:
                destination = within(serial / relative, serial)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
            require(not (serial / "kernel/build").exists(), "Fresh build must be absent")
            report["copied_serial_files"] = {relative: sha(serial / relative) for relative in config["serial_files"]}
        elif args.phase in ("serial", "fusion"):
            require(args.architecture_evidence is not None, "Actual official architecture evidence is required")
            architecture_path = within(args.architecture_evidence, root)
            architecture = json.loads(architecture_path.read_text())
            for key in ("full_soc", "full_soc_source"):
                require(
                    type(architecture.get(key)) is str and bool(architecture[key].strip()),
                    "Missing actual architecture field: " + key,
                )
            require(
                type(architecture.get("npu_arch")) in (int, str) and str(architecture["npu_arch"]).isdigit(),
                "Missing actual npu_arch",
            )
            require(
                architecture["full_soc"] == "Ascend910_9362"
                and str(architecture["npu_arch"]) == "2201"
                and architecture.get("short_soc") == "Ascend910_93",
                "Actual official architecture differs from the independently observed A3 target",
            )
            cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
            version_file = cann / "aarch64-linux/ascend_toolkit_install.info"
            version_text = version_file.read_text()
            match = re.search(r"^version=(\d+)\.(\d+)\.(\d+)", version_text, re.M)
            require(
                match is not None and tuple(map(int, match.groups())) >= (9, 1, 0),
                "Fixed fusion source requires CANN >= 9.1.0",
            )
            # CPU import only, with no device backend autoload or device call.
            import torch

            abi = torch.compiled_with_cxx11_abi()
            require(abi is True, "Current serial CMake requires actual torch C++11 ABI=1")
            report["build_environment"] = {
                "architecture": architecture,
                "architecture_evidence": {"path": str(architecture_path), "sha256": sha(architecture_path)},
                "cann_root": str(cann),
                "cann_version": ".".join(match.groups()),
                "cann_version_file_sha256": sha(version_file),
                "torch_version": str(torch.__version__),
                "torch_cxx11_abi": abi,
            }
            if args.phase == "serial":
                for relative, expected in config["serial_files"].items():
                    require(sha(serial / relative) == expected, "Copied serial source changed")
                build = serial / "kernel/build"
                require(not build.exists(), "Refuse old build or old shared objects")
                argv = [
                    "cmake",
                    "-S",
                    str(serial / "kernel"),
                    "-B",
                    str(build),
                    "-DSOC_VERSION=" + architecture["full_soc"],
                    "-DASCEND_CANN_PACKAGE_PATH=" + str(cann),
                    "-DCMAKE_BUILD_TYPE=Release",
                ]
                report["configure_argv"] = argv
                save()
                run(argv, root, out / "serial-configure.log")
                run(["cmake", "--build", str(build), "--parallel", "4"], root, out / "serial-build.log")
                binaries = list(build.glob("hc_pre_ext*.so"))
                require(len(binaries) == 1, "Exactly one new serial extension required")
                report["new_serial_binary"] = {"path": str(binaries[0]), "sha256": sha(binaries[0])}
            else:
                require(
                    not (repo / "csrc/build_out").exists() and not (repo / "csrc/build").exists(),
                    "Refuse a pre-existing fusion build directory",
                )
                argv = ["bash", "build.sh", "--pkg", "--ops=hc_pre", "--soc=ascend910_93", "-j4"]
                report["build_argv"] = argv
                save()
                run(argv, repo / "csrc", out / "fusion-build.log")
                packages = sorted((repo / "csrc").rglob("*.run"))
                require(bool(packages), "HC package not found; inspect build log")
                report["new_fusion_packages"] = [{"path": str(path), "sha256": sha(path)} for path in packages]
                report["installation"] = "PENDING_EXPLICIT_TASK_LOCAL_VENDOR_INSTALL"
        else:
            require(args.vendor is not None, "Actual task-local installed vendor required")
            vendor = within(args.vendor, root)
            baseline = vendor / "op_api/lib/libcust_opapi.so"
            serial_binaries = list((serial / "kernel/build").glob("hc_pre_ext*.so"))
            require(
                baseline.is_file() and len(serial_binaries) == 1,
                "Both newly built implementation binaries are required",
            )
            paths = [config_path, serial_binaries[0]]
            paths.extend(serial / rel for rel in config["serial_files"])
            paths.extend(path for path in (repo / "csrc/moe/hc_pre").rglob("*") if path.is_file())
            paths.append(repo / "csrc/torch_binding.cpp")
            paths.append(repo / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py")
            paths.extend(path for path in vendor.rglob("*") if path.is_file())
            inventory = {
                "source_commit": actual_commit,
                "baseline_source_commit": BASELINE_COMMIT,
                "target_platform": "A3",
                "source_config_sha256": sha(config_path),
                "files": {path.relative_to(root).as_posix(): sha(path) for path in sorted(set(paths))},
                "loaded_binary_verified": False,
                "NPU_executed": False,
                "scope": "New GitHub tree and build products; actual process mappings still pending",
            }
            inventory_path = out / "source-binary-inventory.json"
            inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
            report["inventory"] = {"path": str(inventory_path), "sha256": sha(inventory_path)}
        report["status"] = "BUILD_PHASE_COMPLETE_RUNTIME_PENDING"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"status": report["status"], "phase": args.phase, "evidence": str(out)}))
        return 0
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc(), finished_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
