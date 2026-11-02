"""Capture actual owned-container A3 metadata without reading operator sources.

Uses the unmodified official architecture probe and a fresh device guard. No
SSH client, source scan, proxy script, tensor allocation or performance test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from resource_guard_a3 import gate, read_snapshot, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collection-epoch", required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    require(root == Path("/home/s30083169/perf-serial-graph/hc-a3-20261102"), "Use only the new A3 task root")
    require(out.is_relative_to(root / "outputs"), "Environment output outside task")
    require(os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0", "Disable automatic device backend loading")
    container = os.environ.get("PERF_CONTAINER_ID", "")
    require(
        re.fullmatch("[a-f0-9]{64}", container) is not None and container.startswith(socket.gethostname()),
        "Actual full container ID must match its default hostname",
    )
    require(re.fullmatch("[A-Za-z0-9_.-]+", args.collection_epoch) is not None, "Invalid collection epoch")
    out.mkdir(parents=True, exist_ok=False)
    gate(out, "environment-before", set())
    snapshot = read_snapshot(out, "environment-before")
    raw_path = out / "official-architecture.raw.txt"
    stderr_path = out / "official-architecture.stderr.txt"
    probe = Path(__file__).with_name("get_npu_arch.py")
    with raw_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(
            [sys.executable, str(probe), "--json"], stdout=stdout, stderr=stderr, timeout=40, check=False
        )
    require(result.returncode == 0, "Official architecture probe failed; inspect preserved raw output")
    raw = raw_path.read_text()
    architecture = json.loads(raw)
    for key in ("full_soc", "full_soc_source"):
        require(
            type(architecture.get(key)) is str and bool(architecture[key].strip()),
            "Official architecture evidence missing: " + key,
        )
    require(
        type(architecture.get("npu_arch")) in (int, str) and str(architecture["npu_arch"]).isdigit(),
        "Official npu_arch missing",
    )
    require(
        architecture["full_soc"] == "Ascend910_9362"
        and str(architecture["npu_arch"]) == "2201"
        and architecture.get("short_soc") == "Ascend910_93",
        "Actual official architecture differs from the independently observed A3 target",
    )
    # Preserve the exact official JSON; normalize only the separate environment.
    architecture_path = out / "official-architecture.json"
    architecture_path.write_text(raw)
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    version_path = cann / "aarch64-linux/ascend_toolkit_install.info"
    version_text = version_path.read_text()
    (out / "cann-install-info.txt").write_text(version_text)
    version = re.search(r"^version=(\d+\.\d+\.\d+)", version_text, re.M)
    require(version is not None, "Actual CANN version missing")
    import torch

    environment = {
        "schema_version": 1,
        "target_platform": "A3",
        "host": os.environ["PERF_HOST"],
        "container_id": container,
        "collection_epoch": args.collection_epoch,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "full_soc": architecture["full_soc"],
        "full_soc_source": architecture["full_soc_source"],
        "npu_arch": str(architecture["npu_arch"]),
        "npu_arch_source": architecture.get("npu_arch_source"),
        "cann_version": version[1],
        "cann_root": str(cann),
        "torch_version": str(torch.__version__),
        "torch_npu_version": importlib.metadata.version("torch_npu"),
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
        "physical_device": snapshot["physical_device"],
        "npu_id": snapshot["npu_id"],
        "chip_id": snapshot["chip_id"],
        "logical_device": 0,
        "logical_device_verification": "Pending actual torch_npu runtime check before tensors",
        "device_mapping": {
            "npu_id": snapshot["npu_id"],
            "chip_id": snapshot["chip_id"],
            "chip_logic_id": snapshot["physical_device"],
            "chip_physic_id": snapshot["physical_device"],
            "device_node": snapshot["device_node"],
            "source": snapshot["mapping_source"],
        },
        "official_architecture": {
            "raw_path": str(raw_path),
            "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "json_path": str(architecture_path),
            "json_sha256": hashlib.sha256(architecture_path.read_bytes()).hexdigest(),
            "normalization": "npu_arch=str(actual official value); original raw JSON retained",
        },
        "NPU_workload_executed": False,
        "accepted_correctness": False,
        "precision_acceptance": "DEFERRED_BY_USER",
    }
    gate(out, "environment-after", set())
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "ACTUAL_ENVIRONMENT_CAPTURED",
                "path": str(out / "environment.json"),
                "full_soc": environment["full_soc"],
                "npu_arch": environment["npu_arch"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
