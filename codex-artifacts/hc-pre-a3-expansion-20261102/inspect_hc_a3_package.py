"""Capture the actual newly built HC package's install help, without installing.

Installer arguments and task-local destination must be chosen from this actual
evidence, not guessed. No SSH, source scan, proxy or NPU action.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    package, out = args.package.resolve(), args.out.resolve()
    if root != Path("/home/s30083169/perf-serial-graph/hc-a3-20261102"):
        raise ValueError("Use only the new A3 task root")
    if not package.is_relative_to(root / "dependencies/vllm-ascend-current-main/csrc"):
        raise ValueError("Package outside the new build tree")
    if not out.is_relative_to(root / "outputs"):
        raise ValueError("Evidence outside task outputs")
    build = json.loads(args.build_record.read_text())
    if build.get("phase") != "fusion" or build.get("status") != "BUILD_PHASE_COMPLETE_RUNTIME_PENDING":
        raise ValueError("A completed new fusion build record is required")
    identity = {"path": str(package), "sha256": sha(package)}
    if identity not in build["new_fusion_packages"]:
        raise ValueError("Actual package differs from the new build record")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "installer-help.txt").open("w") as stream:
        result = subprocess.run(
            ["bash", str(package), "--help"], cwd=root, stdout=stream, stderr=subprocess.STDOUT, timeout=30, check=False
        )
    report = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "status": "HELP_CAPTURED_REQUIRES_REVIEW" if result.returncode == 0 else "HELP_FAILED",
        "package": identity,
        "build_record_sha256": sha(args.build_record),
        "help_returncode": result.returncode,
        "help_sha256": sha(out / "installer-help.txt"),
        "installation_executed": False,
        "NPU_executed": False,
        "next": (
            "Review actual help; select an explicit install destination under the new task root. No global install."
        ),
    }
    (out / "installer-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out)}))
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
