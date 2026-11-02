"""Recover metadata after the known post-build directory-hashing failure only.

No compilation, installation, network, resource query or NPU action. The failed
record/log remain untouched. Original build success is a control-flow inference,
not a fabricated raw return-code field. New metadata has separate provenance.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from build_hc_a3 import BASELINE_COMMIT, package_files, require, sha, within

BUILD_COMMIT = "8db8c0af1efe96667367a5fb08d90ee8d74fc0e4"
BUILDER_SHA = "a1e11eefdc97f534109ae0e60bd9826278a510177b6513ec4fd208751a6308e4"
SOURCE_CONFIG_SHA = "a1999a5666355b9b1daff913891f376ae8a9f54f6289001d0e6ccb22047fea48"
ARTIFACT = "codex-artifacts/hc-pre-a3-expansion-20261102/"
BUILD_ARGV = ["bash", "build.sh", "--pkg", "--ops=hc_pre", "--soc=ascend910_93", "-j4"]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def validate_failed_record(record, original_builder):
    require(
        record.get("phase") == "fusion" and record.get("status") == "FAILED", "Expected original failed fusion metadata"
    )
    require(record.get("sync_commit") == BUILD_COMMIT, "Unexpected original build commit")
    require(record.get("baseline_source_commit") == BASELINE_COMMIT, "Unexpected operator source")
    require(record.get("build_argv") == BUILD_ARGV, "Unexpected original build command")
    require(record.get("builder", {}).get("sha256") == BUILDER_SHA, "Original builder identity differs")
    require(
        record.get("source_config", {}).get("sha256") == SOURCE_CONFIG_SHA, "Original source config identity differs"
    )
    require(hashlib.sha256(original_builder).hexdigest() == BUILDER_SHA, "Original Git builder bytes differ")
    error = record.get("error", "")
    require(
        "IsADirectoryError" in error and 'report["new_fusion_packages"]' in error,
        "Failure was not the known post-build enumeration error",
    )
    require("_CPack_Packages" in error and ".run" in error, "Unexpected failed path")
    environment = record.get("build_environment", {})
    require(environment.get("architecture", {}).get("full_soc") == "Ascend910_9362", "Wrong actual SoC")
    require(str(environment.get("architecture", {}).get("npu_arch")) == "2201", "Wrong actual arch")
    require(environment.get("cann_version") == "9.2.0", "Wrong CANN build environment")
    require(
        record.get("NPU_executed") is False and record.get("accepted_correctness") is False,
        "Unexpected numerical acceptance",
    )


def recover(root, repo, failed_path, sync_commit, out):
    head = git(repo, "rev-parse", "HEAD").decode().strip()
    require(head == sync_commit, "Actual helper checkout differs from fixed commit")
    require(
        subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"]).returncode == 0, "Tracked sources changed"
    )
    original_builder = git(repo, "show", BUILD_COMMIT + ":" + ARTIFACT + "build_hc_a3.py")
    failed_bytes = failed_path.read_bytes()
    record = json.loads(failed_bytes)
    validate_failed_record(record, original_builder)
    csrc_tree = git(repo, "rev-parse", BUILD_COMMIT + ":csrc").decode().strip()
    require(
        csrc_tree == git(repo, "rev-parse", "HEAD:csrc").decode().strip(), "Operator source tree changed since build"
    )
    require(
        csrc_tree == git(repo, "rev-parse", BASELINE_COMMIT + ":csrc").decode().strip(),
        "Operator source differs from baseline freeze",
    )
    config_path = within(record["source_config"]["path"], repo)
    require(config_path == repo / ARTIFACT / "source-config.json", "Unexpected original source config path")
    require(sha(config_path) == record["source_config"]["sha256"], "Original source configuration changed")
    config = json.loads(config_path.read_text())
    source = within(config_path.parent / config["serial_source"], repo)
    expected_inputs = {
        (source / relative).relative_to(root).as_posix(): digest for relative, digest in config["serial_files"].items()
    }
    require(record.get("new_source_inputs") == expected_inputs, "Original source whitelist is incomplete or changed")
    for relative, expected in record["new_source_inputs"].items():
        require(sha(within(root / relative, repo)) == expected, "Original serial source changed")
    log = failed_path.parent / "fusion-build.log"
    log_text = log.read_text(errors="strict")
    require(
        "Self-extractable archive successfully created" in log_text,
        "No actual successful CPack creation in original log",
    )
    package = package_files(repo / "csrc")[0]
    digest = sha(package)
    copies = []
    directories = []
    for path in sorted((repo / "csrc").rglob(package.name)):
        within(path, repo / "csrc")
        require(not path.is_symlink(), "Unexpected package symlink")
        if path.is_dir():
            directories.append(str(path))
        elif path.is_file():
            require(sha(path) == digest, "Same-named package copies disagree")
            copies.append({"path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    require(len(copies) >= 2, "Independent build/CPack package copy missing")
    require(failed_path.read_bytes() == failed_bytes, "Original failed record changed during recovery")
    # Preserve original source/build fields and create a separate derived record.
    recovered = dict(record)
    recovered.update(
        status="BUILD_PHASE_COMPLETE_RUNTIME_PENDING",
        record_kind="RECOVERED_POST_BUILD_METADATA",
        new_fusion_packages=[{"path": str(package), "sha256": digest}],
        installation="PENDING_EXPLICIT_TASK_LOCAL_VENDOR_INSTALL",
        original_failure={
            "path": str(failed_path),
            "sha256": hashlib.sha256(failed_bytes).hexdigest(),
            "status": "FAILED",
            "error": record["error"],
        },
        original_build_log={"path": str(log), "sha256": sha(log)},
        build_success_evidence={
            "inferred_returncode": 0,
            "raw_returncode_field_recorded": False,
            "basis": (
                "Original fixed builder run() requires returncode==0 before the observed package hashing traceback; "
                "original CPack log and identical actual package copies corroborate completion."
            ),
        },
        recovery={
            "written_at": datetime.now(timezone.utc).isoformat(),
            "helper_commit": head,
            "helper_sha256": sha(Path(__file__)),
            "csrc_tree": csrc_tree,
            "package_copies": copies,
            "ignored_same_named_directories": directories,
            "compilation_executed": False,
            "installation_executed": False,
            "NPU_executed": False,
        },
    )
    recovered.pop("error")
    out.mkdir(parents=True, exist_ok=False)
    target = out / "build-record.json"
    target.write_text(json.dumps(recovered, indent=2) + "\n")
    require(failed_path.read_bytes() == failed_bytes, "Original failed record changed")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sync-commit", required=True)
    parser.add_argument("--failed-build-record", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == Path("/home/s30083169/perf-serial-graph/hc-a3-20261102"), "Use only the actual A3 task root")
    repo = root / "dependencies/vllm-ascend-current-main"
    failed = within(args.failed_build_record, root / "outputs/build-evidence")
    require(failed.is_file() and not args.failed_build_record.is_symlink(), "Actual original build record required")
    out = within(args.out, root / "outputs/build-evidence")
    target = recover(root, repo, failed, args.sync_commit, out)
    print(
        json.dumps(
            {
                "status": "BUILD_PHASE_COMPLETE_RUNTIME_PENDING",
                "build_record": str(target),
                "build_commit": BUILD_COMMIT,
                "helper_commit": args.sync_commit,
                "NPU_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
