"""Fresh A3 selected-chip resource observations. No torch, source reads or device allocation.

Both Chip Logic ID and Chip Physic ID must match the selected device. A process
snapshot is not a resource lock; missing or contradictory evidence is rejected.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if condition is not True:
        raise RuntimeError("RESOURCE_GATE_FAILED: " + message)


def selected_device():
    require(os.environ.get("PERF_HOST") == "80.48.33.118", "A3 host differs")
    npu, chip, physical = (int(os.environ[key]) for key in ("PERF_NPU_ID", "PERF_CHIP_ID", "PERF_PHYSICAL_DEVICE"))
    require(0 <= npu < 8 and 0 <= chip < 2 and 0 <= physical < 16, "A3 selected-chip identity invalid")
    require(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(physical), "Runtime visible selection differs")
    return npu, chip, physical


def parse_snapshot(text, mapping, npu, chip, physical):
    require(type(npu) is int and type(chip) is int and type(physical) is int, "Device identifiers must be integers")
    require(0 <= npu < 8 and 0 <= chip < 2 and 0 <= physical < 16, "A3 selected-chip identity invalid")
    mappings = [
        tuple(map(int, m.groups()))
        for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+Ascend\S*\s*$", mapping, re.M)
    ]
    selected = [row for row in mappings if row[:2] == (npu, chip)]
    require(selected == [(npu, chip, physical, physical)], "Actual selected Logic/Physic ID mapping absent or changed")
    current_npu, current_health, current_name = None, None, None
    selected_rows = []
    for line in text.splitlines():
        head = re.match(r"^\|\s*(\d+)\s+(Ascend\S+)\s*\|\s*(\S+)\s*\|", line)
        if head:
            current_npu, current_name, current_health = int(head[1]), head[2], head[3]
        memory = re.match(r"^\|\s*(\d+)\s+(\d+)\s*\|[^|]*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
        if memory and current_npu == npu and int(memory[1]) == chip:
            selected_rows.append((int(memory[2]), int(memory[3]), int(memory[4]), current_health, current_name))
    require(len(selected_rows) == 1, "Selected chip health/HBM row absent or duplicate")
    observed_physical, used, total, health, name = selected_rows[0]
    require(observed_physical == physical and health == "OK", "Selected chip physical identity/health invalid")
    require(0 <= used <= total and total > 0, "Selected chip HBM values invalid")
    require(re.search(r"Process\s+id", text, re.I) is not None, "Process table header absent")
    records = [tuple(map(int, m.groups())) for m in re.finditer(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", text, re.M)]
    no_process = [int(m[1]) for m in re.finditer(r"^\|\s*No running processes found in NPU\s+(\d+)\s*\|", text, re.M)]
    require(len(no_process) == len(set(no_process)), "Duplicate no-process rows")
    process_npus = {row[0] for row in records}
    require(not (process_npus & set(no_process)), "Contradictory process evidence")
    require(process_npus | set(no_process) == set(range(8)), "Full A3 process-table coverage absent")
    require(all(0 <= row[0] < 8 and 0 <= row[1] < 2 and row[2] > 0 for row in records), "Invalid process row")
    pids = {str(pid) for owner_npu, owner_chip, pid in records if (owner_npu, owner_chip) == (npu, chip)}
    return {
        "target_platform": "A3",
        "npu_id": npu,
        "chip_id": chip,
        "physical_device": physical,
        "device_node": "/dev/davinci" + str(physical),
        "mapping_source": "actual npu-smi info -m Chip Logic/Physic IDs",
        "health": health,
        "short_name_observed": name,
        "full_soc_inferred": False,
        "HBM_used_MiB": used,
        "HBM_total_MiB": total,
        "process_ids": sorted(pids),
        "process_evidence_present": True,
    }


def read_snapshot(out, label):
    return json.loads((Path(out) / ("device-" + label + ".json")).read_text())


def verify_runtime_selection(torch, out):
    """Record the actual framework view below a fresh gate, before tensors.

    A privileged container exposes all devices. This checks runtime visibility,
    not physical isolation or host PID ownership, which the executor also checks.
    """
    npu, chip, physical = selected_device()
    count = torch.npu.device_count()
    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "ASCEND_RT_VISIBLE_DEVICES": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "framework_device_count": count,
        "physical_device": physical,
        "npu_id": npu,
        "chip_id": chip,
        "process_pid_in_container": os.getpid(),
        "physical_isolation": False,
        "status": "CHECKING",
    }
    path = Path(out) / "runtime-device-selection.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    require(type(count) is int and count == 1, "Actual framework visibility is not one selected chip")
    torch.npu.set_device(0)
    record["framework_current_device"] = torch.npu.current_device()
    record["status"] = "PASS" if record["framework_current_device"] == 0 else "FAILED"
    path.write_text(json.dumps(record, indent=2) + "\n")
    require(record["status"] == "PASS", "Actual framework current device differs")
    return record


def observe_command(directory, stem, argv):
    """Preserve command bytes and exit/exception evidence before failing closed."""
    directory = Path(directory)
    stdout_path = directory / (stem + ".txt")
    stderr_path = directory / (stem + ".stderr.txt")
    metadata_path = directory / (stem + "-command.json")
    record = {
        "argv": list(argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "STARTED",
        "returncode": None,
        "timed_out": False,
        "exception_type": None,
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n")
    text = None
    try:
        # Files are open before invocation, so timeout preserves partial bytes.
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=15, check=False)
        record["returncode"] = completed.returncode
        record["status"] = "OK" if completed.returncode == 0 else "FAILED_EXIT"
        if completed.returncode == 0:
            text = stdout_path.read_text()
    except subprocess.TimeoutExpired as error:
        record.update(status="FAILED_TIMEOUT", timed_out=True, exception_type=type(error).__name__, error=str(error))
    except Exception as error:
        record.update(status="FAILED_EXCEPTION", exception_type=type(error).__name__, error=str(error))
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
            record[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        metadata_path.write_text(json.dumps(record, indent=2) + "\n")
    require(record["status"] == "OK", "Command observation failed: " + stem + " " + record["status"])
    return text


def gate(out, label, expected=None):
    require(type(label) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", label) is not None, "Invalid evidence label")
    npu, chip, physical = selected_device()
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    text = observe_command(directory, "device-" + label, ["npu-smi", "info"])
    mapping = observe_command(directory, "device-" + label + "-mapping", ["npu-smi", "info", "-m"])
    snapshot = parse_snapshot(text, mapping, npu, chip, physical)
    snapshot["disk_available_bytes"] = shutil.disk_usage(directory).free
    require(snapshot["disk_available_bytes"] >= 32 * 1024**3, "Task partition has less than 32GiB available")
    if expected == set():
        require(snapshot["HBM_used_MiB"] <= 8192, "Selected chip pre-run HBM exceeds 8GiB")
    snapshot.update(
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        host=os.environ["PERF_HOST"],
        snapshot_is_lock=False,
    )
    pids = set(snapshot["process_ids"])
    snapshot["expected_process_ids"] = None if expected is None else sorted(expected)
    snapshot["ownership_matches_expected"] = None if expected is None else pids == expected
    (directory / ("device-" + label + ".json")).write_text(json.dumps(snapshot, indent=2) + "\n")
    if expected is not None:
        require(type(expected) is set and all(type(pid) is str for pid in expected), "Expected PID set invalid")
        require(pids == expected, "A3 process ownership changed: " + str(sorted(pids)))
    return pids
