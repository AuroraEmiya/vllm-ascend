"""Fresh A2 resource observations. No torch, source reads or device allocation.

The physical id is the actual Chip Logic ID from npu-smi info -m. A process
snapshot is not a resource lock; missing or contradictory evidence is rejected.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def require(condition, message):
    if condition is not True:
        raise RuntimeError("RESOURCE_GATE_FAILED: " + message)


def selected_device():
    require(os.environ.get("PERF_HOST") == "80.48.37.140", "A2 host differs")
    npu, chip, physical = (int(os.environ[key]) for key in ("PERF_NPU_ID", "PERF_CHIP_ID", "PERF_PHYSICAL_DEVICE"))
    require(0 <= npu < 8 and chip == 0 and 0 <= physical < 8, "A2 device identity invalid")
    return npu, chip, physical


def parse_snapshot(text, mapping, npu, chip, physical):
    require(type(npu) is int and type(chip) is int and type(physical) is int, "Device identifiers must be integers")
    require(0 <= npu < 8 and chip == 0 and 0 <= physical < 8, "A2 device identity invalid")
    mappings = [
        tuple(map(int, match.groups()))
        for match in re.finditer(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+Ascend\s+\S+\s*$", mapping, re.M)
    ]
    selected = [row for row in mappings if row[:2] == (npu, chip)]
    require(selected == [(npu, chip, physical)], "Actual Chip Logic ID mapping absent or changed")
    current_npu = None
    health, hbm = [], []
    for line in text.splitlines():
        head = re.match(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\S+)\s*\|", line)
        if head and not head[2].isdigit():
            current_npu = int(head[1])
            if current_npu == npu:
                health.append({"short_name_observed": head[2], "health": head[3]})
        memory = re.match(r"^\|\s*(\d+)\s*\|[^|]*\|\s*\d+\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
        if memory and current_npu == npu and int(memory[1]) == chip:
            hbm.append((int(memory[2]), int(memory[3])))
    require(len(health) == 1 and health[0]["health"] == "OK", "Selected NPU health absent, duplicate or not OK")
    require(
        len(hbm) == 1 and 0 <= hbm[0][0] <= hbm[0][1] and hbm[0][1] > 0,
        "Selected A2 single-chip HBM row absent or invalid",
    )
    require(re.search(r"Process\s+id", text, re.I) is not None, "Process table header absent")
    pids = set(re.findall(r"^\|\s*" + str(npu) + r"\s+" + str(chip) + r"\s*\|\s*(\d+)\s*\|", text, re.M))
    no_process = len(re.findall(r"^\|\s*No running processes found in NPU\s+" + str(npu) + r"\s*\|", text, re.M))
    require(
        (bool(pids) and no_process == 0) or (not pids and no_process == 1),
        "Selected process evidence absent or contradictory",
    )
    return {
        "target_platform": "A2",
        "npu_id": npu,
        "chip_id": chip,
        "physical_device": physical,
        "device_node": "/dev/davinci" + str(npu),
        "mapping_source": "actual npu-smi info -m Chip Logic ID",
        "health": health[0]["health"],
        "short_name_observed": health[0]["short_name_observed"],
        "full_soc_inferred": False,
        "HBM_used_MiB": hbm[0][0],
        "HBM_total_MiB": hbm[0][1],
        "process_ids": sorted(pids),
        "process_evidence_present": True,
    }


def read_snapshot(out, label):
    return json.loads((Path(out) / ("device-" + label + ".json")).read_text())


def gate(out, label, expected=None):
    require(type(label) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", label) is not None, "Invalid evidence label")
    npu, chip, physical = selected_device()
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=15)
    (directory / ("device-" + label + ".txt")).write_text(text)
    (directory / ("device-" + label + "-mapping.txt")).write_text(mapping)
    snapshot = parse_snapshot(text, mapping, npu, chip, physical)
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
        require(pids == expected, "A2 process ownership changed: " + str(sorted(pids)))
    return pids
