"""Combine complete per-rank Ascend profiler CSVs without modifying raw traces.

Usage: python summarize_profiles.py --manifest collection.json --output out
Manifest must explicitly identify each trace's variant, bs, phase, repeat and rank.
Exit 2 means incomplete collection or unparseable kernel duration; outputs remain.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


META = ("variant", "bs", "phase", "repeat", "rank", "node", "device", "main_sha", "patch_sha", "actual_bs")
KEY = ("bs", "phase", "repeat", "node", "rank", "device", "kernel", "shape", "dtype",
       "input_format", "output_shape", "output_dtype", "output_format")


def write_csv(path, records, required=()):
    fields = list(dict.fromkeys([*required, *(k for row in records for k in row)]))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def first(row, aliases):
    for key in aliases:
        if row.get(key, "").strip():
            return row[key]
    return "unknown"


def duration_us(row):
    # Only explicitly unit-labelled fields are accepted; ambiguous Duration fails.
    for key, scale in (("Duration(us)", 1), ("Duration (us)", 1),
                       ("Task Duration(us)", 1), ("Task Duration (us)", 1),
                       ("Duration(ns)", .001), ("Duration(ms)", 1000)):
        if row.get(key, "").strip():
            value = float(row[key]) * scale
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}: {row[key]!r}")
            return value
    raise ValueError("No recognized unit-labelled duration column")


def summarize(manifest_path, output):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    output.mkdir(parents=True, exist_ok=True)
    errors, inventory, groups, kernels = list(manifest.get("failures", [])), [], defaultdict(list), []
    seen = set()
    roots = set()
    for entry in manifest["traces"]:
        for field in ("variant", "bs", "phase", "repeat", "rank", "node", "path"):
            if field not in entry:
                raise ValueError(f"Trace missing {field}: {entry}")
        identity = tuple(str(entry[k]) for k in ("variant", "bs", "phase", "repeat", "node", "rank"))
        if identity in seen:
            errors.append({"trace": identity, "error": "duplicate trace identity"})
            continue
        seen.add(identity)
        root = Path(entry["path"])
        if not root.is_absolute():
            root = manifest_path.parent / root
        root = root.resolve()
        if root in roots:
            errors.append({"trace": identity, "error": "raw trace path reused"})
            continue
        roots.add(root)
        parsed = root / "ASCEND_PROFILER_OUTPUT"
        if not parsed.is_dir():
            parsed = root
        files = [p for p in sorted(parsed.rglob("*.csv"))
                 if output.resolve() not in p.resolve().parents] if parsed.is_dir() else []
        names = {p.name for p in files}
        for name in manifest.get("required_csv", ["kernel_details.csv", "operator_details.csv"]):
            if name not in names:
                errors.append({"trace": identity, "error": f"missing {name}"})
        for path in files:
            count = 0
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    for line, raw in enumerate(reader, 2):
                        if None in raw:
                            raise ValueError(f"Malformed CSV at line {line}")
                        row = {f"meta_{k}": entry.get(k, "unknown") for k in META}
                        row.update(source_file=str(path), source_line=line)
                        row.update({f"raw_{k}": v for k, v in raw.items()})
                        groups[path.name].append(row)
                        count += 1
                        if path.name == "kernel_details.csv":
                            kernel = {k: str(entry.get(k, "unknown")) for k in META}
                            kernel.update(kernel=first(raw, ("Name", "Kernel Name", "Op Name", "OP Name")),
                                          shape=first(raw, ("Input Shapes", "Input Shape", "InputShapes")),
                                          dtype=first(raw, ("Input Data Types", "Input Dtypes", "Input Data Type")),
                                          input_format=first(raw, ("Input Formats", "Input Format")),
                                          output_shape=first(raw, ("Output Shapes", "Output Shape", "OutputShapes")),
                                          output_dtype=first(raw, ("Output Data Types", "Output Dtypes", "Output Data Type")),
                                          output_format=first(raw, ("Output Formats", "Output Format")),
                                          source_file=str(path), source_line=line)
                            device = first(raw, ("Device_id", "Device ID", "Device Id", "Device id", "Device"))
                            if device != "unknown":
                                kernel["device"] = device
                            try:
                                kernel["duration_us"] = duration_us(raw)
                            except (ValueError, TypeError) as exc:
                                kernel["duration_us"] = ""
                                errors.append({"file": str(path), "line": line, "error": str(exc)})
                            kernels.append(kernel)
            except (OSError, UnicodeError, ValueError, csv.Error) as exc:
                errors.append({"file": str(path), "error": str(exc)})
            inventory.append({"trace": identity, "file": str(path), "rows": count})
            if not count and path.name in manifest.get("required_csv", ["kernel_details.csv", "operator_details.csv"]):
                errors.append({"file": str(path), "error": "required CSV is empty"})
    expected = {tuple(str(item[k]) for k in ("variant", "bs", "phase", "repeat", "node", "rank"))
                for item in manifest.get("expected_traces", [])}
    for missing in sorted(expected - seen):
        errors.append({"trace": missing, "error": "expected rank/window not supplied"})
    if not expected:
        errors.append({"error": "expected_traces absent: completeness cannot be certified"})
    if seen - expected:
        errors.append({"error": "unexpected trace identities", "traces": sorted(seen - expected)})
    for name, rows in groups.items():
        write_csv(output / ("all-" + name), rows)
    write_csv(output / "kernel-instances-normalized.csv", kernels, (*META, "kernel", "shape", "dtype", "duration_us"))
    buckets = defaultdict(list)
    invalid = set()
    for row in kernels:
        key = (row["variant"], *(row[k] for k in KEY))
        if row["duration_us"] == "":
            invalid.add(key)
        else:
            buckets[key].append(row["duration_us"])
    summaries = []
    for key in sorted(set(buckets) | invalid):
        values = sorted(buckets[key])
        record = dict(zip(("variant", *KEY), key))
        record.update(valid_duration_count=len(values), complete=key not in invalid)
        if values and key not in invalid:
            record.update(total_us=sum(values), mean_us=statistics.mean(values),
                          median_us=statistics.median(values), min_us=values[0], max_us=values[-1],
                          p90_us=values[max(0, math.ceil(.9 * len(values)) - 1)])
        summaries.append(record)
    write_csv(output / "all-operator-summary.csv", summaries, ("variant", *KEY))
    paired = defaultdict(dict)
    for row in summaries:
        paired[tuple(row[k] for k in KEY)][row["variant"]] = row
    comparison = []
    for key, sides in sorted(paired.items()):
        base, cand = sides.get("baseline"), sides.get("candidate")
        row = dict(zip(KEY, key))
        row["status"] = "paired" if base and cand else "removed" if base else "added" if cand else "other_variant"
        for label, side in (("baseline", base), ("candidate", cand)):
            if side:
                row.update({f"{label}_{k}": side[k] for k in ("valid_duration_count", "complete", "total_us", "median_us") if k in side})
        if base and cand and base["complete"] and cand["complete"]:
            row["delta_total_us"] = cand["total_us"] - base["total_us"]
            if cand["total_us"] > 0:
                row["total_speedup"] = base["total_us"] / cand["total_us"]
            row["call_count_matches"] = base["valid_duration_count"] == cand["valid_duration_count"]
        comparison.append(row)
    write_csv(output / "baseline-vs-candidate.csv", comparison, KEY)
    audit = {"complete": not errors, "errors": errors, "inventory": inventory,
             "kernel_rows": len(kernels), "manifest": str(manifest_path.resolve()),
             "scope": "All supplied CSV rows; original traces unchanged. Kernel sums are not wall time."}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(summarize(args.manifest, args.output))
