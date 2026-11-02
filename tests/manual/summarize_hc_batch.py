# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Map sequential, live Graph model IDs to cases and retain all execution rows."""

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from summarize_hc_profile import END, START, numeric, summarize


def batch_reports(raw_path, manifest):
    with raw_path.open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    rows.sort(key=lambda r: numeric(r["Task Start Time(us)"]))
    groups = defaultdict(list)
    for row in rows:
        if row["Op Name"] in START:
            groups[row["Model ID"]].append(row)
    # Uncaptured warm-up uses a reserved model ID. Live Graph IDs are unique
    # because the application retains every graph until profiling finishes.
    live = [(model, starts) for model, starts in groups.items() if model != "4294967295"]
    live.sort(key=lambda item: numeric(item[1][0]["Task Start Time(us)"]))
    expected = manifest["ordered_profiles"]
    if len(live) != len(expected):
        raise ValueError(f"Expected {len(expected)} live graphs, observed {len(live)}")
    reports = []
    previous_end = float("-inf")
    for (model_id, starts), item in zip(live, expected):
        if len(starts) != item["graph_execution_calls"]:
            raise ValueError(f"Graph {model_id}: incomplete replay count")
        first = numeric(starts[0]["Task Start Time(us)"])
        gate_rows = [r for r in rows if r["Model ID"] == model_id and r["Op Name"] in END]
        if len(gate_rows) != len(starts):
            raise ValueError(f"Graph {model_id}: missing gate end")
        last = max(numeric(r["Task Start Time(us)"]) for r in gate_rows)
        if first <= previous_end:
            raise ValueError("Case Graph execution windows overlap")
        # Include every execution task in the window, including other streams
        # or auxiliary model IDs. Unknown tasks invalidate the existing strict
        # classifier; they are not silently filtered.
        selected = [r for r in rows if first <= numeric(r["Task Start Time(us)"]) <= last]
        if len([r for r in selected if r["Op Name"] in START]) != len(starts):
            raise ValueError("Mixed HC paths in a Graph execution window")
        with tempfile.TemporaryDirectory(prefix="hc-profile-") as tmp:
            path = Path(tmp) / "window.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
            report = summarize(path, retained_calls=item["profile_replays"])
        report.update(
            label=item["label"],
            raw_trace=str(raw_path),
            graph_model_id=model_id,
            graph_execution_calls=len(starts),
            window_start_us=first,
            window_end_us=last,
            capture_protocol="one process, sequential live Graphs, 5 warm + 16 retained replays",
        )
        reports.append(report)
        previous_end = last
    return {
        "profiles": reports,
        "source_csv": str(raw_path),
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_execution_rows": len(rows),
        "uncaptured_hc_warmup_calls": len(groups.get("4294967295", [])),
        "profile_manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = list(args.profile_root.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"))
    if len(sources) != 1:
        raise ValueError(f"Expected one full execution CSV, got {len(sources)}")
    result = batch_reports(sources[0], json.loads(args.manifest.read_text()))
    args.out.write_text(json.dumps(result, indent=2))
    for report in result["profiles"]:
        print(json.dumps({k: report[k] for k in ("label", "valid", "c_share_pct", "graph_model_id")}))
    if any(not r["valid"] for r in result["profiles"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
