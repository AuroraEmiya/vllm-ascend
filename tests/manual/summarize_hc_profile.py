# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Retain complete HC profiler call sequences and classify every execution task."""

import argparse
import csv
import json
import statistics
from pathlib import Path

START = {"qwen_hc_combine_norm", "_hc_combine_norm_kernel"}
END = {"qwen_hc_gate_mix", "_hc_gate_mix_kernel"}
OWNED = START | END | {"qwen_hc_silu", "_hc_silu_kernel"}


def classify(row):
    name = row["Op Name"]
    kind = row.get("OP Type", "")
    if "MatMul" in kind or "Matmul" in name or "MatMul" in name:
        return "C_all_matmul"
    if name in OWNED:
        return "V_owned_stage"
    if any(x in name for x in ("TransData", "Transpose", "Cast", "Reshape", "ViewCopy")):
        return "V_layout"
    if "Memcpy" in name or "MEMCPY" in row.get("Task Type", ""):
        return "V_copy"
    if any(x in name for x in ("Fill", "Memset")):
        return "V_other"
    return "unclassified"


def numeric(value):
    return float(value.replace("\\t", "").strip())


def summarize(path, retained_calls=16):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows.sort(key=lambda r: numeric(r["Task Start Time(us)"]))
    starts = [i for i, row in enumerate(rows) if row["Op Name"] in START]
    selected = starts[-retained_calls:]
    sequences = []
    for index in selected:
        next_start = next((i for i in starts if i > index), len(rows))
        call = rows[index:next_start]
        names = [r["Op Name"] for r in call]
        classes = [classify(r) for r in call]
        matmuls = sum(c == "C_all_matmul" for c in classes)
        stage_order = [c for c in classes if c in ("C_all_matmul", "V_owned_stage")]
        complete = (
            bool(call)
            and call[-1]["Op Name"] in END
            and matmuls == 2
            and stage_order == ["V_owned_stage", "C_all_matmul", "V_owned_stage", "C_all_matmul", "V_owned_stage"]
            and "unclassified" not in classes
        )
        c_us = sum(numeric(r["Task Duration(us)"]) for r, c in zip(call, classes) if c == "C_all_matmul")
        v_us = sum(numeric(r["Task Duration(us)"]) for r, c in zip(call, classes) if c.startswith("V_"))
        sequences.append(
            {
                "complete": complete,
                "c_total_us": c_us,
                "v_total_us": v_us,
                "task_sum_us": c_us + v_us,
                "names": names,
                "classification": classes,
                "tasks": [
                    {
                        "name": r["Op Name"],
                        "duration_us": numeric(r["Task Duration(us)"]),
                        "start_us": numeric(r["Task Start Time(us)"]),
                        "class": c,
                        "model_id": r["Model ID"],
                        "stream_id": r["Stream ID"],
                        "task_id": r["Task ID"],
                    }
                    for r, c in zip(call, classes)
                ],
            }
        )
    valid = len(sequences) == retained_calls and all(x["complete"] for x in sequences)
    c_mean = statistics.mean(s["c_total_us"] for s in sequences) if valid else None
    v_mean = statistics.mean(s["v_total_us"] for s in sequences) if valid else None
    return {
        "raw_trace": str(path),
        "valid": valid,
        "complete_sequences": sequences,
        "retained_calls": len(sequences),
        "requested_calls": retained_calls,
        "c_total_us": c_mean,
        "v_total_us": v_mean,
        "task_sum_us": c_mean + v_mean if valid else None,
        "c_share_pct": 100 * c_mean / (c_mean + v_mean) if valid else None,
        "aggregation": "mean of per-complete-call task sums; Graph event latency separate",
        "all_execution_rows": len(rows),
        "omitted_prefix_rows": selected[0] if selected else len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.profile_root.glob("decode-*/PROF_*/mindstudio_profiler_output/op_summary*.csv")):
        reports.append({"label": path.relative_to(args.profile_root).parts[0], **summarize(path)})
    args.out.write_text(json.dumps({"profiles": reports}, indent=2))
    for report in reports:
        print(json.dumps({k: report[k] for k in ("label", "valid", "c_total_us", "v_total_us", "c_share_pct")}))
    if not reports or any(not r["valid"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
