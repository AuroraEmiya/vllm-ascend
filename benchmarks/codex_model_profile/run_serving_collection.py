"""Run official vLLM bench serve; strictly bracket optional profiling via HTTP.

Requires an already running task-owned server. Does not start/stop the server,
change model config, analyse/delete traces, or equate concurrency to actual BS.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path


def http(base, path, body=None, timeout=600):
    request = urllib.request.Request(base.rstrip("/") + path,
                                     data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {"status": response.status, "body": response.read().decode()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="kimi-k3.1-pro")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    p.add_argument("--bs", type=int, choices=[32, 64, 128], required=True)
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-root", type=Path)
    p.add_argument("--expected-ranks", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.profile and not args.profile_root:
        p.error("--profile requires --profile-root shared with server")
    record = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "actual_bs": "unverified: see per-step diagnostic JSONL", "events": []}
    command = ["vllm", "bench", "serve", "--base-url", args.base_url,
               "--backend", "openai", "--endpoint", "/v1/completions",
               "--model", args.model, "--tokenizer", args.tokenizer,
               "--tokenizer-mode", "kimi_k3", "--trust-remote-code",
               "--dataset-name", "random", "--random-input-len", str(args.input_len),
               "--random-output-len", str(args.output_len), "--num-prompts", str(args.num_prompts),
               "--max-concurrency", str(args.bs), "--request-rate", "inf", "--ignore-eos",
               "--seed", "0", "--save-result", "--save-detailed", "--result-dir", str(args.out),
               "--result-filename", "benchmark.json", "--percentile-metrics", "ttft,tpot,itl,e2el"]
    record["command"] = command
    started = False
    before = set(args.profile_root.glob("**/*ascend_pt")) if args.profile else set()
    try:
        record["health"] = http(args.base_url, "/health")
        if args.profile:
            record["events"].append({"event": "start_profile_request", "unix_ns": time.time_ns()})
            # Empty object is accepted by the verified vLLM0.33 API. No unverified overrides.
            record["start_response"] = http(args.base_url, "/start_profile", {})
            started = True
        record["benchmark_start_ns"] = time.time_ns()
        with (args.out / "benchmark.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        record["benchmark_end_ns"] = time.time_ns()
        record["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"bench serve exited {result.returncode}")
    except Exception as exc:
        record["failure"] = repr(exc)
    finally:
        if started:
            try:
                record["stop_response"] = http(args.base_url, "/stop_profile", {}, timeout=1800)
            except Exception as exc:
                record["stop_failure"] = repr(exc)
        record["finished_ns"] = time.time_ns()
        if args.profile:
            after = set(args.profile_root.glob("**/*ascend_pt"))
            record["new_raw_trace_directories"] = [str(x) for x in sorted(after - before)]
            record["all_raw_trace_directories"] = [str(x) for x in sorted(after)]
            record["rank_coverage"] = "pending explicit rank metadata audit; directory count is not rank identity"
        (args.out / "collection-run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 2 if "failure" in record or "stop_failure" in record else 0


if __name__ == "__main__":
    raise SystemExit(main())
