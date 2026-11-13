# SPDX-License-Identifier: Apache-2.0
"""Invoke one public combine call; msprof controls warmup and kernel replay."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--case", type=json.loads, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location("validation_oracle", args.oracle_file)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    from vllm_ascend.ops.triton.dcp import dcp_a2a as module

    launches = []

    def record(kernel, grid, *values, **options):
        compiled = kernel[grid](*values, **options)
        cache_hash = compiled.metadata.hash
        cache_key = base64.b32encode(bytes.fromhex(cache_hash)).decode().rstrip("=")
        directory = Path(os.environ["TRITON_CACHE_DIR"]) / cache_key
        binaries = list(directory.glob("*.npubin"))
        assert len(binaries) == 1, (directory, binaries)
        binary = binaries[0]
        launches.append(
            {
                "kernel": kernel.__name__,
                "grid": list(grid),
                "options": options,
                "metadata_hash": cache_hash,
                "cache_key": cache_key,
                "binary": str(binary),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
        )
        return compiled

    module._launch_kernel = record
    recv, local, local_lse, expected = oracle.make_inputs(args.case, False)
    actual = module.fused_dcp_lse_combine(recv, args.case[2], 1, local_output=local, local_lse=local_lse)
    torch.npu.synchronize()
    actual = actual.cpu()
    error = (actual.double() - expected).abs()
    assert torch.isfinite(actual).all()
    ratio = (error <= 2e-5 + 0.004 * expected.abs()).double().mean().item()
    maximum_error = error.max().item()
    assert ratio >= 0.99
    assert maximum_error <= 0.02
    assert len(launches) == 1
    report = {
        "case": args.case,
        "application_calls": 1,
        "precision": "passed",
        "match_ratio": ratio,
        "maximum_error": maximum_error,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "module": module.__file__,
        "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "launches": launches,
        "oracle_file": str(args.oracle_file),
        "oracle_sha256": hashlib.sha256(args.oracle_file.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"precision": "passed", "application_calls": 1, "launches": launches}))


if __name__ == "__main__":
    main()
