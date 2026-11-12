# SPDX-License-Identifier: Apache-2.0
"""Profile public owner/combine calls with independent CPU test oracles.

Use the same runner and oracle files against each separately installed source tree.
This emits raw identity/precision evidence; msprof op supplies device timings.
"""

import argparse
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
    parser.add_argument("--kind", choices=("owner", "combine"), required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    parser.add_argument("--case", type=json.loads, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    torch.npu.set_device(0)
    spec = importlib.util.spec_from_file_location("validation_oracle", args.oracle_file)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    launches = []

    def record(kernel, grid, values, options):
        compiled = kernel[grid](*values, **options)
        launches.append({"kernel": kernel.__name__, "grid": list(grid), "options": options})
        return compiled

    if args.kind == "owner":
        from vllm_ascend.ops.triton import sharded_indexer as module

        original = module._merge_owner_prefix

        class OwnerKernel:
            def __getitem__(self, grid):
                return lambda *values, **options: record(original, grid, values, options)

        module._merge_owner_prefix = OwnerKernel()
        rows, world, owner, interleave, distribution = args.case
        cpu_packed, expected = oracle.make_inputs(rows, world, owner, interleave, distribution, False)
        packed = cpu_packed.npu()
        output = torch.empty_like(expected, device="npu")

        def invoke():
            return module.merge_packed_candidates(
                packed,
                rows,
                world,
                2048,
                local_rank=owner,
                interleave=interleave,
                out=output,
                presorted=True,
                owner_sharded=True,
            )

        def check(actual):
            assert torch.equal(actual.cpu(), expected)

    else:
        from vllm_ascend.ops.triton.dcp import dcp_a2a as module

        module._launch_kernel = lambda kernel, grid, *values, **options: record(kernel, grid, values, options)
        recv, local, local_lse, expected = oracle.make_inputs(args.case, False)

        def invoke():
            return module.fused_dcp_lse_combine(
                recv,
                args.case[2],
                1,
                local_output=local,
                local_lse=local_lse,
            )

        def check(actual):
            actual = actual.cpu()
            error = (actual.double() - expected).abs()
            assert torch.isfinite(actual).all()
            assert (error <= 2e-5 + 0.004 * expected.abs()).double().mean() >= 0.99
            assert error.max() <= 0.02

    for _ in range(args.samples):
        check(invoke())
        torch.npu.synchronize()
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    binaries = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in cache.rglob("*.npubin")}
    report = {
        "kind": args.kind,
        "case": args.case,
        "samples": args.samples,
        "precision": "passed",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "module": module.__file__,
        "module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "launches": launches,
        "binaries": binaries,
        "oracle_file": str(args.oracle_file),
        "oracle_sha256": hashlib.sha256(args.oracle_file.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"precision": "passed", "launches": len(launches), "binaries": len(binaries)}))


if __name__ == "__main__":
    main()
