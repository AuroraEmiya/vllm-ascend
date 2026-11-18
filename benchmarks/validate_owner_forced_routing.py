# SPDX-License-Identifier: Apache-2.0
"""Ablate only the owner-prefix host routing gates using a frozen kernel.

The public wrapper still allocates and launches the operator. Validation-only
interception selects the existing serial/parallel specialization and its tile.
PARALLEL_PEERS=True retains the kernel's data-dependent skewed-row fallback.
Device timings come from msprof op, not this program's wall-clock duration.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton

from vllm_ascend.ops.triton import sharded_indexer as module

TOPK = 2048
MAIN_COMMIT = "6385299e417741fc576c30c95f7860fd69dbec24"
N4_COMMIT = "53b4021ab8a75ad132158119e664ce6cc6e9da87"
WORLDS = (1, 2, 3, 4, 8, 16)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protection_cases():
    """All owners plus tails, equal scores, empty owners and skewed rows."""
    cases = [(2, world, owner, 768, "random", True) for world in WORLDS for owner in range(world)]
    cases += [
        (41, world, min(3, world - 1), 768, distribution, True)
        for world in WORLDS
        for distribution in ("ties", "short", "mixed", "own_high", "peer_high")
    ]
    return cases


def cpu_check(packed, expected, rows, world):
    assert packed.shape == (world * rows, TOPK, 2)
    assert packed.dtype == torch.int32 and packed.is_contiguous()
    assert expected.shape == (rows, 1, TOPK)
    invalid = expected == -1
    assert not ((~invalid[..., 1:]) & invalid[..., :-1]).any()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("main", "serial", "parallel"), required=True)
    parser.add_argument("--oracle-file", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", nargs=5, metavar=("ROWS", "WORLD", "OWNER", "INTERLEAVE", "DISTRIBUTION"))
    group.add_argument("--protection", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("validation_oracle", args.oracle_file)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    cases = protection_cases() if args.protection else [tuple(map(int, args.case[:4])) + (args.case[4], False)]
    report = dict(
        mode=args.mode,
        baseline_commit=MAIN_COMMIT if args.mode == "main" else N4_COMMIT,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        module=module.__file__,
        module_sha256=digest(module.__file__),
        harness_sha256=digest(__file__),
        oracle_file=str(args.oracle_file),
        oracle_sha256=digest(args.oracle_file),
        seed="21584+rows; timing repetitions reuse this input; rank streams are symmetric",
        launches=[],
        cases=[],
        precision="started",
    )
    original = module._merge_owner_prefix

    class OwnerKernel:
        def __getitem__(self, grid):
            def launch(*values, **options):
                values = list(values)
                public_tile = values[7]
                public_parallel = options.get("PARALLEL_PEERS")
                if args.mode != "main":
                    parallel = args.mode == "parallel"
                    divisor = values[3]
                    tile_input = TOPK // divisor if parallel else 2 * TOPK // divisor
                    values[7] = min(TOPK, triton.next_power_of_2(max(1, tile_input)))
                    options["PARALLEL_PEERS"] = parallel
                record = dict(
                    kernel=original.__name__,
                    grid=list(grid),
                    rows=values[2],
                    world=values[3],
                    owner=values[4],
                    interleave=values[5],
                    tile=values[7],
                    block_ranks=values[8],
                    public_tile=public_tile,
                    public_parallel=public_parallel,
                    options=dict(options),
                )
                report["launches"].append(record)
                compiled = original[grid](*values, **options)
                record["compiled_name"] = compiled.name
                return compiled

            return launch

    module._merge_owner_prefix = OwnerKernel()
    try:
        if not args.cpu_only:
            torch.npu.set_device(0)
        for rows, world, owner, interleave, distribution, edge in cases:
            case = dict(
                rows=rows, world=world, owner=owner, interleave=interleave, distribution=distribution, edge=edge
            )
            report["cases"].append(case)
            packed, expected = oracle.make_inputs(rows, world, owner, interleave, distribution, edge)
            cpu_check(packed, expected, rows, world)
            case["packed_sha256"] = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
            case["expected_sha256"] = hashlib.sha256(expected.numpy().tobytes()).hexdigest()
            case["physical_shape"] = list(packed.shape)
            if not args.cpu_only:
                device_packed = packed.npu()
                output = torch.empty_like(expected, device="npu")
                actual = module.merge_packed_candidates(
                    device_packed,
                    rows,
                    world,
                    TOPK,
                    local_rank=owner,
                    interleave=interleave,
                    out=output,
                    presorted=True,
                    owner_sharded=True,
                )
                torch.npu.synchronize()
                assert torch.equal(actual.cpu(), expected), f"Exact oracle mismatch: {case}"
            case["precision"] = "passed"
        report["precision"] = "cpu_passed" if args.cpu_only else "passed"
    except Exception:
        report["precision"] = "failed"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        cache = Path(os.environ["TRITON_CACHE_DIR"])
        report["artifacts"] = {str(path): digest(path) for path in cache.rglob("*") if path.is_file()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(precision=report["precision"], cases=len(cases), launches=len(report["launches"]))))


if __name__ == "__main__":
    main()
