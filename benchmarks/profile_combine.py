# SPDX-License-Identifier: Apache-2.0
"""One actual public combine call, or an explicitly labelled C02 diagnostic."""

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_oracle(path):
    spec = importlib.util.spec_from_file_location("combine_independent_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return oracle


def tensor_identity(tensor):
    if tensor is None:
        return None
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "logical_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def run_case(
    source_root, oracle_file, case, mode, edge=False, expected_commit=None, expected_kernel=None, dump_ir=None
):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    from vllm_ascend.ops.triton.dcp import dcp_a2a as module

    actual_source = Path(module.__file__).resolve()
    assert actual_source.is_relative_to(source_root), (actual_source, source_root)
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if expected_commit is not None:
        assert commit == expected_commit, (commit, expected_commit)
    assert len(case) == 6 and case[0] >= 1
    assert not edge or case[0] >= 5, "The reused edge oracle requires T>=5 to avoid overlapping edge rows"
    recv, local, local_lse, expected = load_oracle(oracle_file).make_inputs(case, edge)
    inputs = {
        "recv": tensor_identity(recv),
        "local_output": tensor_identity(local),
        "local_lse": tensor_identity(local_lse),
    }
    launches = []

    def record(kernel, grid, *values, **options):
        if expected_kernel is not None:
            assert kernel.__name__ == expected_kernel, (kernel.__name__, expected_kernel)
        compiled = kernel[grid](*values, **options)
        cache_hash = compiled.metadata.hash
        cache_key = base64.b32encode(bytes.fromhex(cache_hash)).decode().rstrip("=")
        directory = Path(os.environ["TRITON_CACHE_DIR"]) / cache_key
        binaries = list(directory.glob("*.npubin"))
        assert len(binaries) == 1, (directory, binaries)
        entry = {
            "kernel": kernel.__name__,
            "grid": list(grid),
            "options": options,
            "metadata_hash": cache_hash,
            "cache_key": cache_key,
            "binary": str(binaries[0]),
            "binary_sha256": digest(binaries[0]),
            "binary_size": binaries[0].stat().st_size,
        }
        if dump_ir is not None:
            dump_ir.mkdir(parents=True, exist_ok=True)
            artifacts = []
            for path in directory.iterdir():
                if path.is_file():
                    shutil.copy2(path, dump_ir / path.name)
                    artifacts.append({"name": path.name, "sha256": digest(path), "bytes": path.stat().st_size})
            entry["artifacts"] = artifacts
        launches.append(entry)
        return compiled

    original_launch = module._launch_kernel
    module._launch_kernel = record
    try:
        if mode == "public":
            actual = module.fused_dcp_lse_combine(recv, case[2], 1, local_output=local, local_lse=local_lse)
        else:
            assert mode == "direct_c02"
            assert recv.dtype == torch.bfloat16 and recv.is_contiguous()
            module.init_device_properties_triton()
            tokens, heads, dim, world, _, _ = case
            actual = torch.full((tokens, heads, dim), float("nan"), dtype=recv.dtype, device=recv.device)
            record(
                module._fused_dcp_lse_combine_row_weights,
                (min(module.get_vectorcore_num(), tokens * heads),),
                recv,
                actual,
                local if local is not None else recv,
                local_lse if local_lse is not None else recv,
                *(local.stride() if local is not None else (0, 0, 0)),
                *(local_lse.stride()[:2] if local_lse is not None else (0, 0)),
                *recv.stride(),
                *actual.stride(),
                dim,
                heads,
                tokens * heads,
                DCP_SIZE=world,
                HAS_LOCAL=local is not None,
                BLOCK_D=1 << (dim - 1).bit_length(),
                multibuffer=False,
            )
        torch.npu.synchronize()
    finally:
        module._launch_kernel = original_launch
    assert len(launches) == 1, launches
    actual_cpu = actual.cpu()
    error = (actual_cpu.double() - expected).abs()
    finite = bool(torch.isfinite(actual_cpu).all())
    ratio = (error <= 2e-5 + 0.004 * expected.abs()).double().mean().item()
    maximum_error = error.max().item()
    assert finite and ratio >= 0.99 and maximum_error <= 0.02, (finite, ratio, maximum_error)
    profile = getattr(module, "get_current_hardware_profile", lambda: None)()
    return {
        "case": list(case),
        "mode": mode,
        "public_route_claim": mode == "public",
        "edge": edge,
        "application_calls": 1,
        "precision": "passed",
        "finite": finite,
        "match_ratio": ratio,
        "maximum_error": maximum_error,
        "commit": commit,
        "source_root": str(source_root),
        "module": str(actual_source),
        "module_sha256": digest(actual_source),
        "kernel_source_sha256": digest(actual_source.parent / "dcp_a2a_batched.py"),
        "launches": launches,
        "inputs": inputs,
        "output_shape": list(actual.shape),
        "output_stride": list(actual.stride()),
        "vector_cores": module.get_vectorcore_num(),
        "historical_policy_value": getattr(profile, "prefer_row_weighted_dcp_combine", None),
        "oracle_file": str(oracle_file),
        "oracle_sha256": digest(oracle_file),
        "runner_sha256": digest(__file__),
        "scalar_settings": {"scatter_dim": 1, "return_lse": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--oracle-file", type=Path, default=Path(__file__).with_name("oracle_reference.py"))
    parser.add_argument("--case", type=json.loads, required=True)
    parser.add_argument("--mode", choices=["public", "direct_c02"], default="public")
    parser.add_argument("--expect-kernel")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--dump-ir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_case(
        args.source_root,
        args.oracle_file,
        args.case,
        args.mode,
        args.edge,
        args.expected_commit,
        args.expect_kernel,
        args.dump_ir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"precision": report["precision"], "mode": args.mode, "launches": report["launches"]}))


if __name__ == "__main__":
    main()
