# SPDX-License-Identifier: Apache-2.0
"""Deterministic public-API runner for paired ``msprof op`` DCP pack captures.

Run this unchanged with either the baseline or candidate package on PYTHONPATH.
The output JSON records precision and compiled-kernel identity, never timing.
Timing comes from the profiler's retained CSVs, separately for each variant.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.dcp import dcp_a2a

CASES = {
    "actual144": (144, 96, 512, "contiguous"),
    "actual152": (152, 96, 512, "contiguous"),
    "actual114": (114, 64, 512, "contiguous"),
    "tail7": (7, 64, 512, "contiguous"),
    "tail129": (129, 64, 512, "contiguous"),
    "dim256": (41, 32, 256, "contiguous"),
    "dim2048": (41, 32, 2048, "contiguous"),
    "feature144": (144, 96, 512, "feature_stride"),
    "head144": (144, 96, 512, "head_major"),
}


def make_layout(value, layout):
    if layout == "feature_stride":
        return torch.stack((value, value), -1).npu()[..., 0]
    if layout == "head_major":
        return value.transpose(0, 1).contiguous().npu().transpose(0, 1)
    return value.npu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--from-stats", action="store_true")
    parser.add_argument("--invocations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.invocations < 20:
        parser.error("at least 20 invocations are required for warm-up=10, launch-count=10")
    torch.npu.set_device(0)
    tokens, heads, dim, layout = CASES[args.case]
    generator = torch.Generator().manual_seed(21586)
    output = torch.randn((tokens, heads, dim), generator=generator).bfloat16()
    output.flatten()[:5] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=torch.bfloat16)
    lse = torch.randn((tokens, heads, 1), generator=generator) * 8
    lse.flatten()[:8] = torch.tensor([0.0, -0.0, torch.inf, -torch.inf, torch.nan, 70_000.0, -70_000.0, 1e-40])
    sums = torch.rand((tokens, heads, 1), generator=generator) + 0.5
    device_output = make_layout(output, layout)
    device_lse = make_layout(lse, layout)
    device_sums = make_layout(sums, "feature_stride") if args.from_stats else None
    compiled = []
    original = dcp_a2a._launch_kernel

    def record(kernel, grid, *inputs, **options):
        result = kernel[grid](*inputs, **options)
        if not compiled:
            asm = getattr(result, "asm", {})
            compiled.append(
                {
                    "name": kernel.__name__,
                    "hash": getattr(result, "hash", None),
                    "grid": grid,
                    "options": options,
                    "asm_sha256": {
                        key: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
                        for key, value in asm.items()
                        if isinstance(value, (str, bytes))
                    },
                }
            )
        return result

    dcp_a2a._launch_kernel = record
    try:
        packed = None
        for _ in range(args.invocations):
            packed = dcp_a2a.pack_dcp_output_lse(device_output, device_lse, 8, 1, softmax_sum=device_sums)
        torch.npu.synchronize()
    finally:
        dcp_a2a._launch_kernel = original
    wire = packed.cpu().reshape(heads, tokens, dim + 4).transpose(0, 1).contiguous()
    payload_exact = torch.equal(wire[..., :dim].contiguous().view(torch.int16), output.view(torch.int16))
    fields = wire[..., dim:].float()
    code, high, middle, low = fields.unbind(-1)
    significand = high.double() * 65536 + middle.double() * 256 + low.double()
    decoded = code.sign().double() * torch.ldexp(significand, code.abs().int() - 151)
    expected = lse[..., 0].double()
    if args.from_stats:
        expected = expected + sums[..., 0].double().log()
    finite = expected.isfinite()
    sentinel_exact = torch.equal(code == 0, ~finite)
    errors = (decoded[finite] - expected[finite]).abs()
    matched = errors <= 2**-16 + 2**-10 * expected[finite].abs()
    match_ratio = matched.double().mean().item()
    max_error = errors.max().item()
    wire_exact = None
    if not args.from_stats:
        raw = lse[..., 0].contiguous().view(torch.int32).long() & 0xFFFFFFFF
        exponent = (raw >> 23) & 255
        fraction = raw & 0x7FFFFF
        sig = fraction | torch.where(exponent == 0, 0, 0x800000)
        sig = torch.where(finite, sig, 0)
        exp_code = (exponent.clamp_min(1) + 1) * torch.where(raw >> 31 == 0, 1, -1)
        exp_code = torch.where(raw & 0x7FFFFFFF == 0, 128, exp_code)
        exp_code = torch.where(finite, exp_code, 0)
        expected_fields = torch.stack((exp_code, (sig >> 16) & 255, (sig >> 8) & 255, sig & 255), -1).float()
        wire_exact = torch.equal(fields, expected_fields)
    passed = payload_exact and sentinel_exact and match_ratio >= 0.99 and max_error <= 0.01
    if wire_exact is not None:
        passed = passed and wire_exact
    source = Path(dcp_a2a.__file__).resolve()
    report = {
        "case": args.case,
        "shape": [tokens, heads, dim],
        "layout": layout,
        "from_stats": args.from_stats,
        "invocations": args.invocations,
        "package_source": str(source),
        "package_commit": subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "strides": {"output": list(device_output.stride()), "lse": list(device_lse.stride())},
        "compiled": compiled,
        "precision": {
            "passed": passed,
            "payload_bit_exact": payload_exact,
            "sentinel_exact": sentinel_exact,
            "wire_bit_exact": wire_exact,
            "lse_match_ratio": match_ratio,
            "lse_max_abs_error": max_error,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("public API pack precision failed")


if __name__ == "__main__":
    main()
