# SPDX-License-Identifier: Apache-2.0
"""Black-box ABI probe using only the published MlaProlog equations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch_npu

QUERY_RANK = 1536
KV_RANK = 512
NOPE_DIM = 128
ROPE_DIM = 64


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def rotate_interleaved(x):
    paired = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)


def rms(x, gamma, eps=1e-5):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma


def compare(actual, expected):
    value = actual.cpu().double()
    error = (value - expected).abs()
    allowed = 2**-6 + 2**-6 * expected.abs()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_error": error.max().item() if error.numel() else 0.0,
        "matched_ratio": (error <= allowed).double().mean().item() if error.numel() else 1.0,
    }


def make_data(batch=1, hidden=7168, heads=32, seed=20261102):
    generator = torch.Generator().manual_seed(seed)

    def normal(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    angles = torch.randn((batch, 1, ROPE_DIM // 2), generator=generator)
    return {
        "x": normal((batch, 1, hidden)),
        "wdq": normal((hidden, QUERY_RANK), hidden**-0.5),
        "wuq": normal((QUERY_RANK, heads * (NOPE_DIM + ROPE_DIM)), QUERY_RANK**-0.5),
        "wuk": normal((heads, NOPE_DIM, KV_RANK), NOPE_DIM**-0.5),
        "wdkv": normal((hidden, KV_RANK + ROPE_DIM), hidden**-0.5),
        "gq": normal((QUERY_RANK,), 0.1) + 1,
        "gkv": normal((KV_RANK,), 0.1) + 1,
        "sin": torch.cat((angles.sin(), angles.sin()), -1).to(torch.bfloat16),
        "cos": torch.cat((angles.cos(), angles.cos()), -1).to(torch.bfloat16),
        "index": (torch.arange(batch, dtype=torch.int64) * 3 + 1).view(batch, 1),
    }


def projections(data, heads):
    x, wdq, wuq, wuk, wdkv = (data[name].double() for name in ("x", "wdq", "wuq", "wuk", "wdkv"))
    cq = rms(x @ wdq, data["gq"].double())
    up = (cq @ wuq).reshape(x.shape[0], 1, heads, NOPE_DIM + ROPE_DIM)
    query = torch.einsum("bsnd,ndh->bsnh", up[..., :NOPE_DIM], wuk)
    kvkr = x @ wdkv
    kv = rms(kvkr[..., :KV_RANK], data["gkv"].double())
    return query, up[..., NOPE_DIM:], kv, kvkr[..., KV_RANK:]


def run_probe(args):
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    data = make_data(args.batch, args.hidden, args.heads)
    device = {name: tensor.npu() for name, tensor in data.items()}
    weights_nz = {name: torch_npu.npu_format_cast(device[name], 29) for name in ("wdq", "wuq", "wdkv")}
    slots = math.ceil(max(16, args.batch * 3 + 2) / 16) * 16
    cache_shape = (slots // 16, 16, 1)
    kv = torch.full((*cache_shape, KV_RANK), 0.125, dtype=torch.bfloat16, device="npu")
    kr = torch.full((*cache_shape, ROPE_DIM), -0.125, dtype=torch.bfloat16, device="npu")
    outputs = torch_npu.npu_mla_prolog(
        device["x"],
        weights_nz["wdq"],
        weights_nz["wuq"],
        device["wuk"],
        weights_nz["wdkv"],
        device["gq"],
        device["gkv"],
        device["sin"],
        device["cos"],
        device["index"],
        kv,
        kr,
    )
    torch.npu.synchronize()
    query, qr, expected_kv, raw_kr = projections(data, args.heads)
    cache_kv = kv.cpu().reshape(-1, KV_RANK)[data["index"].flatten()]
    cache_kr = kr.cpu().reshape(-1, ROPE_DIM)[data["index"].flatten()]
    qsin, qcos = data["sin"].double().unsqueeze(-2), data["cos"].double().unsqueeze(-2)
    variants = {}
    for name, transform, rotate in (
        ("half", lambda x: x, rotate_half),
        ("interleaved", lambda x: x, rotate_interleaved),
        ("pair_to_half", lambda x: x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).flatten(-2), rotate_half),
    ):
        qbase, kbase = transform(qr), transform(raw_kr)
        qexpected = qbase * qcos + rotate(qbase) * qsin
        kexpected = kbase * data["cos"].double() + rotate(kbase) * data["sin"].double()
        variants[name] = {
            "query_rope": compare(outputs[1], qexpected),
            "kr_cache_written": compare(cache_kr, kexpected.flatten(0, 1)),
        }
    touched = torch.zeros(slots, dtype=torch.bool)
    touched[data["index"].flatten()] = True
    maps = Path("/proc/self/maps").read_text().splitlines()
    libs = sorted(
        {row.split()[-1] for row in maps if row.split()[-1].endswith(".so") and ("opapi" in row or "torch_npu" in row)}
    )
    return {
        "schema": str(torch.ops.npu.npu_mla_prolog.default._schema),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "batch": args.batch,
        "hidden": args.hidden,
        "heads": args.heads,
        "outputs": [
            {
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "aliases_kv": out.data_ptr() == kv.data_ptr(),
                "aliases_kr": out.data_ptr() == kr.data_ptr(),
            }
            for out in outputs
        ],
        "query": compare(outputs[0], query),
        "kv_cache_written": compare(cache_kv, expected_kv.flatten(0, 1)),
        "returned_kv_matches_input": torch.equal(outputs[2].cpu(), kv.cpu()),
        "returned_kr_matches_input": torch.equal(outputs[3].cpu(), kr.cpu()),
        "rope_variants": variants,
        "kv_unwritten_exact": bool((kv.cpu().reshape(-1, KV_RANK)[~touched] == 0.125).all()),
        "kr_unwritten_exact": bool((kr.cpu().reshape(-1, ROPE_DIM)[~touched] == -0.125).all()),
        "loaded_libraries": [
            {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in libs
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--heads", type=int, default=32)
    args = parser.parse_args()
    result = run_probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "loaded_libraries"}))
