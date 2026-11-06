# SPDX-License-Identifier: Apache-2.0
"""Independent CPU FP64 reference; never used in the timed NPU pipeline."""
import math
import torch
from .pipeline import QUERY_RANK, KV_RANK, NOPE_DIM, ROPE_DIM, validate_cache_index


def rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


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


def rope_reference(x, sine, cosine):
    base = x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).flatten(-2)
    return base * cosine + rotate_half(base) * sine


def oracle(data, heads, initial_kv, initial_kr):
    query, raw_qr, kv, raw_kr = projections(data, heads)
    sine, cosine = data["sin"].double(), data["cos"].double()
    query_rope = rope_reference(raw_qr, sine.unsqueeze(-2), cosine.unsqueeze(-2))
    kr = rope_reference(raw_kr, sine, cosine)
    output_kv, output_kr = initial_kv.double().clone(), initial_kr.double().clone()
    index = data["index"].flatten()
    output_kv.view(-1, KV_RANK)[index] = kv.flatten(0, 1)
    output_kr.view(-1, ROPE_DIM)[index] = kr.flatten(0, 1)
    return query, query_rope, output_kv, output_kr


def check_outputs(outputs, caches, expected, data, initial_kv, initial_kr):
    results = {}
    names = ("query", "query_rope", "kv_cache_returned", "kr_cache_returned")
    for name, value, reference in zip(names, outputs, expected):
        if list(value.shape) != list(reference.shape) or value.dtype != torch.bfloat16:
            raise AssertionError(f"{name} ABI differs: {value.shape}, {value.dtype}, expected {reference.shape}")
        result = compare(value, reference)
        result["finite"] = bool(torch.isfinite(value.cpu().float()).all())
        result["pass"] = result["matched_ratio"] >= 0.99 and result["max_abs_error"] <= 1 and result["finite"]
        results[name] = result
    slots = initial_kv.numel() // KV_RANK
    touched = torch.zeros(slots, dtype=torch.bool)
    touched[data["index"].flatten()] = True
    for name, value, returned, reference, initial, width in (
        ("kv_cache", caches[0], outputs[2], expected[2], initial_kv, KV_RANK),
        ("kr_cache", caches[1], outputs[3], expected[3], initial_kr, ROPE_DIM),
    ):
        actual = value.cpu()
        written = compare(actual.reshape(-1, width)[touched], reference.reshape(-1, width)[touched])
        untouched_exact = torch.equal(actual.reshape(-1, width)[~touched], initial.reshape(-1, width)[~touched])
        returned_untouched_exact = torch.equal(
            returned.cpu().reshape(-1, width)[~touched], initial.reshape(-1, width)[~touched]
        )
        results[name + "_state"] = {
            "written": written,
            "untouched_exact": untouched_exact,
            "returned_untouched_exact": returned_untouched_exact,
            "returned_aliases_input": returned.data_ptr() == value.data_ptr(),
            "pass": written["matched_ratio"] >= 0.99
            and written["max_abs_error"] <= 1
            and untouched_exact
            and returned_untouched_exact,
        }
    results["pass"] = all(result["pass"] for result in results.values())
    return results


def build_case(case):
    data = make_data(case["batch"], case["hidden"], case["heads"], case["seed"])
    if case.get("zeros"):
        data["x"].zero_()
    if case.get("zero_gamma"):
        data["gq"].zero_()
        data["gkv"].zero_()
    block_size = case.get("block_size", 16)
    slots = math.ceil(max(block_size, case["batch"] * 3 + 2) / block_size) * block_size
    if case.get("index_mode") == "reverse":
        data["index"] = (slots - 1 - data["index"]).contiguous()
    if case.get("index_mode") == "edges":
        data["index"] = torch.linspace(0, slots - 1, case["batch"], dtype=torch.int64).reshape(-1, 1)
    validate_cache_index(data["index"], slots)
    generator = torch.Generator().manual_seed(case["seed"] + 100)
    initial_kv = torch.randn((slots // block_size, block_size, 1, KV_RANK), generator=generator).to(torch.bfloat16)
    initial_kr = torch.randn((slots // block_size, block_size, 1, ROPE_DIM), generator=generator).to(torch.bfloat16)
    return data, initial_kv, initial_kr
