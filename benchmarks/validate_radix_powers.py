# SPDX-License-Identifier: Apache-2.0
"""Validation-only power-of-two row sweep of unchanged candidate-pack kernels.

verify mode records actual public dispatch and checks public/direct outputs.
profile mode invokes only the direct production kernel, checking every output.
Timing comes exclusively from external msprof op; no production code is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SHA = {
    "main": "49857caf73aecdd119d276fe807108d2777ea36f20adefad3067938f080f9de3",
    "fixed20": "c1ceb44e21f6402ec706562797b3c4aa5d4c211d69a780ab6e4773cb4ab9f644",
}
ROWS = (2, 4, 8, 16, 32, 64, 128, 256)
K, WORLD, RANK, INTERLEAVE, SEED = 2048, 8, 7, 768, 20878


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def tensor_sha(torch, tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def inputs(torch, rows, dtype):
    id_generator = torch.Generator().manual_seed(SEED)
    score_generator = torch.Generator().manual_seed(SEED + 1)
    ids = torch.stack([torch.randperm(K, generator=id_generator, dtype=torch.int64) * 509 for _ in range(rows)]).to(torch.int32)
    ids[:, -1] = (1 << 20) - 1
    ids[:, 7] = -1
    ids = ids.view(rows, 1, K)
    scores = torch.randint(-3, 4, ids.shape, generator=score_generator).to(dtype)
    scores[..., :7] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0., 1., -1.], dtype=dtype)
    scores[..., 8] = torch.nextafter(torch.tensor(1., dtype=dtype), torch.tensor(2., dtype=dtype))
    scores[..., -1] = 1.
    # CPU oracle independently sorts numeric scores descending and IDs ascending.
    valid = (ids >= 0) & torch.isfinite(scores)
    values = torch.where(valid, scores.float(), -float("inf"))
    key_ids = torch.where(valid, ids, torch.iinfo(torch.int32).max)
    by_id = torch.argsort(key_ids, stable=True, dim=-1)
    by_score = torch.argsort(values.gather(-1, by_id), descending=True, stable=True, dim=-1)
    order = by_id.gather(-1, by_score)
    mapped = ids + (torch.div(ids, INTERLEAVE, rounding_mode="floor") * (WORLD - 1) + RANK) * INTERLEAVE
    mapped = torch.where(valid, mapped, -1)
    oracle = torch.stack((mapped.gather(-1, order), values.gather(-1, order).contiguous().view(torch.int32)), -1)[:, 0]
    identity = {"ids_sha256": tensor_sha(torch, ids), "scores_sha256": tensor_sha(torch, scores),
                "oracle_sha256": tensor_sha(torch, oracle), "id_max": int(ids.max()),
                "id_min": int(ids.min()), "valid_id_min": int(ids[ids >= 0].min()),
                "valid_unique_ids_per_row": [int(torch.unique(row[row >= 0]).numel()) for row in ids],
                "score_distribution": "seeded integer ties -3..3; nan/+inf/-inf/+0/-0/nextafter(1)/invalid-id probes"}
    return ids.to("npu"), scores.to("npu"), oracle, identity


def compiled_record(kernel, directory):
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, value in getattr(kernel, "asm", {}).items():
        if not isinstance(value, (str, bytes)):
            continue
        data = value.encode() if isinstance(value, str) else value
        filename = "compiled." + name.replace("/", "_")
        (directory / filename).write_bytes(data)
        files[name] = {"file": filename, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {"hash": str(getattr(kernel, "hash", None)),
            "metadata": str(getattr(kernel, "metadata", None)), "files": files}


class LaunchRecorder:
    """Observe one actual wrapper call while forwarding the original launch."""
    def __init__(self, kernel):
        self.kernel, self.calls = kernel, []

    def __getitem__(self, grid):
        original = self.kernel[grid]
        def launch(*args, **kwargs):
            self.calls.append({"grid": list(grid), "kwargs": kwargs})
            return original(*args, **kwargs)
        return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(EXPECTED_SHA), required=True)
    parser.add_argument("--mode", choices=("verify", "profile"), required=True)
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--dtype", choices=("fp32", "bf16", "both"), default="both")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_values = [int(value) for value in args.rows.split(",")]
    assert rows_values and len(set(rows_values)) == len(rows_values) and set(rows_values) <= set(ROWS)
    dtypes = ("fp32", "bf16") if args.dtype == "both" else (args.dtype,)
    assert args.samples == 10, "Formal profile uses ten real application invocations"
    assert args.mode != "profile" or (len(rows_values) == 1 and len(dtypes) == 1)
    import torch
    import torch_npu
    from vllm_ascend.ops.triton import sharded_indexer as op
    source = Path(op.__file__).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == EXPECTED_SHA[args.variant], (str(source), source_sha)
    torch.npu.set_device(0)
    op.init_device_properties_triton()
    cores = op.get_vectorcore_num()
    kernel = op._pack_sorted_candidates
    report = {"variant": args.variant, "mode": args.mode, "source": str(source), "source_sha256": source_sha,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "soc_version": int(torch_npu._C._npu_get_soc_version()),
              "device_name": torch.npu.get_device_name(0), "vector_cores": cores,
              "K": K, "WORLD": WORLD, "RANK": RANK, "INTERLEAVE": INTERLEAVE,
              "seeds": {"ids": SEED, "scores": SEED + 1},
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing": "external msprof op only; no host timing", "cases": []}
    save(args.output, report)
    for rows in rows_values:
        for name in dtypes:
            dtype = torch.float32 if name == "fp32" else torch.bfloat16
            ids, scores, oracle, identity = inputs(torch, rows, dtype)
            packed = torch.empty((rows, K, 2), dtype=torch.int32, device="npu")
            kwargs = {"multibuffer": False}
            if args.variant == "fixed20":
                kwargs["SCORE_BITS"] = 32 if name == "fp32" else 16
            row = {"rows": rows, "dtype": name, "input_identity": identity,
                   "ids_stride": list(ids.stride()), "scores_stride": list(scores.stride()),
                   "grid": [min(rows, cores)], "launch_kwargs": kwargs,
                   "direct_kernel": "_pack_sorted_candidates", "checked_direct_calls": 0}
            report["cases"].append(row)
            if args.mode == "verify":
                proxy = LaunchRecorder(kernel)
                op._pack_sorted_candidates = proxy
                try:
                    public = op.pack_local_candidates(ids, scores, WORLD, RANK, INTERLEAVE,
                                                      sort_candidates=True, fuse_sort=True)
                    torch.npu.synchronize()
                    assert torch.equal(public.cpu(), oracle), "public CPU oracle mismatch"
                finally:
                    op._pack_sorted_candidates = kernel
                row["public_fused"] = bool(proxy.calls)
                row["public_sorted_kernel_launches"] = proxy.calls
                row["public_cpu_oracle_bit_exact"] = True
            for sample in range(args.samples if args.mode == "profile" else 1):
                compiled = kernel[(min(rows, cores),)](
                    ids, scores, packed, rows, ids.stride(0), ids.stride(-1),
                    scores.stride(0), scores.stride(-1), WORLD, RANK, INTERLEAVE, 11, **kwargs)
                torch.npu.synchronize()
                assert torch.equal(packed.cpu(), oracle), f"direct CPU oracle mismatch: {rows}/{name}/{sample}"
                row["checked_direct_calls"] += 1
                if sample == 0:
                    row["compiled"] = compiled_record(compiled, args.output.parent / (args.output.stem + "-compiled") / f"r{rows}-{name}")
                save(args.output, report)
            row["cpu_oracle_bit_exact"] = True
            save(args.output, report)
            print(json.dumps({"rows": rows, "dtype": name, "checked_direct_calls": row["checked_direct_calls"],
                              "public_fused": row.get("public_fused"), "grid": row["grid"]}), flush=True)
    report["complete"] = True
    save(args.output, report)


if __name__ == "__main__":
    main()
