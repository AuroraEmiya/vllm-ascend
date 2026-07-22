#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark for MiniMax-M3 decode score and full index-decode paths.

The benchmark separates two timing boundaries:

1. ``score_pipeline``
   Launch only the fused ``_decode_index_score_kernel`` into a preallocated
   score tensor.  Allocation, torch.topk, and the final invalid-index mask are
   excluded.

2. ``full_index``
   Call the public ``minimax_m3_index_decode`` API with a reusable final output
   tensor.  This includes score allocation/launch, torch.topk, output copy, and
   the final invalid-index mask.

The script is intended for release-style modules whose decode-score kernel has
``SCORE_BLOCK_COUNT`` and ``NUM_CHUNKS`` constexpr launch arguments, including
``msa_m3_triton_release_v1.py`` and ``msa_m3_triton_release_v2.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Literal

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


BLOCK_SIZE = 128
DEFAULT_ALIGNMENT = 16
Stage = Literal["score_pipeline", "full_index"]


@dataclass(frozen=True)
class DecodeCase:
    name: str
    seq_lens: tuple[int, ...]
    decode_query_len: int = 1
    heads: int = 4
    head_dim: int = 128
    topk: int = 16
    init_blocks: int = 4
    local_blocks: int = 8
    max_seq_len: int | None = None
    permute_pages: bool = False

    @property
    def resolved_max_seq_len(self) -> int:
        return self.max_seq_len if self.max_seq_len is not None else max(self.seq_lens)


CASES: tuple[DecodeCase, ...] = (
    DecodeCase(
        name="q1_short_4k",
        seq_lens=(4096, 4096, 4096, 4096),
    ),
    DecodeCase(
        name="q1_ragged_8k",
        seq_lens=(129, 4097, 4253, 8191),
        permute_pages=True,
    ),
    DecodeCase(
        name="q4_ragged",
        seq_lens=(131, 1027, 4099),
        decode_query_len=4,
        permute_pages=True,
    ),
    DecodeCase(
        name="q1_init_local_overlap",
        seq_lens=(257, 513),
        heads=2,
        init_blocks=4,
        local_blocks=4,
    ),
    DecodeCase(
        name="q1_long_128k",
        seq_lens=(131072, 131071, 130945, 129001),
        permute_pages=True,
    ),
    # This case magnifies the expected benefit of skipping forced-block QK.
    DecodeCase(
        name="q1_local32_16k",
        seq_lens=(16384, 16384, 16384, 16384),
        local_blocks=32,
    ),
)
CASES_BY_NAME = {case.name: case for case in CASES}


@dataclass
class Payload:
    case: DecodeCase
    idx_q: torch.Tensor
    index_k_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor

    @property
    def total_q(self) -> int:
        return self.idx_q.shape[0]

    @property
    def max_seq_len(self) -> int:
        return self.case.resolved_max_seq_len

    @property
    def max_block_count(self) -> int:
        return ceil_div(self.max_seq_len, BLOCK_SIZE)


@dataclass
class Implementation:
    label: str
    path: Path
    module: ModuleType
    score: torch.Tensor
    index_out: torch.Tensor
    score_stride: int
    num_chunks: int


@dataclass
class Timing:
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    samples_us: list[float]


@dataclass
class Validation:
    score_max_abs: float
    score_max_rel: float
    score_exact_special_mask: bool
    index_exact: bool


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def dtype_from_name(name: str) -> torch.dtype:
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    key = name.strip().lower()
    if key not in choices:
        raise ValueError(f"unsupported dtype {name!r}; use bf16 or fp16")
    return choices[key]


def load_module(path: Path, ordinal: int) -> ModuleType:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    name = f"_msa_index_ab_{ordinal}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_payload(
    case: DecodeCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Payload:
    if case.decode_query_len <= 0:
        raise ValueError("decode_query_len must be positive")
    if any(length < 0 for length in case.seq_lens):
        raise ValueError("seq_lens must be nonnegative")
    if case.resolved_max_seq_len < max(case.seq_lens):
        raise ValueError("max_seq_len cannot be smaller than max(seq_lens)")

    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)

    request_count = len(case.seq_lens)
    total_q = request_count * case.decode_query_len
    max_blocks = max(1, ceil_div(case.resolved_max_seq_len, BLOCK_SIZE))
    num_pages = request_count * max_blocks

    physical_pages = torch.arange(num_pages, dtype=torch.int32, device="cpu")
    if case.permute_pages:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1009)
        physical_pages = physical_pages[
            torch.randperm(num_pages, generator=generator)
        ]

    block_table = physical_pages.reshape(request_count, max_blocks).to(device)
    index_k_cache = torch.randn(
        (num_pages, BLOCK_SIZE, case.head_dim),
        dtype=dtype,
        device=device,
    )
    idx_q = torch.randn(
        (total_q, case.heads, case.head_dim),
        dtype=dtype,
        device=device,
    )
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=device)
    return Payload(case, idx_q, index_k_cache, block_table, seq_lens)


def detect_aicore_count(module: ModuleType) -> int:
    cached = getattr(module, "_detected_aicore_count", None)
    if callable(cached):
        try:
            return max(0, int(cached()))
        except Exception:
            pass

    getter = getattr(module, "get_aicore_num", None)
    initializer = getattr(module, "init_device_properties_triton", None)
    if not callable(getter):
        return 0
    try:
        return max(0, int(getter()))
    except AssertionError:
        if callable(initializer):
            try:
                initializer()
                return max(0, int(getter()))
            except Exception:
                return 0
    except Exception:
        return 0
    return 0


def choose_num_chunks(module: ModuleType, total_q: int, score_stride: int) -> int:
    detected = detect_aicore_count(module)
    fallback = int(
        getattr(module, "DECODE_SCORE_FALLBACK_PROGRAM_COUNT", 32)
    )
    max_chunks = int(getattr(module, "DECODE_SCORE_MAX_CHUNK_COUNT", 256))
    target_programs = detected if detected > 0 else fallback
    return max(
        1,
        min(
            ceil_div(target_programs, max(1, total_q)),
            max_chunks,
            score_stride,
        ),
    )


def prepare_implementation(
    path: Path,
    ordinal: int,
    payload: Payload,
) -> Implementation:
    module = load_module(path, ordinal)
    kernel = getattr(module, "_decode_index_score_kernel", None)
    public = getattr(module, "minimax_m3_index_decode", None)
    if kernel is None or not callable(public):
        raise AttributeError(
            f"{path.name} must expose _decode_index_score_kernel and "
            "minimax_m3_index_decode"
        )

    # Reject incompatible legacy kernels early with a useful error.
    kernel_text = str(inspect.signature(kernel.fn if hasattr(kernel, "fn") else kernel))
    if "SCORE_BLOCK_COUNT" not in kernel_text or "NUM_CHUNKS" not in kernel_text:
        raise TypeError(
            f"{path.name} does not use the release fused-score ABI "
            "(missing SCORE_BLOCK_COUNT/NUM_CHUNKS)"
        )

    alignment = int(
        getattr(module, "SCORE_BLOCK_STRIDE_ALIGNMENT", DEFAULT_ALIGNMENT)
    )
    score_stride = round_up(max(1, payload.max_block_count), alignment)
    score = torch.full(
        (payload.case.heads, payload.total_q, score_stride),
        float("nan"),
        dtype=torch.float32,
        device=payload.idx_q.device,
    )
    index_out = torch.empty(
        (payload.case.heads, payload.total_q, payload.case.topk),
        dtype=torch.int32,
        device=payload.idx_q.device,
    )
    num_chunks = choose_num_chunks(module, payload.total_q, score_stride)
    return Implementation(
        label=path.name,
        path=path,
        module=module,
        score=score,
        index_out=index_out,
        score_stride=score_stride,
        num_chunks=num_chunks,
    )


def launch_score(impl: Implementation, payload: Payload) -> torch.Tensor:
    module = impl.module
    case = payload.case
    module._decode_index_score_kernel[(payload.total_q, impl.num_chunks)](
        payload.idx_q,
        payload.index_k_cache,
        impl.score,
        payload.block_table,
        payload.seq_lens,
        case.heads,
        case.head_dim,
        case.decode_query_len,
        case.init_blocks,
        case.local_blocks,
        payload.idx_q.stride(0),
        payload.idx_q.stride(1),
        payload.idx_q.stride(2),
        payload.index_k_cache.stride(0),
        payload.index_k_cache.stride(1),
        payload.index_k_cache.stride(2),
        impl.score.stride(0),
        impl.score.stride(1),
        impl.score.stride(2),
        payload.block_table.stride(0),
        BLOCK_SIZE_K=int(getattr(module, "SPARSE_BLOCK_SIZE", BLOCK_SIZE)),
        SCORE_BLOCK_COUNT=impl.score_stride,
        NUM_CHUNKS=impl.num_chunks,
    )
    return impl.score


def call_full_index(impl: Implementation, payload: Payload) -> torch.Tensor:
    case = payload.case
    function = impl.module.minimax_m3_index_decode
    signature = inspect.signature(function)
    kwargs: dict[str, Any] = {
        "decode_query_len": case.decode_query_len,
        "max_decode_query_len": case.decode_query_len,
        "out": impl.index_out,
        "sm_scale": case.head_dim ** -0.5,
    }
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(
        payload.idx_q,
        payload.index_k_cache,
        payload.block_table,
        payload.seq_lens,
        payload.max_seq_len,
        case.topk,
        case.init_blocks,
        case.local_blocks,
        case.heads,
        **kwargs,
    )


def special_score_mask(payload: Payload, score_stride: int) -> torch.Tensor:
    """Mask positions whose values are fixed constants or -inf by semantics."""
    case = payload.case
    mask = torch.zeros(
        (case.heads, payload.total_q, score_stride),
        dtype=torch.bool,
        device="cpu",
    )
    for request_id, seq_len in enumerate(case.seq_lens):
        for query_offset in range(case.decode_query_len):
            query_id = request_id * case.decode_query_len + query_offset
            kv_len = max(seq_len - case.decode_query_len + query_offset + 1, 0)
            valid = ceil_div(kv_len, BLOCK_SIZE)
            local_start = max(valid - case.local_blocks, 0)
            for block_id in range(score_stride):
                if (
                    block_id >= valid
                    or block_id < case.init_blocks
                    or block_id >= local_start
                ):
                    mask[:, query_id, block_id] = True
    return mask


def validate_pair(
    baseline: Implementation,
    candidate: Implementation,
    payload: Payload,
    *,
    atol: float,
    rtol: float,
) -> Validation:
    baseline.score.fill_(float("nan"))
    candidate.score.fill_(float("nan"))
    launch_score(baseline, payload)
    launch_score(candidate, payload)
    synchronize(payload.idx_q.device)

    left = baseline.score.detach().cpu()
    right = candidate.score.detach().cpu()
    if torch.isnan(left).any() or torch.isnan(right).any():
        raise AssertionError("NaN remains in score output; score row was not fully owned")
    if not torch.equal(torch.isneginf(left), torch.isneginf(right)):
        raise AssertionError("score -inf masks differ")

    finite = torch.isfinite(left) & torch.isfinite(right)
    difference = (left[finite] - right[finite]).abs()
    if difference.numel():
        denominator = left[finite].abs().clamp_min(1e-12)
        relative = difference / denominator
        max_abs = float(difference.max())
        max_rel = float(relative.max())
        if not torch.allclose(left[finite], right[finite], atol=atol, rtol=rtol):
            worst = int(torch.argmax(difference))
            raise AssertionError(
                "score mismatch: "
                f"baseline={float(left[finite][worst])}, "
                f"candidate={float(right[finite][worst])}, "
                f"abs={float(difference[worst])}, rel={float(relative[worst])}"
            )
    else:
        max_abs = 0.0
        max_rel = 0.0

    special = special_score_mask(payload, baseline.score_stride)
    exact_special = bool(torch.equal(left[special], right[special]))
    if not exact_special:
        raise AssertionError("init/local/tail score constants differ")

    left_idx = call_full_index(baseline, payload)
    right_idx = call_full_index(candidate, payload)
    synchronize(payload.idx_q.device)
    index_exact = bool(torch.equal(left_idx, right_idx))
    if not index_exact:
        mismatch = torch.nonzero(left_idx != right_idx, as_tuple=False)[0].tolist()
        raise AssertionError(f"top-k index mismatch at {mismatch}")

    return Validation(max_abs, max_rel, exact_special, index_exact)


def benchmark_balanced(
    implementations: list[Implementation],
    payload: Payload,
    function: Callable[[Implementation, Payload], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Timing]:
    for impl in implementations:
        for _ in range(warmup):
            function(impl, payload)
    synchronize(payload.idx_q.device)

    samples: dict[str, list[float]] = {impl.label: [] for impl in implementations}
    for repeat in range(repeats):
        order = implementations if repeat % 2 == 0 else list(reversed(implementations))
        for impl in order:
            synchronize(payload.idx_q.device)
            start = time.perf_counter()
            for _ in range(iters):
                function(impl, payload)
            synchronize(payload.idx_q.device)
            samples[impl.label].append((time.perf_counter() - start) * 1e6 / iters)

    results: dict[str, Timing] = {}
    for label, values in samples.items():
        results[label] = Timing(
            mean_us=statistics.fmean(values),
            median_us=statistics.median(values),
            min_us=min(values),
            max_us=max(values),
            std_us=statistics.pstdev(values) if len(values) > 1 else 0.0,
            samples_us=values,
        )
    return results


def qk_block_accounting(case: DecodeCase) -> tuple[int, int, float]:
    """Return old-QK blocks, v2 normal-QK blocks, and theoretical skip ratio."""
    all_valid = 0
    normal = 0
    for seq_len in case.seq_lens:
        for query_offset in range(case.decode_query_len):
            kv_len = max(seq_len - case.decode_query_len + query_offset + 1, 0)
            valid = ceil_div(kv_len, BLOCK_SIZE)
            local_start = max(valid - case.local_blocks, 0)
            normal_count = max(0, local_start - min(case.init_blocks, local_start))
            all_valid += valid
            normal += normal_count
    skipped = all_valid - normal
    ratio = skipped / all_valid if all_valid else 0.0
    return all_valid, normal, ratio


def parse_cases(args: argparse.Namespace) -> list[DecodeCase]:
    if args.custom_seq_lens:
        seq_lens = tuple(int(value) for value in args.custom_seq_lens.split(","))
        return [
            DecodeCase(
                name="custom",
                seq_lens=seq_lens,
                decode_query_len=args.decode_query_len,
                heads=args.heads,
                head_dim=args.head_dim,
                topk=args.topk,
                init_blocks=args.init_blocks,
                local_blocks=args.local_blocks,
                max_seq_len=args.max_seq_len,
                permute_pages=args.permute_pages,
            )
        ]

    names: list[str] = []
    if args.case:
        names.extend(args.case)
    if args.cases:
        names.extend(item.strip() for item in args.cases.split(",") if item.strip())
    if args.all_cases:
        names = [case.name for case in CASES]
    if not names:
        names = ["q1_short_4k", "q1_long_128k", "q4_ragged"]

    unknown = [name for name in names if name not in CASES_BY_NAME]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    result: list[DecodeCase] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            result.append(CASES_BY_NAME[name])
            seen.add(name)
    return result


def print_table(
    case: DecodeCase,
    stage: Stage,
    implementations: list[Implementation],
    timings: dict[str, Timing],
) -> None:
    baseline_mean = timings[implementations[0].label].mean_us
    print(f"\n{case.name} / {stage}")
    print(
        f"{'implementation':36} {'mean_us':>11} {'median_us':>11} "
        f"{'min_us':>11} {'max_us':>11} {'std_us':>10} {'speedup':>9}"
    )
    print("-" * 111)
    for impl in implementations:
        timing = timings[impl.label]
        speedup = baseline_mean / timing.mean_us
        print(
            f"{impl.label:36} {timing.mean_us:11.3f} "
            f"{timing.median_us:11.3f} {timing.min_us:11.3f} "
            f"{timing.max_us:11.3f} {timing.std_us:10.3f} {speedup:8.3f}x"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "msa_m3_triton_release_v1.py",
            "msa_m3_triton_release_v2.py",
        ],
        help="release-style implementation files; first file is speedup baseline",
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--cases", help="comma-separated built-in case names")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("score_pipeline", "full_index", "both"),
        default="both",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path)

    custom = parser.add_argument_group("custom case")
    custom.add_argument("--custom-seq-lens", help="comma-separated sequence lengths")
    custom.add_argument("--decode-query-len", type=int, default=1)
    custom.add_argument("--heads", type=int, default=4)
    custom.add_argument("--head-dim", type=int, default=128)
    custom.add_argument("--topk", type=int, default=16)
    custom.add_argument("--init-blocks", type=int, default=4)
    custom.add_argument("--local-blocks", type=int, default=8)
    custom.add_argument("--max-seq-len", type=int)
    custom.add_argument("--permute-pages", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list_cases:
        for case in CASES:
            old_qk, new_qk, skip = qk_block_accounting(case)
            print(
                f"{case.name:26} seq={case.seq_lens} qlen={case.decode_query_len} "
                f"init={case.init_blocks} local={case.local_blocks} "
                f"QK blocks {old_qk}->{new_qk} skip={skip:.1%}"
            )
        return 0
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be >=0 and iters/repeats must be >0")

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None or not hasattr(torch, "npu"):
            raise RuntimeError("NPU benchmark requires torch_npu")
        if not torch.npu.is_available():
            raise RuntimeError("no available Ascend NPU")
        torch.npu.set_device(device)

    dtype = dtype_from_name(args.dtype)
    cases = parse_cases(args)
    paths = [Path(token).expanduser().resolve() for token in args.files]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    stages: list[Stage]
    if args.stage == "both":
        stages = ["score_pipeline", "full_index"]
    else:
        stages = [args.stage]

    records: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        payload = build_payload(
            case,
            device=device,
            dtype=dtype,
            seed=args.seed + case_index * 17,
        )
        implementations = [
            prepare_implementation(path, ordinal, payload)
            for ordinal, path in enumerate(paths)
        ]

        # All compared release files must use the same score layout.
        strides = {impl.score_stride for impl in implementations}
        if len(strides) != 1:
            raise ValueError(f"score strides differ across implementations: {strides}")

        old_qk, new_qk, skip_ratio = qk_block_accounting(case)
        print(
            f"\nCASE {case.name}: q={tuple(payload.idx_q.shape)} "
            f"cache={tuple(payload.index_k_cache.shape)} "
            f"max_seq_len={payload.max_seq_len} topk={case.topk}"
        )
        print(
            f"QK block accounting per all queries: v1={old_qk}, v2={new_qk}, "
            f"theoretical skipped={old_qk - new_qk} ({skip_ratio:.2%})"
        )
        print(
            "chunks: "
            + ", ".join(f"{impl.label}={impl.num_chunks}" for impl in implementations)
        )

        validation: Validation | None = None
        if args.validate and len(implementations) >= 2:
            validation = validate_pair(
                implementations[0],
                implementations[1],
                payload,
                atol=args.atol,
                rtol=args.rtol,
            )
            print(
                f"VALID {implementations[0].label} vs {implementations[1].label}: "
                f"score_max_abs={validation.score_max_abs:.6g} "
                f"score_max_rel={validation.score_max_rel:.6g} "
                f"special_exact={validation.score_exact_special_mask} "
                f"index_exact={validation.index_exact}"
            )

        for stage in stages:
            function = launch_score if stage == "score_pipeline" else call_full_index
            timings = benchmark_balanced(
                implementations,
                payload,
                function,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
            print_table(case, stage, implementations, timings)
            baseline_mean = timings[implementations[0].label].mean_us
            for impl in implementations:
                record: dict[str, Any] = {
                    "case": asdict(case),
                    "stage": stage,
                    "implementation": impl.label,
                    "path": str(impl.path),
                    "dtype": str(dtype),
                    "device": str(device),
                    "idx_q_shape": list(payload.idx_q.shape),
                    "cache_shape": list(payload.index_k_cache.shape),
                    "score_stride": impl.score_stride,
                    "num_chunks": impl.num_chunks,
                    "qk_blocks_v1": old_qk,
                    "qk_blocks_v2": new_qk,
                    "qk_skip_ratio": skip_ratio,
                    "timing": asdict(timings[impl.label]),
                    "speedup_vs_first": baseline_mean / timings[impl.label].mean_us,
                }
                if validation is not None:
                    record["validation"] = asdict(validation)
                records.append(record)

        del implementations, payload
        if device.type == "npu":
            torch.npu.empty_cache()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
