#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness and performance harness for standalone MiniMax-M3 score files.

Capabilities
------------
* one Python implementation file or multiple files in the same run;
* prefill, decode, or both paths;
* one named case, repeated ``--case`` selections, comma-separated ``--cases``,
  or all registered cases;
* independent PyTorch FP32 references;
* reusable output/workspace buffers;
* average timing with warmup, inner iterations, and repeated samples;
* JSON output suitable for later A3 tuning comparisons.

Every implementation file must expose ``prefill_score`` and/or ``decode_score``
with the standalone API used by ``index_score_a3_baseline.py``.  For fair
benchmarking, the selected function must accept an ``out`` tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Literal

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


BLOCK_SIZE = 128
DEFAULT_ALIGNMENT = 16
Mode = Literal["prefill", "decode"]


@dataclass(frozen=True)
class PrefillSpec:
    name: str
    q_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    heads: int
    head_dim: int
    permute_pages: bool = False

    @property
    def mode(self) -> Mode:
        return "prefill"

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(
            prefix + query
            for prefix, query in zip(self.prefix_lens, self.q_lens)
        )


@dataclass(frozen=True)
class DecodeSpec:
    name: str
    seq_lens: tuple[int, ...]
    decode_query_len: int
    heads: int
    head_dim: int
    init_blocks: int
    local_blocks: int
    permute_pages: bool = False

    @property
    def mode(self) -> Mode:
        return "decode"


CaseSpec = PrefillSpec | DecodeSpec


CASE_SPECS: tuple[CaseSpec, ...] = (
    PrefillSpec(
        name="prefill_aligned_small",
        q_lens=(128,),
        prefix_lens=(1024,),
        heads=2,
        head_dim=128,
    ),
    PrefillSpec(
        name="prefill_unaligned_ragged",
        q_lens=(17, 73, 129),
        prefix_lens=(0, 511, 2049),
        heads=2,
        head_dim=128,
    ),
    PrefillSpec(
        name="prefill_permuted_pages",
        q_lens=(96, 64),
        prefix_lens=(1003, 4097),
        heads=4,
        head_dim=128,
        permute_pages=True,
    ),
    PrefillSpec(
        name="prefill_scalar_d1",
        q_lens=(96, 64),
        prefix_lens=(1024, 2050),
        heads=2,
        head_dim=1,
        permute_pages=True,
    ),
    PrefillSpec(
        name="prefill_long_context",
        q_lens=(256,),
        prefix_lens=(8192,),
        heads=4,
        head_dim=128,
        permute_pages=True,
    ),
    DecodeSpec(
        name="decode_zero_short",
        seq_lens=(0, 1, 127, 128),
        decode_query_len=1,
        heads=2,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
    ),
    DecodeSpec(
        name="decode_aligned_q1",
        seq_lens=(4096, 4096, 4096, 4096),
        decode_query_len=1,
        heads=4,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
    ),
    DecodeSpec(
        name="decode_unaligned_ragged_q1",
        seq_lens=(129, 4097, 4253, 8191),
        decode_query_len=1,
        heads=4,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
    ),
    DecodeSpec(
        name="decode_q4_ragged",
        seq_lens=(131, 1027, 4099),
        decode_query_len=4,
        heads=4,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
    ),
    DecodeSpec(
        name="decode_permuted_pages",
        seq_lens=(6145, 8193),
        decode_query_len=1,
        heads=4,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
        permute_pages=True,
    ),
    DecodeSpec(
        name="decode_init_local_overlap",
        seq_lens=(257, 513),
        decode_query_len=1,
        heads=2,
        head_dim=128,
        init_blocks=4,
        local_blocks=4,
    ),
    DecodeSpec(
        name="decode_long_context",
        seq_lens=(131072, 131071, 130945, 129001),
        decode_query_len=1,
        heads=4,
        head_dim=128,
        init_blocks=4,
        local_blocks=8,
        permute_pages=True,
    ),
)

CASES_BY_NAME = {case.name: case for case in CASE_SPECS}


@dataclass
class CasePayload:
    spec: CaseSpec
    dtype: torch.dtype
    device: torch.device
    idx_q: torch.Tensor
    index_k_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    cu_seqlens_q: torch.Tensor | None = None
    prefix_lens: torch.Tensor | None = None
    max_query_len: int | None = None

    @property
    def mode(self) -> Mode:
        return self.spec.mode

    @property
    def total_query_tokens(self) -> int:
        return self.idx_q.shape[0]

    @property
    def heads(self) -> int:
        return self.idx_q.shape[1]

    @property
    def head_dim(self) -> int:
        return self.idx_q.shape[2]

    @property
    def max_block_count(self) -> int:
        return _ceil_div(self.max_seq_len, BLOCK_SIZE)


@dataclass
class RunState:
    label: str
    module: ModuleType
    function: Any
    output: torch.Tensor
    workspace: dict[str, torch.Tensor] | None
    score_stride: int


@dataclass
class ValidationResult:
    max_abs: float
    max_rel: float
    compared_values: int
    negative_infinity_values: int


@dataclass
class TimingResult:
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    samples_us: list[float]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if normalized not in choices:
        raise ValueError(
            f"unsupported dtype {name!r}; choose bf16 or fp16"
        )
    return choices[normalized]


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu is unavailable")
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_block_table_and_cache(
    *,
    request_count: int,
    max_seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    permute_pages: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_blocks = max(1, _ceil_div(max_seq_len, BLOCK_SIZE))
    num_pages = request_count * max_blocks
    physical = torch.arange(
        num_pages,
        dtype=torch.int32,
        device="cpu",
    )
    if permute_pages:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 1009)
        physical = physical[torch.randperm(num_pages, generator=generator)]
    block_table = physical.reshape(request_count, max_blocks).to(device)
    index_k_cache = torch.randn(
        (num_pages, BLOCK_SIZE, head_dim),
        dtype=dtype,
        device=device,
    )
    return block_table, index_k_cache


def build_case(
    spec: CaseSpec,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> CasePayload:
    _seed_everything(seed, device)
    if isinstance(spec, PrefillSpec):
        if len(spec.q_lens) != len(spec.prefix_lens):
            raise ValueError(f"invalid prefill spec {spec.name}")
        request_count = len(spec.q_lens)
        seq_lens_tuple = spec.seq_lens
        max_seq_len = max(seq_lens_tuple)
        total_q = sum(spec.q_lens)
        cumulative = [0]
        for query_length in spec.q_lens:
            cumulative.append(cumulative[-1] + query_length)
        idx_q = torch.randn(
            (total_q, spec.heads, spec.head_dim),
            dtype=dtype,
            device=device,
        )
        block_table, index_k_cache = _build_block_table_and_cache(
            request_count=request_count,
            max_seq_len=max_seq_len,
            head_dim=spec.head_dim,
            dtype=dtype,
            device=device,
            permute_pages=spec.permute_pages,
            seed=seed,
        )
        return CasePayload(
            spec=spec,
            dtype=dtype,
            device=device,
            idx_q=idx_q,
            index_k_cache=index_k_cache,
            block_table=block_table,
            seq_lens=torch.tensor(
                seq_lens_tuple,
                dtype=torch.int32,
                device=device,
            ),
            max_seq_len=max_seq_len,
            cu_seqlens_q=torch.tensor(
                cumulative,
                dtype=torch.int32,
                device=device,
            ),
            prefix_lens=torch.tensor(
                spec.prefix_lens,
                dtype=torch.int32,
                device=device,
            ),
            max_query_len=max(spec.q_lens),
        )

    request_count = len(spec.seq_lens)
    max_seq_len = max(spec.seq_lens)
    total_q = request_count * spec.decode_query_len
    idx_q = torch.randn(
        (total_q, spec.heads, spec.head_dim),
        dtype=dtype,
        device=device,
    )
    block_table, index_k_cache = _build_block_table_and_cache(
        request_count=request_count,
        max_seq_len=max_seq_len,
        head_dim=spec.head_dim,
        dtype=dtype,
        device=device,
        permute_pages=spec.permute_pages,
        seed=seed,
    )
    return CasePayload(
        spec=spec,
        dtype=dtype,
        device=device,
        idx_q=idx_q,
        index_k_cache=index_k_cache,
        block_table=block_table,
        seq_lens=torch.tensor(
            spec.seq_lens,
            dtype=torch.int32,
            device=device,
        ),
        max_seq_len=max_seq_len,
    )


def _load_module(token: str, ordinal: int) -> tuple[str, ModuleType]:
    path = Path(token).expanduser()
    if path.suffix == ".py" or path.exists():
        if not path.exists():
            raise FileNotFoundError(f"implementation file not found: {path}")
        digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
        module_name = f"_index_score_impl_{ordinal}_{digest}"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot import implementation file: {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return path.name, module
    module = importlib.import_module(token)
    return token, module


def load_implementations(
    tokens: Iterable[str],
    selected_modes: set[Mode],
) -> list[tuple[str, ModuleType]]:
    loaded: list[tuple[str, ModuleType]] = []
    for ordinal, token in enumerate(tokens):
        label, module = _load_module(token, ordinal)
        for mode in selected_modes:
            function_name = f"{mode}_score"
            if not callable(getattr(module, function_name, None)):
                raise AttributeError(
                    f"{label} does not expose callable {function_name}"
                )
            signature = inspect.signature(getattr(module, function_name))
            if "out" not in signature.parameters:
                raise TypeError(
                    f"{label}.{function_name} must accept out= for fair timing"
                )
        loaded.append((label, module))
    if not loaded:
        raise ValueError("at least one implementation file is required")
    return loaded


def _score_stride_for_module(
    module: ModuleType,
    max_block_count: int,
) -> int:
    alignment = int(
        getattr(module, "SCORE_BLOCK_STRIDE_ALIGNMENT", DEFAULT_ALIGNMENT)
    )
    if alignment <= 0:
        raise ValueError(
            f"invalid SCORE_BLOCK_STRIDE_ALIGNMENT={alignment}"
        )
    return _round_up(max(1, max_block_count), alignment)


def _prepare_state(
    label: str,
    module: ModuleType,
    payload: CasePayload,
) -> RunState:
    function = getattr(module, f"{payload.mode}_score")
    score_stride = _score_stride_for_module(
        module,
        payload.max_block_count,
    )
    output = torch.full(
        (payload.heads, payload.total_query_tokens, score_stride),
        float("nan"),
        dtype=torch.float32,
        device=payload.device,
    )
    workspace: dict[str, torch.Tensor] | None = None
    if payload.mode == "decode":
        maker = getattr(module, "make_decode_workspace", None)
        if callable(maker):
            workspace = maker(
                total_query_tokens=payload.total_query_tokens,
                score_block_stride=score_stride,
                device=payload.device,
            )
    else:
        maker = getattr(module, "make_prefill_workspace", None)
        if callable(maker):
            workspace = maker(
                payload.index_k_cache,
                head_dim=payload.head_dim,
            )
    return RunState(
        label=label,
        module=module,
        function=function,
        output=output,
        workspace=workspace,
        score_stride=score_stride,
    )


def _supported_kwargs(function: Any, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {
        key: value
        for key, value in values.items()
        if key in signature.parameters
    }


def invoke(state: RunState, payload: CasePayload) -> torch.Tensor:
    common_optional = {
        "num_kv_heads": payload.heads,
        "sm_scale": payload.head_dim ** -0.5,
        "out": state.output,
        "workspace": state.workspace,
    }
    if payload.mode == "prefill":
        assert payload.cu_seqlens_q is not None
        assert payload.prefix_lens is not None
        assert payload.max_query_len is not None
        optional = _supported_kwargs(state.function, common_optional)
        result = state.function(
            payload.idx_q,
            payload.index_k_cache,
            payload.block_table,
            payload.cu_seqlens_q,
            payload.seq_lens,
            payload.prefix_lens,
            payload.max_query_len,
            payload.max_seq_len,
            **optional,
        )
    else:
        assert isinstance(payload.spec, DecodeSpec)
        optional = _supported_kwargs(
            state.function,
            {
                **common_optional,
                "max_decode_query_len": payload.spec.decode_query_len,
            },
        )
        result = state.function(
            payload.idx_q,
            payload.index_k_cache,
            payload.block_table,
            payload.seq_lens,
            payload.max_seq_len,
            payload.spec.init_blocks,
            payload.spec.local_blocks,
            payload.spec.decode_query_len,
            **optional,
        )
    if result.data_ptr() != state.output.data_ptr():
        raise RuntimeError(
            f"{state.label}.{payload.mode}_score did not reuse the supplied out"
        )
    return result


def _prefill_reference(
    payload: CasePayload,
    score_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(payload.spec, PrefillSpec)
    assert payload.cu_seqlens_q is not None
    assert payload.prefix_lens is not None

    query = payload.idx_q.detach().cpu().float()
    cache = payload.index_k_cache.detach().cpu().float()
    block_table = payload.block_table.detach().cpu().long()
    cu = payload.cu_seqlens_q.detach().cpu().long()
    seq_lens = payload.seq_lens.detach().cpu().long()
    prefix_lens = payload.prefix_lens.detach().cpu().long()

    expected = torch.full(
        (payload.heads, payload.total_query_tokens, score_stride),
        float("nan"),
        dtype=torch.float32,
    )
    defined = torch.zeros_like(expected, dtype=torch.bool)

    for request_id in range(len(payload.spec.q_lens)):
        sequence_start = int(cu[request_id])
        query_length = int(cu[request_id + 1] - cu[request_id])
        sequence_length = int(seq_lens[request_id])
        prefix_length = int(prefix_lens[request_id])
        for query_offset in range(query_length):
            query_id = sequence_start + query_offset
            visible_key_end = min(
                sequence_length,
                prefix_length + query_offset + 1,
            )
            visible_blocks = _ceil_div(visible_key_end, BLOCK_SIZE)
            for block_id in range(visible_blocks):
                page_id = int(block_table[request_id, block_id])
                key_start = block_id * BLOCK_SIZE
                valid_positions = min(
                    BLOCK_SIZE,
                    visible_key_end - key_start,
                    sequence_length - key_start,
                )
                if valid_positions <= 0:
                    continue
                keys = cache[page_id, :valid_positions]
                for head_id in range(payload.heads):
                    score = torch.mv(
                        keys,
                        query[query_id, head_id],
                    ).max()
                    expected[head_id, query_id, block_id] = score
                    defined[head_id, query_id, block_id] = True
    return expected, defined


def _decode_reference(
    payload: CasePayload,
    score_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(payload.spec, DecodeSpec)
    spec = payload.spec
    query = payload.idx_q.detach().cpu().float()
    cache = payload.index_k_cache.detach().cpu().float()
    block_table = payload.block_table.detach().cpu().long()
    seq_lens = payload.seq_lens.detach().cpu().long()

    expected = torch.full(
        (payload.heads, payload.total_query_tokens, score_stride),
        float("nan"),
        dtype=torch.float32,
    )
    defined = torch.zeros_like(expected, dtype=torch.bool)
    max_blocks = payload.max_block_count
    expected[:, :, :max_blocks] = float("-inf")
    defined[:, :, :max_blocks] = True

    for request_id, sequence_length_value in enumerate(seq_lens.tolist()):
        sequence_length = int(sequence_length_value)
        for query_offset in range(spec.decode_query_len):
            query_id = request_id * spec.decode_query_len + query_offset
            query_position = (
                sequence_length - spec.decode_query_len + query_offset
            )
            kv_length = max(query_position + 1, 0)
            valid_blocks = _ceil_div(kv_length, BLOCK_SIZE)
            local_start = max(valid_blocks - spec.local_blocks, 0)
            for block_id in range(valid_blocks):
                if block_id >= max_blocks:
                    raise ValueError(
                        f"case {spec.name} exceeds max_seq_len contract"
                    )
                page_id = int(block_table[request_id, block_id])
                key_start = block_id * BLOCK_SIZE
                valid_positions = min(
                    BLOCK_SIZE,
                    kv_length - key_start,
                )
                keys = cache[page_id, :valid_positions]
                for head_id in range(payload.heads):
                    score = torch.mv(
                        keys,
                        query[query_id, head_id],
                    ).max()
                    if block_id < spec.init_blocks:
                        score = torch.tensor(1e30, dtype=torch.float32)
                    if block_id >= local_start:
                        score = torch.tensor(1e29, dtype=torch.float32)
                    expected[head_id, query_id, block_id] = score
    return expected, defined


def reference(
    payload: CasePayload,
    score_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.mode == "prefill":
        return _prefill_reference(payload, score_stride)
    return _decode_reference(payload, score_stride)


def validate(
    state: RunState,
    payload: CasePayload,
    *,
    atol: float,
    rtol: float,
) -> ValidationResult:
    state.output.fill_(float("nan"))
    actual_device = invoke(state, payload)
    _synchronize(payload.device)
    actual = actual_device.detach().cpu()
    expected, defined = reference(payload, state.score_stride)

    actual_defined = actual[defined]
    expected_defined = expected[defined]
    if torch.isnan(actual_defined).any():
        first = int(torch.nonzero(torch.isnan(actual_defined), as_tuple=False)[0])
        raise AssertionError(
            f"{state.label} {payload.spec.name}: NaN in defined score domain "
            f"at flattened defined index {first}"
        )
    if torch.isposinf(actual_defined).any():
        raise AssertionError(
            f"{state.label} {payload.spec.name}: unexpected +inf in score output"
        )

    expected_neg_inf = torch.isneginf(expected_defined)
    actual_neg_inf = torch.isneginf(actual_defined)
    if not torch.equal(actual_neg_inf, expected_neg_inf):
        mismatch = int(
            torch.nonzero(actual_neg_inf != expected_neg_inf, as_tuple=False)[0]
        )
        raise AssertionError(
            f"{state.label} {payload.spec.name}: -inf mask mismatch at "
            f"flattened defined index {mismatch}"
        )

    finite_mask = torch.isfinite(expected_defined)
    actual_finite = actual_defined[finite_mask]
    expected_finite = expected_defined[finite_mask]
    if actual_finite.numel():
        difference = (actual_finite - expected_finite).abs()
        denominator = expected_finite.abs().clamp_min(1e-12)
        relative = difference / denominator
        max_abs = float(difference.max())
        max_rel = float(relative.max())
        close = torch.isclose(
            actual_finite,
            expected_finite,
            atol=atol,
            rtol=rtol,
        )
        if not bool(close.all()):
            worst = int(torch.argmax(difference))
            raise AssertionError(
                f"{state.label} {payload.spec.name}: score mismatch; "
                f"actual={float(actual_finite[worst])}, "
                f"expected={float(expected_finite[worst])}, "
                f"abs={float(difference[worst])}, "
                f"rel={float(relative[worst])}, atol={atol}, rtol={rtol}"
            )
    else:
        max_abs = 0.0
        max_rel = 0.0

    return ValidationResult(
        max_abs=max_abs,
        max_rel=max_rel,
        compared_values=int(finite_mask.sum()),
        negative_infinity_values=int(expected_neg_inf.sum()),
    )


def benchmark(
    state: RunState,
    payload: CasePayload,
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> TimingResult:
    for _ in range(warmup):
        invoke(state, payload)
    _synchronize(payload.device)

    samples: list[float] = []
    for _ in range(repeats):
        _synchronize(payload.device)
        start = time.perf_counter()
        for _ in range(iters):
            invoke(state, payload)
        _synchronize(payload.device)
        elapsed = time.perf_counter() - start
        samples.append(elapsed * 1e6 / iters)

    return TimingResult(
        mean_us=statistics.fmean(samples),
        median_us=statistics.median(samples),
        min_us=min(samples),
        max_us=max(samples),
        std_us=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        samples_us=samples,
    )


def _parse_cases(args: argparse.Namespace) -> list[CaseSpec]:
    selected_names: list[str] = []
    if args.case:
        selected_names.extend(args.case)
    if args.cases:
        selected_names.extend(
            item.strip()
            for item in args.cases.split(",")
            if item.strip()
        )

    allowed_modes: set[Mode]
    if args.mode == "both":
        allowed_modes = {"prefill", "decode"}
    else:
        allowed_modes = {args.mode}

    if args.all_cases:
        selected_names = [
            case.name
            for case in CASE_SPECS
            if case.mode in allowed_modes
        ]
    elif not selected_names:
        selected_names = []
        if "prefill" in allowed_modes:
            selected_names.append("prefill_aligned_small")
        if "decode" in allowed_modes:
            selected_names.append("decode_aligned_q1")

    unknown = [name for name in selected_names if name not in CASES_BY_NAME]
    if unknown:
        raise ValueError(f"unknown case names: {unknown}")

    deduplicated: list[CaseSpec] = []
    seen: set[str] = set()
    for name in selected_names:
        case = CASES_BY_NAME[name]
        if case.mode not in allowed_modes:
            raise ValueError(
                f"case {name} is {case.mode}, incompatible with --mode {args.mode}"
            )
        if name not in seen:
            deduplicated.append(case)
            seen.add(name)
    return deduplicated


def _list_cases() -> None:
    for mode in ("prefill", "decode"):
        print(f"{mode}:")
        for case in CASE_SPECS:
            if case.mode == mode:
                print(f"  {case.name}")


def _format_table(
    rows: list[tuple[str, str, TimingResult]],
) -> str:
    header = (
        f"{'case':32} {'implementation':32} "
        f"{'mean_us':>11} {'median_us':>11} {'min_us':>11} "
        f"{'max_us':>11} {'std_us':>11}"
    )
    lines = [header, "-" * len(header)]
    for case_name, label, result in rows:
        lines.append(
            f"{case_name:32} {label:32} "
            f"{result.mean_us:11.3f} {result.median_us:11.3f} "
            f"{result.min_us:11.3f} {result.max_us:11.3f} "
            f"{result.std_us:11.3f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=False,
        help="one or more implementation .py files or importable modules",
    )
    parser.add_argument(
        "--mode",
        choices=("prefill", "decode", "both"),
        default="both",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="select one case; repeat for multiple cases",
    )
    parser.add_argument(
        "--cases",
        help="comma-separated case names",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="run every registered case compatible with --mode",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list case names and exit",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_cases:
        _list_cases()
        return 0
    if not args.files:
        parser.error("--files is required unless --list-cases is used")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("warmup must be >=0; iters/repeats must be >0")

    device = torch.device(args.device)
    if device.type == "npu":
        if torch_npu is None or not hasattr(torch, "npu"):
            raise RuntimeError(
                "Ascend benchmark requires torch_npu and torch.npu"
            )
        if not torch.npu.is_available():
            raise RuntimeError("no available Ascend NPU")
        torch.npu.set_device(device)

    dtype = _dtype_from_name(args.dtype)
    cases = _parse_cases(args)
    selected_modes = {case.mode for case in cases}
    implementations = load_implementations(args.files, selected_modes)

    timing_rows: list[tuple[str, str, TimingResult]] = []
    records: list[dict[str, Any]] = []

    for case_index, spec in enumerate(cases):
        payload = build_case(
            spec,
            dtype=dtype,
            device=device,
            seed=args.seed + case_index * 17,
        )
        states = [
            _prepare_state(label, module, payload)
            for label, module in implementations
        ]
        print(
            f"\nCASE {spec.name} mode={spec.mode} "
            f"q={tuple(payload.idx_q.shape)} "
            f"cache={tuple(payload.index_k_cache.shape)} "
            f"max_seq_len={payload.max_seq_len}"
        )

        validation_by_label: dict[str, ValidationResult] = {}
        if args.validate:
            for state in states:
                result = validate(
                    state,
                    payload,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                validation_by_label[state.label] = result
                print(
                    f"VALID {state.label}: max_abs={result.max_abs:.6g} "
                    f"max_rel={result.max_rel:.6g} "
                    f"finite={result.compared_values} "
                    f"neg_inf={result.negative_infinity_values}"
                )

        timing_by_label: dict[str, TimingResult] = {}
        if args.benchmark:
            for state in states:
                result = benchmark(
                    state,
                    payload,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                timing_by_label[state.label] = result
                timing_rows.append((spec.name, state.label, result))

        for state in states:
            record: dict[str, Any] = {
                "case": spec.name,
                "mode": spec.mode,
                "implementation": state.label,
                "dtype": str(dtype),
                "device": str(device),
                "idx_q_shape": list(payload.idx_q.shape),
                "index_k_cache_shape": list(payload.index_k_cache.shape),
                "block_table_shape": list(payload.block_table.shape),
                "max_seq_len": payload.max_seq_len,
                "max_block_count": payload.max_block_count,
                "score_stride": state.score_stride,
                "warmup": args.warmup,
                "iters": args.iters,
                "repeats": args.repeats,
                "spec": asdict(spec),
            }
            if state.label in validation_by_label:
                record["validation"] = asdict(
                    validation_by_label[state.label]
                )
            if state.label in timing_by_label:
                record["timing"] = asdict(timing_by_label[state.label])
            records.append(record)

        del states, payload
        if device.type == "npu":
            torch.npu.empty_cache()

    if timing_rows:
        print("\n" + _format_table(timing_rows))

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
