# SPDX-License-Identifier: Apache-2.0
"""Independent A3 precision/state tests and reproducible Graph/msprof entry.

The CPU FP64 oracle follows the public Compressor contract; no fused golden,
compute kernel or tiling implementation is imported. Only BF16 is timed against
the installed fused binary; FP16 accuracy uses the independent oracle.
"""

import argparse
import importlib
import json
import statistics
from functools import partial
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.compressor_serial import compressor_serial, prepare_compressor_serial_weights

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an A3 NPU")


def _reference(values, reverse_overlap=False):
    x, wkv, wgate, old, ape, norm, sin, cos, table, cu, used, starts, rope_dim, ratio, coff, eps, rotary, cache_mode = (
        values
    )
    assert x.ndim == 2 and cache_mode == 1 and rotary == 2 and rope_dim == 64
    x, wkv, wgate, ape, norm, sin, cos = (v.cpu().double() for v in (x, wkv, wgate, ape, norm, sin, cos))
    old = old.cpu().double()
    table, cu, starts = (v.cpu().long() for v in (table, cu, starts))
    used = None if used is None else used.cpu().long()
    dim = norm.numel()
    kv, scores = x @ wkv.T, x @ wgate.T
    block_size = old.shape[1]
    state = old.clone()
    output = torch.zeros(cos.shape[0], dim, dtype=torch.float64)
    out_count = 0
    written = torch.zeros(old.shape[:2], dtype=torch.bool)
    for batch in range(starts.numel()):
        begin, stop = int(cu[batch]), int(cu[batch + 1])
        count = stop - begin if used is None else int(used[batch])
        start = int(starts[batch])
        for token in range(count):
            position = start + token
            scores[begin + token] += ape[position % ratio]
        group_count = (start + count) // ratio - start // ratio
        for group in range(group_count):
            end = (start // ratio + group + 1) * ratio
            data, gates = [], []
            for half in range(coff):
                channel_half = coff - 1 - half if reverse_overlap else half
                channels = slice(channel_half * dim, (channel_half + 1) * dim)
                for position in range(end - coff * ratio + half * ratio, end - coff * ratio + (half + 1) * ratio):
                    if position < 0:
                        data.append(torch.zeros(dim, dtype=torch.float64))
                        gates.append(torch.full((dim,), -float("inf"), dtype=torch.float64))
                    elif position < start:
                        page = int(table[batch, position // block_size])
                        cached = old[page, position % block_size]
                        data.append(cached[: coff * dim][channels])
                        gates.append(cached[coff * dim :][channels])
                    else:
                        row = begin + position - start
                        data.append(kv[row, channels])
                        gates.append(scores[row, channels])
            compressed = (torch.stack(gates).softmax(0) * torch.stack(data)).sum(0)
            result = compressed * torch.rsqrt(compressed.square().mean() + eps) * norm
            tail = result[-rope_dim:].reshape(-1, 2)
            rotated = torch.stack((-tail[:, 1], tail[:, 0]), dim=-1).flatten()
            result[-rope_dim:] = result[-rope_dim:] * cos[out_count] + rotated * sin[out_count]
            output[out_count] = result
            out_count += 1
        for token in range(count):
            position = start + token
            page = int(table[batch, position // block_size])
            if page:
                state[page, position % block_size] = torch.cat((kv[begin + token], scores[begin + token]))
                written[page, position % block_size] = True
    return output, state, out_count, written


def _make_case(lengths, starts, ratio, coff, dim, hidden=1024, block_size=128, seed=271):
    torch.manual_seed(seed)
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    positions = torch.tensor(starts, dtype=torch.int32)
    # Public ABI uses max(start_pos) + max(input sequence length), not max of pairwise sums.
    table_cols = (max(starts) + max(lengths) + block_size - 1) // block_size
    table = torch.arange(1, len(lengths) * table_cols + 1, dtype=torch.int32).reshape(len(lengths), table_cols)
    state = torch.randn(table.numel() + 1, block_size, 2 * coff * dim) * 0.2
    x = torch.randn(total, hidden).to(torch.bfloat16)
    wkv = (torch.randn(coff * dim, hidden) / hidden**0.5).to(torch.bfloat16)
    wgate = (torch.randn(coff * dim, hidden) / hidden**0.5).to(torch.bfloat16)
    ape = torch.randn(ratio, coff * dim) * 0.1
    norm = 1 + torch.randn(dim) * 0.1
    rows = min(total, total // ratio + len(lengths))
    angles = torch.randn(rows, 32).repeat_interleave(2, -1)
    return (
        x,
        wkv,
        wgate,
        state,
        ape,
        norm,
        angles.sin(),
        angles.cos(),
        table,
        cu,
        None,
        positions,
        64,
        ratio,
        coff,
        1e-6,
        2,
        1,
    )


def _compare(actual, golden, atol, rtol, max_limit):
    actual = actual.cpu().double()
    assert actual.shape == golden.shape
    if not actual.numel():
        return {"pass": True, "max_abs": 0.0, "matched": 1.0}
    error = (actual - golden).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(golden).all())
    matched = float((error <= atol + rtol * golden.abs()).double().mean())
    maximum = float(error.max())
    return {
        "pass": finite and matched >= 0.99 and maximum <= max_limit,
        "finite": finite,
        "matched": matched,
        "max_abs": maximum,
    }


_FAMILIES = (("c4li", 4, 2, 128), ("c4a", 4, 2, 512), ("c128a", 128, 1, 512))
_SCENARIOS = (
    "empty",
    "cross-page",
    "partial-used",
    "zero-page",
    "permuted-pages",
    "near-zero-spike",
    "negative-gates",
    "empty-request",
)
_GUARD_ELEMENTS = 64
_GRAPH_CALLS = 20


@pytest.fixture(scope="module", autouse=True)
def _extensions():
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    assert hasattr(torch.ops._C_ascend, "compressor_serial_pool"), "rebuild vllm-ascend with the A3 serial ops"


def _logical_case(family, dtype, scenario):
    _, ratio, coff, dim = family
    lengths, starts = [ratio + 1], [ratio - 1]
    if scenario == "empty":
        lengths, starts = [0], [0]
    elif scenario == "negative-gates":
        lengths, starts = [ratio], [0]
    elif scenario == "partial-used":
        lengths, starts = [ratio + 3, 3, ratio // 2], [ratio - 1, 0, 2 * ratio - 1]
    elif scenario == "empty-request":
        lengths, starts = [0, ratio + 1, 2], [0, 0, ratio - 1]
    elif scenario == "zero-page":
        lengths, starts = [1], [ratio - 1]
    cpu = list(_make_case(lengths, starts, ratio, coff, dim, block_size=3, seed=919))
    for index in (0, 1, 2):
        cpu[index] = cpu[index].to(dtype)
    if scenario == "partial-used":
        cpu[10] = torch.tensor([ratio, 0, ratio // 2], dtype=torch.int32)
    elif scenario == "zero-page":
        cpu[8].zero_()
    elif scenario == "permuted-pages":
        cpu[8] = cpu[8].flip(-1).contiguous()
    elif scenario == "near-zero-spike":
        cpu[0].mul_(1e-4)
        cpu[4].mul_(1000)
    elif scenario == "negative-gates":
        cpu[2].zero_()
        cpu[4].fill_(-1e35)
    return cpu


def _device_case(cpu):
    values = [v.npu() if isinstance(v, torch.Tensor) else v for v in cpu]
    guard = torch.full((cpu[3].numel() + 2 * _GUARD_ELEMENTS,), 3.25, device="npu")
    values[3] = guard[_GUARD_ELEMENTS:-_GUARD_ELEMENTS].view_as(cpu[3])
    values[3].copy_(cpu[3])
    return values, guard


def _assert_result(output, values, cpu, expected, guard=None):
    golden, state, valid, written = expected
    tolerance = 2**-6 if cpu[0].dtype == torch.bfloat16 else 2**-9
    limit = 1 if cpu[0].dtype == torch.bfloat16 else 0.1
    assert output.shape == golden.shape and output.dtype == cpu[0].dtype
    assert _compare(output[:valid], golden[:valid], tolerance, tolerance, limit)["pass"]
    assert bool(output[valid:].eq(0).all()), "candidate padding must be zero"
    actual_state = values[3].cpu()
    assert _compare(actual_state[written], state[written], 2**-16, 2**-10, 0.01)["pass"]
    assert torch.equal(actual_state[~written], cpu[3][~written])
    for index, value in enumerate(values):
        if isinstance(value, torch.Tensor) and index != 3:
            assert torch.equal(value.cpu(), cpu[index]), f"input{index} modified"
    if guard is not None:
        assert bool(guard[:_GUARD_ELEMENTS].eq(3.25).all())
        assert bool(guard[-_GUARD_ELEMENTS:].eq(3.25).all())


@pytest.mark.parametrize("family", _FAMILIES, ids=lambda f: f[0])
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("scenario", _SCENARIOS)
@pytest.mark.parametrize("packed", (False, True))
def test_function_accuracy_state(family, dtype, scenario, packed):
    cpu = _logical_case(family, dtype, scenario)
    expected = _reference(cpu)
    values, guard = _device_case(cpu)
    weights = prepare_compressor_serial_weights(values[1], values[2]) if packed else None
    output = compressor_serial(*values, packed_weight=weights)
    torch.npu.synchronize()
    _assert_result(output, values, cpu, expected, guard)


@pytest.mark.parametrize("family", _FAMILIES, ids=lambda f: f[0])
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_graph_reads_updated_inputs(family, dtype):
    cpu = _logical_case(family, dtype, "cross-page")
    values, guard = _device_case(cpu)
    weights = prepare_compressor_serial_weights(values[1], values[2])
    function = partial(compressor_serial, packed_weight=weights)
    function(*values)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = function(*values)
    for change in (0, 0.125, -0.25):
        cpu[0].add_(change)
        values[0].copy_(cpu[0])
        values[3].copy_(cpu[3])
        graph.replay()
        torch.npu.synchronize()
        _assert_result(output, values, cpu, _reference(cpu), guard)


@pytest.mark.parametrize("invalid", ("rope", "cache-mode", "dtype", "bsh", "missing-start", "metadata-dtype"))
def test_invalid_static_contract(invalid):
    values = _logical_case(_FAMILIES[0], torch.bfloat16, "cross-page")
    if invalid == "rope":
        values[12] = 32
    elif invalid == "cache-mode":
        values[17] = 2
    elif invalid == "dtype":
        values[5] = values[5].to(torch.bfloat16)
    elif invalid == "bsh":
        values[0] = values[0].unsqueeze(0)
    elif invalid == "missing-start":
        values[11] = None
    elif invalid == "metadata-dtype":
        values[9] = values[9].to(torch.int64)
    with pytest.raises(ValueError):
        compressor_serial(*values)


def test_vector_meta():
    kv = torch.empty((17, 2048), device="meta")
    cache = torch.empty((7, 128, 2048), device="meta")
    ape = torch.empty((4, 1024), device="meta")
    cu = torch.empty((3,), device="meta", dtype=torch.int32)
    starts = torch.empty((2,), device="meta", dtype=torch.int32)
    table = torch.empty((2, 2), device="meta", dtype=torch.int32)
    pool = torch.ops._C_ascend.compressor_serial_pool(kv, kv, cache, ape, cu, starts, cu, table, 4, 2, 6, False)
    norm = torch.empty((512,), device="meta")
    rope = torch.empty((6, 64), device="meta")
    x = torch.empty((17, 4096), device="meta", dtype=torch.bfloat16)
    result = torch.ops._C_ascend.compressor_serial_finalize(
        pool, kv, kv, ape, norm, rope, rope, table, cu, starts, cu, x, cache, 4, 2, False, 1e-6
    )
    assert pool.shape == result.shape == (6, 512)
    assert pool.dtype == torch.float32 and result.dtype == torch.bfloat16


def _business_cases(long_prefill=False):
    for hidden in (4096, 7168):
        for family, ratio, coff, dim in _FAMILIES:
            modes = [
                ("decode1", [1], [ratio - 1]),
                ("decode16", [1] * 16, [ratio - 1] * 16),
                ("prefill256", [256], [0]),
                ("prefill1024", [1024], [0]),
            ]
            if long_prefill:
                modes += [("prefill4096", [4096], [0]), ("prefill8192", [8192], [0])]
            for mode, lengths, starts in modes:
                yield f"h{hidden}-{family}-{mode}", _make_case(lengths, starts, ratio, coff, dim, hidden=hidden)


def _graph_times(functions, values):
    graphs, outputs = {}, {}
    for name, function in functions.items():
        for _ in range(5):
            function(*values)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(_GRAPH_CALLS):
                outputs[name] = function(*values)
        graphs[name] = graph
    torch.npu.synchronize()
    result = {}
    for group in (0, 1):
        samples = {name: [] for name in functions}
        names = list(functions)
        for trial in range(25):
            for name in names if (trial + group) % 2 == 0 else names[::-1]:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                if trial >= 5:
                    samples[name].append(start.elapsed_time(end) * 1000 / _GRAPH_CALLS)
        result[str(group)] = {
            "samples_us": samples,
            "means_us": {name: statistics.mean(v) for name, v in samples.items()},
            "std_us": {name: statistics.pstdev(v) for name, v in samples.items()},
        }
    return result, outputs


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile", choices=("original", "candidate"))
    parser.add_argument("--case")
    parser.add_argument("--long-prefill", action="store_true")
    parser.add_argument("--output", default="compressor_serial_graph.json")
    args = parser.parse_args()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    records = []
    for name, cpu in _business_cases(args.long_prefill):
        if args.case is not None and args.case != name:
            continue
        values, guard = _device_case(cpu)
        weights = prepare_compressor_serial_weights(values[1], values[2])
        functions = {
            "original": torch.ops._C_ascend.compressor,
            "candidate": partial(compressor_serial, packed_weight=weights),
        }
        expected = _reference(cpu)
        for label, function in functions.items():
            values[3].copy_(cpu[3])
            output = function(*values)
            torch.npu.synchronize()
            if label == "candidate":
                _assert_result(output, values, cpu, expected, guard)
            else:
                valid = expected[2]
                assert _compare(output[:valid], expected[0][:valid], 2**-6, 2**-6, 1)["pass"]
                assert _compare(values[3].cpu()[expected[3]], expected[1][expected[3]], 2**-16, 2**-10, 0.01)["pass"]
        if args.profile:
            function = functions[args.profile]
            for _ in range(5):
                function(*values)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                function(*values)
            for _ in range(25):
                graph.replay()
            torch.npu.synchronize()
            print("PROFILE", name, args.profile, "25 replays; retain last20", flush=True)
            return
        timing, outputs = _graph_times(functions, values)
        _assert_result(outputs["candidate"], values, cpu, expected, guard)
        speed = min(v["means_us"]["original"] / v["means_us"]["candidate"] for v in timing.values())
        records.append(
            {
                "case": name,
                "graph_calls": _GRAPH_CALLS,
                "graph_event": timing,
                "min_speed": speed,
                "prepared_weight_bytes": weights.numel() * weights.element_size(),
            }
        )
        Path(args.output).write_text(json.dumps(records, indent=2))
        print(name, "Graph fused/candidate", round(speed, 4), flush=True)
    if not records:
        raise ValueError("no matching case")


if __name__ == "__main__":
    _main()
