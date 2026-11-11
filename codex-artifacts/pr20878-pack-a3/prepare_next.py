"""Extract frozen definitions and generate auditable first candidates."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parts = ['import torch\nfrom vllm.triton_utils import triton, tl\nfrom baseline_kernels import _lse_pack_dim\nMAX_UNROLLED_COMBINE_ELEMENTS = 36 * 1024\n']
index = []
for path, names in [
    ('sharded_indexer.py', ['_merge_owner_prefix']),
    ('dcp/dcp_a2a_batched.py', ['_fused_dcp_lse_combine_batched_kernel']),
    ('dcp/dcp_a2a.py', ['_fused_dcp_lse_combine_kernel', '_launch_fused_dcp_lse_combine']),
]:
    source = (ROOT / 'source/vllm_ascend/ops/triton' / path).read_text(encoding='utf-8')
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            text = '\n'.join(source.splitlines()[start - 1:node.end_lineno]) + '\n'
            parts.append(text)
            index.append(dict(function=node.name, source=path, sha256=hashlib.sha256(text.encode()).hexdigest()))
baseline = '\n'.join(parts)
(ROOT / 'next_baseline_kernels.py').write_text(baseline, encoding='utf-8')
for entry in index:
    a = next(n for n in ast.parse(baseline).body if isinstance(n, ast.FunctionDef) and n.name == entry['function'])
    original = (ROOT / 'source/vllm_ascend/ops/triton' / entry['source']).read_text(encoding='utf-8')
    b = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == entry['function'])
    assert ast.dump(a) == ast.dump(b)
(ROOT / 'next-source-index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')

owner = parts[1].replace('_merge_owner_prefix(', '_merge_owner_parallel(')
begin = owner.index('            for rank in range(WORLD):')
end = owner.index('            selected = ', begin)
owner = owner[:begin] + '''            # All peers evaluate this pivot together. Inactive peers stop loading
            # after their first partial tile; the shared cursor remains bounded.
            active = (ranks < WORLD) & (ranks != OWNER)
            start = tl.full((), 0, tl.int32)
            peer_count = tl.zeros((BLOCK_RANKS,), tl.float32)
            while (start < K) & (tl.sum(active.to(tl.int32), 0) > 0):
                offsets = ((peer_rank[:, None] * ROWS + row) * K + start) * 2 + lanes[None, :]
                pair = tl.load(packed + offsets, active[:, None], 0)
                raw_ids, bits = tl.split(tl.reshape(pair, (BLOCK_RANKS, TILE, 2)))
                ids = raw_ids.to(tl.float32)
                scores = bits.to(tl.float32, bitcast=True)
                valid = active[:, None] & (ids >= 0) & (tl.abs(scores) < float("inf"))
                better = valid & ((scores > pivot_score) | ((scores == pivot_score) & (ids < pivot_id)))
                amount = tl.sum(better.to(tl.float32), 1)
                peer_count += amount
                active = active & (amount == TILE)
                start += TILE
            count += tl.sum(peer_count, 0)
''' + owner[end:]

combine = '''@triton.jit
def _fused_dcp_lse_combine_cached(
    recv_ptr, output_ptr, local_output_ptr, local_lse_ptr,
    local_output_stride_t: tl.constexpr, local_output_stride_h: tl.constexpr, local_output_stride_d: tl.constexpr,
    local_lse_stride_t: tl.constexpr, local_lse_stride_h: tl.constexpr,
    recv_stride_rank: tl.constexpr, recv_stride_scatter: tl.constexpr,
    recv_stride_replicated: tl.constexpr, recv_stride_d: tl.constexpr,
    output_stride_t: tl.constexpr, output_stride_h: tl.constexpr, output_stride_d: tl.constexpr,
    head_dim: tl.constexpr, num_heads: tl.constexpr, total_rows,
    DCP_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_ROWS: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True, RANK_UNROLL: tl.constexpr = 1,
):
    # Cache only the small LSE matrix; keep the wide output streaming by rank.
    BLOCK_RANKS: tl.constexpr = triton.next_power_of_2(DCP_SIZE)
    ranks = tl.arange(0, BLOCK_RANKS)[None, :]
    d_offsets = tl.arange(0, BLOCK_D)[None, :]
    for row_start in range(tl.program_id(0) * BLOCK_ROWS, total_rows, tl.num_programs(0) * BLOCK_ROWS):
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        tokens = rows // num_heads
        heads = rows % num_heads
        mask = rows < total_rows
        bases = ranks * recv_stride_rank + heads[:, None] * recv_stride_scatter + tokens[:, None] * recv_stride_replicated
        packed = tl.load(recv_ptr + bases[:, :, None] + (head_dim + tl.arange(0, 4)[None, None, :]) * recv_stride_d,
                         mask[:, None, None] & (ranks[:, :, None] < DCP_SIZE), 0).to(tl.float32)
        even, odd = tl.split(tl.reshape(packed, (BLOCK_ROWS * BLOCK_RANKS, 2, 2)))
        code, mid = tl.split(even)
        hi, lo = tl.split(odd)
        significand = hi * 65536.0 + mid * 256.0 + lo
        lse_bits = (tl.maximum(tl.abs(code).to(tl.int32) - 2, 0) << 23) + significand.to(tl.int32)
        lse_bits = tl.where(significand != 0.0, lse_bits, 0)
        lses = tl.reshape(tl.where(code < 0.0, -1.0, 1.0) * lse_bits.to(tl.float32, bitcast=True), (BLOCK_ROWS, BLOCK_RANKS))
        valid = tl.reshape(code != 0.0, (BLOCK_ROWS, BLOCK_RANKS)) & (ranks < DCP_SIZE) & (tl.abs(lses) < float("inf"))
        max_lse = tl.max(tl.where(valid, lses, -float("inf")), 1)
        if HAS_LOCAL:
            local_lse = tl.load(local_lse_ptr + tokens * local_lse_stride_t + heads * local_lse_stride_h, mask, 0).to(tl.float32)
            local_valid = tl.abs(local_lse) < float("inf")
            max_lse = tl.maximum(max_lse, tl.where(local_valid, local_lse, -float("inf")))
        safe_max = tl.where(max_lse != -float("inf"), max_lse, 0.0)
        weights = tl.where(valid, tl.exp(lses - safe_max[:, None]), 0.0)
        weight_sum = tl.sum(weights, 1)[:, None]
        merged = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)
        d_mask = mask[:, None] & (d_offsets < head_dim)
        for rank in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
            weight = tl.sum(tl.where(ranks == rank, weights, 0.0), 1)[:, None]
            rank_valid = tl.sum((valid & (ranks == rank)).to(tl.int32), 1)[:, None] > 0
            recv_base = rank * recv_stride_rank + heads[:, None] * recv_stride_scatter + tokens[:, None] * recv_stride_replicated
            values = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_d, d_mask, 0).to(tl.float32)
            merged += tl.where(rank_valid, values, 0.0) * weight
        if HAS_LOCAL:
            weight = tl.where(local_valid, tl.exp(local_lse - safe_max), 0.0)[:, None]
            local_base = tokens[:, None] * local_output_stride_t + heads[:, None] * local_output_stride_h
            values = tl.load(local_output_ptr + local_base + d_offsets * local_output_stride_d, d_mask, 0).to(tl.float32)
            merged += tl.where(local_valid[:, None], values, 0.0) * weight
            weight_sum += weight
        merged /= tl.where(weight_sum > 0.0, weight_sum, 1.0)
        out = tokens[:, None] * output_stride_t + heads[:, None] * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + out, merged, d_mask)
'''
(ROOT / 'next_candidate_kernels.py').write_text('from vllm.triton_utils import triton, tl\n\n' + owner + '\n' + combine, encoding='utf-8')
runner = (ROOT / 'run_general.py').read_text(encoding='utf-8').replace('import general_benchmark as benchmark', 'import next_benchmark as benchmark').replace('TARGET_OPERATOR = "PR20878PackOpt"', 'TARGET_OPERATOR = "PR20878NextOpt"')
(ROOT / 'run_next.py').write_text(runner, encoding='utf-8')
print('NEXT_BASELINE_AST_EQUAL', len(index))
