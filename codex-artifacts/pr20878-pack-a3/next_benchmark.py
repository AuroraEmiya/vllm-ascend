"""Owner-prefix and combine: deterministic inputs and independent CPU references.

144/152 owner rows and combine local H=12 follow the recorded call chain.
These are isolated shape reproductions, not a model replay or captured payloads.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch_npu

import next_baseline_kernels as base
import next_candidate_kernels as cand
from benchmark import binary_identity, cores, encode_cpu

ROOT = Path(__file__).resolve().parent
SPECS = {}
for t in (144, 152):
    SPECS[f'o{t}'] = dict(kind='owner', t=t, world=8, owner=0, distribution='random')
    SPECS[f'c{t}'] = dict(kind='combine', t=t, h=12, d=512, world=8, local=True, layout='contiguous')
SPECS['c114'] = dict(kind='combine', t=114, h=8, d=512, world=8, local=False, layout='contiguous')
for t in (1, 39, 40, 41, 257):
    SPECS[f'ot{t}'] = dict(kind='owner', t=t, world=8, owner=3, distribution='random')
for world, owner in ((4, 1), (16, 15), (3, 2)):
    SPECS[f'ow{world}'] = dict(kind='owner', t=41, world=world, owner=owner, distribution='random')
for distribution in ('ties', 'own_high', 'peer_high', 'short', 'mixed'):
    SPECS['o' + distribution] = dict(kind='owner', t=41, world=8, owner=0, distribution=distribution)
for t, h, d, world, local in ((20, 8, 64, 4, False), (41, 7, 128, 3, True), (127, 16, 256, 16, True), (129, 12, 512, 4, False), (257, 12, 512, 8, True)):
    SPECS[f'cx{t}'] = dict(kind='combine', t=t, h=h, d=d, world=world, local=local, layout='contiguous')
SPECS['cstrided'] = dict(kind='combine', t=41, h=12, d=512, world=8, local=True, layout='headmajor')


def make_case(key, edge=False):
    case_id, impl = key.split(':')
    s = SPECS[case_id]
    g = torch.Generator().manual_seed(20878 + s['t'])
    t, w = s['t'], s['world']
    c = dict(case_id=case_id, impl=impl, spec=s, compiled=[], vector_cores=cores())
    if s['kind'] == 'owner':
        k, interleave = 2048, 768
        local_ids = torch.arange(k, dtype=torch.int64)
        ids = torch.stack([local_ids + ((local_ids // interleave) * (w - 1) + rank) * interleave for rank in range(w)])[:, None, :].expand(w, t, k).clone()
        scores = torch.randn((w, t, k), generator=g)
        distribution = s['distribution']
        if distribution == 'ties': scores = scores.round()
        if distribution == 'own_high': scores[s['owner']] += 20
        if distribution == 'peer_high': scores[(s['owner'] + 1) % w] += 20
        if distribution in ('short', 'mixed'):
            for rank in range(w):
                n = (97 + rank * 23) if distribution == 'short' else (rank * 307) % k
                ids[rank, :, n:] = -1
                scores[rank, :, n:] = -float('inf')
        if edge:
            scores[:, :, :8] = torch.tensor([0., -0., float('nan'), float('inf'), -float('inf'), 1., 1., 1.])
            ids[:, :, 9] = -1
            # Include entirely invalid rows in otherwise well-populated tests.
            ids[:, 0, :] = -1
        valid = (ids >= 0) & torch.isfinite(scores)
        scores = torch.where(valid, scores, -float('inf'))
        ids = torch.where(valid, ids, -1)
        id_order = torch.argsort(torch.where(valid, ids, torch.iinfo(torch.int64).max), dim=-1, stable=True)
        score_order = torch.argsort(torch.gather(scores, -1, id_order), dim=-1, descending=True, stable=True)
        order = torch.gather(id_order, -1, score_order)
        ids = torch.gather(ids, -1, order)
        scores = torch.gather(scores, -1, order)
        pair = torch.stack((ids.int(), scores.contiguous().view(torch.int32)), -1)
        flat_ids = ids.permute(1, 0, 2).reshape(t, w * k)
        flat_scores = scores.permute(1, 0, 2).reshape(t, w * k)
        by_id = torch.argsort(torch.where(flat_ids >= 0, flat_ids, torch.iinfo(torch.int64).max), dim=-1, stable=True)
        by_score = torch.argsort(torch.gather(flat_scores, -1, by_id), descending=True, stable=True, dim=-1)
        winners = torch.gather(flat_ids, -1, torch.gather(by_id, -1, by_score)[:, :k])
        is_owner = (winners >= 0) & ((winners // interleave) % w == s['owner'])
        selected_count = is_owner.sum(-1)
        own = ids[s['owner']]
        block = own // interleave
        expected = torch.where(torch.arange(k)[None, :] < selected_count[:, None], block // w * interleave + own - block * interleave, -1).int()
        c.update(packed=pair.npu(), expected=expected, out=torch.empty((t, k), device='npu', dtype=torch.int32))
    else:
        h, d = s['h'], s['d']
        values = torch.randn((w, t, h, d), generator=g).bfloat16()
        lses = torch.randn((w, t, h), generator=g) * 4
        local_values = torch.randn((t, h, d), generator=g).bfloat16()
        local_lse = torch.randn((t, h), generator=g) * 4
        if edge:
            lses[:, 0, 0] = -float('inf')
            local_lse[0, 0] = float('nan')
            values[:, 0, 0] = float('nan')
            local_values[0, 0] = float('inf')
            lses[0, -1, 0] = float('nan')
            lses[-1, -1, -1] = float('inf')
            values[0, -1, 0] = float('nan')
            if t > 3:
                lses[:, 2, 0] = 200000.
                lses[:, 3, 0] = -200000.
                local_lse[2, 0] = 200000.
                local_lse[3, 0] = -200000.
        wire = torch.cat((values, encode_cpu(lses[..., None])), -1).permute(0, 2, 1, 3).contiguous()
        all_lses = torch.cat((lses, local_lse[None]), 0) if s['local'] else lses
        all_values = torch.cat((values, local_values[None]), 0) if s['local'] else values
        valid = torch.isfinite(all_lses)
        all_lses = torch.where(valid, all_lses.double(), -float('inf'))
        maximum = all_lses.max(0).values
        maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
        weights = torch.where(valid, torch.exp(all_lses - maximum), 0.)
        numerator = (torch.where(valid[..., None], all_values.double(), 0.) * weights[..., None]).sum(0)
        denominator = weights.sum(0)
        expected = numerator / torch.where(denominator > 0, denominator, 1.)[..., None]
        if s['layout'] == 'headmajor':
            local_device = local_values.transpose(0, 1).contiguous().npu().transpose(0, 1)
        else:
            local_device = local_values.npu()
        c.update(recv=wire.npu(), local_output=local_device if s['local'] else None,
                 local_lse=local_lse.npu() if s['local'] else None, expected=expected,
                 out=torch.empty((t, h, d), device='npu', dtype=torch.bfloat16))
    return c


def invoke(c):
    s = c['spec']
    if s['kind'] == 'owner':
        fn = base._merge_owner_prefix if c['impl'] == 'baseline' else cand._merge_owner_parallel
        tile = min(2048, 2 ** math.ceil(math.log2(max(1, (2 if c['impl'] == 'baseline' else 1) * 2048 // s['world']))))
        binary = fn[(min(s['t'], c['vector_cores']),)](c['packed'], c['out'], s['t'], WORLD=s['world'], OWNER=s['owner'], INTERLEAVE=768, LOG_BLOCK=11, TILE=tile, BLOCK_RANKS=2 ** math.ceil(math.log2(s['world'])), multibuffer=False)
        c['compiled'] = [binary]
    else:
        def launch(fn, grid, *args, **kwargs):
            assert fn is base._fused_dcp_lse_combine_batched_kernel
            if c['impl'] != 'baseline' and (c['local_output'] is None or c['local_output'].stride(1) == s['d']): fn = cand._fused_dcp_lse_combine_cached
            c['compiled'] = [fn[grid](*args, **kwargs)]
        base._launch_fused_dcp_lse_combine(c['recv'], c['out'], s['d'], 1, False, c['local_output'], c['local_lse'], c['vector_cores'], launch)
    return c['out']


def check(c):
    actual, expected = c['out'].cpu(), c['expected']
    if c['spec']['kind'] == 'owner':
        return dict(passed=torch.equal(actual, expected), mismatches=int((actual != expected).sum()), rule='exact_global_topk_owner_prefix')
    error = (actual.double() - expected).abs()
    allowed = .00002 + .004 * expected.abs()
    ratio = float((error <= allowed).double().mean())
    return dict(passed=bool(torch.isfinite(actual).all()) and ratio >= .99 and float(error.max()) <= .02,
                match_ratio=ratio, max_error=float(error.max()), rule='CPU_FP64_weighted_merge;99pct_rtol0.004_atol2e-5_max0.02')


def kernel_name(params):
    key, impl = params.split(':')
    if SPECS[key]['kind'] == 'owner': return '_merge_owner_prefix' if impl == 'baseline' else '_merge_owner_parallel'
    return '_fused_dcp_lse_combine_batched_kernel' if impl == 'baseline' or SPECS[key]['layout'] != 'contiguous' else '_fused_dcp_lse_combine_cached'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', default='all')
    p.add_argument('--out', default='results/optimization/next/validation.json')
    p.add_argument('--impl', default='both')
    a = p.parse_args()
    torch.npu.set_device(0)
    result = []
    for key in SPECS if a.cases == 'all' else a.cases.split(','):
        for edge in (False, True):
            for impl in ('baseline', 'candidate') if a.impl == 'both' else (a.impl,):
                c = make_case(key + ':' + impl, edge)
                try:
                    invoke(c)
                    torch.npu.synchronize()
                    r = check(c)
                except Exception as exc:
                    r = dict(passed=False, error_type=type(exc).__name__, error=str(exc))
                record = dict(case_id=key, impl=impl, edge=edge, spec=c['spec'], **r)
                result.append(record)
                print(json.dumps(record), flush=True)
                path = Path(a.out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(status='running', cases=result), indent=2))
                if not r['passed']: raise RuntimeError('Validation failed ' + key + ':' + impl)
    sources = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in ('next_baseline_kernels.py', 'next_candidate_kernels.py', 'next_benchmark.py')}
    path.write_text(json.dumps(dict(status='passed', cases=result, binaries=binary_identity(), source_sha256=sources), indent=2))
    print('VALIDATION_PASSED', len(result), flush=True)


if __name__ == '__main__': main()
