# SPDX-License-Identifier: Apache-2.0
"""Paired stage experiments. No matrix work is included in stage latency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['norm', 'split', 'layout', 'all'], default='all')
    p.add_argument('--rows', default='1,8,31,32,64,128,512,2048')
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--modes', default='baseline', help='baseline or comma-separated kernel mode integers')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--calls', type=int, default=50)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError(out/'result.json')
    result = {'started': datetime.now(timezone.utc).isoformat(), 'args': vars(a) | {'output': str(out)}, 'cases': []}
    try:
        import torch
        import torch_npu
        import mla_superkernel
        from mla_superkernel.reference import rms, rope_reference, compare
        torch.set_num_threads(4); torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        result['environment'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__, 'device': torch.npu.get_device_name(0)}
        gen = torch.Generator().manual_seed(20261106)
        stream = torch.npu.Stream()
        rng = random.Random(42)
        stages = ['norm','split','layout'] if a.stage == 'all' else [a.stage]
        for rows in map(int, a.rows.split(',')):
            slots = ((3 * rows + 15)//16)*16
            def rnd(shape): return torch.randn(shape, generator=gen).bfloat16()
            sine = rnd((rows,64)); cosine = rnd((rows,64))
            index = torch.linspace(slots-1,0,rows,dtype=torch.int64)
            gq, gkv = rnd((1536,)), rnd((512,))
            d = {'dummy': torch.empty(1,dtype=torch.bfloat16).npu(), 'sine':sine.npu(), 'cosine':cosine.npu(),
                 'index': index.npu(), 'gq':gq.npu(), 'gkv':gkv.npu()}
            for stage in stages:
                baseline = (7 if rows >= 32 else 4) if stage=='norm' else 10 if stage=='split' else 2
                modes = [baseline] if a.modes=='baseline' else list(dict.fromkeys([baseline]+list(map(int,a.modes.split(',')))))
                if stage == 'norm':
                    x = rnd((rows,2112)); expected = [rms(x[:,:1536].double(),gq.double()),
                        rms(x[:,1536:2048].double(),gkv.double()), rope_reference(x[:,2048:].double(),sine.double(),cosine.double())]
                elif stage == 'split':
                    x = rnd((rows,a.heads*192)); v=x.view(rows,a.heads,192)
                    expected = [v[:,:,:128].permute(1,0,2).contiguous(), rope_reference(v[:,:,128:].double(),sine.double().unsqueeze(1),cosine.double().unsqueeze(1))]
                else:
                    x = rnd((a.heads*rows,512)); expected = [x.view(a.heads,rows,512).transpose(0,1).contiguous()]
                xd = x.npu(); graphs={}; outputs={}; stats={}
                for mode in modes:
                    yshape = (rows,1536) if stage=='norm' else (a.heads,rows,128) if stage=='split' else (rows,a.heads,512)
                    y=torch.empty(yshape,device='npu',dtype=torch.bfloat16)
                    qr=torch.empty((rows,a.heads,64),device='npu',dtype=torch.bfloat16)
                    kv=torch.full((slots,512),-3.25,device='npu',dtype=torch.bfloat16)
                    kr=torch.full((slots,64),-3.25,device='npu',dtype=torch.bfloat16)
                    op=getattr(torch.ops._C_ascend, {'norm':'mla_norm_cache_out','split':'mla_split_rope_out','layout':'mla_layout_out'}[stage])
                    def call():
                        op(xd,d['dummy'],d['gq'],d['gkv'],d['sine'],d['cosine'],d['index'],mode,rows,a.heads,slots,1e-5,1e-5,y,qr,kv,kr)
                    call();torch.npu.synchronize()
                    # One-call correctness is independent of the repeated timing graph.
                    actual = [y,kv[index],kr[index]] if stage=='norm' else [y,qr] if stage=='split' else [y]
                    checks=[]
                    for k,(value,ref) in enumerate(zip(actual,expected)):
                        exact = stage=='layout' or (stage=='split' and k==0)
                        c=compare(value,ref.double()); c['exact_required']=exact
                        c['pass']=torch.equal(value.cpu(),ref) if exact else c['matched_ratio']>=.99 and c['max_abs_error']<=1
                        checks.append(c)
                    if stage=='norm':
                        touched=torch.zeros(slots,dtype=torch.bool);touched[index]=True
                        cache_ok=bool((kv.cpu()[~touched]==-3.25).all() and (kr.cpu()[~touched]==-3.25).all())
                    else: cache_ok=bool((kv.cpu()==-3.25).all() and (kr.cpu()==-3.25).all())
                    stats[mode]={'precision':checks,'cache_untouched_exact':cache_ok,'pass':all(c['pass'] for c in checks) and cache_ok,'samples_us':[]}
                    graph=torch.npu.NPUGraph();stream.wait_stream(torch.npu.current_stream())
                    with torch.npu.graph(graph,stream=stream):
                        for _ in range(a.calls): call()
                    graphs[mode]=graph;outputs[mode]=(y,qr,kv,kr)
                    for _ in range(5): graph.replay()
                    torch.npu.synchronize()
                for _ in range(a.samples):
                    order=modes.copy();rng.shuffle(order)
                    for mode in order:
                        begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        with torch.npu.stream(stream):
                            begin.record();graphs[mode].replay();end.record()
                        end.synchronize();stats[mode]['samples_us'].append(begin.elapsed_time(end)*1000/a.calls)
                for mode,v in stats.items():
                    v['median_us']=statistics.median(v['samples_us'])
                    v['baseline_over_candidate']=statistics.median(stats[baseline]['samples_us'])/v['median_us']
                record={'rows':rows,'heads':a.heads,'stage':stage,'baseline_mode':baseline,'modes':stats}
                result['cases'].append(record)
                (out/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'stage':stage,'rows':rows,'modes':{k:{'us':round(v['median_us'],4),'ratio':round(v['baseline_over_candidate'],4),'pass':v['pass']} for k,v in stats.items()}}),flush=True)
                del graphs,outputs
                torch.npu.synchronize()
        result['loaded_binaries']={}
        for line in Path('/proc/self/maps').read_text().splitlines():
            f=Path(line.split()[-1])
            if f.is_file() and f.suffix=='.so' and any(s in str(f) for s in ('mla_superkernel_native','mlask_transformer','libopapi')):
                result['loaded_binaries'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        result['status']='PASS' if all(v['pass'] for c in result['cases'] for v in c['modes'].values()) else 'PRECISION_FAILURE'
    except Exception as exc:
        result.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        print(result['traceback'],file=sys.stderr)
    result['finished']=datetime.now(timezone.utc).isoformat()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': sys.exit(main())
