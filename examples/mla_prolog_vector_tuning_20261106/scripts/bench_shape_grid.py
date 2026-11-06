# SPDX-License-Identifier: Apache-2.0
"""Run the declared B/S grid in one process, keeping every case and failure."""
import argparse
import gc
import json
from pathlib import Path
import sys
import bench_pipeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--selection',default='all',help='all, decode, sk, guards, or BxS comma-separated')
    p.add_argument('--mode',choices=['S0','S1','S2'],default='S0')
    p.add_argument('--backend',choices=['v3_nz','torch_matmul_nz'],default='v3_nz')
    p.add_argument('--vector-modes',default='adaptive')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--refresh-check',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((Path(__file__).resolve().parents[1]/'shape-plan.json').read_text())
    cases=plan['cases']
    if a.selection=='decode':cases=[c for c in cases if c['sequence']==1]
    elif a.selection=='sk':cases=[dict(batch=b,sequence=s) for b,s in [(1,1),(8,1),(32,1),(128,1),(1,8),(2,16),(16,8),(1,33),(64,32)]]
    elif a.selection=='guards':
        cases=[dict(batch=b,sequence=s,hidden=h,heads=n,block_size=bs,index_mode='reverse')
               for b,s,h,n,bs in [(1,1,1024,1,16),(2,4,8192,8,128),(3,17,4096,128,128),(16,8,7168,1,128),(1,129,7680,64,16)]]
    elif a.selection!='all':cases=[dict(zip(('batch','sequence'),map(int,item.split('x')))) for item in a.selection.split(',')]
    results=[]
    for number,c in enumerate(cases,1):
        name=f'b{c["batch"]}-s{c["sequence"]}'
        if a.selection=='guards':name+=f'-h{c["hidden"]}-n{c["heads"]}'
        target=out/name
        sys.argv=['bench_pipeline.py','--batch',str(c['batch']),'--sequence',str(c['sequence']),
                  '--hidden',str(c.get('hidden',7168)),'--heads',str(c.get('heads',32)),
                  '--block-size',str(c.get('block_size',16)),'--index-mode',c.get('index_mode','edges'),
                  '--mode',a.mode,'--backend',a.backend,'--vector-modes',a.vector_modes,'--output',str(target)]
        if a.profile and number==1:sys.argv+=['--profile']
        if a.refresh_check and number==1:sys.argv+=['--refresh-check']
        code=bench_pipeline.main()
        result=json.loads((target/'result.json').read_text())
        results.append(dict(case=c,status=result['status'],exit_code=code,path=name+'/result.json',
                            variants={k:{x:v[x] for x in ('status','median_us','error') if x in v} for k,v in result.get('variants',{}).items()}))
        (out/'partial.json').write_text(json.dumps({'args':vars(a)|{'output':str(out)},'planned':len(cases),'cases':results},indent=2)+'\n')
        print(json.dumps({'progress':f'{number}/{len(cases)}','last':results[-1]}),flush=True)
        import torch
        torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
        if a.mode!='S0':torch._dynamo.reset()
        if any(token in str(result) for token in ('507034','aicore error','device-side assert','507035')):
            raise RuntimeError('Device execution failure; stop the owned suite and preserve this result')
    summary={'args':vars(a)|{'output':str(out)},'planned':len(cases),'cases':results,
             'status':'PASS' if all(c['status']=='PASS' for c in results) else 'CASE_FAILURE'}
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
