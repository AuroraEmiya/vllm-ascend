"""Synthetic generalization tests; real shape provenance remains in the frozen report."""
import argparse, hashlib, json, math, os
from pathlib import Path
import torch
import torch_npu
import baseline_kernels as base
import candidate_kernels as cand
import dcp_iter4
from benchmark import cores, encode_cpu, binary_identity

ROOT = Path(__file__).resolve().parent
SPECS = json.loads((ROOT / 'general-cases.json').read_text())


def strided(value, mode):
    if mode == 'contiguous':
        return value.npu()
    if mode == 'feature2':
        result = torch.empty((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype, device='npu')[..., ::2]
        result.copy_(value)
        return result
    if mode == 'headmajor':
        return value.transpose(0, 1).contiguous().npu().transpose(0, 1)
    raise ValueError(mode)


def make_case(key, edge=False):
    case_id, impl = key.split(':')
    s = SPECS[case_id]
    t = s['t']; g = torch.Generator().manual_seed(20878 + t)
    c = dict(case_id=case_id, impl=impl, spec=s, vector_cores=cores(), compiled=[])
    if s['kind'] == 'sort':
        k=s['k']; dtype=getattr(torch,s['dtype'])
        ids=torch.stack([torch.randperm(max(16384,k*2),generator=g)[:k] for _ in range(t)]).to(torch.int32)
        scores=torch.randn((t,k),generator=g).to(dtype)
        scores[:,20:25]=0.;scores[:,25:30]=-0.
        ids[:,0]=-1;scores[:,1]=float('nan');scores[:,2]=float('inf');scores[:,3]=-float('inf')
        if s.get('distribution')=='ties': scores[:,30:]=torch.round(scores[:,30:])
        if s.get('distribution')=='ordered': scores[:,30:]=torch.arange(k-30).to(dtype)
        if edge:
            ids[:,4:8]=torch.tensor([8388607,8388608,16777215,16777216],dtype=torch.int32)
            if dtype==torch.float32:
                bits=torch.tensor([0x3f800000,0x3f800001,0x3f800002,0xbf800001,0x00800000,0x7f7fffff],dtype=torch.int64).to(torch.int32)
                scores[:,8:14]=bits.view(torch.float32)
        c.update(cpu_ids=ids,cpu_scores=scores,indices=strided(ids,s['layout']),scores=strided(scores,s['layout']),out=torch.empty((t,k,2),device='npu',dtype=torch.int32))
    else:
        h,d=s['h'],s['d']; output=torch.randn((t,h,d),generator=g).to(torch.bfloat16)
        lse=torch.randn((t,h,1),generator=g)*4
        sums=torch.rand((t,h,1),generator=g)*16+.1 if s['stats'] else None
        if edge and not s['stats']:
            bits=torch.tensor([0,0x80000000,1,0x007fffff,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00000],dtype=torch.int64).to(torch.int32)
            lse.reshape(-1)[:bits.numel()]=bits.view(torch.float32)
            wire_bits=torch.tensor([0,32768,1,127,32639,32640,65408,32704],dtype=torch.int32).to(torch.int16)
            output.reshape(-1)[:wire_bits.numel()]=wire_bits.view(torch.bfloat16)
        c.update(cpu_output=output,cpu_lse=lse,cpu_sum=sums,output=strided(output,s['layout']),lse=lse.npu(),sums=None if sums is None else sums.npu(),out=torch.empty((s['world'],h//s['world'],t,d+4),dtype=torch.bfloat16,device='npu'))
    return c


def invoke(c):
    s=c['spec']; t=s['t']; nc=c['vector_cores']
    if s['kind']=='sort':
        ix,sc=c['indices'],c['scores'];fn=base._pack_sorted_candidates if c['impl']=='baseline' else cand._pack_sorted_candidates_radix
        result=fn[(min(t,nc),)](ix,sc,c['out'],t,INDEX_ROW=ix.stride(0),INDEX_COL=ix.stride(1),SCORE_ROW=sc.stride(0),SCORE_COL=sc.stride(1),WORLD=s['world'],RANK=s['rank'],INTERLEAVE=s['interleave'],LOG_BLOCK=int(math.log2(s['k'])),multibuffer=False)
        c['compiled']=[result]
    else:
        def launch(fn,grid,*args,**kwargs):
            if c['impl']!='baseline' and fn is base._pack_dcp_output_lse_batched_kernel:
                fn=dcp_iter4._pack_dcp_recurrence
            c['compiled']=[fn[grid](*args,**kwargs)]
        base._launch_pack_dcp_output_lse(c['output'],c['lse'],c['sums'],c['out'],1,nc,launch)
    return c['out']


def check(c):
    s=c['spec'];a=c['out'].cpu()
    if s['kind']=='sort':
        ids=c['cpu_ids'].long();score=c['cpu_scores'].float()
        valid=(ids>=0)&torch.isfinite(score);score=torch.where(valid,score,-float('inf'))
        ids_order=torch.argsort(torch.where(valid,ids,torch.iinfo(torch.int64).max),dim=1,stable=True)
        order=torch.gather(ids_order,1,torch.argsort(torch.gather(score,1,ids_order),dim=1,descending=True,stable=True))
        ids=torch.gather(ids,1,order);score=torch.gather(score,1,order)
        mapped=ids+(ids//s['interleave']*(s['world']-1)+s['rank'])*s['interleave']
        mapped=torch.where(torch.isfinite(score),mapped,-1).int()
        expected=torch.stack((mapped,score.contiguous().view(torch.int32)),dim=-1)
        return dict(passed=torch.equal(a,expected),mismatches=int((a!=expected).sum()),rule='bit_exact_pairs')
    t,h,d=s['t'],s['h'],s['d'];matrix=a.reshape(h,t,d+4).permute(1,0,2)
    copied=torch.equal(matrix[:,:,:d].contiguous().view(torch.int16),c['cpu_output'].contiguous().view(torch.int16));fields=matrix[:,:,d:]
    if not s['stats']:
        exact=torch.equal(fields,encode_cpu(c['cpu_lse']))
        return dict(passed=copied and exact,output_exact=copied,wire_exact=exact)
    f=fields.double();code=f[:,:,0];sig=f[:,:,1]*65536+f[:,:,2]*256+f[:,:,3]
    decoded=torch.sign(code)*torch.ldexp(sig,(code.abs()-151).int());decoded=torch.where((code==128)&(sig==0),0.,decoded)
    expected=c['cpu_lse'].double().squeeze(-1)+torch.log(c['cpu_sum'].double().squeeze(-1))
    error=(decoded-expected).abs();ratio=float((error<=2**-16+2**-10*expected.abs()).double().mean());maximum=float(error.max())
    return dict(passed=copied and ratio>=.99 and maximum<=.01,output_exact=copied,lse_match_ratio=ratio,lse_max_error=maximum)


def kernel_name(params):
    case_id,impl=params.split(':');s=SPECS[case_id]
    if s['kind']=='sort':return '_pack_sorted_candidates' if impl=='baseline' else '_pack_sorted_candidates_radix'
    batched=s['t']*s['h']>=8*cores() and (s['layout']!='feature2' or (s['d']==512 and s['t']>=128))
    if impl!='baseline' and batched:return '_pack_dcp_recurrence'
    return '_pack_dcp_output_lse_batched_kernel' if batched else '_pack_dcp_output_lse_kernel'


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='all');p.add_argument('--out',default='results/optimization/validation.json');a=p.parse_args()
    torch.npu.set_device(0);results=[]
    selected=list(SPECS) if a.cases=='all' else a.cases.split(',')
    for case_id in selected:
        s=SPECS[case_id]
        for edge in ((False,True) if s['kind']=='sort' or not s['stats'] else (False,)):
            for impl in ('baseline','candidate'):
                c=make_case(case_id+':'+impl,edge)
                try:
                    invoke(c);torch.npu.synchronize();r=check(c)
                except Exception as exc:
                    # Compiler errors have not launched a device task. Preserve and
                    # continue independent cases; runtime/precision errors still stop.
                    if type(exc).__name__ not in ('MLIRCompilationError','CompilationError','OutOfResources'):raise
                    r=dict(passed=False,status='compile_failed',error_type=type(exc).__name__,error=str(exc))
                record=dict(case_id=case_id,impl=impl,edge=edge,spec=s,**r,strides={k:list(c[k].stride()) for k in ('indices','scores','output','lse','sums','out') if k in c and c[k] is not None})
                results.append(record);print(json.dumps(record),flush=True)
                path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(status='running' if r['passed'] else 'failed',cases=results),indent=2))
                if not r['passed'] and r.get('status')!='compile_failed':raise RuntimeError('precision failed '+case_id+':'+impl)
    value=dict(status='passed' if all(r['passed'] for r in results) else 'complete_with_compile_failures',cases=results,vector_cores=cores(),binaries=binary_identity(),source_sha256=hashlib.sha256((ROOT/'dcp_iter4.py').read_bytes()).hexdigest(), source_files_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('baseline_kernels.py','candidate_kernels.py','dcp_iter4.py','general_benchmark.py')})
    Path(a.out).write_text(json.dumps(value,indent=2));print('VALIDATION_'+value['status'].upper(),len(results),flush=True)


if __name__=='__main__':main()
