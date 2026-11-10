"""Pinned PR microkernel inputs, independent CPU oracles and binary identity."""
import argparse, hashlib, json, math, os
from pathlib import Path
import torch
import torch_npu
import baseline_kernels as kernels

ROOT=Path(__file__).resolve().parent
SPECS={
 'sort144fp32':dict(kind='sort',t=144,dtype='float32',role='A3_native'),
 'sort152fp32':dict(kind='sort',t=152,dtype='float32',role='A3_native'),
 'dcp144stats':dict(kind='dcp',t=144,h=96,stats=True,role='A3_native'),
 'dcp152stats':dict(kind='dcp',t=152,h=96,stats=True,role='A3_native'),
 'dcp114lse':dict(kind='dcp',t=114,h=64,stats=False,role='A3_native'),
 'sort144bf16':dict(kind='sort',t=144,dtype='bfloat16',role='950_dtype_reference_on_A3'),
 'sort152bf16':dict(kind='sort',t=152,dtype='bfloat16',role='950_dtype_reference_on_A3'),
}

def cores():
    prop=kernels.triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
    return int(prop['num_vectorcore'])

def make_case(case_id,edge=False):
    s=SPECS[case_id];g=torch.Generator().manual_seed(20878+s['t'])
    t=s['t'];c=dict(case_id=case_id,spec=s,vector_cores=cores(),compiled=[])
    if s['kind']=='sort':
        # Unique IDs make the score-desc / ID-asc order fully defined, even for ties.
        ids=torch.stack([torch.randperm(16384,generator=g)[:2048] for _ in range(t)]).to(torch.int32).reshape(t,1,2048)
        scores=torch.randn((t,1,2048),generator=g).to(getattr(torch,s['dtype']))
        scores[:,:,20:25]=0
        scores[:,:,25:30]=-0.0
        ids[:,:,0]=-1;scores[:,:,1]=float('nan');scores[:,:,2]=float('inf');scores[:,:,3]=-float('inf')
        if edge: ids[:,:,4:8]=torch.tensor([8388607,8388608,16777215,16777216],dtype=torch.int32)
        c.update(cpu_ids=ids,cpu_scores=scores,indices=ids.npu(),scores=scores.npu(),out=torch.empty((t,2048,2),device='npu',dtype=torch.int32))
        c['launch_config']=dict(grid=[min(t,c['vector_cores'])],WORLD=8,RANK=0,INTERLEAVE=768,LOG_BLOCK=11,multibuffer=False,branch='radix_4pass' if edge and s['dtype']=='bfloat16' else ('radix_3pass' if s['dtype']=='bfloat16' else 'exact_bitonic_66stage'))
    else:
        h=s['h'];output=torch.randn((t,h,512),generator=g).to(torch.bfloat16)
        lse=torch.randn((t,h,1),generator=g,dtype=torch.float32)*4
        sums=torch.rand((t,h,1),generator=g,dtype=torch.float32)*16+.1 if s['stats'] else None
        if edge and not s['stats']:
            raw=torch.tensor([0,0x80000000,1,0x007fffff,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00000],dtype=torch.int64).to(torch.int32)
            lse.reshape(-1)[:raw.numel()]=raw.view(torch.float32)
        c.update(cpu_output=output,cpu_lse=lse,cpu_sum=sums,output=output.npu(),lse=lse.npu(),sums=None if sums is None else sums.npu(),out=torch.empty((8,h//8,t,516),dtype=torch.bfloat16,device='npu'))
        c['launch_config']=dict(grid=[c['vector_cores']],BLOCK_ROWS=32 if t>=128 else 8,BLOCK_D=512,OUTPUT_HEAD_MAJOR=True,FROM_STATS=s['stats'],LSE_PACK_DIM=4,scatter_dim=1)
    return c

def invoke(c):
    if c['spec']['kind']=='sort':
        ix,sc=c['indices'],c['scores']
        k=kernels._pack_sorted_candidates[(min(c['spec']['t'],c['vector_cores']),)](ix,sc,c['out'],c['spec']['t'],INDEX_ROW=ix.stride(0),INDEX_COL=ix.stride(2),SCORE_ROW=sc.stride(0),SCORE_COL=sc.stride(2),WORLD=8,RANK=0,INTERLEAVE=768,LOG_BLOCK=11,multibuffer=False)
        c['compiled']=[k]
    else:
        def launch(fn,grid,*args,**kwargs):
            assert fn is kernels._pack_dcp_output_lse_batched_kernel,'production dispatcher selected a different kernel'
            k=fn[grid](*args,**kwargs);c['compiled']=[k]
        kernels._launch_pack_dcp_output_lse(c['output'],c['lse'],c['sums'],c['out'],1,c['vector_cores'],launch)
    return c['out']

def encode_cpu(lse):
    raw=lse.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
    exponent=(raw>>23)&255;fraction=raw&0x7fffff;finite=exponent<255
    sig=fraction|torch.where(exponent==0,0,0x800000)
    sig=torch.where(finite,sig,0)
    code=(torch.maximum(exponent,torch.ones_like(exponent))+1).to(torch.float32)
    code=torch.where(raw>>31!=0,-code,code)
    code=torch.where((raw&0x7fffffff)==0,128.,code)
    code=torch.where(finite,code,0.)
    return torch.stack((code,((sig>>16)&255).float(),((sig>>8)&255).float(),(sig&255).float()),dim=-1).squeeze(-2).to(torch.bfloat16)

def check(c):
    s=c['spec'];actual=c['out'].cpu()
    if s['kind']=='sort':
        ids=c['cpu_ids'].reshape(s['t'],2048).to(torch.int64);scores=c['cpu_scores'].float().reshape(s['t'],2048)
        valid=(ids>=0)&torch.isfinite(scores);scores=torch.where(valid,scores,-float('inf'))
        keys=torch.where(valid,ids,torch.iinfo(torch.int64).max)
        by_id=torch.argsort(keys,dim=-1,stable=True)
        by_score=torch.argsort(torch.gather(scores,1,by_id),dim=-1,descending=True,stable=True)
        order=torch.gather(by_id,1,by_score);ids=torch.gather(ids,1,order);scores=torch.gather(scores,1,order)
        mapped=ids+(ids//768*7)*768
        mapped=torch.where(torch.isfinite(scores),mapped,-1).to(torch.int32)
        expected=torch.stack((mapped,scores.contiguous().view(torch.int32)),dim=-1)
        ok=torch.equal(actual,expected)
        return dict(passed=ok,rule='bit_exact_pairs',mismatch_elements=int((actual!=expected).sum()))
    t,h=s['t'],s['h'];matrix=actual.reshape(h,t,516).permute(1,0,2)
    copy_ok=torch.equal(matrix[:,:,:512],c['cpu_output'])
    fields=matrix[:,:,512:]
    if not s['stats']:
        expected=encode_cpu(c['cpu_lse']);lse_ok=torch.equal(fields,expected)
        return dict(passed=copy_ok and lse_ok,output_bit_exact=copy_ok,lse_fields_bit_exact=lse_ok)
    f=fields.double();code=f[:,:,0];sig=f[:,:,1]*65536+f[:,:,2]*256+f[:,:,3]
    decoded=torch.sign(code)*torch.ldexp(sig,(code.abs()-151).to(torch.int32))
    decoded=torch.where((code==128)&(sig==0),0.,decoded)
    expected=c['cpu_lse'].double().squeeze(-1)+torch.log(c['cpu_sum'].double().squeeze(-1))
    error=(decoded-expected).abs();allowed=2**-16+2**-10*expected.abs()
    ratio=float((error<=allowed).double().mean());max_error=float(error.max())
    return dict(passed=copy_ok and ratio>=.99 and max_error<=.01,output_bit_exact=copy_ok,lse_match_ratio=ratio,lse_max_abs_error=max_error,lse_oracle='CPU float64 max+log(sum)',rtol=2**-10,atol=2**-16)

def binary_identity():
    cache=Path(os.environ.get('TRITON_CACHE_DIR','/work/.triton-cache'))
    return {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in cache.rglob('*') if f.is_file() and f.suffix=='.npubin'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/validation.json');p.add_argument('--case',default=None);a=p.parse_args()
    torch.npu.set_device(0);results=[]
    for case_id in ([a.case] if a.case else SPECS):
        for edge in ((False,True) if SPECS[case_id]['kind']=='sort' or not SPECS[case_id]['stats'] else (False,)):
            c=make_case(case_id,edge);invoke(c);torch.npu.synchronize();r=check(c)
            record=dict(case_id=case_id,edge=edge,**r,launch_config=c['launch_config'],strides={k:list(c[k].stride()) for k in ('indices','scores','output','lse','sums','out') if k in c and c[k] is not None})
            results.append(record);print(json.dumps(record),flush=True)
            if not r['passed']:raise RuntimeError('precision failed '+case_id)
    binaries=binary_identity()
    value=dict(status='passed',cases=results,binaries=binaries,source_sha256=hashlib.sha256((ROOT/'baseline_kernels.py').read_bytes()).hexdigest(),vector_cores=cores())
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print('VALIDATION_PASSED',len(results),'BINARIES',len(binaries),flush=True)

if __name__=='__main__':main()
