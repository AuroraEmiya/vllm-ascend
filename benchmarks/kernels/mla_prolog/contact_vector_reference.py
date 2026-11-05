# SPDX-License-Identifier: Apache-2.0
"""CPU-only public equations and state oracle; never used inside device forward."""
import math
import torch

def make_case(c, seed_offset=0):
    b,h,n,q,k,d,r=(c[x] for x in ('batch','hidden','heads','query_rank','kv_rank','nope_dim','rope_dim'))
    g=torch.Generator().manual_seed(c['seed']+seed_offset)
    def normal(shape,scale=1.):
        return (torch.randn(shape,generator=g)*scale).to(torch.bfloat16)
    angles=torch.randn((b,1,r//2),generator=g)
    data=dict(x=normal((b,1,h)),wdq=normal((h,q),h**-.5),wuq=normal((q,n*(d+r)),q**-.5),
        wuk=normal((n,d,k),d**-.5),wdkv=normal((h,k+r),h**-.5),gq=normal((q,),.1)+1,
        gkv=normal((k,),.1)+1,sin=torch.cat((angles.sin(),angles.sin()),-1).bfloat16(),
        cos=torch.cat((angles.cos(),angles.cos()),-1).bfloat16(),index=(torch.arange(b)*3+1).reshape(b,1))
    if c.get('zeros'):data['x'].zero_()
    if c.get('zero_gamma'):data['gq'].zero_();data['gkv'].zero_()
    page=c['block_size']; slots=math.ceil(max(page,b*3+2)/page)*page
    if seed_offset:data['index']=(slots-1-data['index']).contiguous()
    assert data['index'].unique().numel()==b and data['index'].min()>=0 and data['index'].max()<slots
    return data,normal((slots//page,page,1,k)),normal((slots//page,page,1,r))

def rope(x,sine,cosine):
    base=x.reshape(*x.shape[:-1],-1,2).transpose(-1,-2).flatten(-2)
    left,right=base.chunk(2,-1)
    return base*cosine+torch.cat((-right,left),-1)*sine

def oracle(data,c,initial_kv,initial_kr):
    x,wdq,wuq,wuk,wdkv=(data[z].double() for z in ('x','wdq','wuq','wuk','wdkv'))
    def rms(z,g):return z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*g.double()
    cq=rms(x@wdq,data['gq'])
    up=(cq@wuq).reshape(c['batch'],1,c['heads'],c['nope_dim']+c['rope_dim'])
    query=torch.einsum('bsnd,ndh->bsnh',up[...,:c['nope_dim']],wuk)
    sine,cosine=data['sin'].double(),data['cos'].double()
    qr=rope(up[...,c['nope_dim']:],sine.unsqueeze(-2),cosine.unsqueeze(-2))
    down=x@wdkv
    kv=rms(down[...,:c['kv_rank']],data['gkv'])
    kr=rope(down[...,c['kv_rank']:],sine,cosine)
    out_kv,out_kr=initial_kv.double().clone(),initial_kr.double().clone()
    out_kv.reshape(-1,c['kv_rank'])[data['index'].flatten()]=kv.flatten(0,1)
    out_kr.reshape(-1,c['rope_dim'])[data['index'].flatten()]=kr.flatten(0,1)
    return query,qr,out_kv,out_kr

def compare(x,ref):
    value=x.cpu().double();error=(value-ref).abs();allowed=2**-6*(1+ref.abs())
    finite=bool(torch.isfinite(value).all()); matched=(error<=allowed).double().mean().item() if error.numel() else 1.
    maximum=error.max().item() if error.numel() else 0.
    return dict(shape=list(x.shape),dtype=str(x.dtype),max_abs_error=maximum,matched_ratio=matched,
        finite=finite,passed=finite and matched>=.99 and maximum<=1)

def check(outputs,caches,ref,data,c,initials):
    assert len(outputs)==5,'public V3 five-output ABI'
    details={}
    for name,x,want in zip(('query','query_rope'),outputs[:2],ref[:2]):
        assert x.dtype==torch.bfloat16 and list(x.shape)==list(want.shape)
        details[name]=compare(x,want)
    for name,z,dtype in zip(('dequant_scale_q_nope','query_norm','dequant_scale_q_norm'),outputs[2:],
        (torch.float32,torch.bfloat16,torch.float32)):
        assert z.dtype==dtype and z.numel()==0
        details[name]=dict(shape=list(z.shape),dtype=str(z.dtype),passed=True)
    touched=torch.zeros(initials[0].numel()//c['kv_rank'],dtype=torch.bool)
    touched[data['index'].flatten()]=True
    for name,x,want,initial,width in zip(('kv_cache','kr_cache'),caches,ref[2:],initials,(c['kv_rank'],c['rope_dim'])):
        result=compare(x,want)
        result['untouched_exact']=torch.equal(x.cpu().reshape(-1,width)[~touched],initial.reshape(-1,width)[~touched])
        result['written']=compare(x.cpu().reshape(-1,width)[touched],want.reshape(-1,width)[touched])
        result['passed'] &= result['untouched_exact'] and result['written']['passed']
        details[name]=result
    details['passed']=all(v['passed'] for v in details.values())
    return details
