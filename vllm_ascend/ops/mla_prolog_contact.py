# SPDX-License-Identifier: Apache-2.0
"""BF16 S1 Contact serial: four torch.matmul and three Ascend C Vector calls.

Vector addressing owns all layout conversion. Dense projection weights use
the public V3 NZ input ABI; per-head UK is ND. Load before Graph capture.
There is no prepared tensor/weight ownership or SuperKernel scope/grouping.
"""
from pathlib import Path
import os
import torch

_extension=None
def load_contact_extension():
    """Use the integrated binding, or build the identical bridge for isolation."""
    global _extension
    if hasattr(torch.ops._C_ascend,'contact_norm_out'):return _extension
    import torch_npu
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[2]
    npu=Path(torch_npu.__file__).resolve().parent
    cann=Path(os.environ['ASCEND_HOME_PATH'])
    _extension=load(name='contact_vector_binding',sources=[
        str(root/'csrc/attention/mla_prolog_contact/torch_binding.cpp'),
        str(root/'csrc/aclnn_torch_adapter/NPUBridge.cpp'),
        str(root/'csrc/aclnn_torch_adapter/NPUStorageImpl.cpp')],
        extra_include_paths=[str(npu/'include'),str(cann/'include'),str(cann/'include/aclnn')],
        extra_cflags=['-O3','-DCONTACT_JIT_MODULE'],
        extra_ldflags=[f'-L{npu}/lib','-ltorch_npu',f'-L{cann}/lib64','-lascendcl'],verbose=True)
    return _extension

def mla_prolog_contact(x,wdq,wuq,wuk,wdkv,gq,gkv,sine,cosine,index,kv_cache,kr_cache):
    """V3 five-output ABI, PA_BSND nonquant and query_norm_flag=False."""
    b,s,h=x.shape
    n,d,k=wuk.shape
    r=sine.shape[-1]
    if s!=1:raise ValueError('Contact optimized path supports decode S1')
    dq=torch.matmul(x,wdq)
    dkv=torch.matmul(x,wdkv)
    cq=torch.empty(dq.shape,dtype=x.dtype,device=x.device)
    torch.ops._C_ascend.contact_norm_out(dq,dkv,gq,gkv,sine,cosine,index,kv_cache,kr_cache,cq)
    up=torch.matmul(cq,wuq)
    qc=torch.empty((n,b,d),dtype=x.dtype,device=x.device)
    qr=torch.empty((b,1,n,r),dtype=x.dtype,device=x.device)
    torch.ops._C_ascend.contact_split_out(up,sine,cosine,n,d,r,qc,qr)
    query_nt=torch.matmul(qc,wuk)
    query=torch.empty((b,1,n,k),dtype=x.dtype,device=x.device)
    torch.ops._C_ascend.contact_layout_out(query_nt,query)
    empty_float=torch.empty(0,dtype=torch.float32,device=x.device)
    empty_bf16=torch.empty(0,dtype=torch.bfloat16,device=x.device)
    return query,qr,empty_float,empty_bf16,empty_float
