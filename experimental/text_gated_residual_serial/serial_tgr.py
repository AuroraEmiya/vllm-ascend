"""Serial C+V TextGatedResidual, model-specific BF16 forward domain.

C is public torch.mm. V is XTgrVector, an independently written Ascend C op.
Weight objects and their preparation belong to the caller; no global cache.
"""
import math
import torch
import torch_npu
import tgr_ops


class TextGatedResidualSerial:
    def __init__(self,w_down,w_up,w_gate=None,channels=4,eps=1e-6):
        if channels!=4 or not math.isfinite(eps) or eps<=0:
            raise ValueError('supports channels=4 and finite positive eps')
        for w,shape in ((w_down,(320,10240)),(w_up,(10240,320)),(w_gate,(4,10240))):
            if w is not None and (tuple(w.shape)!=shape or w.dtype!=torch.bfloat16 or not w.is_contiguous()):
                raise ValueError('weights must be contiguous BF16 with model-specific shapes')
            if w is not None and w.device!=w_down.device:
                raise ValueError('all weights must be on the same device')
        self.wd=w_down.t(); self.wu=w_up.t(); self.wg=None if w_gate is None else w_gate.t()
        self.device=w_down.device; self.eps=eps; self.channels=channels
        self.dummy=torch.empty(1,device=self.device,dtype=torch.bfloat16)
        self.dummy_gamma=torch.empty(1,device=self.device,dtype=torch.float32)

    def __call__(self,x,gamma=None,need_backward=False):
        if x.ndim!=3 or x.shape[-1]!=10240 or not 1<=x.shape[0]*x.shape[1]<=2048 or x.shape[0]<1 or x.shape[1]<1:
            raise ValueError('x must be [B,S,10240], B,S>0, B*S<=2048')
        if x.dtype!=torch.bfloat16 or x.device!=self.device or not x.is_contiguous():
            raise ValueError('x must be contiguous BF16 on the weight device')
        if gamma is not None and (gamma.dtype!=torch.float32 or tuple(gamma.shape)!=(10240,) or gamma.device!=x.device or not gamma.is_contiguous()):
            raise ValueError('gamma must be contiguous FP32[10240] on the input device')
        b,s,_=x.shape; n=b*s
        xf=x.view(n,10240)
        empty=x.new_empty(0); empty_f=x.new_empty(0,dtype=torch.float32)
        norm=torch.empty_like(xf); inv=x.new_empty((n,4),dtype=torch.float32)
        torch.ops.custom.tgr_vector_out(xf,self.dummy_gamma if gamma is None else gamma,
            self.dummy,self.dummy,0,4,self.eps,norm,inv,empty)
        down=torch.mm(norm,self.wd)
        act=torch.empty_like(down)
        torch.ops.custom.tgr_vector_out(down,self.dummy_gamma,self.dummy,self.dummy,
            1,4,self.eps,act,empty_f,empty)
        up=torch.mm(act,self.wu)
        gate=self.dummy if self.wg is None else torch.mm(norm,self.wg)
        mixed=x.new_empty((n,2560))
        inject=empty if self.wg is None else x.new_empty((n,4))
        torch.ops.custom.tgr_vector_out(norm,self.dummy_gamma,up,gate,
            2,4,self.eps,mixed,empty_f,inject)
        return (mixed.view(b,s,2560),inject if self.wg is None else inject.view(b,s,4),
                inv.view(b,s,4) if need_backward else empty_f,
                down.view(b,s,320) if need_backward else empty,
                up.view(b,s,10240) if need_backward else empty)


def text_gated_residual(x,w_down,w_up,w_gate=None,channels=4,gamma=None,eps=1e-6,need_backward=False):
    """Public convenience interface; prepared class is preferred for repeated calls."""
    out=TextGatedResidualSerial(w_down,w_up,w_gate,channels,eps)(x,gamma,need_backward)
    if need_backward:
        return out if w_gate is not None else (out[0],out[2],out[3],out[4])
    return out[:2] if w_gate is not None else out[0]
