// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#include "kernel_operator.h"
#include "../qwen_hc_launch.h"

using namespace AscendC;

namespace {
constexpr uint32_t VECTOR_CORES = 40;
constexpr uint32_t TILE = 512;

__aicore__ inline uint32_t Align(uint32_t n) { return (n + 15) / 16 * 16; }

template<HardEvent E> __aicore__ inline void Sync() {
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

template<typename T>
__aicore__ inline void Load(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset,
                            uint32_t count) {
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(Align(count) - count), 0};
    DataCopyPad(dst, src[offset], copy, pad);
}
template<typename T>
__aicore__ inline void Store(GlobalTensor<T> dst, uint32_t offset, LocalTensor<T> src,
                             uint32_t count) {
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[offset], src, copy);
}

template<typename T>
class CombineNorm {
public:
    __aicore__ inline void Run(GM_ADDR residual, GM_ADDR block, GM_ADDR injection,
        GM_ADDR weight, GM_ADDR combined, GM_ADDR normalized, HcParams p) {
        GlobalTensor<T> r, b, i, w, s, n;
        r.SetGlobalBuffer((__gm__ T*)residual);
        b.SetGlobalBuffer((__gm__ T*)block);
        i.SetGlobalBuffer((__gm__ T*)injection);
        w.SetGlobalBuffer((__gm__ T*)weight);
        s.SetGlobalBuffer((__gm__ T*)combined);
        n.SetGlobalBuffer((__gm__ T*)normalized);
        const uint32_t size = (p.hidden + 63) / 64 * 64;
        TPipe pipe;
        TBuf<TPosition::VECCALC> rt, bt, wt, nt, it, xf, tmp, work, small, broad;
        pipe.InitBuffer(rt, size * sizeof(T));
        pipe.InitBuffer(bt, size * sizeof(T));
        pipe.InitBuffer(wt, size * sizeof(T));
        pipe.InitBuffer(nt, size * sizeof(T));
        pipe.InitBuffer(it, 16 * sizeof(T));
        pipe.InitBuffer(xf, size * sizeof(float));
        pipe.InitBuffer(tmp, size * sizeof(float));
        pipe.InitBuffer(work, size * sizeof(float));
        pipe.InitBuffer(small, 32 * sizeof(float));
        pipe.InitBuffer(broad, 64 * sizeof(float));
        auto lr = rt.Get<T>(); auto lb = bt.Get<T>(); auto lw = wt.Get<T>();
        auto ln = nt.Get<T>(); auto li = it.Get<T>();
        auto x = xf.Get<float>(); auto t = tmp.Get<float>(); auto ws = work.Get<float>();
        auto sc = small.Get<float>();
        auto bc = broad.Get<float>();
        for (uint32_t g = GetBlockIdx(); g < p.rows * p.groups; g += GetBlockNum()) {
            const uint32_t row = g / p.groups, group = g % p.groups;
            // Reuse is safe only after the previous MTE3 output finishes.
            Sync<HardEvent::MTE3_MTE2>();
            Load(lr, r, row * p.residual_stride + group * p.hidden, p.hidden);
            Load(lb, b, row * p.block_stride, p.hidden);
            Load(lw, w, p.shared_weight ? 0 : group * p.hidden, p.hidden);
            if (p.has_injection) Load(li, i, row * p.injection_stride + group, 1);
            Sync<HardEvent::MTE2_V>();
            Cast(x, lr, RoundMode::CAST_NONE, p.hidden);
            Cast(t, lb, RoundMode::CAST_NONE, p.hidden);
            PipeBarrier<PIPE_V>();
            if (p.has_injection) {
                Cast(sc, li, RoundMode::CAST_NONE, 16);
                PipeBarrier<PIPE_V>();
                Muls(sc, sc, -1.0f / p.groups, 16);
                PipeBarrier<PIPE_V>();
                Exp(sc, sc, 16); PipeBarrier<PIPE_V>();
                Adds(sc, sc, 1.0f, 16); PipeBarrier<PIPE_V>();
                Reciprocal(sc, sc, 16); PipeBarrier<PIPE_V>();
                Muls(sc, sc, 2.0f, 16);
                PipeBarrier<PIPE_V>();
                // Brcb replicates the first FP32 value into one 32B block.
                // Reuse that block in every lane/repeat, keeping the reduction
                // result in V rather than stalling the scalar pipeline.
                Brcb(bc, sc, 1, {1, 8}); PipeBarrier<PIPE_V>();
                MulAddDst(x, t, bc, uint64_t(64), (p.hidden + 63) / 64,
                    {1, 1, 0, 8, 8, 0}); PipeBarrier<PIPE_V>();
            }
            if (!p.has_injection) { Add(x, x, t, p.hidden); PipeBarrier<PIPE_V>(); }
            // Preserve the original low-precision combine materialization before RMS.
            Cast(lr, x, RoundMode::CAST_RINT, p.hidden); PipeBarrier<PIPE_V>();
            Cast(x, lr, RoundMode::CAST_NONE, p.hidden); PipeBarrier<PIPE_V>();
            Mul(t, x, x, p.hidden); PipeBarrier<PIPE_V>();
            ReduceSum(sc, t, ws, static_cast<int32_t>(p.hidden)); PipeBarrier<PIPE_V>();
            Muls(sc, sc, 1.0f / p.hidden, 8); PipeBarrier<PIPE_V>();
            Adds(sc, sc, p.eps, 8); PipeBarrier<PIPE_V>();
            Rsqrt(sc, sc, 8);
            PipeBarrier<PIPE_V>();
            Brcb(bc, sc, 1, {1, 8}); PipeBarrier<PIPE_V>();
            Mul(x, x, bc, uint64_t(64), (p.hidden + 63) / 64,
                {1, 1, 0, 8, 8, 0});
            Cast(t, lw, RoundMode::CAST_NONE, p.hidden); PipeBarrier<PIPE_V>();
            MulAddDst(x, x, t, p.hidden); PipeBarrier<PIPE_V>();
            Cast(ln, x, RoundMode::CAST_RINT, p.hidden);
            Sync<HardEvent::V_MTE3>();
            Store(s, g * p.hidden, lr, p.hidden);
            Store(n, g * p.hidden, ln, p.hidden);
        }
        Sync<HardEvent::MTE3_S>();
    }
};

template<typename T>
class HcSilu {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, HcParams p) {
        GlobalTensor<T> in, out;
        in.SetGlobalBuffer((__gm__ T*)input); out.SetGlobalBuffer((__gm__ T*)output);
        TPipe pipe; TBuf<TPosition::VECCALC> it, out_buf, xf, tf;
        const uint32_t size = Align(p.hidden);
        pipe.InitBuffer(it, size * sizeof(T)); pipe.InitBuffer(out_buf, size * sizeof(T));
        pipe.InitBuffer(xf, size * 4); pipe.InitBuffer(tf, size * 4);
        auto a = it.Get<T>(); auto b = out_buf.Get<T>();
        auto x = xf.Get<float>(); auto t = tf.Get<float>();
        for (uint32_t row = GetBlockIdx(); row < p.rows; row += GetBlockNum()) {
            Sync<HardEvent::MTE3_MTE2>();
            Load(a, in, row * p.residual_stride, p.hidden);
            Sync<HardEvent::MTE2_V>();
            Cast(x, a, RoundMode::CAST_NONE, p.hidden); PipeBarrier<PIPE_V>();
            Muls(x, x, 1.0f / p.groups, p.hidden); PipeBarrier<PIPE_V>();
            Muls(t, x, -1.0f, p.hidden); PipeBarrier<PIPE_V>();
            Exp(t, t, p.hidden); PipeBarrier<PIPE_V>();
            Adds(t, t, 1.0f, p.hidden); PipeBarrier<PIPE_V>();
            Reciprocal(t, t, p.hidden); PipeBarrier<PIPE_V>();
            Mul(x, x, t, p.hidden); PipeBarrier<PIPE_V>();
            Cast(b, x, RoundMode::CAST_RINT, p.hidden);
            Sync<HardEvent::V_MTE3>();
            Store(out, row * p.hidden, b, p.hidden);
        }
        Sync<HardEvent::MTE3_S>();
    }
};

template<typename T>
class GateMix {
public:
    __aicore__ inline void Run(GM_ADDR values, GM_ADDR gates, GM_ADDR output, HcParams p) {
        GlobalTensor<T> in, gate, out;
        in.SetGlobalBuffer((__gm__ T*)values);gate.SetGlobalBuffer((__gm__ T*)gates);
        out.SetGlobalBuffer((__gm__ T*)output);
        TPipe pipe; TBuf<TPosition::VECCALC> it, gt, out_buf, xf, tf, af;
        pipe.InitBuffer(it, TILE * sizeof(T)); pipe.InitBuffer(gt, TILE * sizeof(T));
        pipe.InitBuffer(out_buf, TILE * sizeof(T)); pipe.InitBuffer(xf, TILE * 4);
        pipe.InitBuffer(tf, TILE * 4);pipe.InitBuffer(af, TILE * 4);
        auto a=it.Get<T>(); auto b=gt.Get<T>(); auto o=out_buf.Get<T>();
        auto x=xf.Get<float>(); auto t=tf.Get<float>(); auto acc=af.Get<float>();
        const uint32_t tiles = (p.hidden + TILE - 1) / TILE;
        for (uint32_t task=GetBlockIdx();task<p.rows*tiles;task+=GetBlockNum()) {
            const uint32_t row=task/tiles, offset=(task%tiles)*TILE;
            const uint32_t count=(p.hidden-offset<TILE) ? p.hidden-offset : TILE;
            Sync<HardEvent::MTE3_V>();
            Duplicate(acc,0.0f,count);PipeBarrier<PIPE_V>();
            for (uint32_t k=0;k<p.groups;k++) {
                Sync<HardEvent::V_MTE2>();
                Load(a,in,row*p.residual_stride+k*p.hidden+offset,count);
                Load(b,gate,row*p.block_stride+k*p.hidden+offset,count);
                Sync<HardEvent::MTE2_V>();
                Cast(x,a,RoundMode::CAST_NONE,count);
                Cast(t,b,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
                Muls(t,t,-1.0f,count);PipeBarrier<PIPE_V>();
                Exp(t,t,count);PipeBarrier<PIPE_V>();
                Adds(t,t,1.0f,count);PipeBarrier<PIPE_V>();
                Reciprocal(t,t,count);PipeBarrier<PIPE_V>();
                Mul(x,x,t,count);PipeBarrier<PIPE_V>();
                Add(acc,acc,x,count);PipeBarrier<PIPE_V>();
            }
            Muls(acc,acc,1.0f/p.groups,count);PipeBarrier<PIPE_V>();
            Cast(o,acc,RoundMode::CAST_RINT,count);
            Sync<HardEvent::V_MTE3>();
            Store(out,row*p.hidden+offset,o,count);
        }
        Sync<HardEvent::MTE3_S>();
    }
};
// Whole rows of HC branches share block DMA and the flat elementwise chain.
// Each branch keeps its own RMS reduction and the BF16 combine boundary.
template<typename T>
class BatchedCombineNorm {
public:
    __aicore__ inline void Run(GM_ADDR residual,GM_ADDR block,GM_ADDR injection,
        GM_ADDR weight,GM_ADDR combined,GM_ADDR normalized,HcParams p) {
        GlobalTensor<T> r,b,i,w,s,n;
        r.SetGlobalBuffer((__gm__ T*)residual);b.SetGlobalBuffer((__gm__ T*)block);
        i.SetGlobalBuffer((__gm__ T*)injection);w.SetGlobalBuffer((__gm__ T*)weight);
        s.SetGlobalBuffer((__gm__ T*)combined);n.SetGlobalBuffer((__gm__ T*)normalized);
        const uint32_t flat=p.row_batch*p.hidden;
        TPipe pipe;
        TBuf<TPosition::VECCALC> rt,bt,wt,nt,it,xf,tmp,work,small,broad;
        pipe.InitBuffer(rt,flat*sizeof(T)+256);
        pipe.InitBuffer(bt,p.hidden*sizeof(T)+256);
        pipe.InitBuffer(wt,flat*sizeof(T)+256);
        pipe.InitBuffer(nt,flat*sizeof(T)+256);
        pipe.InitBuffer(it,16*sizeof(T)+256);
        pipe.InitBuffer(xf,flat*sizeof(float)+256);
        pipe.InitBuffer(tmp,flat*sizeof(float)+256);
        pipe.InitBuffer(work,p.hidden*sizeof(float)+256);
        pipe.InitBuffer(small,32*sizeof(float));
        pipe.InitBuffer(broad,64*sizeof(float));
        auto lr=rt.Get<T>();auto lb=bt.Get<T>();auto lw=wt.Get<T>();
        auto ln=nt.Get<T>();auto li=it.Get<T>();
        auto x=xf.Get<float>();auto t=tmp.Get<float>();auto ws=work.Get<float>();
        auto sc=small.Get<float>();auto bc=broad.Get<float>();
        const uint32_t chunks=p.groups/p.row_batch;
        for(uint32_t task=GetBlockIdx();task<p.rows*chunks;task+=GetBlockNum()) {
            const uint32_t row=task/chunks,first=(task%chunks)*p.row_batch;
            Sync<HardEvent::MTE3_MTE2>();
            Load(lr,r,row*p.residual_stride+first*p.hidden,flat);
            Load(lb,b,row*p.block_stride,p.hidden);
            Load(lw,w,p.shared_weight?0:first*p.hidden,p.shared_weight?p.hidden:flat);
            if(p.has_injection)Load(li,i,row*p.injection_stride+first,p.row_batch);
            Sync<HardEvent::MTE2_V>();
            Cast(x,lr,RoundMode::CAST_NONE,flat);
            Cast(ws,lb,RoundMode::CAST_NONE,p.hidden);
            if(p.has_injection) {
                Cast(sc,li,RoundMode::CAST_NONE,16);PipeBarrier<PIPE_V>();
                Muls(sc,sc,-1.0f/p.groups,16);PipeBarrier<PIPE_V>();
                Exp(sc,sc,16);PipeBarrier<PIPE_V>();
                Adds(sc,sc,1.0f,16);PipeBarrier<PIPE_V>();
                Reciprocal(sc,sc,16);PipeBarrier<PIPE_V>();
                Muls(sc,sc,2.0f,16);
            } else {
                Duplicate(sc,1.0f,16);
            }
            PipeBarrier<PIPE_V>();
            Brcb(bc,sc,1,{1,8});PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<p.row_batch;j++) {
                MulAddDst(x[j*p.hidden],ws,bc[j*8],uint64_t(64),p.hidden/64,
                    {1,1,0,8,8,0});
            }
            PipeBarrier<PIPE_V>();
            Cast(lr,x,RoundMode::CAST_RINT,flat);PipeBarrier<PIPE_V>();
            Cast(x,lr,RoundMode::CAST_NONE,flat);PipeBarrier<PIPE_V>();
            Mul(t,x,x,flat);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<p.row_batch;j++) {
                ReduceSum(sc[j*8],t[j*p.hidden],ws,static_cast<int32_t>(p.hidden));
                PipeBarrier<PIPE_V>();
            }
            const uint32_t scalars=p.row_batch*8;
            Muls(sc,sc,1.0f/p.hidden,scalars);PipeBarrier<PIPE_V>();
            Adds(sc,sc,p.eps,scalars);PipeBarrier<PIPE_V>();
            Rsqrt(sc,sc,scalars);PipeBarrier<PIPE_V>();
            for(uint32_t j=0;j<p.row_batch;j++) {
                Brcb(bc,sc[j*8],1,{1,8});PipeBarrier<PIPE_V>();
                Mul(x[j*p.hidden],x[j*p.hidden],bc,uint64_t(64),p.hidden/64,
                    {1,1,0,8,8,0});PipeBarrier<PIPE_V>();
            }
            if(p.shared_weight) {
                Cast(ws,lw,RoundMode::CAST_NONE,p.hidden);PipeBarrier<PIPE_V>();
                for(uint32_t j=0;j<p.row_batch;j++) {
                    MulAddDst(x[j*p.hidden],x[j*p.hidden],ws,p.hidden);
                }
                PipeBarrier<PIPE_V>();
            } else {
                Cast(t,lw,RoundMode::CAST_NONE,flat);PipeBarrier<PIPE_V>();
                MulAddDst(x,x,t,flat);PipeBarrier<PIPE_V>();
            }
            Cast(ln,x,RoundMode::CAST_RINT,flat);
            Sync<HardEvent::V_MTE3>();
            Store(s,row*p.groups*p.hidden+first*p.hidden,lr,flat);
            Store(n,row*p.groups*p.hidden+first*p.hidden,ln,flat);
        }
        Sync<HardEvent::MTE3_S>();
    }
};

template<typename T>
class BatchedSilu {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, HcParams p) {
        GlobalTensor<T> in, out;
        in.SetGlobalBuffer((__gm__ T*)input);
        out.SetGlobalBuffer((__gm__ T*)output);
        const uint32_t capacity = p.row_batch * p.hidden;
        TPipe pipe;
        TBuf<TPosition::VECCALC> it, out_buf, xf, tf;
        pipe.InitBuffer(it, capacity * sizeof(T) + 256);
        pipe.InitBuffer(out_buf, capacity * sizeof(T) + 256);
        pipe.InitBuffer(xf, capacity * sizeof(float) + 256);
        pipe.InitBuffer(tf, capacity * sizeof(float) + 256);
        auto a=it.Get<T>(); auto b=out_buf.Get<T>();
        auto x=xf.Get<float>(); auto t=tf.Get<float>();
        const uint32_t tasks=(p.rows+p.row_batch-1)/p.row_batch;
        for(uint32_t task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
            const uint32_t row=task*p.row_batch;
            const uint32_t rows=(p.rows-row<p.row_batch)?p.rows-row:p.row_batch;
            const uint32_t count=rows*p.hidden;
            Sync<HardEvent::MTE3_MTE2>();
            DataCopyExtParams cp{static_cast<uint16_t>(rows),p.hidden*uint32_t(sizeof(T)),
                (p.residual_stride-p.hidden)*uint32_t(sizeof(T)),0,0};
            DataCopyPadExtParams<T> pad{false,0,0,0};
            DataCopyPad(a,in[row*p.residual_stride],cp,pad);
            Sync<HardEvent::MTE2_V>();
            Cast(x,a,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
            Muls(x,x,1.0f/p.groups,count); PipeBarrier<PIPE_V>();
            Muls(t,x,-1.0f,count); PipeBarrier<PIPE_V>();
            Exp(t,t,count); PipeBarrier<PIPE_V>();
            Adds(t,t,1.0f,count); PipeBarrier<PIPE_V>();
            Reciprocal(t,t,count); PipeBarrier<PIPE_V>();
            Mul(x,x,t,count); PipeBarrier<PIPE_V>();
            Cast(b,x,RoundMode::CAST_RINT,count);
            Sync<HardEvent::V_MTE3>();
            Store(out,row*p.hidden,b,count);
        }
        Sync<HardEvent::MTE3_S>();
    }
};

// HC count is small in the model. Batch [K,tile] DMA and activation instead of
// issuing a separate copy/activation/synchronization chain for every branch.
template<typename T>
class BatchedGateMix {
public:
    __aicore__ inline void Run(GM_ADDR values, GM_ADDR gates, GM_ADDR output,HcParams p) {
        GlobalTensor<T> in, gate, out;
        in.SetGlobalBuffer((__gm__ T*)values);
        gate.SetGlobalBuffer((__gm__ T*)gates);
        out.SetGlobalBuffer((__gm__ T*)output);
        TPipe pipe;
        TBuf<TPosition::VECCALC> it,gt,out_buf,xf,tf,af;
        const uint32_t capacity=p.groups*p.tile;
        pipe.InitBuffer(it,capacity*sizeof(T)+256);
        pipe.InitBuffer(gt,capacity*sizeof(T)+256);
        pipe.InitBuffer(out_buf,p.tile*sizeof(T)+256);
        pipe.InitBuffer(xf,capacity*sizeof(float)+256);
        pipe.InitBuffer(tf,capacity*sizeof(float)+256);
        pipe.InitBuffer(af,p.tile*sizeof(float)+256);
        auto a=it.Get<T>(); auto b=gt.Get<T>(); auto o=out_buf.Get<T>();
        auto x=xf.Get<float>();auto t=tf.Get<float>();auto acc=af.Get<float>();
        const uint32_t tiles=(p.hidden+p.tile-1)/p.tile;
        for(uint32_t task=GetBlockIdx();task<p.rows*tiles;task+=GetBlockNum()) {
            const uint32_t row=task/tiles,offset=(task%tiles)*p.tile;
            const uint32_t count=(p.hidden-offset<p.tile)?p.hidden-offset:p.tile;
            const uint32_t aligned=Align(count), flat=p.groups*aligned;
            Sync<HardEvent::MTE3_MTE2>();
            DataCopyExtParams cp{static_cast<uint16_t>(p.groups),count*uint32_t(sizeof(T)),
                (p.hidden-count)*uint32_t(sizeof(T)),0,0};
            DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(aligned-count),0};
            DataCopyPad(a,in[row*p.residual_stride+offset],cp,pad);
            DataCopyPad(b,gate[row*p.block_stride+offset],cp,pad);
            Sync<HardEvent::MTE2_V>();
            Cast(x,a,RoundMode::CAST_NONE,flat);
            Cast(t,b,RoundMode::CAST_NONE,flat); PipeBarrier<PIPE_V>();
            Muls(t,t,-1.0f,flat); PipeBarrier<PIPE_V>();
            Exp(t,t,flat); PipeBarrier<PIPE_V>();
            Adds(t,t,1.0f,flat); PipeBarrier<PIPE_V>();
            Reciprocal(t,t,flat); PipeBarrier<PIPE_V>();
            Mul(x,x,t,flat); PipeBarrier<PIPE_V>();
            Duplicate(acc,0.0f,count); PipeBarrier<PIPE_V>();
            if(p.groups==4) {
                // Two independent pair sums, then one final K reduction.
                Add(acc,x,x[aligned],count);
                Add(t,x[2*aligned],x[3*aligned],count);PipeBarrier<PIPE_V>();
                Add(acc,acc,t,count);PipeBarrier<PIPE_V>();
            } else {
                for(uint32_t k=0;k<p.groups;k++) {
                    Add(acc,acc,x[k*aligned],count);PipeBarrier<PIPE_V>();
                }
            }
            Muls(acc,acc,1.0f/p.groups,count); PipeBarrier<PIPE_V>();
            Cast(o,acc,RoundMode::CAST_RINT,count);
            Sync<HardEvent::V_MTE3>();
            Store(out,row*p.hidden+offset,o,count);
        }
        Sync<HardEvent::MTE3_S>();
    }
};
} // namespace

extern "C" __global__ __aicore__ void qwen_hc_combine_norm(
    GM_ADDR r,GM_ADDR b,GM_ADDR i,GM_ADDR w,GM_ADDR s,GM_ADDR n,HcParams p) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(p.row_batch>1) {
        BatchedCombineNorm<bfloat16_t> kernel;kernel.Run(r,b,i,w,s,n,p);
    } else {
        CombineNorm<bfloat16_t> kernel;kernel.Run(r,b,i,w,s,n,p);
    }
}
extern "C" __global__ __aicore__ void qwen_hc_silu(GM_ADDR x,GM_ADDR y,HcParams p) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(p.row_batch>1) {
        BatchedSilu<bfloat16_t> kernel;kernel.Run(x,y,p);
    } else {
        HcSilu<bfloat16_t> kernel;kernel.Run(x,y,p);
    }
}
extern "C" __global__ __aicore__ void qwen_hc_gate_mix(GM_ADDR x,GM_ADDR g,GM_ADDR y,HcParams p) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(p.tile>0 && p.groups<=8) {
        BatchedGateMix<bfloat16_t> kernel;kernel.Run(x,g,y,p);
    } else {
        GateMix<bfloat16_t> kernel;kernel.Run(x,g,y,p);
    }
}
void qwen_hc_combine_norm_launch(uint32_t cores, void* stream,void* r,void* b,void* i,
                                void* w,void* s,void* n,HcParams p) {
    qwen_hc_combine_norm<<<cores,nullptr,stream>>>((GM_ADDR)r,(GM_ADDR)b,(GM_ADDR)i,
                                                  (GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)n,&p);
}
void qwen_hc_silu_launch(uint32_t cores,void* stream,void* x,void* y,HcParams p) {
    qwen_hc_silu<<<cores,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)y,&p);
}
void qwen_hc_gate_mix_launch(uint32_t cores,void* stream,void* x,void* g,void* y,HcParams p) {
    qwen_hc_gate_mix<<<cores,nullptr,stream>>>((GM_ADDR)x,(GM_ADDR)g,(GM_ADDR)y,&p);
}
