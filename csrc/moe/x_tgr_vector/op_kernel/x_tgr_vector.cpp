/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "x_tgr_vector_tiling.h"

using namespace AscendC;

// Independent branch rows for RMS, disjoint linear tiles for SiLU, and disjoint
// [token, hidden-tile] ownership for mixing. No cross-core synchronization.
class TgrVector {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate,
                               GM_ADDR y, GM_ADDR inv, GM_ADDR inject,
                               const XTgrVectorTilingData& data) {
        t=data;
        xG.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        gG.SetGlobalBuffer((__gm__ float*)gamma);
        uG.SetGlobalBuffer((__gm__ bfloat16_t*)aux);
        gateG.SetGlobalBuffer((__gm__ bfloat16_t*)gate);
        yG.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        invG.SetGlobalBuffer((__gm__ float*)inv);
        injectG.SetGlobalBuffer((__gm__ bfloat16_t*)inject);
        // Mix stages all four branches together. Other modes retain the
        // 51,232-byte layout. Tiling validates the mode-specific UB demand.
        const uint32_t count=t.mode==2?t.channels*t.tile:2560;
        const uint32_t sumCount=t.mode==2?t.tile:2560;
        pipe.InitBuffer(b0,count*2); pipe.InitBuffer(b1,count*2);
        pipe.InitBuffer(f0,count*4); pipe.InitBuffer(f1,count*4);
        pipe.InitBuffer(f2,sumCount*4); pipe.InitBuffer(f3,count*4);
        pipe.InitBuffer(scalar,32);
    }
    __aicore__ inline void Process() {
        for (uint32_t unit=GetBlockIdx(); unit<t.units; unit+=GetBlockNum()) {
            if(t.mode==0 || t.mode==4) Norm(unit);
            else if(t.mode==1) Activation(unit);
            else if(t.mode==3 || t.mode==5) PackedActivation(unit);
            else Mix(unit);
        }
    }
private:
    __aicore__ inline void Barrier() { PipeBarrier<PIPE_V>(); }
    template<HardEvent event> __aicore__ inline void Fence() {
        const auto id=static_cast<event_t>(pipe.FetchEventID(event));
        SetFlag<event>(id); WaitFlag<event>(id);
    }
    __aicore__ inline void Sigmoid(LocalTensor<float> v,uint32_t n) {
        Muls(v,v,-1.0f,n); Barrier(); Exp(v,v,n); Barrier();
        Adds(v,v,1.0f,n); Barrier();
        auto one=f3.Get<float>(); Duplicate(one,1.0f,n); Barrier();
        Div(v,one,v,n); Barrier();
    }
    __aicore__ inline void Norm(uint32_t row) {
        uint32_t h=t.width/t.channels,off=row*h;
        auto xb=b0.Get<bfloat16_t>(),yb=b1.Get<bfloat16_t>();
        auto z=f0.Get<float>(),q=f1.Get<float>(),tmp=f2.Get<float>();
        auto r=scalar.Get<float>();
        auto g=f3.Get<float>();
        DataCopy(xb,xG[off],h);
        if(t.hasGamma) DataCopy(g,gG[(row%t.channels)*h],h);
        Fence<HardEvent::MTE2_V>();
        Cast(z,xb,RoundMode::CAST_NONE,h); Barrier();
        Mul(q,z,z,h); Barrier();
        ReduceSum(r,q,tmp,h); Barrier();
        Muls(r,r,1.0f/h,1); Barrier(); Adds(r,r,t.eps,1); Barrier();
        // The basic FP32 Rsqrt approximation does not meet the auxiliary
        // inverse-RMS contract. Sqrt+Div is the documented precise path.
        Sqrt(r,r,1); Barrier();
        Duplicate(q,1.0f,8); Barrier();
        Div(r,q,r,1); Barrier();
        // Broadcast r[0] into the first FP32 block, then reuse that block
        // across all vector repeats. No V-to-scalar round trip is needed.
        Brcb(q,r,1,{1,8}); Barrier();
        Mul(z,z,q,64,h/64,{1,1,0,8,8,0}); Barrier();
        if(t.hasGamma) {
            Adds(g,g,1.0f,h); Barrier(); Mul(z,z,g,h); Barrier();
        }
        Cast(yb,z,RoundMode::CAST_RINT,h); Fence<HardEvent::V_MTE3>();
        DataCopy(yG[off],yb,h);
        if(t.mode==0) {
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(invG[row],r,cp);
        }
        // Both the BF16 output and FP32 scalar are still read by MTE3.
        // Order the next iteration's input DMA after those reads complete.
        Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Activation(uint32_t unit) {
        uint32_t off=unit*t.tile,n=t.rows*t.width-off;
        n=n<t.tile?n:t.tile;
        auto xb=b0.Get<bfloat16_t>(),yb=b1.Get<bfloat16_t>();
        auto z=f0.Get<float>(),q=f1.Get<float>();
        DataCopy(xb,xG[off],n); Fence<HardEvent::MTE2_V>();
        Cast(z,xb,RoundMode::CAST_NONE,n); Barrier();
        Muls(z,z,1.0f/t.channels,n); Barrier();
        Muls(q,z,1.0f,n); Barrier(); Sigmoid(q,n);
        Mul(z,z,q,n); Barrier(); Cast(yb,z,RoundMode::CAST_RINT,n); Fence<HardEvent::V_MTE3>();
        DataCopy(yG[off],yb,n); Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void PackedActivation(uint32_t row) {
        // One aligned projection contains [down320, gate4, zero padding12].
        // The complete reduction remains in public mm. This V stage owns the
        // activation and gate outputs for its row and reads only live columns.
        auto xb=b0.Get<bfloat16_t>(),yb=b1.Get<bfloat16_t>();
        auto z=f0.Get<float>(),q=f1.Get<float>();
        DataCopy(xb,xG[row*t.width],320);
        DataCopyExtParams cp{1,8,0,0,0};
        DataCopyPadExtParams<bfloat16_t> pad{true,0,0,(bfloat16_t)0};
        DataCopyPad(xb[320],xG[row*t.width+320],cp,pad);
        Fence<HardEvent::MTE2_V>();
        Cast(z,xb,RoundMode::CAST_NONE,320); Barrier();
        Muls(z,z,1.0f/t.channels,320); Barrier();
        Muls(q,z,1.0f,320); Barrier(); Sigmoid(q,320);
        Mul(z,z,q,320); Barrier(); Cast(yb,z,RoundMode::CAST_RINT,320);
        Fence<HardEvent::V_MTE3>(); DataCopy(yG[row*320],yb,320);
        Cast(q,xb[320],RoundMode::CAST_NONE,4); Barrier();
        Muls(q,q,1.0f/t.channels,4); Barrier(); Sigmoid(q,4);
        Muls(q,q,2.0f,4); Barrier(); Cast(xb,q,RoundMode::CAST_RINT,4);
        // A padded internal row keeps independent cores' small gate stores
        // in separate 32-byte blocks. The Python API returns the live view.
        Fence<HardEvent::V_MTE3>(); DataCopyPad(injectG[row*(t.mode==5?16:4)],xb,cp);
        Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Mix(uint32_t unit) {
        uint32_t h=t.width/t.channels,tiles=(h+t.tile-1)/t.tile;
        uint32_t row=unit/tiles,col=(unit%tiles)*t.tile,n=h-col;
        n=n<t.tile?n:t.tile;
        auto xb=b0.Get<bfloat16_t>(),ub=b1.Get<bfloat16_t>();
        auto z=f0.Get<float>(),q=f1.Get<float>(),sum=f2.Get<float>();
        // DataCopy strides and lengths are in 32-byte blocks. All supported
        // tiles divide the hidden width and are multiples of 16 BF16 values.
        const DataCopyParams cp{static_cast<uint16_t>(t.channels),
            static_cast<uint16_t>(n/16),static_cast<uint16_t>((h-n)/16),0};
        const uint32_t off=row*t.width+col,all=n*t.channels;
        DataCopy(xb,xG[off],cp); DataCopy(ub,uG[off],cp); Fence<HardEvent::MTE2_V>();
        Cast(z,xb,RoundMode::CAST_NONE,all); Cast(q,ub,RoundMode::CAST_NONE,all); Barrier();
        Sigmoid(q,all); Mul(z,z,q,all); Barrier();
        Add(sum,z,z[n],n); Barrier();
        Add(sum,sum,z[2*n],n); Barrier(); Add(sum,sum,z[3*n],n); Barrier();
        Muls(sum,sum,1.0f/t.channels,n); Barrier();
        Cast(xb,sum,RoundMode::CAST_RINT,n); Fence<HardEvent::V_MTE3>();
        DataCopy(yG[row*h+col],xb,n);
        if(t.hasGate && col==0) {
            Fence<HardEvent::MTE3_MTE2>();
            DataCopyExtParams cp{1,t.channels*2,0,0,0};
            DataCopyPadExtParams<bfloat16_t> pad{true,0,0,(bfloat16_t)0};
            DataCopyPad(xb,gateG[row*t.channels],cp,pad); Fence<HardEvent::MTE2_V>();
            Cast(q,xb,RoundMode::CAST_NONE,t.channels); Barrier();
            Muls(q,q,1.0f/t.channels,t.channels); Barrier(); Sigmoid(q,t.channels);
            Muls(q,q,2.0f,t.channels); Barrier();
            Cast(xb,q,RoundMode::CAST_RINT,t.channels); Fence<HardEvent::V_MTE3>();
            DataCopyPad(injectG[row*t.channels],xb,cp);
        }
        Fence<HardEvent::MTE3_MTE2>();
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> b0,b1,f0,f1,f2,f3,scalar;
    GlobalTensor<bfloat16_t> xG,uG,gateG,yG,injectG;
    GlobalTensor<float> gG,invG;
    XTgrVectorTilingData t;
};

extern "C" __global__ __aicore__ void x_tgr_vector(GM_ADDR x, GM_ADDR gamma, GM_ADDR aux, GM_ADDR gate, GM_ADDR y, GM_ADDR inv, GM_ADDR inject, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(XTgrVectorTilingData);
    GET_TILING_DATA_WITH_STRUCT(XTgrVectorTilingData, tilingData, tiling);
    TgrVector op;
    op.Init(x,gamma,aux,gate,y,inv,inject,tilingData);
    op.Process();
}
