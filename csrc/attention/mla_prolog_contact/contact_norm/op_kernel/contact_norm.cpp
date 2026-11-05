// Licensed under CANN Open Software License Agreement Version 2.0.
#include "contact_serial_common.h"
#include "contact_norm_tiling.h"
using namespace AscendC;
using namespace ContactSerial;
namespace {
class ContactNormKernel {
 public:
  __aicore__ inline void Init(GM_ADDR dq,GM_ADDR dkv,GM_ADDR gq,GM_ADDR gkv,GM_ADDR sine,GM_ADDR cosine,
      GM_ADDR index,GM_ADDR cq,GM_ADDR kv,GM_ADDR kr,const ContactNormDeviceData& data) {
    t=data;
    dqG.SetGlobalBuffer((__gm__ bfloat16_t*)dq);dkvG.SetGlobalBuffer((__gm__ bfloat16_t*)dkv);
    gqG.SetGlobalBuffer((__gm__ bfloat16_t*)gq);gkvG.SetGlobalBuffer((__gm__ bfloat16_t*)gkv);
    sinG.SetGlobalBuffer((__gm__ bfloat16_t*)sine);cosG.SetGlobalBuffer((__gm__ bfloat16_t*)cosine);
    indexG.SetGlobalBuffer((__gm__ int64_t*)index);
    cqG.SetGlobalBuffer((__gm__ bfloat16_t*)cq);kvG.SetGlobalBuffer((__gm__ bfloat16_t*)kv);krG.SetGlobalBuffer((__gm__ bfloat16_t*)kr);
    pipe.InitBuffer(inputQ,2,t.inElements*2);pipe.InitBuffer(outputQ,2,t.outElements*2);
    pipe.InitBuffer(valueB,t.qRank*4);pipe.InitBuffer(gammaB,t.qRank*4);
    pipe.InitBuffer(squareB,t.qRank*4);pipe.InitBuffer(reduceB,t.qRank*4);pipe.InitBuffer(sumB,128);
    pipe.InitBuffer(ropeB,t.rope*4);pipe.InitBuffer(sinB,t.rope*4);
    pipe.InitBuffer(cosB,t.rope*4);pipe.InitBuffer(tmpB,t.rope*4);
  }
  __aicore__ inline void Process() {
    uint32_t unit=GetBlockIdx();if(unit>=t.units)return;
    CopyIn(unit);
    for(;unit<t.units;unit+=GetBlockNum()) {
      if(unit+GetBlockNum()<t.units)CopyIn(unit+GetBlockNum());
      const int64_t slot=Compute();CopyOut(unit,slot);
    }
  }
 private:
  __aicore__ inline void CopyIn(uint32_t row) {
    auto x=inputQ.AllocTensor<bfloat16_t>();
    const uint32_t q=t.qRank,k=t.kvRank,r=t.rope;
    Load(x,dqG[row*q],q);Load(x[q],gqG,q);
    Load(x[2*q],dkvG[row*(k+r)],k+r);Load(x[2*q+k+r],gkvG,k);
    Load(x[2*q+2*k+r],sinG[row*r],r);Load(x[2*q+2*k+2*r],cosG[row*r],r);
    auto idx=x[2*q+2*k+3*r].ReinterpretCast<int64_t>();
    DataCopyPad(idx,indexG[row],DataCopyExtParams{1,8,0,0,0},DataCopyPadExtParams<int64_t>{false,0,0,0});
    inputQ.EnQue(x);
  }
  __aicore__ inline void Normalize(LocalTensor<float> v,LocalTensor<float> gamma,uint32_t width) {
    auto square=squareB.Get<float>(),reduce=reduceB.Get<float>(),sum=sumB.Get<float>();
    Mul(square,v,v,width);Duplicate(sum,0.0f,8);PipeBarrier<PIPE_V>();
    ReduceSum<float,true>(sum,square,reduce,width);PipeBarrier<PIPE_V>();
    const float reciprocalWidth=1.0f/static_cast<float>(static_cast<int32_t>(width));
    Muls(sum,sum,reciprocalWidth,8);PipeBarrier<PIPE_V>();
    Adds(sum,sum,1e-5f,8);PipeBarrier<PIPE_V>();
    Rsqrt(reduce,sum,8);PipeBarrier<PIPE_V>();
    Brcb(square,reduce,1,{1,8});PipeBarrier<PIPE_V>();
    BinaryRepeatParams inverseBroadcast(1,1,0,8,8,0);
    Mul(v,v,square,uint64_t(64),static_cast<uint8_t>(width/64),inverseBroadcast);PipeBarrier<PIPE_V>();
    Mul(v,v,gamma,width);PipeBarrier<PIPE_V>();
  }
  __aicore__ inline int64_t Compute() {
    const uint32_t q=t.qRank,k=t.kvRank,r=t.rope,half=r/2;
    auto x=inputQ.DeQue<bfloat16_t>();auto out=outputQ.AllocTensor<bfloat16_t>();
    auto v=valueB.Get<float>(),gamma=gammaB.Get<float>();
    Cast(v,x,RoundMode::CAST_NONE,q);Cast(gamma,x[q],RoundMode::CAST_NONE,q);PipeBarrier<PIPE_V>();
    Normalize(v,gamma,q);Cast(out,v,RoundMode::CAST_RINT,q);PipeBarrier<PIPE_V>();
    Cast(v,x[2*q],RoundMode::CAST_NONE,k+r);Cast(gamma,x[2*q+k+r],RoundMode::CAST_NONE,k);PipeBarrier<PIPE_V>();
    Normalize(v,gamma,k);Cast(out[q],v,RoundMode::CAST_RINT,k);PipeBarrier<PIPE_V>();
    auto rope=ropeB.Get<float>(),sine=sinB.Get<float>(),cosine=cosB.Get<float>(),tmp=tmpB.Get<float>();
    Cast(sine,x[2*q+2*k+r],RoundMode::CAST_NONE,r);Cast(cosine,x[2*q+2*k+2*r],RoundMode::CAST_NONE,r);
    uint64_t reserved=0;
    GatherMask(rope,v[k],static_cast<uint8_t>(1),true,r,GatherMaskParams(1,1,8,0),reserved);
    GatherMask(rope[half],v[k],static_cast<uint8_t>(2),true,r,GatherMaskParams(1,1,8,0),reserved);PipeBarrier<PIPE_V>();
    Mul(v[k],rope,cosine,half);Mul(tmp,rope[half],sine,half);PipeBarrier<PIPE_V>();
    Sub(v[k],v[k],tmp,half);PipeBarrier<PIPE_V>();
    Mul(v[k+half],rope[half],cosine[half],half);Mul(tmp,rope,sine[half],half);PipeBarrier<PIPE_V>();
    Add(v[k+half],v[k+half],tmp,half);PipeBarrier<PIPE_V>();
    Cast(out[q+k],v[k],RoundMode::CAST_RINT,r);PipeBarrier<PIPE_V>();
    SyncLoadToScalar(pipe);
    const int64_t slot=x[2*q+2*k+3*r].ReinterpretCast<int64_t>().GetValue(0);
    inputQ.FreeTensor(x);outputQ.EnQue(out);return slot;
  }
  __aicore__ inline void CopyOut(uint32_t row,int64_t slot) {
    auto out=outputQ.DeQue<bfloat16_t>();
    Store(cqG[row*t.qRank],out,t.qRank);Store(kvG[slot*t.kvRank],out[t.qRank],t.kvRank);
    Store(krG[slot*t.rope],out[t.qRank+t.kvRank],t.rope);outputQ.FreeTensor(out);
  }
  ContactNormDeviceData t;TPipe pipe;
  GlobalTensor<bfloat16_t> dqG,dkvG,gqG,gkvG,sinG,cosG,cqG,kvG,krG;
  GlobalTensor<int64_t> indexG;
  TQue<TPosition::VECIN,2> inputQ;TQue<TPosition::VECOUT,1> outputQ;
  TBuf<TPosition::VECCALC> valueB,gammaB,squareB,reduceB,sumB,ropeB,sinB,cosB,tmpB;
};
}
extern "C" __global__ __aicore__ void contact_norm(GM_ADDR dq,GM_ADDR dkv,GM_ADDR gq,GM_ADDR gkv,
    GM_ADDR sine,GM_ADDR cosine,GM_ADDR index,GM_ADDR kv,GM_ADDR kr,GM_ADDR cq,GM_ADDR kv_out,GM_ADDR kr_out,
    GM_ADDR workspace,GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(ContactNormDeviceData);
  GET_TILING_DATA_WITH_STRUCT(ContactNormDeviceData,t,tiling);KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ContactNormKernel op;op.Init(dq,dkv,gq,gkv,sine,cosine,index,cq,kv_out,kr_out,t);op.Process();
}
