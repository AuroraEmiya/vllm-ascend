// Licensed under CANN Open Software License Agreement Version 2.0.
#include "../../kernel_common.h"
#include "contact_split_tiling.h"
using namespace AscendC;
using namespace ContactSerial;
namespace {
class Split {
 public:
  __aicore__ inline void Init(GM_ADDR x,GM_ADDR sine,GM_ADDR cosine,GM_ADDR qc,GM_ADDR qr,const ContactSplitTilingData& data) {
    t=data;xG.SetGlobalBuffer((__gm__ bfloat16_t*)x);sinG.SetGlobalBuffer((__gm__ bfloat16_t*)sine);
    cosG.SetGlobalBuffer((__gm__ bfloat16_t*)cosine);qcG.SetGlobalBuffer((__gm__ bfloat16_t*)qc);qrG.SetGlobalBuffer((__gm__ bfloat16_t*)qr);
    pipe.InitBuffer(inputQ,2,t.inElements*2);pipe.InitBuffer(outputQ,2,t.outElements*2);
    pipe.InitBuffer(valueB,t.headTile*t.rope*4);pipe.InitBuffer(gatherB,t.headTile*t.rope*4);
    pipe.InitBuffer(resultB,t.headTile*t.rope*4);pipe.InitBuffer(coeffB,2*t.rope*4);
  }
  __aicore__ inline void Process() {
    uint32_t unit=GetBlockIdx();if(unit>=t.units)return;CopyIn(unit);
    for(;unit<t.units;unit+=GetBlockNum()) {
      if(unit+GetBlockNum()<t.units)CopyIn(unit+GetBlockNum());
      Compute(unit);CopyOut(unit);
    }
  }
 private:
  __aicore__ inline uint32_t Count(uint32_t unit) {
    const uint32_t head=(unit%((t.heads+t.headTile-1)/t.headTile))*t.headTile;
    return t.heads-head<t.headTile?t.heads-head:t.headTile;
  }
  __aicore__ inline void CopyIn(uint32_t unit) {
    const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=unit/tiles,head=(unit%tiles)*t.headTile;
    const uint32_t count=Count(unit),d=t.nope,r=t.rope,ht=t.headTile;
    const uint64_t offset=(static_cast<uint64_t>(row)*t.heads+head)*(d+r);
    auto in=inputQ.AllocTensor<bfloat16_t>();
    DataCopyPad(in,xG[offset],DataCopyExtParams{static_cast<uint16_t>(count),d*2,r*2,0,0},DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
    DataCopyPad(in[ht*d],xG[offset+d],DataCopyExtParams{static_cast<uint16_t>(count),r*2,d*2,0,0},DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
    Load(in[ht*(d+r)],sinG[row*r],r);Load(in[ht*(d+r)+r],cosG[row*r],r);inputQ.EnQue(in);
  }
  __aicore__ inline void Compute(uint32_t unit) {
    const uint32_t h=Count(unit),d=t.nope,r=t.rope,half=r/2,count=h*r,ht=t.headTile;
    auto in=inputQ.DeQue<bfloat16_t>();auto out=outputQ.AllocTensor<bfloat16_t>();
    DataCopy(out,in,h*d);
    auto value=valueB.Get<float>(),gather=gatherB.Get<float>(),result=resultB.Get<float>(),coeff=coeffB.Get<float>();
    Cast(value,in[ht*d],RoundMode::CAST_NONE,count);
    Cast(coeff,in[ht*(d+r)],RoundMode::CAST_NONE,2*r);PipeBarrier<PIPE_V>();
    uint64_t reserved=0;
    GatherMask(gather,value,static_cast<uint8_t>(1),true,count,GatherMaskParams(1,1,8,0),reserved);
    GatherMask(gather[count/2],value,static_cast<uint8_t>(2),true,count,GatherMaskParams(1,1,8,0),reserved);PipeBarrier<PIPE_V>();
    BinaryRepeatParams broadcast{1,1,1,static_cast<uint8_t>(r/8),static_cast<uint8_t>(half/8),0};
    const uint8_t repeats=static_cast<uint8_t>(h);const uint64_t mask=half;
    Mul(result,gather,coeff[r],mask,repeats,broadcast);
    Mul(result[half],gather[count/2],coeff[r+half],mask,repeats,broadcast);
    Mul(value,gather[count/2],coeff,mask,repeats,broadcast);
    Mul(value[half],gather,coeff[half],mask,repeats,broadcast);PipeBarrier<PIPE_V>();
    BinaryRepeatParams pair{1,1,1,static_cast<uint8_t>(r/8),static_cast<uint8_t>(r/8),static_cast<uint8_t>(r/8)};
    Sub(result,result,value,mask,repeats,pair);Add(result[half],result[half],value[half],mask,repeats,pair);PipeBarrier<PIPE_V>();
    Cast(out[ht*d],result,RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
    inputQ.FreeTensor(in);outputQ.EnQue(out);
  }
  __aicore__ inline void CopyOut(uint32_t unit) {
    const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=unit/tiles,head=(unit%tiles)*t.headTile;
    const uint32_t count=Count(unit);auto out=outputQ.DeQue<bfloat16_t>();
    DataCopyPad(qcG[(static_cast<uint64_t>(head)*t.rows+row)*t.nope],out,
        DataCopyExtParams{static_cast<uint16_t>(count),t.nope*2,0,(t.rows-1)*t.nope*2,0});
    Store(qrG[(static_cast<uint64_t>(row)*t.heads+head)*t.rope],out[t.headTile*t.nope],count*t.rope);
    outputQ.FreeTensor(out);
  }
  ContactSplitTilingData t;TPipe pipe;GlobalTensor<bfloat16_t> xG,sinG,cosG,qcG,qrG;
  TQue<TPosition::VECIN,2> inputQ;TQue<TPosition::VECOUT,1> outputQ;
  TBuf<TPosition::VECCALC> valueB,gatherB,resultB,coeffB;
};
}
extern "C" __global__ __aicore__ void contact_split(GM_ADDR x,GM_ADDR sine,GM_ADDR cosine,GM_ADDR qc,GM_ADDR qr,
    GM_ADDR workspace,GM_ADDR tiling) {
  GET_TILING_DATA(t,tiling);KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  Split op;op.Init(x,sine,cosine,qc,qr,t);op.Process();
}
