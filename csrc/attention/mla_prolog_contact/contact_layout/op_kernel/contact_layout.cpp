// Licensed under CANN Open Software License Agreement Version 2.0.
#include "contact_serial_common.h"
#include "contact_layout_tiling.h"
using namespace AscendC;
using namespace ContactSerial;
namespace {
class ContactLayoutKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const ContactLayoutDeviceData& data) {
    t=data;xG.SetGlobalBuffer((__gm__ bfloat16_t*)x);yG.SetGlobalBuffer((__gm__ bfloat16_t*)y);
    pipe.InitBuffer(inputQ,2,t.inElements*2);pipe.InitBuffer(outputQ,2,t.outElements*2);
  }
  __aicore__ inline void Process() {
    uint32_t unit=GetBlockIdx();if(unit>=t.units)return;CopyIn(unit);
    for(;unit<t.units;unit+=GetBlockNum()) {
      if(unit+GetBlockNum()<t.units)CopyIn(unit+GetBlockNum());
      const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=unit/tiles,head=(unit%tiles)*t.headTile;
      const uint32_t count=t.heads-head<t.headTile?t.heads-head:t.headTile;
      auto in=inputQ.DeQue<bfloat16_t>();auto out=outputQ.AllocTensor<bfloat16_t>();
      DataCopy(out,in,count*t.kvRank);inputQ.FreeTensor(in);outputQ.EnQue(out);
      out=outputQ.DeQue<bfloat16_t>();
      Store(yG[(static_cast<uint64_t>(row)*t.heads+head)*t.kvRank],out,count*t.kvRank);outputQ.FreeTensor(out);
    }
  }
 private:
  __aicore__ inline void CopyIn(uint32_t unit) {
    const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=unit/tiles,head=(unit%tiles)*t.headTile;
    const uint32_t count=t.heads-head<t.headTile?t.heads-head:t.headTile;
    auto in=inputQ.AllocTensor<bfloat16_t>();
    DataCopyPad(in,xG[(static_cast<uint64_t>(head)*t.rows+row)*t.kvRank],
        DataCopyExtParams{static_cast<uint16_t>(count),t.kvRank*2,(t.rows-1)*t.kvRank*2,0,0},DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
    inputQ.EnQue(in);
  }
  ContactLayoutDeviceData t;TPipe pipe;GlobalTensor<bfloat16_t> xG,yG;
  TQue<TPosition::VECIN,2> inputQ;TQue<TPosition::VECOUT,1> outputQ;
};
}
extern "C" __global__ __aicore__ void contact_layout(GM_ADDR x,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(ContactLayoutDeviceData);
  GET_TILING_DATA_WITH_STRUCT(ContactLayoutDeviceData,t,tiling);KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ContactLayoutKernel op;op.Init(x,y,t);op.Process();
}
