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
    pipe.InitBuffer(copyQ,2,t.inElements*2);
  }
  __aicore__ inline void Process() {
    uint32_t unit=GetBlockIdx();if(unit>=t.units)return;CopyIn(unit);
    for(;unit<t.units;unit+=GetBlockNum()) {
      if(unit+GetBlockNum()<t.units)CopyIn(unit+GetBlockNum());
      const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=(unit/tiles)*t.rowTile,head=(unit%tiles)*t.headTile;
      const uint32_t count=t.heads-head<t.headTile?t.heads-head:t.headTile;
      const uint32_t rowCount=t.rows-row<t.rowTile?t.rows-row:t.rowTile;
      auto in=copyQ.DeQue<bfloat16_t>();
      if(t.rowTile==1)Store(yG[(static_cast<uint64_t>(row)*t.heads+head)*t.kvRank],in,count*t.kvRank);
      else for(uint32_t rr=0;rr<rowCount;++rr) {
        // UB is [head,row,KV]; GM is [row,head,KV]. UB stride is in 32B blocks.
        DataCopyPad(yG[(static_cast<uint64_t>(row+rr)*t.heads+head)*t.kvRank],in[rr*t.kvRank],
            DataCopyExtParams{static_cast<uint16_t>(count),t.kvRank*2,(rowCount-1)*t.kvRank*2/32,0,0});
      }
      copyQ.FreeTensor(in);
    }
  }
 private:
  __aicore__ inline void CopyIn(uint32_t unit) {
    const uint32_t tiles=(t.heads+t.headTile-1)/t.headTile,row=(unit/tiles)*t.rowTile,head=(unit%tiles)*t.headTile;
    const uint32_t count=t.heads-head<t.headTile?t.heads-head:t.headTile;
    const uint32_t rowCount=t.rows-row<t.rowTile?t.rows-row:t.rowTile;
    auto in=copyQ.AllocTensor<bfloat16_t>();
    DataCopyPad(in,xG[(static_cast<uint64_t>(head)*t.rows+row)*t.kvRank],
        DataCopyExtParams{static_cast<uint16_t>(count),rowCount*t.kvRank*2,(t.rows-rowCount)*t.kvRank*2,0,0},DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
    copyQ.EnQue(in);
  }
  ContactLayoutDeviceData t;TPipe pipe;GlobalTensor<bfloat16_t> xG,yG;
  TQueBind<TPosition::VECIN,TPosition::VECOUT,2> copyQ;
};
}
extern "C" __global__ __aicore__ void contact_layout(GM_ADDR x,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(ContactLayoutDeviceData);
  GET_TILING_DATA_WITH_STRUCT(ContactLayoutDeviceData,t,tiling);KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  ContactLayoutKernel op;op.Init(x,y,t);op.Process();
}
