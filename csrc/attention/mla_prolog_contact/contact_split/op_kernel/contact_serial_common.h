// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once
#include "kernel_operator.h"
namespace ContactSerial {
using namespace AscendC;
__aicore__ inline void SyncVectorToScalar(TPipe& pipe) {
  const auto id=static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(id);WaitFlag<HardEvent::V_S>(id);
}
__aicore__ inline void SyncLoadToScalar(TPipe& pipe) {
  const auto id=static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(id);WaitFlag<HardEvent::MTE2_S>(id);
}
__aicore__ inline void Load(LocalTensor<bfloat16_t> dst,GlobalTensor<bfloat16_t> src,uint32_t elements) {
  DataCopy(dst,src,elements);
}
__aicore__ inline void Store(GlobalTensor<bfloat16_t> dst,LocalTensor<bfloat16_t> src,uint32_t elements) {
  DataCopy(dst,src,elements);
}
}
