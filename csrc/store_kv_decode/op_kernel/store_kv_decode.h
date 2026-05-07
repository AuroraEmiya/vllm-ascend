/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file reshape_and_cache_base.h
 * \brief
 */

#ifndef ASCEND_STORE_KV_DECODE_H
#define ASCEND_STORE_KV_DECODE_H

#include "kernel_operator.h"
#include <cstdint>

namespace StoreKVDecode {
using namespace AscendC;

#ifndef STORE_KV_DECODE_TILING_DATA_H_
  #define STORE_KV_DECODE_TILING_DATA_H_
struct StoreKVDecodeTilingData {
  uint32_t tokenSize;
  uint32_t numTokens;
//   uint32_t maxBlockNum;
};
#endif
template <typename T>
class StoreKVDecodeBase {
 public:
  // tiling data
  uint32_t tokenSize = 0;
  uint32_t tokenByteSize = 0;
  uint32_t numTokens = 0;
  uint32_t maxBlockNum = 0;
  // core data
  uint32_t coreId = 0;
  uint32_t blockNum = 0;
  // Global tensor
  AscendC::TPipe* pipeThis;
  AscendC::LocalTensor<T> tokenLocal;
  AscendC::GlobalTensor<T> keyInputGt;
  AscendC::GlobalTensor<T> keyCacheIntputGt;
  AscendC::GlobalTensor<int64_t> slotMappingListGt;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf;
  __aicore__ inline StoreKVDecodeBase() {}

  //   __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16) { return y == 0 ? 0 : (x + y - 1) / y * y; }

  __aicore__ inline void Init(AscendC::TPipe* pipe, StoreKVDecodeTilingData* tilingData) {
    pipeThis = pipe;
    // tiling data
    tokenSize = tilingData->tokenSize;
    tokenByteSize = tokenSize * sizeof(T);
    numTokens = tilingData->numTokens;
    // maxBlockNum = tilingData->maxBlockNum;
    // core data
    coreId = AscendC::GetBlockIdx();
    blockNum = AscendC::GetBlockNum();
  }
  __aicore__ inline void Process(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMappingList) {
    //test only
    // if (coreId >= maxBlockNum) return;
    
    keyInputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyIn));
    keyCacheIntputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyCacheIn));
    slotMappingListGt.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slotMappingList));

    pipeThis->InitBuffer(tokenBuf, tokenByteSize);
    tokenLocal = tokenBuf.Get<T>();

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};  // todo完整块长度
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    for (uint32_t idx = coreId; idx < numTokens; idx += blockNum) {
      int keyCacheInputIdx = slotMappingListGt.GetValue(idx) * tokenSize;

      // invalid slotMappingList idx (-1)
      if (keyCacheInputIdx < 0) continue;

      uint32_t keyInputIdx = idx * tokenSize;
      copyParams.blockLen = tokenByteSize;

      DataCopyPad(tokenLocal, keyInputGt[keyInputIdx], copyParams, padParams);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
      DataCopyPad(keyCacheIntputGt[keyCacheInputIdx], tokenLocal, copyParams);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
  }
};
}  // namespace StoreKVDecode

#endif
