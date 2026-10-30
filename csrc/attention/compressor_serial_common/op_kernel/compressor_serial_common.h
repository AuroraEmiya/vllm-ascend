// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint32_t SMin(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline void MteToVec() {
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}
__aicore__ inline void MteToScalar() {
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}
__aicore__ inline void VecToMte() {
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}
__aicore__ inline void ReuseBuffers() { PipeBarrier<PIPE_ALL>(); }
// Metadata values stay on device. Page IDs are read with MTE2-to-Scalar ordering.
struct CompressorMetadata {
  GlobalTensor<int32_t> cuGm, startGm, usedGm, tableGm;
  TBuf<TPosition::VECCALC> cuBuf, startBuf, usedBuf, pageBuf;
  LocalTensor<int32_t> cu, start, used, page;
  uint32_t batches, tableCols, ratio, hasUsed;
  uint32_t groupRequest, groupBase, groupStop, tokenRequest;
  int32_t groupFirstEnd, tokenBegin, tokenStop;
  bool groupBoundsReady, tokenBoundsReady;
  __aicore__ inline void Init(TPipe& pipe, GM_ADDR cuIn, GM_ADDR startsInput, GM_ADDR usedIn, GM_ADDR tableIn,
                              uint32_t b, uint32_t cols, uint32_t r, uint32_t hu) {
    batches = b;
    tableCols = cols;
    ratio = r;
    hasUsed = hu;
    // Per-invocation state: Graph replay must reread updated metadata.
    groupRequest = 0;
    groupBase = 0;
    groupStop = 0;
    tokenRequest = 0;
    groupBoundsReady = false;
    tokenBoundsReady = false;
    pipe.InitBuffer(cuBuf, 264 * 4);
    pipe.InitBuffer(startBuf, 264 * 4);
    pipe.InitBuffer(usedBuf, 264 * 4);
    pipe.InitBuffer(pageBuf, 32);
    cu = cuBuf.Get<int32_t>();
    start = startBuf.Get<int32_t>();
    used = usedBuf.Get<int32_t>();
    page = pageBuf.Get<int32_t>();
    cuGm.SetGlobalBuffer((__gm__ int32_t*)cuIn);
    startGm.SetGlobalBuffer((__gm__ int32_t*)startsInput);
    usedGm.SetGlobalBuffer((__gm__ int32_t*)usedIn);
    tableGm.SetGlobalBuffer((__gm__ int32_t*)tableIn);
    DataCopyPad(cu, cuGm, DataCopyExtParams{1, (b + 1) * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    DataCopyPad(start, startGm, DataCopyExtParams{1, b * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    if (hu)
      DataCopyPad(used, usedGm, DataCopyExtParams{1, b * 4, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    MteToScalar();
  }
  __aicore__ inline int32_t Count(uint32_t b) {
    return hasUsed ? used.GetValue(b) : cu.GetValue(b + 1) - cu.GetValue(b);
  }
  __aicore__ inline int32_t Page(uint32_t b, uint32_t index) {
    DataCopyPad(page, tableGm[uint64_t(b) * tableCols + index], DataCopyExtParams{1, 4, 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    MteToScalar();
    return page.GetValue(0);
  }
  // Cyclic task assignment gives each AIV nondecreasing group/token queries.
  // Cache one request at a time; repeated channel groups remain on that request.
  __aicore__ inline int32_t Group(uint32_t group, int32_t& end) {
    while (groupRequest < batches) {
      if (!groupBoundsReady) {
        const int32_t position = start.GetValue(groupRequest);
        const int32_t firstWindow = position / int32_t(ratio);
        const int32_t count = (position + Count(groupRequest)) / int32_t(ratio) - firstWindow;
        groupStop = groupBase + uint32_t(count);
        groupFirstEnd = (firstWindow + 1) * int32_t(ratio);
        groupBoundsReady = true;
      }
      if (group < groupStop) {
        end = groupFirstEnd + int32_t(group - groupBase) * int32_t(ratio);
        return int32_t(groupRequest);
      }
      groupBase = groupStop;
      groupRequest++;
      groupBoundsReady = false;
    }
    return -1;
  }
  __aicore__ inline int32_t Token(uint32_t token) {
    while (tokenRequest < batches) {
      if (!tokenBoundsReady) {
        tokenBegin = cu.GetValue(tokenRequest);
        tokenStop = tokenBegin + Count(tokenRequest);
        tokenBoundsReady = true;
      }
      if (token < uint32_t(tokenBegin)) return -1;
      if (token < uint32_t(tokenStop)) return int32_t(tokenRequest);
      tokenRequest++;
      tokenBoundsReady = false;
    }
    return -1;
  }
};
