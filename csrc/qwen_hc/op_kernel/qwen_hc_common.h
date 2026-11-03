// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#pragma once
#include "kernel_operator.h"
#include "../qwen_hc_tiling_data.h"
using namespace AscendC;
namespace vllm_ascend::hc {
__aicore__ inline uint32_t alignElements(uint32_t elementCount) {
  return (elementCount + kBf16BlockElements - 1) / kBf16BlockElements * kBf16BlockElements;
}

template <HardEvent E>
__aicore__ inline void syncPipeline() {
  SetFlag<E>(EVENT_ID0);
  WaitFlag<E>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void loadTile(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t offset, uint32_t count) {
  DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
  DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignElements(count) - count), 0};
  DataCopyPad(dst, src[offset], copy, pad);
}
template <typename T>
__aicore__ inline void storeTile(GlobalTensor<T> dst, uint32_t offset, LocalTensor<T> src, uint32_t count) {
  DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
  DataCopyPad(dst[offset], src, copy);
}

}  // namespace vllm_ascend::hc
