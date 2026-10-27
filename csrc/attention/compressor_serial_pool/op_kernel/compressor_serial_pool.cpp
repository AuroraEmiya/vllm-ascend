// SPDX-License-Identifier: Apache-2.0
#include "compressor_serial_common.h"
extern "C" __global__ __aicore__ void compressor_serial_pool(GM_ADDR kv, GM_ADDR gate, GM_ADDR cache, GM_ADDR ape,
                                                             GM_ADDR cu, GM_ADDR starts, GM_ADDR used, GM_ADDR table,
                                                             GM_ADDR pool, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  const uint32_t C = t.channelTile;
  const uint32_t rows = t.ratio * t.coff, n = rows * C, width = t.coff * t.dim;
  const uint32_t projectionStride = t.tileCols, gateBase = projectionStride - width;
  TPipe pipe;
  TBuf<TPosition::VECCALC> valuesBuf, scoresBuf, biasBuf, tmpBuf, maxBuf, sumBuf, outBuf;
  pipe.InitBuffer(valuesBuf, n * 4);
  pipe.InitBuffer(scoresBuf, n * 4);
  pipe.InitBuffer(biasBuf, n * 4);
  pipe.InitBuffer(tmpBuf, 32768);
  pipe.InitBuffer(maxBuf, C * 4);
  pipe.InitBuffer(sumBuf, C * 4);
  pipe.InitBuffer(outBuf, C * 4);
  auto values = valuesBuf.Get<float>(), scores = scoresBuf.Get<float>(), bias = biasBuf.Get<float>();
  auto tmp = tmpBuf.Get<uint8_t>();
  auto maximum = maxBuf.Get<float>(), sum = sumBuf.Get<float>(), out = outBuf.Get<float>();
  GlobalTensor<float> kvGm, gateGm, cacheGm, apeGm, poolGm;
  kvGm.SetGlobalBuffer((__gm__ float*)kv);
  gateGm.SetGlobalBuffer((__gm__ float*)gate);
  cacheGm.SetGlobalBuffer((__gm__ float*)cache);
  apeGm.SetGlobalBuffer((__gm__ float*)ape);
  poolGm.SetGlobalBuffer((__gm__ float*)pool);
  CompressorMetadata meta;
  meta.Init(pipe, cu, starts, used, table, t.batches, t.tableCols, t.ratio, t.hasUsed);
  for (uint32_t task = GetBlockIdx(); task < t.groups * (t.dim / C); task += GetBlockNum()) {
    uint32_t group = task / (t.dim / C), channel = (task % (t.dim / C)) * C;
    int32_t end = 0, b = meta.Group(group, end);
    if (b < 0) {
      Duplicate(out, 0.0f, C);
      VecToMte();
      DataCopyPad(poolGm[uint64_t(group) * t.dim + channel], out, DataCopyExtParams{1, C * 4, 0, 0, 0});
      ReuseBuffers();
      continue;
    }
    int32_t start = meta.start.GetValue(b), begin = meta.cu.GetValue(b);
    const int32_t first = end - int32_t(rows);
    if (first < 0) {
      const uint32_t missing = uint32_t(-first) * C;
      Duplicate(values, 0.0f, missing);
      Duplicate(scores, -__builtin_inff(), missing);
    }
    if (first < start) Duplicate(bias, 0.0f, n);
    if (first < start || first < 0) {
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    for (uint32_t row = 0; row < rows;) {
      int32_t position = end - int32_t(rows) + row;
      uint32_t half = row / t.ratio;
      if (position < 0) {
        row += SMin(uint32_t(-position), t.ratio - row % t.ratio);
        continue;
      }
      uint32_t count = t.ratio - row % t.ratio;
      if (position < start) {
        count = SMin(count, uint32_t(start - position));
        count = SMin(count, t.blockSize - uint32_t(position) % t.blockSize);
        int32_t page = meta.Page(b, uint32_t(position) / t.blockSize);
        uint64_t offset =
            (uint64_t(page) * t.blockSize + uint32_t(position) % t.blockSize) * 2 * width + half * t.dim + channel;
        DataCopyExtParams cp{uint16_t(count), C * 4, (2 * width - C) * 4, 0, 0};
        DataCopyPad(values[row * C], cacheGm[offset], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(scores[row * C], cacheGm[offset + width], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
      } else {
        uint64_t offset = uint64_t(begin + position - start) * projectionStride + half * t.dim + channel;
        DataCopyExtParams cp{uint16_t(count), C * 4, (projectionStride - C) * 4, 0, 0};
        DataCopyExtParams bp{uint16_t(count), C * 4, (width - C) * 4, 0, 0};
        DataCopyPad(values[row * C], kvGm[offset], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(scores[row * C], gateGm[offset + gateBase], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(bias[row * C], apeGm[uint64_t(position % int32_t(t.ratio)) * width + half * t.dim + channel], bp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
      }
      row += count;
    }
    MteToVec();
    Add(scores, scores, bias, n);
    PipeBarrier<PIPE_V>();
    uint32_t shape[2] = {rows, C};
    ReduceMax<float, Pattern::Reduce::RA, false>(maximum, scores, tmp, shape, true);
    PipeBarrier<PIPE_V>();
    for (uint32_t lane = 0; lane < C; lane += 64)
      Sub(scores[lane], scores[lane], maximum[lane], 64, uint8_t(rows),
          BinaryRepeatParams{1, 1, 1, uint8_t(C / 8), uint8_t(C / 8), 0});
    PipeBarrier<PIPE_V>();
    Exp(scores, scores, n);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::RA, false>(sum, scores, tmp, shape, true);
    PipeBarrier<PIPE_V>();
    Mul(values, values, scores, n);
    PipeBarrier<PIPE_V>();
    ReduceSum<float, Pattern::Reduce::RA, false>(out, values, tmp, shape, true);
    PipeBarrier<PIPE_V>();
    Div(out, out, sum, C);
    VecToMte();
    DataCopyPad(poolGm[uint64_t(group) * t.dim + channel], out, DataCopyExtParams{1, C * 4, 0, 0, 0});
    ReuseBuffers();
  }
}
