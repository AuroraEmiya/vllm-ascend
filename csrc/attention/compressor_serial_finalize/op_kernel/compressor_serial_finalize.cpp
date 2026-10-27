// SPDX-License-Identifier: Apache-2.0
#include "compressor_serial_common.h"
extern "C" __global__ __aicore__ void compressor_serial_finalize(GM_ADDR pool, GM_ADDR kv, GM_ADDR gate, GM_ADDR ape,
                                                                 GM_ADDR norm, GM_ADDR sin, GM_ADDR cos, GM_ADDR table,
                                                                 GM_ADDR cu, GM_ADDR starts, GM_ADDR used, GM_ADDR x,
                                                                 GM_ADDR query, GM_ADDR cache, GM_ADDR workspace,
                                                                 GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  const uint32_t width = t.coff * t.dim, projectionStride = t.tileCols, gateBase = projectionStride - width;
  TPipe pipe;
  TBuf<TPosition::VECCALC> valueBuf, otherBuf, biasBuf, tmpBuf, cosBuf, sinBuf, sumBuf, castBuf;
  pipe.InitBuffer(valueBuf, t.stateTile * width * 4);
  pipe.InitBuffer(otherBuf, t.stateTile * width * 4);
  pipe.InitBuffer(biasBuf, t.stateTile * width * 4);
  pipe.InitBuffer(tmpBuf, 4096);
  pipe.InitBuffer(cosBuf, 256);
  pipe.InitBuffer(sinBuf, 256);
  pipe.InitBuffer(sumBuf, 32);
  pipe.InitBuffer(castBuf, 1024);
  auto value = valueBuf.Get<float>(), other = otherBuf.Get<float>(), bias = biasBuf.Get<float>();
  auto tmp = tmpBuf.Get<float>(), cosine = cosBuf.Get<float>(), sine = sinBuf.Get<float>(), sum = sumBuf.Get<float>();
  auto cast = castBuf.Get<DTYPE_QUERY>();
  GlobalTensor<float> poolGm, kvGm, gateGm, apeGm, normGm, sinGm, cosGm, cacheGm;
  GlobalTensor<DTYPE_QUERY> queryGm;
  poolGm.SetGlobalBuffer((__gm__ float*)pool);
  kvGm.SetGlobalBuffer((__gm__ float*)kv);
  gateGm.SetGlobalBuffer((__gm__ float*)gate);
  apeGm.SetGlobalBuffer((__gm__ float*)ape);
  normGm.SetGlobalBuffer((__gm__ float*)norm);
  sinGm.SetGlobalBuffer((__gm__ float*)sin);
  cosGm.SetGlobalBuffer((__gm__ float*)cos);
  cacheGm.SetGlobalBuffer((__gm__ float*)cache);
  queryGm.SetGlobalBuffer((__gm__ DTYPE_QUERY*)query);
  TBuf<TPosition::VECCALC> indexBuf, signBuf;
  pipe.InitBuffer(indexBuf, 256);
  pipe.InitBuffer(signBuf, 256);
  auto indexValues = indexBuf.Get<int32_t>();
  auto indices = indexValues.ReinterpretCast<uint32_t>();
  auto signs = signBuf.Get<float>();
  // One 32-byte base pattern; extend byte offsets with Vector instructions.
  for (uint32_t i = 0; i < 8; i++) indexValues.SetValue(i, int32_t((i ^ 1u) * sizeof(float)));
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  for (uint32_t block = 1; block < 8; block++) Adds(indexValues[block * 8], indexValues, int32_t(block * 32), 8);
  Duplicate(signs, 1.0f, 64);
  PipeBarrier<PIPE_V>();
  uint64_t evenMask[2] = {0x5555555555555555ULL, 0};
  Duplicate(signs, -1.0f, evenMask, 1, 1, 8);
  PipeBarrier<PIPE_V>();
  CompressorMetadata meta;
  meta.Init(pipe, cu, starts, used, table, t.batches, t.tableCols, t.ratio, t.hasUsed);
  for (uint32_t task = GetBlockIdx(); task < t.groups + (t.tokens + t.stateTile - 1) / t.stateTile;
       task += GetBlockNum()) {
    if (task < t.groups) {
      DataCopy(value, poolGm[uint64_t(task) * t.dim], t.dim);
      DataCopy(other, normGm, t.dim);
      DataCopy(cosine, cosGm[uint64_t(task) * 64], 64);
      DataCopy(sine, sinGm[uint64_t(task) * 64], 64);
      MteToVec();
      Mul(tmp, value, value, t.dim);
      PipeBarrier<PIPE_V>();
      ReduceSum(sum, tmp, bias, t.dim);
      PipeBarrier<PIPE_V>();
      Muls(sum, sum, 1.0f / t.dim, 1);
      PipeBarrier<PIPE_V>();
      Adds(sum, sum, t.eps, 1);
      PipeBarrier<PIPE_V>();
      Rsqrt(sum, sum, 1);
      PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      Muls(value, value, sum.GetValue(0), t.dim);
      PipeBarrier<PIPE_V>();
      Mul(value, value, other, t.dim);
      PipeBarrier<PIPE_V>();
      Gather(other, value[t.dim - 64], indices, 0, 64);
      PipeBarrier<PIPE_V>();
      Mul(other, other, signs, 64);
      PipeBarrier<PIPE_V>();
      Mul(other, other, sine, 64);
      Mul(value[t.dim - 64], value[t.dim - 64], cosine, 64);
      PipeBarrier<PIPE_V>();
      Add(value[t.dim - 64], value[t.dim - 64], other, 64);
      PipeBarrier<PIPE_V>();
      Cast(cast, value, RoundMode::CAST_RINT, t.dim);
      VecToMte();
      DataCopy(queryGm[uint64_t(task) * t.dim], cast, t.dim);
    } else {
      uint32_t token = (task - t.groups) * t.stateTile;
      const uint32_t stop = SMin(token + t.stateTile, t.tokens);
      while (token < stop) {
        int32_t b = meta.Token(token);
        if (b < 0) {
          token++;
          continue;
        }
        uint32_t position = meta.start.GetValue(b) + token - meta.cu.GetValue(b);
        uint32_t count = SMin(stop - token, uint32_t(meta.cu.GetValue(b) + meta.Count(b)) - token);
        count = SMin(count, t.blockSize - position % t.blockSize);
        count = SMin(count, t.ratio - position % t.ratio);
        int32_t page = meta.Page(b, position / t.blockSize);
        if (page <= 0 || uint32_t(page) >= t.blocks) {
          token += count;
          continue;
        }
        DataCopyExtParams cp{uint16_t(count), width * 4, (projectionStride - width) * 4, 0, 0};
        DataCopyPad(value, kvGm[uint64_t(token) * projectionStride], cp, DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(other, gateGm[uint64_t(token) * projectionStride + gateBase], cp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopy(bias, apeGm[uint64_t(position % t.ratio) * width], count * width);
        MteToVec();
        Add(other, other, bias, count * width);
        VecToMte();
        uint64_t offset = (uint64_t(page) * t.blockSize + position % t.blockSize) * 2 * width;
        DataCopyExtParams put{uint16_t(count), width * 4, 0, width * 4, 0};
        DataCopyPad(cacheGm[offset], value, put);
        DataCopyPad(cacheGm[offset + width], other, put);
        ReuseBuffers();
        token += count;
      }
    }
    ReuseBuffers();
  }
}
