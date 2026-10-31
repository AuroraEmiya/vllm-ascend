// SPDX-License-Identifier: Apache-2.0
#include "compressor_serial.h"
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
namespace vllm_ascend {
namespace {
void CheckMetadata(const at::Tensor& cu, const at::Tensor& starts, const at::Tensor& used, const at::Tensor& table,
                   bool hasUsed) {
  TORCH_CHECK(table.dim() == 2 && table.size(0) > 0 && table.size(0) <= 256, "table requires B1..256");
  const int64_t batches = table.size(0);
  TORCH_CHECK(cu.dim() == 1 && cu.numel() == batches + 1 && starts.dim() == 1 && starts.numel() == batches,
              "invalid request metadata shapes");
  TORCH_CHECK(!hasUsed || (used.dim() == 1 && used.numel() == batches), "invalid seqused shape");
  for (const auto* tensor : {&cu, &starts, &used, &table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt && tensor->is_contiguous() && tensor->device() == table.device(),
                "metadata must be contiguous INT32 on the same device");
  }
}
void CheckProjection(const at::Tensor& kv, const at::Tensor& gate, const at::Tensor& ape, int64_t ratio, int64_t coff) {
  TORCH_CHECK(kv.dim() == 2 && gate.sizes() == kv.sizes() && ape.dim() == 2, "invalid projection shapes");
  TORCH_CHECK(coff > 0 && ape.size(1) % coff == 0, "invalid coff");
  const int64_t dim = ape.size(1) / coff;
  TORCH_CHECK((ratio == 4 && coff == 2 && (dim == 128 || dim == 512)) || (ratio == 128 && coff == 1 && dim == 512),
              "only C4Li/C4A/C128A are supported");
  TORCH_CHECK(ape.size(0) == ratio && (kv.size(1) == ape.size(1) || kv.size(1) == 2 * ape.size(1)),
              "invalid APE/projection width");
  for (const auto* tensor : {&kv, &gate, &ape}) {
    TORCH_CHECK(tensor->scalar_type() == at::kFloat && tensor->is_contiguous() && tensor->device() == kv.device(),
                "projections and APE must be contiguous FP32 on one device");
  }
}
}  // namespace

at::Tensor compressor_serial_pool(const at::Tensor& kv, const at::Tensor& gate, const at::Tensor& cache,
                                  const at::Tensor& ape, const at::Tensor& cu, const at::Tensor& starts,
                                  const at::Tensor& used, const at::Tensor& table, int64_t ratio, int64_t coff,
                                  int64_t groups, bool hasUsed) {
  CheckProjection(kv, gate, ape, ratio, coff);
  CheckMetadata(cu, starts, used, table, hasUsed);
  TORCH_CHECK(cache.dim() == 3 && cache.size(0) > 0 && cache.size(1) >= 1 && cache.size(1) <= 1024 &&
                  cache.size(2) == 2 * ape.size(1) && cache.scalar_type() == at::kFloat && cache.is_contiguous() &&
                  cache.device() == kv.device() && table.device() == kv.device(),
              "invalid FP32 state cache");
  TORCH_CHECK(groups >= 0 && groups <= kv.size(0), "invalid output capacity");
  auto out = at::empty({groups, ape.size(1) / coff}, kv.options());
  int64_t has_used = hasUsed;
  EXEC_NPU_CMD(aclnnCompressorSerialPool, kv, gate, cache, ape, cu, starts, used, table, ratio, coff, groups, has_used,
               out);
  return out;
}
at::Tensor compressor_serial_finalize(const at::Tensor& pool, const at::Tensor& kv, const at::Tensor& gate,
                                      const at::Tensor& ape, const at::Tensor& norm, const at::Tensor& sin,
                                      const at::Tensor& cos, const at::Tensor& table, const at::Tensor& cu,
                                      const at::Tensor& starts, const at::Tensor& used, const at::Tensor& x,
                                      at::Tensor& cache, int64_t ratio, int64_t coff, bool hasUsed, double eps) {
  CheckProjection(kv, gate, ape, ratio, coff);
  CheckMetadata(cu, starts, used, table, hasUsed);
  TORCH_CHECK(pool.dim() == 2 && pool.size(1) == ape.size(1) / coff && norm.dim() == 1 && norm.numel() == pool.size(1),
              "invalid pooled/norm shape");
  TORCH_CHECK(sin.dim() == 2 && sin.size(0) == pool.size(0) && sin.size(1) == 64 && cos.sizes() == sin.sizes(),
              "invalid RoPE shape");
  TORCH_CHECK(
      x.dim() == 2 && x.size(0) == kv.size(0) && (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
      "x must be BF16/FP16 TND");
  TORCH_CHECK(cache.dim() == 3 && cache.size(0) > 0 && cache.size(1) >= 1 && cache.size(1) <= 1024 &&
                  cache.size(2) == 2 * ape.size(1),
              "invalid state shape");
  const at::Tensor* floatInputs[] = {&pool, &norm, &sin, &cos, &cache};
  for (const auto* tensor : floatInputs) {
    TORCH_CHECK(tensor->scalar_type() == at::kFloat && tensor->is_contiguous() && tensor->device() == kv.device(),
                "pool, norm, RoPE and state must be contiguous FP32 on one device");
  }
  TORCH_CHECK(x.is_contiguous() && x.device() == kv.device() && table.device() == kv.device(),
              "invalid x/metadata device");
  auto out = at::empty(pool.sizes(), x.options());
  int64_t has_used = hasUsed, block_size = cache.size(1), blocks = cache.size(0);
  EXEC_NPU_CMD(aclnnCompressorSerialFinalize, pool, kv, gate, ape, norm, sin, cos, table, cu, starts, used, x, ratio,
               coff, has_used, block_size, blocks, eps, out, cache);
  return out;
}
}  // namespace vllm_ascend

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "compressor_serial_pool(Tensor kv, Tensor gate, Tensor cache, Tensor ape, Tensor cu, Tensor starts, Tensor used, "
      "Tensor table, int ratio, int coff, int groups, bool has_used) -> Tensor");
  ops.impl("compressor_serial_pool", torch::kPrivateUse1, &vllm_ascend::compressor_serial_pool);
  ops.def(
      "compressor_serial_finalize(Tensor pool, Tensor kv, Tensor gate, Tensor ape, Tensor norm, Tensor sin, Tensor "
      "cos, Tensor table, Tensor cu, Tensor starts, Tensor used, Tensor x, Tensor(a!) cache, int ratio, int coff, bool "
      "has_used, float eps) -> Tensor");
  ops.impl("compressor_serial_finalize", torch::kPrivateUse1, &vllm_ascend::compressor_serial_finalize);
}

namespace vllm_ascend::meta {
at::Tensor compressor_serial_pool(const at::Tensor& kv, const at::Tensor&, const at::Tensor&, const at::Tensor& ape,
                                  const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
                                  int64_t coff, int64_t groups, bool) {
  return at::empty_symint({c10::SymInt(groups), ape.sym_size(1) / coff}, kv.options());
}
at::Tensor compressor_serial_finalize(const at::Tensor& pool, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                      const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor& x,
                                      at::Tensor&, int64_t, int64_t, bool, double) {
  return at::empty_symint(pool.sym_sizes(), x.options());
}
}  // namespace vllm_ascend::meta
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("compressor_serial_pool", &vllm_ascend::meta::compressor_serial_pool);
  ops.impl("compressor_serial_finalize", &vllm_ascend::meta::compressor_serial_finalize);
}
