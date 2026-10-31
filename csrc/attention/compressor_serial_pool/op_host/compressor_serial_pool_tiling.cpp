// SPDX-License-Identifier: Apache-2.0
#include "compressor_serial_pool_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context) {
  const auto* attrs = context->GetAttrs();
  if (!attrs) return ge::GRAPH_FAILED;

  const uint32_t ratio = *attrs->GetInt(0), coff = *attrs->GetInt(1);
  const auto& kv = context->GetInputShape(0)->GetStorageShape();
  const auto& cache = context->GetInputShape(2)->GetStorageShape();
  const auto& table = context->GetInputShape(7)->GetStorageShape();
  const uint32_t groups = *attrs->GetInt(2), hasUsed = *attrs->GetInt(3);
  const uint32_t tokens = kv.GetDim(0), dim = context->GetInputShape(3)->GetStorageShape().GetDim(1) / coff;
  const uint32_t blocks = cache.GetDim(0), blockSize = cache.GetDim(1);
  const uint32_t batches = table.GetDim(0), tableCols = table.GetDim(1), tileCols = kv.GetDim(1);
  const uint32_t channelTile = ratio == 4 ? dim : 64;
  const uint32_t stateTile = 1;
  const float eps = 0;
  uint32_t work = groups * (dim / channelTile);
  uint64_t required = (3 * ratio * coff * channelTile + 3 * channelTile + 3 * 264 + 16) * 4 + 32768;

  if ((ratio != 4 && ratio != 128) || (coff != 1 && coff != 2) || (dim != 128 && dim != 512) ||
      (coff == 2 && ratio != 4) || batches < 1 || batches > 256 || blockSize < 1 || blockSize > 1024)
    return ge::GRAPH_FAILED;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  uint32_t cores = platform.GetCoreNumAiv();
  if (cores == 0 || ub < required) return ge::GRAPH_FAILED;
  CompressorSerialPoolTilingData data;
  data.set_tokens(tokens);
  data.set_groups(groups);
  data.set_batches(batches);
  data.set_dim(dim);
  data.set_coff(coff);
  data.set_ratio(ratio);
  data.set_blockSize(blockSize);
  data.set_tableCols(tableCols);
  data.set_hasUsed(hasUsed);
  data.set_blocks(blocks);
  data.set_tileCols(tileCols);
  data.set_channelTile(channelTile);
  data.set_stateTile(stateTile);
  data.set_eps(eps);
  data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(data.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = 0;
  context->SetBlockDim(std::max(1u, std::min(work, cores)));
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
struct CompressorSerialPoolCompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(CompressorSerialPool).Tiling(Tiling).TilingParse<CompressorSerialPoolCompileInfo>(Parse);
}  // namespace optiling
