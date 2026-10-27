// SPDX-License-Identifier: Apache-2.0
#include "compressor_serial_finalize_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* context) {
  const auto* attrs = context->GetAttrs();
  if (!attrs) return ge::GRAPH_FAILED;

  const uint32_t ratio = *attrs->GetInt(0), coff = *attrs->GetInt(1), hasUsed = *attrs->GetInt(2);
  const auto& pool = context->GetInputShape(0)->GetStorageShape();
  const auto& kv = context->GetInputShape(1)->GetStorageShape();
  const auto& table = context->GetInputShape(7)->GetStorageShape();
  const uint32_t tokens = kv.GetDim(0), groups = pool.GetDim(0), dim = pool.GetDim(1);
  const uint32_t blockSize = *attrs->GetInt(3), blocks = *attrs->GetInt(4);
  const uint32_t batches = table.GetDim(0), tableCols = table.GetDim(1), tileCols = kv.GetDim(1);
  const uint32_t channelTile = 64;
  const float eps = *attrs->GetFloat(5);
  const uint32_t deviceCores = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAiv();
  const uint32_t stateTile = tokens <= deviceCores ? 1u : 8u;
  uint32_t work = groups + (tokens + stateTile - 1) / stateTile;
  uint64_t required = 3 * stateTile * coff * dim * 4 + 16384;

  if ((ratio != 4 && ratio != 128) || (coff != 1 && coff != 2) || (dim != 128 && dim != 512) ||
      (coff == 2 && ratio != 4) || batches < 1 || batches > 256 || blockSize < 1 || blockSize > 1024)
    return ge::GRAPH_FAILED;
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
  uint32_t cores = platform.GetCoreNumAiv();
  if (cores == 0 || ub < required) return ge::GRAPH_FAILED;
  CompressorSerialFinalizeTilingData data;
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
struct CompressorSerialFinalizeCompileInfo {};
static ge::graphStatus Parse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(CompressorSerialFinalize).Tiling(Tiling).TilingParse<CompressorSerialFinalizeCompileInfo>(Parse);
}  // namespace optiling
