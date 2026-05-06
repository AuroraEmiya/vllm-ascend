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
 * \file reshape_and_cache_tiling.cc
 * \brief
 */
#include "read_kv_cache_common.h"
#include <chrono>
namespace optiling {

static ge::graphStatus ReadKVCacheTilingFunc(gert::TilingContext* context) {
  std::string op_type(context->GetNodeType());
  ge::graphStatus ret;

  if (op_type == "ReadKVCache") {
    ReadKVCacheCommonTiling ReadKVCacheCommonTiling(context);
    ret = ReadKVCacheCommonTiling.DoTiling();
  } else {
    printf("[ZTLOG] no  ReadKVCacheCommonTiling\n");
    return ge::GRAPH_FAILED;
  }
  return ret;
}

struct Tiling4ReadKVCacheCompileInfo {
  uint32_t coreNum;
  uint64_t ubSizePlatForm;
  uint32_t sysWorkspaceSize;
};
static ge::graphStatus TilingParseForReadKVCache(gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(ReadKVCache)
    .Tiling(ReadKVCacheTilingFunc)
    .TilingParse<Tiling4ReadKVCacheCompileInfo>(TilingParseForReadKVCache);  //?
}  // namespace optiling
