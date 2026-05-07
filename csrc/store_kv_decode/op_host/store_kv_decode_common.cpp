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
 * \file write_cache_by_group_list_common.cpp
 * \brief
 */
#include "exe_graph/runtime/extended_kernel_context.h"
#include "store_kv_decode_common.h"
#include <chrono>

#include <cstdint>
#include <cstdio>
namespace optiling {

void StoreKVDecodeCommonTiling::PrintTilingData() {
  std::cout << context_->GetNodeName() << " Start WriteStoreKVDecodeListTilingData printing" << std::endl;
  std::cout << "params.tokenSize " << params.tokenSize << std::endl;
  std::cout << "params.numTokens " << params.numTokens << std::endl;
  //   std::cout << "params.numCache " << params.numCache << std::endl;
  std::cout << context_->GetNodeName() << " End WriteStoreKVDecodeListTilingData printing" << std::endl;
}

void StoreKVDecodeCommonTiling::SetTiling() {
  if (params.tokenSize == 0)
    printf("[LOG] params.tokenSize==0 \n");
  else
    tilingData_.set_tokenSize(params.tokenSize);

  if (params.numTokens <= 0)
    printf("[LOG] params.numTokens<=0 \n");
  else
    tilingData_.set_numTokens(params.numTokens);

  //   if (params.numCache <= 0)
  //     printf("[ZTLOG] params.numCache<=0 \n");
  //   else
  //     tilingData_.set_numCache(params.numCache);

  size_t* workspaceSize = context_->GetWorkspaceSizes(1);
  *workspaceSize = params.workspaceSize + params.sysWorkspaceSize;
  context_->SetTilingKey(params.tilingKey);
  if (params.coreNum <= 0)
    printf("[ZTLOG] params.coreNum<=0 \n");
  else
    context_->SetBlockDim(params.coreNum);

  tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

ge::graphStatus StoreKVDecodeCommonTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
  params.coreNum = ascendcPlatform.GetCoreNum();
  OP_CHECK_IF((params.coreNum == 0),
              VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "Failed to get core num."),
              return ge::GRAPH_FAILED);
  params.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StoreKVDecodeCommonTiling::DoCommonTiling() {
  auto kShape = context_->GetInputShape(DIM_0);
  auto kDimNum = kShape->GetStorageShape().GetDimNum();
  if (kDimNum < 2 || kDimNum > 7) {
    printf("[ERROR] StoreKVDecode Input kDimNum < 2 || kDimNum>7");
  } else {
     params.numTokens = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(0);
     uint32_t token_size = 1;
    for (int i = 1; i < kDimNum; i++) {
      int get_shape = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
      if (get_shape <= 0) {
        printf("[ERROR] Invalid shape dim value!\n");
      } else {
        token_size *= get_shape;
      }
    }
    params.tokenSize = token_size;
  }

  auto kCacheShape = context_->GetInputShape(DIM_1);
  auto kCacheDimNum = kCacheShape->GetStorageShape().GetDimNum();
  if (kCacheDimNum < 2 || kCacheDimNum > 7) {
    printf("[ERROR] StoreKVDecode Input kCacheDimNum < 2 ");
  }  // else {
  //     params.numCache = kCacheShape->GetStorageShape().GetDim(0) * kCacheShape->GetStorageShape().GetDim(1);
  //   }

  auto xDataType = context_->GetInputDesc(DIM_0)->GetDataType();
  if (xDataType == ge::DataType::DT_INT8) {
    params.tilingKey = 1;
  } else if (xDataType == ge::DataType::DT_FLOAT16 || xDataType == ge::DataType::DT_BF16) {
    params.tilingKey = 2;
  } else if (xDataType == ge::DataType::DT_INT32 || xDataType == ge::DataType::DT_UINT32) {
    params.tilingKey = 4;
  } else {
    OP_LOGE(context_->GetNodeName(), "Unsupported type.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StoreKVDecodeCommonTiling::DoTiling() {
  auto ret = GetPlatformInfo();
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  ret = DoCommonTiling();
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  SetTiling();

  // PrintTilingData();
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
