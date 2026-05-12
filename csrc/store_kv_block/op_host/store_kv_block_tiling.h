/**
 * @file store_kv_block_tiling.h
 * @brief StoreKVBlock tiling data structure
 */

#ifndef STORE_KV_BLOCK_TILING_H
#define STORE_KV_BLOCK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKVBlockTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rowBytes);
    TILING_DATA_FIELD_DEF(uint32_t, maxGroupLen);
    TILING_DATA_FIELD_DEF(uint32_t, groupCount);
    TILING_DATA_FIELD_DEF(uint32_t, coreCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKVBlock, StoreKVBlockTilingData)
}

struct storeKVBlockCompileInfo {};

#endif // STORE_KV_BLOCK_TILING_H
