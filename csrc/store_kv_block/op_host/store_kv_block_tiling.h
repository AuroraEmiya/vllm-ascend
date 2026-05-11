#ifndef STORE_KV_BLOCK_TILING_H
#define STORE_KV_BLOCK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKVBlockTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, numGroups);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKVBlock, StoreKVBlockTilingData)
}

struct storeKVBlockCompileInfo {};

#endif // STORE_KV_BLOCK_TILING_H
