/**
 * @file store_kv_decode_tiling.h
 * @brief StoreKVDecode tiling data — single-token copy, single core
 */

#ifndef STORE_KV_DECODE_TILING_H
#define STORE_KV_DECODE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKVDecodeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rowBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKVDecode, StoreKVDecodeTilingData)
}

struct storeKVDecodeCompileInfo {};

#endif // STORE_KV_DECODE_TILING_H
