/**
 * @file store_kv_block_def.cpp
 * @brief StoreKVBlock OpDef registration
 *
 * Grouped KV cache store operator for the P stage.
 * Uses precomputed group metadata (groupLen, groupKeyIdx, groupKeyCacheIdx)
 * to perform coalesced copies from KV input into the KV cache.
 */

#include "register/op_def_registry.h"

namespace ops {
class StoreKVBlock : public OpDef {
public:
    explicit StoreKVBlock(const char* name) : OpDef(name)
    {
        this->Input("keyIn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("keyCacheIn")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("groupLen")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("groupKeyIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("groupKeyCacheIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("keyCacheOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("blockSize").Int();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend910b", aicore_config);
    }
};

OP_ADD(StoreKVBlock);
}
