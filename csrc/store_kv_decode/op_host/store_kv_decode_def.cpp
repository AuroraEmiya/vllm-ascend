/**
 * @file store_kv_decode_def.cpp
 * @brief StoreKVDecode OpDef — single-token KV cache store for the D (decode) stage
 */

#include "register/op_def_registry.h"

namespace ops {
class StoreKVDecode : public OpDef {
public:
    explicit StoreKVDecode(const char* name) : OpDef(name)
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

        this->Input("slotPos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("keyCacheOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

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

OP_ADD(StoreKVDecode);
}
