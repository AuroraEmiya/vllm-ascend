// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops {
class XMlaNormCache : public OpDef {
 public:
  explicit XMlaNormCache(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma_q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gamma_kv")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("sin")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("cos")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("index")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("normalized_q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("kv_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("kr_cache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("rows").Int();
    this->Attr("cache_slots").Int();
    this->Attr("eps_q").Float();
    this->Attr("eps_kv").Float();
    this->AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(XMlaNormCache);
}  // namespace ops
