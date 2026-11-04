// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops {
class XMlaNormCache : public OpDef {
 public:
  explicit XMlaNormCache(const char* name) : OpDef(name) {
    for (const char* input : {"x", "aux", "gamma_q", "gamma_kv", "sin", "cos"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_BF16})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Input("index")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    for (const char* output : {"y", "query_rope", "kv_cache", "kr_cache"}) {
      this->Output(output)
          .ParamType(REQUIRED)
          .DataType({ge::DT_BF16})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Attr("mode").Int();
    this->Attr("rows").Int();
    this->Attr("heads").Int();
    this->Attr("cache_slots").Int();
    this->Attr("eps_q").Float();
    this->Attr("eps_kv").Float();
    this->AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(XMlaNormCache);
}  // namespace ops
