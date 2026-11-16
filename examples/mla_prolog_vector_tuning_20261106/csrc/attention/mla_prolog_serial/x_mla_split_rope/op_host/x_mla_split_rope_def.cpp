// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"

namespace ops {
class XMlaSplitRope : public OpDef {
 public:
  explicit XMlaSplitRope(const char* name) : OpDef(name) {
    this->Input("x")
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
    this->Output("head_q")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("query_rope")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("rows").Int();
    this->Attr("heads").Int();
    this->AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(XMlaSplitRope);
}  // namespace ops
