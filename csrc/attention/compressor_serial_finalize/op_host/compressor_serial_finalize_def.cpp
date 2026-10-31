// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class CompressorSerialFinalize : public OpDef {
 public:
  explicit CompressorSerialFinalize(const char* name) : OpDef(name) {
    this->Input("pool").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("kv").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("gate").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("ape").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("norm").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("sin").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("cos").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("cu").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("starts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("used").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
    this->Output("query").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
    this->Output("cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Attr("ratio").AttrType(REQUIRED).Int();
    this->Attr("coff").AttrType(REQUIRED).Int();
    this->Attr("has_used").AttrType(REQUIRED).Int();
    this->Attr("block_size").AttrType(REQUIRED).Int();
    this->Attr("blocks").AttrType(REQUIRED).Int();
    this->Attr("eps").AttrType(REQUIRED).Float();
    this->AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(CompressorSerialFinalize);
}  // namespace ops
