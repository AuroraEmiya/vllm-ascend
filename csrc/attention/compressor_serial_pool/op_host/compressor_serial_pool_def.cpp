// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class CompressorSerialPool : public OpDef {
 public:
  explicit CompressorSerialPool(const char* name) : OpDef(name) {
    this->Input("kv").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("gate").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("ape").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Input("cu").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("starts").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("used").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Input("table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
    this->Output("pool").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
    this->Attr("ratio").AttrType(REQUIRED).Int();
    this->Attr("coff").AttrType(REQUIRED).Int();
    this->Attr("groups").AttrType(REQUIRED).Int();
    this->Attr("has_used").AttrType(REQUIRED).Int();
    this->AICore().AddConfig("ascend910_93");
  }
};
OP_ADD(CompressorSerialPool);
}  // namespace ops
