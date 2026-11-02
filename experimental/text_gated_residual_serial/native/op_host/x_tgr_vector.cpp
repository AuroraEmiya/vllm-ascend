
#include "../op_kernel/x_tgr_vector_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    auto* t = context->GetTilingData<XTgrVectorTilingData>();
    const auto& s = context->GetInputShape(0)->GetStorageShape();
    const auto* a = context->GetAttrs();
    const int64_t mode = *a->GetInt(0), channels = *a->GetInt(1);
    const float eps = *a->GetFloat(2);
    if (s.GetDimNum()!=2 || s.GetDim(0)<1 || s.GetDim(0)>2048 ||
        channels!=4 || mode<0 || mode>2 || !std::isfinite(eps) || eps<=0) return ge::GRAPH_FAILED;
    const uint32_t n=s.GetDim(0), d=s.GetDim(1);
    if ((mode==1 && d!=320) || (mode!=1 && d!=10240)) return ge::GRAPH_FAILED;
    const auto gammaSize=context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    const auto gateSize=context->GetInputShape(3)->GetStorageShape().GetShapeSize();
    if (mode==0 && gammaSize!=1 && gammaSize!=d) return ge::GRAPH_FAILED;
    if (mode==2 && context->GetInputShape(2)->GetStorageShape().GetShapeSize()!=n*d) return ge::GRAPH_FAILED;
    if (mode==2 && gateSize!=1 && gateSize!=n*channels) return ge::GRAPH_FAILED;
    t->rows=n; t->width=d; t->channels=channels; t->mode=mode; t->eps=eps;
    t->hasGamma=(gammaSize==d); t->hasGate=(mode==2 && gateSize==n*channels);
    t->tile=256;
    t->units=mode==0 ? n*channels : (mode==1 ? (n*d+t->tile-1)/t->tile : n*((d/channels+t->tile-1)/t->tile));
    auto p=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub=0; p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if (ub<51232 || p.GetCoreNumAiv()==0) return ge::GRAPH_FAILED;
    context->SetBlockDim(std::min(t->units,p.GetCoreNumAiv()));
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto& x=*context->GetInputShape(0);
    const auto mode=*context->GetAttrs()->GetInt(0), c=*context->GetAttrs()->GetInt(1);
    if (x.GetDimNum()!=2 || c!=4) return GRAPH_FAILED;
    *context->GetOutputShape(0)=x;
    if (mode==2) context->GetOutputShape(0)->SetDim(1,x.GetDim(1)/c);
    *context->GetOutputShape(1)=mode==0 ? gert::Shape({x.GetDim(0),c}) : gert::Shape({0});
    bool gate=mode==2 && context->GetInputShape(3)->GetShapeSize()==x.GetDim(0)*c;
    *context->GetOutputShape(2)=gate ? gert::Shape({x.GetDim(0),c}) : gert::Shape({0});
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class XTgrVector : public OpDef {
public:
    explicit XTgrVector(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("aux")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("inv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("inject")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("mode").Int();
        this->Attr("channels").Int();
        this->Attr("eps").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(XTgrVector);
}
