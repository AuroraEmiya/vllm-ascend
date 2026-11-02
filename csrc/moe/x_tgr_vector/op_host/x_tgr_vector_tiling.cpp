/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "x_tgr_vector_tiling.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    XTgrVectorTilingData t;
    const auto& s = context->GetInputShape(0)->GetStorageShape();
    const auto* a = context->GetAttrs();
    const int64_t mode = *a->GetInt(0), channels = *a->GetInt(1);
    const float eps = *a->GetFloat(2);
    if (s.GetDimNum()!=2 || s.GetDim(0)<1 || s.GetDim(0)>2048 ||
        channels!=4 || mode<0 || mode>5 || !std::isfinite(eps) || eps<=0) return ge::GRAPH_FAILED;
    const uint32_t n=s.GetDim(0), d=s.GetDim(1);
    if ((mode==1 && d!=320) || ((mode==3 || mode==5) && d!=336) || ((mode==0 || mode==2 || mode==4) && d!=10240)) return ge::GRAPH_FAILED;
    const auto gammaSize=context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    const auto gateSize=context->GetInputShape(3)->GetStorageShape().GetShapeSize();
    if ((mode==0 || mode==4) && gammaSize!=1 && gammaSize!=d) return ge::GRAPH_FAILED;
    if (mode==2 && context->GetInputShape(2)->GetStorageShape().GetShapeSize()!=n*d) return ge::GRAPH_FAILED;
    if (mode==2 && gateSize!=1 && gateSize!=n*channels) return ge::GRAPH_FAILED;
    t.set_rows(n); t.set_width(d); t.set_channels(channels); t.set_mode(mode); t.set_eps(eps);
    t.set_hasGamma((gammaSize==d)); t.set_hasGate((mode==2 && gateSize==n*channels));
    // Keep enough independent work for small decode batches. At larger
    // batches, process a larger tile to amortize each DMA/vector fence.
    uint32_t tile=256;
    if (mode==2) {
        if (n>=40) tile=2560;
        else if (n>=20) tile=1280;
        else if (n>=10) tile=640;
        else if (n>=5) tile=320;
    } else if (mode==1) {
        tile=std::min(2560u,std::max(256u,((n*d+40*32-1)/(40*32))*32));
    }
    auto p=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub=0; p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    if (ub<51232 || p.GetCoreNumAiv()==0) return ge::GRAPH_FAILED;
    // Two BF16 and three FP32 four-branch tiles, one FP32 sum tile,
    // plus a 32-byte scalar block: at most 174,112 bytes on this path.
    if (mode==2) {
        while (tile>32 && 68ull*tile+32>ub) tile/=2;
        if (68ull*tile+32>ub || 2560%tile!=0) return ge::GRAPH_FAILED;
    }
    t.set_tile(tile);
    t.set_units((mode==3 || mode==5) ? n : ((mode==0 || mode==4) ? n*channels : (mode==1 ? (n*d+t.get_tile()-1)/t.get_tile() : n*((d/channels+t.get_tile()-1)/t.get_tile()))));
    t.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(t.GetDataSize());
    context->SetBlockDim(std::min(t.get_units(),p.GetCoreNumAiv()));
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(XTgrVector).Tiling(TilingFunc);
}
