// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "contact_split_tiling.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());
  uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
  const uint32_t cores=p.GetCoreNumAiv(); if(!cores || !ub) return ge::GRAPH_FAILED;
  uint32_t b=0,n=0,q=0,k=0,d=0,r=0,ht=1,units=0,in=0,out=0;uint64_t peak=0;
  const auto& s=context->GetInputShape(0)->GetStorageShape();
  const auto* a=context->GetAttrs();
  if(s.GetDimNum()!=3 || s.GetDim(1)!=1) return ge::GRAPH_FAILED;
  b=s.GetDim(0);n=*a->GetInt(0);d=*a->GetInt(1);r=*a->GetInt(2);
  if ((n!=8 && n!=16 && n!=32 && n!=64) || (d!=128 && d!=192) || r!=64 || s.GetDim(2)!=n*(d+r)) return ge::GRAPH_FAILED;
  if (context->GetInputShape(1)->GetStorageShape().GetShapeSize()!=b*r ||
      context->GetInputShape(2)->GetStorageShape().GetShapeSize()!=b*r) return ge::GRAPH_FAILED;
  ht=std::min(16u,n); units=b*((n+ht-1)/ht);
  in=ht*r+2*r;out=ht*r;
  peak=4ull*ht*d+4ull*in+4ull*out+4ull*(3*ht*r+2*r);
  if(b<1 || b>64 || !units || peak>ub) return ge::GRAPH_FAILED;
  ContactSplitTilingData t;
  t.set_rows(b);t.set_heads(n);t.set_qRank(q);t.set_kvRank(k);t.set_nope(d);t.set_rope(r);
  t.set_units(units);t.set_headTile(ht);t.set_inElements(in);t.set_outElements(out);
  t.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  context->SetBlockDim(std::min(units,cores));context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0]=0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) {return ge::GRAPH_SUCCESS;}
IMPL_OP_OPTILING(ContactSplit).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}
