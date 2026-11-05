// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "contact_norm_tiling.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());
  uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
  const uint32_t cores=p.GetCoreNumAiv(); if(!cores || !ub) return ge::GRAPH_FAILED;
  uint32_t b=0,n=0,q=0,k=0,d=0,r=0,ht=1,units=0,in=0,out=0;uint64_t peak=0;
  const auto& s=context->GetInputShape(0)->GetStorageShape();
  if (s.GetDimNum()!=3 || s.GetDim(1)!=1) return ge::GRAPH_FAILED;
  b=s.GetDim(0);q=s.GetDim(2);k=context->GetInputShape(3)->GetStorageShape().GetShapeSize();
  r=context->GetInputShape(4)->GetStorageShape().GetDim(2);
  if ((q!=1536 && q!=2048) || k!=512 || r!=64) return ge::GRAPH_FAILED;
  if (context->GetInputShape(2)->GetStorageShape().GetShapeSize()!=q ||
      context->GetInputShape(1)->GetStorageShape().GetShapeSize()!=b*(k+r) ||
      context->GetInputShape(4)->GetStorageShape().GetShapeSize()!=b*r ||
      context->GetInputShape(5)->GetStorageShape().GetShapeSize()!=b*r ||
      context->GetInputShape(6)->GetStorageShape().GetShapeSize()!=b) return ge::GRAPH_FAILED;
  for (int i=7;i<9;++i) {
    const auto& c=context->GetInputShape(i)->GetStorageShape();
    if(c.GetDimNum()!=4 || c.GetDim(2)!=1 || c.GetDim(3)!=(i==7?k:r) ||
       (c.GetDim(1)!=16 && c.GetDim(1)!=128) || c.GetDim(0)*c.GetDim(1)<b) return ge::GRAPH_FAILED;
  }
  units=2*b; in=2*q+2*k+3*r+16;out=q+k+r;
  peak=4ull*in+4ull*out+16ull*q+20ull*r+128;
  if(b<1 || b>64 || !units || peak>ub) return ge::GRAPH_FAILED;
  ContactNormTilingData t;
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
IMPL_OP_OPTILING(ContactNorm).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}
