// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "contact_layout_tiling.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  platform_ascendc::PlatformAscendC p(context->GetPlatformInfo());
  uint64_t ub=0;p.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
  const uint32_t cores=p.GetCoreNumAiv(); if(!cores || !ub) return ge::GRAPH_FAILED;
  uint32_t b=0,n=0,q=0,k=0,d=0,r=0,ht=1,units=0,in=0,out=0;uint64_t peak=0;
  const auto& s=context->GetInputShape(0)->GetStorageShape();
  if(s.GetDimNum()!=3) return ge::GRAPH_FAILED;
  n=s.GetDim(0);b=s.GetDim(1);k=s.GetDim(2);
  if ((n!=8 && n!=16 && n!=32 && n!=64) || k!=512) return ge::GRAPH_FAILED;
  ht=std::min(16u,n);
  uint32_t rowTile=1;const uint32_t headTiles=(n+ht-1)/ht;
  while(rowTile<4) {
    const uint32_t next=rowTile*2;
    if(((b+next-1)/next)*headTiles<cores || 4ull*ht*next*k>ub)break;
    rowTile=next;
  }
  units=((b+rowTile-1)/rowTile)*headTiles;in=out=ht*rowTile*k;
  peak=4ull*in;
  if(b<1 || b>64 || !units || peak>ub) return ge::GRAPH_FAILED;
  ContactLayoutTilingData t;
  t.set_rows(b);t.set_heads(n);t.set_rowTile(rowTile);t.set_kvRank(k);t.set_nope(d);t.set_rope(r);
  t.set_units(units);t.set_headTile(ht);t.set_inElements(in);t.set_outElements(out);
  t.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(t.GetDataSize());
  context->SetBlockDim(std::min(units,cores));context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0]=0;
  return ge::GRAPH_SUCCESS;
}
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext*) {return ge::GRAPH_SUCCESS;}
IMPL_OP_OPTILING(ContactLayout).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
}
