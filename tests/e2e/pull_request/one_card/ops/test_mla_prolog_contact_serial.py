# SPDX-License-Identifier: Apache-2.0
"""Independent public-equation and cache test; no fused source reference."""
import json
from pathlib import Path
import sys
import pytest
import torch
import torch_npu

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'benchmarks/kernels/mla_prolog'))
from contact_vector_reference import make_case,oracle,check
from vllm_ascend.ops.mla_prolog_contact import load_contact_extension,mla_prolog_contact

@pytest.mark.parametrize('case_id',[
    'b1-h7168-n32-p16','b17-h7168-n32-p16','b64-h7168-n32-p16',
    'b16-h6144-n64-p128-q2048-d192'])
@pytest.mark.parametrize('edge',['random','zero_input','zero_gamma'])
def test_contact_serial_state(case_id,edge):
    load_contact_extension();torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    cases=json.loads((ROOT/'benchmarks/kernels/mla_prolog/contact_vector_cases.json').read_text())
    c=next(c for c in cases if c['id']==case_id).copy()
    c.update(zeros=edge=='zero_input',zero_gamma=edge=='zero_gamma')
    data,kv0,kr0=make_case(c);ref=oracle(data,c,kv0,kr0)
    d={k:v.npu() for k,v in data.items()}
    w={k:torch_npu.npu_format_cast(d[k],29) for k in ('wdq','wuq','wdkv')}
    caches=kv0.npu(),kr0.npu()
    out=mla_prolog_contact(d['x'],w['wdq'],w['wuq'],d['wuk'],w['wdkv'],d['gq'],d['gkv'],
        d['sin'],d['cos'],d['index'],*caches)
    torch.npu.synchronize()
    assert check(out,caches,ref,data,c,(kv0,kr0))['passed']
