# SPDX-License-Identifier: Apache-2.0
"""Paired ordinary Graph baseline/candidate measurement, with full state checks."""
import argparse,hashlib,importlib,json,os,statistics,time,traceback
from pathlib import Path
import torch
import torch_npu
from contact_vector_reference import make_case,oracle,check

CALLS=10
SAMPLES=20
CONTACT_HASH='cc4f60f9f9d3cdcb8cf0b95b288b0e149821a066d68d4035041e7514ee2de7ef'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity():
    paths={l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines()
        if '.so' in l and any(z in l for z in ('opapi','nnopbase','contact_vector','optiling','contact_kernels','cust_opsproto'))}
    return dict(torch=torch.__version__,torch_npu=torch_npu.__version__,soc=torch.npu.get_device_name(0),
        hf32=bool(torch.npu.matmul.allow_hf32),jit_compile_false=torch.npu.is_jit_compile_false(),
        mm_bmm_format_nd=torch.npu.get_mm_bmm_format_nd(),
        files={p:sha(p) for p in sorted(paths) if Path(p).is_file()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case-id',action='append');ap.add_argument('--routes',default='contact,v3')
    ap.add_argument('--profile-target');ap.add_argument('--no-timing',action='store_true')
    ap.add_argument('--candidate-id',default='baseline');ap.add_argument('--static',action='store_true')
    a=ap.parse_args();assert not a.out.exists(),'preserve previous attempt'
    routes=a.routes.split(',');assert len(set(routes))==len(routes)
    assert set(routes)<={'contact','v3','candidate'}
    manifest=Path(__file__).with_name('contact_vector_cases.json')
    cases=json.loads(manifest.read_text());requested=set(a.case_id or [c['id'] for c in cases])
    cases=[c for c in cases if c['id'] in requested];assert {c['id'] for c in cases}==requested
    a.out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='STARTED',candidate_id=a.candidate_id,calls_per_graph=CALLS,samples=SAMPLES,
        superkernel=False,static_only=a.static,host_start_ns=time.time_ns(),cases=[],
        sources={str(p):sha(p) for p in (Path(__file__),manifest,Path(__file__).with_name('contact_vector_reference.py'))})
    def save():
        tmp=a.out.with_suffix('.partial');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(a.out)
    save();torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
    contact=importlib.import_module('cann_ops_transformer_custom.ops.attention.mla_prolog_v3_contact.mla_prolog_contact')
    assert sha(contact.__file__)==CONTACT_HASH
    candidate=None
    if 'candidate' in routes:
        from vllm_ascend.ops.mla_prolog_contact import mla_prolog_contact,load_contact_extension
        load_contact_extension();candidate=mla_prolog_contact
        if a.static:
            from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel
            static_kernel._compute_opc_compile_jobs=lambda:'4'
            from vllm_ascend.ops.mla_prolog_contact import compile_contact_static
    for c in cases:
        if a.static:
            # Each shape owns a fresh compile lifecycle outside measured replay.
            torch._dynamo.reset()
            candidate=compile_contact_static()
        row=dict(case=c,status='STARTED',routes={},raw_samples=[],preparation={});result['cases'].append(row);save()
        print('CONTACT_PHASE CASE',c['id'],flush=True)
        data,kv0,kr0=make_case(c);initials=(kv0,kr0);ref=oracle(data,c,*initials)
        d={k:v.npu() for k,v in data.items()}
        nz={k:torch_npu.npu_format_cast(d[k],29) for k in ('wdq','wuq','wdkv')}
        caches={n:(kv0.npu(),kr0.npu()) for n in routes};initial_device=(kv0.npu(),kr0.npu())
        def reset(n):
            for x,z in zip(caches[n],initial_device):x.copy_(z)
        def call(n):
            if n=='contact':
                return contact.mla_prolog(d['x'],nz['wdq'],nz['wuq'],d['wuk'],nz['wdkv'],d['gq'],d['gkv'],
                    *caches[n],rope_sin=d['sin'],rope_cos=d['cos'],cache_index=d['index'],
                    cache_mode='PA_BSND',query_norm_flag=False,enable_superkernel=False)
            if n=='v3':
                return torch_npu.npu_mla_prolog_v3(d['x'],nz['wdq'],nz['wuq'],d['wuk'],nz['wdkv'],d['gq'],d['gkv'],
                    d['sin'],d['cos'],*caches[n],cache_index=d['index'],cache_mode='PA_BSND',query_norm_flag=False,
                    weight_quant_mode=0,kv_cache_quant_mode=0,query_quant_mode=0,qc_qr_scale=1.,kc_scale=1.)
            return candidate(d['x'],nz['wdq'],nz['wuq'],d['wuk'],nz['wdkv'],d['gq'],d['gkv'],
                d['sin'],d['cos'],d['index'],*caches[n])
        stream=torch.npu.Stream();stream.wait_stream(torch.npu.current_stream());graphs={};outputs={}
        with torch.npu.stream(stream):
            for n in routes:
                rr=row['routes'][n]=dict(status='STARTED',checks={})
                try:
                    reset(n);start=time.perf_counter_ns();out=call(n);stream.synchronize()
                    row['preparation'][n+'_first_call_us']=(time.perf_counter_ns()-start)/1000
                    rr['checks']['eager']=check(out,caches[n],ref,data,c,initials)
                    if not rr['checks']['eager']['passed']:rr['status']='FAIL_EAGER';continue
                    for _ in range(10):call(n)
                    stream.synchronize();reset(n);g=torch.npu.NPUGraph();start=time.perf_counter_ns()
                    with torch.npu.graph(g,stream=stream):
                        for _ in range(CALLS):out=call(n)
                    row['preparation'][n+'_capture_us']=(time.perf_counter_ns()-start)/1000
                    for _ in range(10):g.replay()
                    stream.synchronize();rr['checks']['graph']=check(out,caches[n],ref,data,c,initials)
                    if not rr['checks']['graph']['passed']:rr['status']='FAIL_GRAPH';continue
                    graphs[n]=g;outputs[n]=out;rr['status']='CAPTURED'
                except Exception:rr.update(status='ERROR',traceback=traceback.format_exc())
                save()
            row['input_immutable']=all(torch.equal(z.cpu(),data[k]) for k,z in d.items())
            row['weight_immutable']=all(torch.equal(z.cpu(),data[k]) for k,z in nz.items())
            assert row['input_immutable'] and row['weight_immutable']
            updated,_,_=make_case(c,1);updated_ref=oracle(updated,c,*initials)
            for k,z in d.items():z.copy_(updated[k])
            for k,z in nz.items():z.copy_(torch_npu.npu_format_cast(d[k],29))
            stream.synchronize()
            for n,g in graphs.items():
                reset(n);g.replay();stream.synchronize()
                row['routes'][n]['checks']['graph_changed_data']=check(outputs[n],caches[n],updated_ref,updated,c,initials)
            for k,z in d.items():z.copy_(data[k])
            for k,z in nz.items():z.copy_(torch_npu.npu_format_cast(d[k],29))
            stream.synchronize()
            for n,g in graphs.items():
                reset(n);g.replay();stream.synchronize();rr=row['routes'][n]
                rr['checks']['graph_restored']=check(outputs[n],caches[n],ref,data,c,initials)
                rr['status']='PASS' if all(v['passed'] for v in rr['checks'].values()) else 'FAIL_UPDATE'
            valid=[n for n in routes if row['routes'][n]['status']=='PASS']
            if a.profile_target:
                n=a.profile_target;assert n in valid
                row['profile_marker']=dict(route=n,calls_per_graph=CALLS,replays=2,host_begin_ns=time.time_ns())
                for _ in range(2):graphs[n].replay()
                stream.synchronize();row['profile_marker']['host_end_ns']=time.time_ns()
            elif not a.no_timing:
                for rep in range(SAMPLES):
                    off=rep%len(valid) if valid else 0;order=valid[off:]+valid[:off]
                    if rep%2:order.reverse()
                    for n in order:
                        reset(n);start=torch.npu.Event(enable_timing=True);stop=torch.npu.Event(enable_timing=True)
                        start.record(stream);graphs[n].replay();stop.record(stream);stop.synchronize()
                        us=start.elapsed_time(stop)*1000
                        row['raw_samples'].append(dict(round=rep,route=n,graph_us=us,per_call_us=us/CALLS))
                for n in valid:
                    v=[s['per_call_us'] for s in row['raw_samples'] if s['route']==n];q=statistics.quantiles(v,n=10)
                    row['routes'][n]['timing']=dict(median_us=statistics.median(v),p10_us=q[0],p90_us=q[8],min_us=min(v),max_us=max(v))
            row['status']='PASS' if len(valid)==len(routes) else 'PARTIAL_FAILURE'
            print('CONTACT_CASE',c['id'],json.dumps({n:dict(status=v['status'],timing=v.get('timing')) for n,v in row['routes'].items()}),flush=True)
        row['runtime_identity']=identity();save()
        del graphs,outputs,caches,nz,d
    result['status']='PASS' if all(r['status']=='PASS' for r in result['cases']) else 'PARTIAL_FAILURE'
    result['host_end_ns']=time.time_ns();save()
    raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__':main()
