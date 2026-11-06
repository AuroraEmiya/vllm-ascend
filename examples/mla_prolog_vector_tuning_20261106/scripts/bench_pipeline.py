# SPDX-License-Identifier: Apache-2.0
"""Compare the complete original/tuned serial pipeline and installed aclnnMlaProlog."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback


def load_baseline(root, torch, torch_npu):
    from torch.utils.cpp_extension import load
    cann=Path(os.environ['ASCEND_HOME_PATH']); npu=Path(torch_npu.__file__).resolve().parent
    build=root/'build/baseline-binding';build.mkdir(parents=True,exist_ok=True)
    load(name='mla_public_baseline',sources=[str(root/'csrc'/p) for p in ('benchmark/aclnn_mla_prolog.cpp','aclnn_torch_adapter/NPUBridge.cpp','aclnn_torch_adapter/NPUStorageImpl.cpp')],
         extra_include_paths=[str(root/'csrc'),str(cann/'include'),str(cann/'include/aclnn'),str(npu/'include')],
         extra_cflags=['-O2','-std=c++17'],extra_ldflags=['-L'+str(cann/'lib64'),'-lopapi','-lnnopbase','-lascendcl','-L'+str(npu/'lib'),'-ltorch_npu'],
         build_directory=str(build),is_python_module=False,verbose=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--batch',type=int,required=True);p.add_argument('--sequence',type=int,default=1)
    p.add_argument('--hidden',type=int,default=7168);p.add_argument('--heads',type=int,default=32)
    p.add_argument('--block-size',type=int,choices=[16,128],default=16)
    p.add_argument('--index-mode',choices=['edges','reverse','strided'],default='edges')
    p.add_argument('--mode',choices=['S0','S1','S2'],default='S2')
    p.add_argument('--backend',choices=['v3_nz','torch_matmul_nz'],default='v3_nz')
    p.add_argument('--vector-modes',default='baseline')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',action='store_true')
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--refresh-check',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    result={'args':vars(a)|{'output':str(out)},'started':datetime.now(timezone.utc).isoformat(),'variants':{}}
    try:
        cache=out/'fresh-static-cache';cache.mkdir(exist_ok=False);os.chdir(cache)
        import torch
        import torch_npu
        from mla_superkernel import MlaPrologSerial
        from mla_superkernel.reference import build_case,oracle,check_outputs,compare
        torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.config.allow_internal_format=True
        result['environment']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'device':str(torch.npu.get_device_properties(0)),'cann':os.environ['ASCEND_HOME_PATH']}
        root=Path(__file__).resolve().parents[1];load_baseline(root,torch,torch_npu)
        try:
            from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel
            static_kernel._compute_opc_compile_jobs=lambda:'4'
        except ImportError:pass
        case=dict(batch=a.batch,sequence=a.sequence,hidden=a.hidden,heads=a.heads,seed=20261106,block_size=a.block_size,index_mode=a.index_mode)
        data,initial_kv,initial_kr=build_case(case);dev={k:v.npu() for k,v in data.items()}
        pipeline=MlaPrologSerial(*(dev[k] for k in ('wdq','wuq','wuk','wdkv','gq','gkv')))
        pipeline._matrix_backend=a.backend
        pipeline._vector_policy='baseline'
        candidate=copy.copy(pipeline)
        if a.vector_modes=='adaptive':
            candidate._vector_policy='tuned'
        elif a.vector_modes!='baseline':
            candidate._vector_modes=tuple(map(int,a.vector_modes.split(',')))
        weights={k:torch_npu.npu_format_cast(dev[k],29) for k in ('wdq','wuq','wdkv')}
        stream=torch.npu.Stream();graphs={};outputs={};owners={};caches={}
        def check_cpu(outputs_value,cache_value,current_data):
            rows=a.batch*a.sequence
            if rows<=512:
                return check_outputs(outputs_value,cache_value,oracle(current_data,a.heads,initial_kv,initial_kr),current_data,initial_kv,initial_kr)
            # Independent FP64 oracle for selected token rows, including edges.
            selected=sorted(set([0,rows//4,rows//2,rows-1]));small=dict(current_data)
            for key,width in [('x',a.hidden),('sin',64),('cos',64)]:small[key]=current_data[key].reshape(rows,width)[selected].view(1,len(selected),width)
            small['index']=current_data['index'].flatten()[selected].view(1,-1)
            ref=oracle(small,a.heads,initial_kv,initial_kr)
            checks={}
            for key,value,expected in zip(('query','query_rope'),outputs_value[:2],ref[:2]):
                actual=value.view(rows,a.heads,-1)[selected].view(1,len(selected),a.heads,-1)
                c=compare(actual,expected);c['pass']=c['matched_ratio']>=.99 and c['max_abs_error']<=1;checks[key]=c
            slots=initial_kv.numel()//512;touched=torch.zeros(slots,dtype=torch.bool);touched[current_data['index'].flatten()]=True
            for key,value,expected,width,initial in [('kv',cache_value[0],ref[2],512,initial_kv),('kr',cache_value[1],ref[3],64,initial_kr)]:
                cpu=value.cpu().reshape(-1,width);idx=small['index'].flatten()
                c=compare(cpu[idx],expected.reshape(-1,width)[idx]);c['untouched_exact']=torch.equal(cpu[~touched],initial.reshape(-1,width)[~touched]);c['pass']=c['matched_ratio']>=.99 and c['max_abs_error']<=1 and c['untouched_exact'];checks[key]=c
            checks['oracle_coverage']={'kind':'sampled token rows with full untouched cache check','selected':selected,'total_rows':rows}
            checks['pass']=all(v['pass'] for k,v in checks.items() if k!='oracle_coverage')
            return checks
        for label,owner in [('fused',None),('original',pipeline),('candidate',candidate)]:
            r={'status':'STARTED','samples_us':[]};result['variants'][label]=r
            try:
                caches[label]=(initial_kv.npu(),initial_kr.npu())
                if owner is None:
                    def call():return torch.ops.mla_benchmark.fused_public(dev['x'],weights['wdq'],weights['wuq'],dev['wuk'],weights['wdkv'],dev['gq'],dev['gkv'],dev['sin'],dev['cos'],dev['index'],*caches[label])
                    fn=None
                else:
                    fn=owner if a.mode=='S0' else owner.compile_superkernel(matrix_backend=a.backend,optimize=a.mode=='S2',optimize_options={'constant_codegen':1,'split_mode':4,'preload_code':5})
                    def call():return fn(dev['x'],dev['sin'],dev['cos'],dev['index'],*caches[label])
                start=time.perf_counter_ns();call();torch.npu.synchronize();r['first_call_wall_us']=(time.perf_counter_ns()-start)/1000
                stream.wait_stream(torch.npu.current_stream());graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream):value=call()
                if fn is not None and a.mode!='S0':fn.optimize_graph(graph)
                graph.replay();torch.npu.synchronize()
                r['precision']=check_cpu(value,caches[label],data)
                r['cache_return_aliases_input']=[value[k+2].data_ptr()==caches[label][k].data_ptr() for k in (0,1)]
                graphs[label]=graph;outputs[label]=value;owners[label]=fn
                for _ in range(10):graph.replay()
                torch.npu.synchronize();r['status']='PASS' if r['precision']['pass'] and all(r['cache_return_aliases_input']) else 'PRECISION_FAILURE'
            except Exception as exc:r.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        rng=random.Random(42)
        for _ in range(a.samples):
            order=list(graphs);rng.shuffle(order)
            for label in order:
                if result['variants'][label]['status']!='PASS':continue
                begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record()
                    for _ in range(10):graphs[label].replay()
                    end.record()
                end.synchronize();result['variants'][label]['samples_us'].append(begin.elapsed_time(end)*1000/10)
        for label,r in result['variants'].items():
            if r['samples_us']:r['median_us']=statistics.median(r['samples_us'])
            r['preparation']=getattr(owners.get(label),'preparation',None)
        if a.profile and 'candidate' in graphs:
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'trace')),record_shapes=True) as prof:
                graphs['candidate'].replay();torch.npu.synchronize();prof.step()
        # Full output comparison to the installed public API, processed in small CPU chunks.
        if 'fused' in outputs:
            for label in ('original','candidate'):
                if label not in outputs:continue
                comparisons=[]
                for v,ref in zip(outputs[label],outputs['fused']):
                    vf,rf=v.reshape(-1),ref.reshape(-1);matched=total=0;max_abs=0.
                    for start in range(0,vf.numel(),1048576):
                        av=vf[start:start+1048576].cpu().float();rv=rf[start:start+1048576].cpu().float();err=(av-rv).abs()
                        matched+=int((err<=2**-6+2**-6*rv.abs()).sum());total+=err.numel();max_abs=max(max_abs,float(err.max()))
                    comparisons.append({'matched_ratio':matched/total,'max_abs_error':max_abs,'pass':matched/total>=.99 and max_abs<=1})
                result['variants'][label]['full_vs_aclnnMlaProlog']=comparisons
                if not all(c['pass'] for c in comparisons):result['variants'][label]['status']='BASELINE_PRECISION_FAILURE'
        if a.refresh_check:
            addresses={k:v.data_ptr() for k,v in dev.items()}
            prepared=tuple(v.data_ptr() for v in pipeline._matrix_nz+pipeline._matrix_nd)
            newdata,_,_=build_case(case|{'seed':20261107,'index_mode':'reverse'})
            for k in dev:dev[k].copy_(newdata[k])
            pipeline.refresh_weights()
            for k in weights:weights[k].copy_(torch_npu.npu_format_cast(dev[k],29))
            result['refresh_addresses_preserved']=all(dev[k].data_ptr()==v for k,v in addresses.items()) and prepared==tuple(v.data_ptr() for v in pipeline._matrix_nz+pipeline._matrix_nd)
            for label,graph in graphs.items():
                caches[label][0].copy_(initial_kv);caches[label][1].copy_(initial_kr)
                graph.replay();torch.npu.synchronize()
                c=check_cpu(outputs[label],caches[label],newdata)
                result['variants'][label]['same_address_update']=c
                if not c['pass'] or not result['refresh_addresses_preserved']:result['variants'][label]['status']='REFRESH_FAILURE'
        result['loaded_binaries']={}
        result['vector_routes']={'original':pipeline.vector_modes(a.batch*a.sequence),
                                 'candidate':candidate.vector_modes(a.batch*a.sequence)}
        for line in Path('/proc/self/maps').read_text().splitlines():
            f=Path(line.split()[-1])
            if f.is_file() and f.suffix=='.so' and any(v in str(f) for v in ('mla_superkernel_native','mla_public_baseline','mlask_transformer','libopapi','libnnopbase')):result['loaded_binaries'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        result['status']='PASS' if all(r['status']=='PASS' for r in result['variants'].values()) else 'VARIANT_FAILURE'
        if a.mode!='S0':
            from inspect_static_metadata import inspect
            result['static_metadata']=inspect(out)
    except Exception as exc:result.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    result['finished']=datetime.now(timezone.utc).isoformat()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('loaded_binaries',)},indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':sys.exit(main())
