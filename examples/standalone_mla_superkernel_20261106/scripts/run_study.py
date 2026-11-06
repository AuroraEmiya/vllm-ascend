# SPDX-License-Identifier: Apache-2.0
"""One fresh process/cache per mode. Errors are persisted before returning failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',choices=['torch','torch_matmul','torch_matmul_nz','v3','v3_nz'],required=True)
    p.add_argument('--mode',choices=['S0','S1','S2'],required=True)
    p.add_argument('--batch',type=int,choices=[1,32],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report=out/'result.json'
    if report.exists(): raise FileExistsError(report)
    result=dict(backend=a.backend,mode=a.mode,batch=a.batch,hidden=7168,local_heads=32,sequence=1,dtype='BF16',source='own R5; not Contact',status='STARTED')
    try:
        # This cwd is part of the static-child cache identity; never reuse it across sources.
        cache=out/'fresh-static-cache';cache.mkdir(exist_ok=False);os.chdir(cache)
        import torch
        import torch_npu
        torch.set_num_threads(4)
        result['environment']=dict(python=sys.version,torch=torch.__version__,torch_npu=torch_npu.__version__,cann=os.environ.get('ASCEND_HOME_PATH'))
        torch.npu.set_device(0)
        result['device_name']=torch.npu.get_device_name(0)
        torch.npu.config.allow_internal_format=True
        try:
            from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel
            static_kernel._compute_opc_compile_jobs=lambda: '4'
        except ImportError:
            pass
        from mla_superkernel import MlaPrologSerial
        from mla_superkernel.reference import build_case,oracle,check_outputs
        case=dict(batch=a.batch,hidden=7168,heads=32,seed=20261106,block_size=16,index_mode='edges')
        data,initial_kv,initial_kr=build_case(case)
        expected=oracle(data,32,initial_kv,initial_kr)
        dev={k:v.npu() for k,v in data.items()}
        pipeline=MlaPrologSerial(*(dev[k] for k in ('wdq','wuq','wuk','wdkv','gq','gkv')))
        pipeline._matrix_backend=a.backend;pipeline._vector_backend='staged';pipeline._head_layout='row';pipeline._split_head_tile=16
        fn=pipeline if a.mode=='S0' else pipeline.compile_superkernel(matrix_backend=a.backend,optimize=a.mode=='S2',vector_backend='staged',head_layout='row',split_head_tile=16,optimize_options={'constant_codegen':1,'split_mode':4,'preload_code':5})
        caches=(initial_kv.npu(),initial_kr.npu())
        def call(): return fn(dev['x'],dev['sin'],dev['cos'],dev['index'],*caches)
        call();torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        stream=torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph,stream=stream): outputs=call()
        if a.mode!='S0': fn.optimize_graph(graph)
        torch.npu.synchronize()
        graph.replay();torch.npu.synchronize()
        result['initial_correctness']=check_outputs(outputs,caches,expected,data,initial_kv,initial_kr)
        addresses={k:v.data_ptr() for k,v in dev.items()}
        newdata,_,_=build_case({**case,'seed':20261107,'index_mode':'reverse'})
        for k in data: dev[k].copy_(newdata[k])
        pipeline.refresh_weights()
        caches[0].copy_(initial_kv);caches[1].copy_(initial_kr)
        graph.replay();torch.npu.synchronize()
        result['same_address_updates']=check_outputs(outputs,caches,oracle(newdata,32,initial_kv,initial_kr),newdata,initial_kv,initial_kr)
        result['addresses_preserved']=all(dev[k].data_ptr()==v for k,v in addresses.items())
        for _ in range(10): graph.replay()
        torch.npu.synchronize()
        samples=[]
        for _ in range(20):
            begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            with torch.npu.stream(stream):
                begin.record()
                for _ in range(10): graph.replay()
                end.record()
            end.synchronize();samples.append(begin.elapsed_time(end)*1000/10)
        result['graph_event_us']=dict(median=statistics.median(samples),raw=samples,calls_per_sample=10,description='One MLA call per captured graph; 10 replays per Event sample. Separate from profiler stage timing.')
        if a.profile:
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'trace')),record_shapes=True) as prof:
                graph.replay();torch.npu.synchronize();prof.step()
        result['preparation']=getattr(fn,'preparation',None)
        result['loaded_binaries']={}
        for line in Path('/proc/self/maps').read_text().splitlines():
            file=Path(line.split()[-1])
            if file.is_file() and file.suffix=='.so' and any(x in str(file) for x in ('mla_superkernel_native','vllm_ascend_C','serial_transformer','mlask_transformer','libopapi','libnnopbase')):
                result['loaded_binaries'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        result['status']='PASS' if result['initial_correctness']['pass'] and result['same_address_updates']['pass'] and result['addresses_preserved'] else 'CORRECTNESS_FAILURE'
        result['grouping']='REQUIRES_STATIC_METADATA_AND_DEVICE_TRACE_REVIEW'
    except Exception as exc:
        result.update(status='ERROR',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        (out/'error.txt').write_text(result['traceback'],encoding='utf-8')
    finally:
        report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': sys.exit(main())
