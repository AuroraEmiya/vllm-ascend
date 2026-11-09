"""Same-input, whole-call Graph Event comparison of frozen MLA implementations.

Contact sources and the frozen own source are provided by explicit read-only
paths. This harness changes no kernel or matrix implementation. The optional
norm_opt2 + opt4 composition is diagnostic, not a released Contact default.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--sequence', type=int, default=1)
    p.add_argument('--hidden', type=int, default=7168)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--samples', type=int, default=20)
    p.add_argument('--own-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--own-sk', action='store_true')
    p.add_argument('--contact-rank', type=int, choices=[2, 3], default=2)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--norm-opt-build-failed', action='store_true')
    p.add_argument('--norm-opt2-timeout-guard', type=int, default=512,
                   help='Exclude diagnostic composition at/above the observed 2026-11-09 T512 runtime stall; zero disables the guard')
    p.add_argument('--only', default='all')
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    result = {'args': vars(a) | {'own_root': str(a.own_root), 'output': str(out)},
              'started': datetime.now(timezone.utc).isoformat(), 'variants': {}}
    try:
        cache = out / 'static-cache'; cache.mkdir(); os.chdir(cache)
        import torch
        import torch_npu
        from mla_superkernel import MlaPrologSerial
        from mla_superkernel.reference import build_case, oracle, check_outputs, compare
        torch.set_num_threads(4); torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        try:
            from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel
            static_kernel._compute_opc_compile_jobs = lambda: '4'
        except ImportError:
            pass
        result['environment'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'device': str(torch.npu.get_device_properties(0)), 'soc': torch_npu._C._npu_get_soc_version(),
            'cann': os.environ['ASCEND_HOME_PATH'], 'custom_opp': os.environ['ASCEND_CUSTOM_OPP_PATH']}
        result['fused_api_schemas'] = {name: str(getattr(torch.ops.npu, name).default._schema)
            for name in ('npu_mla_prolog_v2', 'npu_mla_prolog_v3')}
        baseline_so = a.own_root / 'build/baseline-binding/mla_public_baseline.so'
        torch.ops.load_library(str(baseline_so))
        case = dict(batch=a.batch, sequence=a.sequence, hidden=a.hidden, heads=a.heads,
                    seed=20261109, block_size=16, index_mode='edges')
        cpu, initial_kv, initial_kr = build_case(case)
        reference = oracle(cpu, a.heads, initial_kv, initial_kr)
        dev = {k: v.npu() for k, v in cpu.items()}
        weights = {k: torch_npu.npu_format_cast(dev[k], 29) for k in ('wdq', 'wuq', 'wdkv')}
        own = MlaPrologSerial(*(dev[k] for k in ('wdq','wuq','wuk','wdkv','gq','gkv')))
        own._matrix_backend = 'v3_nz'; own._vector_policy = 'tuned'
        rows = a.batch * a.sequence
        x = dev['x'].view(rows, a.hidden) if a.contact_rank == 2 else dev['x']
        sine = dev['sin'].view(rows,64) if a.contact_rank == 2 else dev['sin']
        cosine = dev['cos'].view(rows,64) if a.contact_rank == 2 else dev['cos']
        index = dev['index'].view(rows) if a.contact_rank == 2 else dev['index']
        prefix = 'cann_ops_transformer_custom.ops.attention.mla_prolog_v3_contact.'
        modules = {}
        for name in ('mla_prolog_contact','mla_prolog_contact_opt','mla_prolog_contact_opt2',
                     'mla_prolog_contact_opt3','mla_prolog_contact_opt4',
                     'mla_prolog_norm_rope_contact_opt','mla_prolog_norm_rope_contact_opt2'):
            modules[name] = importlib.import_module(prefix + name)
        # The installed package's single broad C++ binding has all V0/V1/V2
        # functions. Reuse it for identical builder sources in setup only.
        broad = modules['mla_prolog_contact']._builder.load()
        for name, module in modules.items():
            if name != 'mla_prolog_norm_rope_contact_opt2':
                module._builder._loaded_ops[module._builder.name] = broad
        modules['mla_prolog_norm_rope_contact_opt2']._builder.load()
        result['contact_python_identity'] = {name: {'path': module.__file__,
            'sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            for name,module in modules.items()}
        result['contact_binding_exports'] = sorted(n for n in dir(broad) if n.startswith('mla_prolog'))
        options = dict(rope_sin=sine, rope_cos=cosine, cache_index=index,
                       cache_mode='PA_BSND', query_norm_flag=False, enable_superkernel=False)
        def contact_call(module, kv, kr):
            return module.mla_prolog(x, weights['wdq'], weights['wuq'], dev['wuk'],
                    weights['wdkv'], dev['gq'], dev['gkv'], kv, kr, **options)
        def compose_norm(norm_name, kv, kr):
            # Diagnostic composition: exact Contact torch.matmul topology,
            # explicitly replacing only the V0 and V1 calls with existing ops.
            dq = torch.matmul(x, weights['wdq'])
            dkv = torch.matmul(x, weights['wdkv'])
            cq = getattr(torch.ops.cann_ops_transformer, norm_name)(
                dq, dkv, dev['gq'], dev['gkv'], sine, cosine, index, kv, kr)
            up = torch.matmul(cq, weights['wuq'])
            qc, qr = torch.ops.cann_ops_transformer.mla_prolog_qc_rope_contact_opt4(
                up, sine, cosine, a.heads, 128, 64, True, 0)
            if a.contact_rank == 2:
                query = torch_npu.npu_transpose_batchmatmul(qc, dev['wuk'], perm_y=[1,0,2])
            else:
                nt = torch.matmul(qc, dev['wuk'])
                query, _ = torch.ops.cann_ops_transformer.mla_prolog_query_contact(
                    nt, False, a.batch, a.sequence, 3)
            return query, qr
        labels = ['aclnnMlaProlog','MlaPrologV2','MlaPrologV3','own_S0']
        if a.own_sk: labels += ['own_S2']
        labels += ['contact','contact_opt','contact_opt2','contact_opt3','contact_opt4',
                   'compose_norm_opt_opt4','compose_norm_opt2_opt4']
        if a.only != 'all': labels = [v for v in labels if v in a.only.split(',')]
        stream = torch.npu.Stream(); graphs = {}; values = {}; caches = {}; owners = {}
        def canonical(value, kvkr):
            return (value[0].view(a.batch,a.sequence,a.heads,512),
                    value[1].view(a.batch,a.sequence,a.heads,64), *kvkr)
        for label in labels:
            r = {'status': 'STARTED', 'samples_us': []}; result['variants'][label] = r
            result['current_variant'] = label
            (out/'result-partial.json').write_text(json.dumps(result,indent=2)+'\n')
            if label == 'MlaPrologV2' and a.sequence > 16:
                r.update(status='SKIPPED_PUBLIC_V2_S_RANGE', reason='Installed Torch-NPU V2 public API doc limits rank3 S to 0..16; preserve this shape for other versions without claiming V2 support')
                continue
            if a.norm_opt_build_failed and label == 'compose_norm_opt_opt4':
                r.update(status='SKIPPED_SOURCE_BUILD_FAILURE', reason='Latest norm_opt kernel has no function name at lines 144-145; original compile error preserved')
                continue
            if a.norm_opt2_timeout_guard and rows >= a.norm_opt2_timeout_guard and label == 'compose_norm_opt2_opt4':
                r.update(status='SKIPPED_OBSERVED_STALL_GUARD', reason='Initial T512 full composition reported a three-minute runtime timeout and was terminated; larger cases excluded after this observed failure, not claimed to pass')
                continue
            # The unmodified Contact V0 documents a T>=128 hang. Keep it as
            # unsupported coverage; do not silently replace it in opt1..4.
            if rows >= 128 and label in ('contact','contact_opt','contact_opt2','contact_opt3','contact_opt4'):
                r.update(status='SKIPPED_DOCUMENTED_V0_HANG', reason='Contact README T>=128 V0 hang; V1-only opt entry still calls that V0')
                continue
            try:
                caches[label] = (initial_kv.npu(), initial_kr.npu())
                kv, kr = caches[label]
                owner = None
                if label == 'aclnnMlaProlog':
                    fn = lambda: torch.ops.mla_benchmark.fused_public(dev['x'],weights['wdq'],weights['wuq'],dev['wuk'],weights['wdkv'],dev['gq'],dev['gkv'],dev['sin'],dev['cos'],dev['index'],kv,kr)
                elif label == 'MlaPrologV2':
                    fn = lambda: torch_npu.npu_mla_prolog_v2(dev['x'],weights['wdq'],weights['wuq'],dev['wuk'],weights['wdkv'],dev['gq'],dev['gkv'],dev['sin'],dev['cos'],dev['index'],kv,kr,cache_mode='PA_BSND')
                elif label == 'MlaPrologV3':
                    fn = lambda: torch_npu.npu_mla_prolog_v3(dev['x'],weights['wdq'],weights['wuq'],dev['wuk'],weights['wdkv'],dev['gq'],dev['gkv'],dev['sin'],dev['cos'],kv,kr,cache_index=dev['index'],cache_mode='PA_BSND',query_norm_flag=False,weight_quant_mode=0,kv_cache_quant_mode=0,query_quant_mode=0,ckvkr_repo_mode=0,quant_scale_repo_mode=0,tile_size=128,qc_qr_scale=1.,kc_scale=1.)
                elif label.startswith('own_'):
                    owner = own if label == 'own_S0' else own.compile_superkernel(matrix_backend='v3_nz',optimize=True,optimize_options={'constant_codegen':1,'split_mode':4,'preload_code':5})
                    fn = lambda: owner(dev['x'],dev['sin'],dev['cos'],dev['index'],kv,kr)
                elif label.startswith('compose_'):
                    norm_name = 'mla_prolog_norm_rope_contact_opt2' if label == 'compose_norm_opt2_opt4' else 'mla_prolog_norm_rope_contact_opt'
                    fn = lambda: compose_norm(norm_name,kv,kr)
                else:
                    name = 'mla_prolog_' + label
                    module = modules[name]
                    fn = lambda: contact_call(module,kv,kr)
                start = time.perf_counter_ns(); eager = fn(); torch.npu.synchronize()
                r['first_call_wall_us'] = (time.perf_counter_ns()-start)/1000
                r['eager_precision'] = check_outputs(canonical(eager,caches[label]),caches[label],reference,cpu,initial_kv,initial_kr)
                stream.wait_stream(torch.npu.current_stream()); graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph,stream=stream): value = fn()
                if label == 'own_S2': owner.optimize_graph(graph)
                graph.replay(); torch.npu.synchronize()
                r['graph_precision'] = check_outputs(canonical(value,caches[label]),caches[label],reference,cpu,initial_kv,initial_kr)
                r['raw_output_metadata'] = [{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for v in value]
                if label.startswith('own_') or label in ('aclnnMlaProlog', 'MlaPrologV2'):
                    r['returned_cache_alias'] = [value[i+2].data_ptr()==caches[label][i].data_ptr() for i in (0,1)]
                if label.startswith('contact'):
                    r['unused_outputs_empty'] = [value[i].numel()==0 for i in (2,3,4)]
                r['status'] = 'PASS' if r['eager_precision']['pass'] and r['graph_precision']['pass'] and all(r.get('returned_cache_alias',[True])) and all(r.get('unused_outputs_empty',[True])) else 'PRECISION_FAILURE'
                graphs[label] = graph; values[label] = value; owners[label] = owner
                for _ in range(10): graph.replay()
                torch.npu.synchronize()
            except Exception as exc:
                r.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
            print(json.dumps({'variant':label,'status':r['status']}),flush=True)
            (out/'result-partial.json').write_text(json.dumps(result,indent=2)+'\n')
        rng = random.Random(20261109)
        for _ in range(a.samples):
            order = list(graphs); rng.shuffle(order)
            for label in order:
                r = result['variants'][label]
                if r['status'] != 'PASS': continue
                begin,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record()
                    for _ in range(10): graphs[label].replay()
                    end.record()
                end.synchronize();r['samples_us'].append(begin.elapsed_time(end)*1000/10)
        if 'aclnnMlaProlog' in values:
            fused = canonical(values['aclnnMlaProlog'],caches['aclnnMlaProlog'])
            for label,value in values.items():
                checks=[]
                for got,want in zip(canonical(value,caches[label]),fused):
                    c=compare(got,want.cpu().double());c['pass']=c['matched_ratio']>=.99 and c['max_abs_error']<=1;checks.append(c)
                r=result['variants'][label];r['full_vs_aclnnMlaProlog']=checks
                if not all(c['pass'] for c in checks):r['status']='BASELINE_PRECISION_FAILURE'
        # Trace each successful full replay under its own name, without using
        # these profiler timings as Event latency or pure Cube-active time.
        if a.profile:
            for label,graph in graphs.items():
                if result['variants'][label]['status']!='PASS':continue
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'trace'/label)),record_shapes=True) as prof:
                    graph.replay();torch.npu.synchronize();prof.step()
        for label,r in result['variants'].items():
            if r['samples_us']:
                r['median_us']=statistics.median(r['samples_us'])
                r['min_us']=min(r['samples_us']);r['max_us']=max(r['samples_us'])
            r['preparation']=getattr(owners.get(label),'preparation',None)
        result['loaded_binaries']={}
        for line in Path('/proc/self/maps').read_text().splitlines():
            f=Path(line.split()[-1])
            if f.is_file() and f.suffix=='.so' and any(s in str(f) for s in ('mla','cust_opapi','optiling','nnopbase','opapi')):
                result['loaded_binaries'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        result['vector_modes']=own.vector_modes(rows)
        statuses=[r['status'] for r in result['variants'].values()]
        result['status']='PASS' if all(s=='PASS' or s.startswith('SKIPPED_') for s in statuses) else 'VARIANT_FAILURE'
    except Exception as exc:
        result.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    result['finished']=datetime.now(timezone.utc).isoformat()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'output':str(out),'variants':{k:{f:v[f] for f in ('status','median_us','error') if f in v} for k,v in result['variants'].items()}}),flush=True)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
