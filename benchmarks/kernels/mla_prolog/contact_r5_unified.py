# SPDX-License-Identifier: Apache-2.0
"""Same-device, unchanged-source MLA serial comparison; no fused-source access.

Run one batch per process. Both serial paths receive identical 3D BF16 input.
Preparation, accuracy, stable-address updates and profiling are outside Events.
R5 native/static/SuperKernel controls all use the frozen staged V + V3 NZ route.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import torch_npu

R5 = Path(os.environ.get('UNIFIED_R5_REPO', '/home/s30083169/dev-mla-prolog/vector-tiling/vllm-ascend'))
sys.path.insert(0, str(R5 / 'benchmarks/kernels/mla_prolog'))
import validate as validation
from preflight import make_data
from vllm_ascend.ops.mla_prolog_serial import MlaPrologSerial, validate_cache_index

CALLS = 10
SAMPLES = 20
ROUTES = ('contact', 'r5_native', 'r5_static', 'r5_superkernel', 'fused_aclnn')
CONTACT_HASH = 'cc4f60f9f9d3cdcb8cf0b95b288b0e149821a066d68d4035041e7514ee2de7ef'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '.so' in line and any(s in line for s in
                    ('opapi', 'nnopbase', 'vllm_ascend', 'mla_prolog_contact', 'cust_opsproto', 'optiling'))})
    return {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'soc': torch.npu.get_device_name(0), 'device_count': torch.npu.device_count(),
            'allow_hf32_matmul': bool(torch.npu.matmul.allow_hf32),
            'files': {p: sha(p) for p in paths if Path(p).is_file()},
            'r5_python': {str(p): sha(p) for p in
                (R5 / 'vllm_ascend/ops/mla_prolog_serial.py',
                 R5 / 'vllm_ascend/compilation/serial_superkernel.py',
                 R5 / 'vllm_ascend/ops/serial_matmul.py', Path(validation.__file__))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--routes', default=','.join(ROUTES))
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--profile-target', choices=ROUTES)
    args = parser.parse_args()
    routes = args.routes.split(',')
    if len(set(routes)) != len(routes) or set(routes) - set(ROUTES):
        parser.error('unknown or duplicate route')
    if args.profile_target and args.profile_target not in routes:
        parser.error('profile target must be selected')
    if args.out.exists():
        parser.error('preserve prior result, select a new output path')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'STARTED', 'case': {'id': f'decode-b{args.batch}', 'batch': args.batch,
              'seq': 1, 'total_tokens': args.batch, 'hidden': 7168, 'heads': 32,
              'block_size': 16, 'seed': 20261105},
              'calls_per_graph': CALLS, 'samples_per_route': SAMPLES,
              'routes': {}, 'raw_samples': [], 'preparation': {},
              'harness_sha256': sha(__file__), 'host_start_ns': time.time_ns()}

    def save():
        temporary = args.out.with_suffix('.partial')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.out)

    def phase(name, **extra):
        print('UNIFIED_PHASE', name, json.dumps(extra), flush=True)
        result['phase'] = name
        save()

    save()
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    from torch_npu.dynamo.npugraph_ex._acl_concrete_graph import static_kernel
    static_kernel._compute_opc_compile_jobs = lambda: '4'
    contact_module = importlib.import_module(
        'cann_ops_transformer_custom.ops.attention.mla_prolog_v3_contact.mla_prolog_contact')
    result['contact_wrapper'] = {'path': contact_module.__file__, 'sha256': sha(contact_module.__file__)}
    assert result['contact_wrapper']['sha256'] == CONTACT_HASH, 'Contact wrapper differs from reviewed source'
    case = result['case']
    data, initial_kv, initial_kr = validation.build_case(case)
    phase('cpu_oracle')
    expected = validation.oracle(data, case['heads'], initial_kv, initial_kr)
    d = {name: value.npu() for name, value in data.items()}
    nz = {name: torch_npu.npu_format_cast(d[name], 29) for name in ('wdq', 'wuq', 'wdkv')}
    initial_device = (initial_kv.npu(), initial_kr.npu())
    caches = {name: (initial_kv.npu(), initial_kr.npu()) for name in routes}
    begin = time.perf_counter_ns()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    owner = MlaPrologSerial(d['wdq'], d['wuq'], d['wuk'], d['wdkv'], d['gq'], d['gkv'])
    owner._matrix_backend = 'v3_nz'
    owner._vector_backend = 'staged'
    owner._head_layout = 'row'
    owner._split_head_tile = 16
    torch.npu.synchronize()
    result['preparation']['r5_weights'] = {'wall_us': (time.perf_counter_ns() - begin) / 1000,
        'allocated_delta_bytes': torch.npu.memory_allocated() - before, 'buffers': owner.prepared_weight_info()}
    prepared = owner.prepared_weight_info()
    compiled = {}
    for name in ('r5_static', 'r5_superkernel'):
        if name in routes:
            compiled[name] = owner.compile_superkernel(
                optimize=name == 'r5_superkernel', vector_backend='staged', matrix_backend='v3_nz',
                head_layout='row', split_head_tile=16,
                optimize_options={'constant_codegen': 1, 'split_mode': 4, 'preload_code': 5})
            result['preparation'][name] = compiled[name].preparation

    def call(name):
        if name == 'contact':
            outputs = contact_module.mla_prolog(
                d['x'], nz['wdq'], nz['wuq'], d['wuk'], nz['wdkv'], d['gq'], d['gkv'],
                *caches[name], rope_sin=d['sin'], rope_cos=d['cos'], cache_index=d['index'],
                cache_mode='PA_BSND', query_norm_flag=False, enable_superkernel=False)
            return outputs[0], outputs[1], *caches[name]
        if name == 'fused_aclnn':
            return torch.ops._C_ascend.mla_prolog_fused_public(
                d['x'], nz['wdq'], nz['wuq'], d['wuk'], nz['wdkv'], d['gq'], d['gkv'],
                d['sin'], d['cos'], d['index'], *caches[name])
        pipeline = owner if name == 'r5_native' else compiled[name]
        return pipeline(d['x'], d['sin'], d['cos'], d['index'], *caches[name])

    def reset(name):
        for target, source in zip(caches[name], initial_device):
            target.copy_(source)

    def check(name, outputs, ref, cpu_data):
        value = validation.check_outputs(outputs, caches[name], ref, cpu_data, initial_kv, initial_kr)
        value['pass'] &= all(value[k]['returned_aliases_input'] for k in ('kv_cache_state', 'kr_cache_state'))
        return value

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graphs, captured = {}, {}
    with torch.npu.stream(stream):
        for name in routes:
            row = result['routes'][name] = {'status': 'STARTED', 'checks': {}}
            phase('eager_and_capture', route=name)
            try:
                reset(name)
                start = time.perf_counter_ns()
                outputs = call(name)
                stream.synchronize()
                result['preparation'][name + '_first_call_wall_us'] = (time.perf_counter_ns() - start) / 1000
                row['checks']['eager'] = check(name, outputs, expected, data)
                if not row['checks']['eager']['pass']:
                    row['status'] = 'FAIL_EAGER'
                    continue
                for _ in range(10):
                    call(name)
                stream.synchronize()
                reset(name)
                graph = torch.npu.NPUGraph()
                start = time.perf_counter_ns()
                with torch.npu.graph(graph, stream=stream):
                    for _ in range(CALLS):
                        outputs = call(name)
                if name in compiled:
                    compiled[name].optimize_graph(graph)
                result['preparation'][name + '_capture_optimize_wall_us'] = (time.perf_counter_ns() - start) / 1000
                for _ in range(10):
                    graph.replay()
                stream.synchronize()
                row['checks']['graph'] = check(name, outputs, expected, data)
                if not row['checks']['graph']['pass']:
                    row['status'] = 'FAIL_GRAPH'
                    continue
                graphs[name], captured[name] = graph, outputs
                row['status'] = 'CAPTURED'
            except Exception:
                row.update(status='ERROR', traceback=traceback.format_exc())
            save()

        result['inputs_immutable_before_update'] = all(torch.equal(t.cpu(), data[n]) for n, t in d.items())
        result['nz_immutable_before_update'] = all(torch.equal(t.cpu(), data[n]) for n, t in nz.items())
        assert result['inputs_immutable_before_update'] and result['nz_immutable_before_update']
        phase('stable_address_update')
        updated = make_data(case['batch'], case['hidden'], case['heads'], case['seed'] + 1)
        slots = initial_kv.numel() // 512
        updated['index'] = (slots - 1 - updated['index']).contiguous()
        validate_cache_index(updated['index'], slots)
        for name, value in d.items():
            value.copy_(updated[name])
        for name, value in nz.items():
            value.copy_(torch_npu.npu_format_cast(d[name], 29))
        owner.refresh_weights()
        stream.synchronize()
        assert owner.prepared_weight_info() == prepared
        updated_expected = validation.oracle(updated, case['heads'], initial_kv, initial_kr)
        for name, graph in graphs.items():
            reset(name)
            graph.replay()
            stream.synchronize()
            result['routes'][name]['checks']['graph_changed_data'] = check(
                name, captured[name], updated_expected, updated)
        for name, value in d.items():
            value.copy_(data[name])
        for name, value in nz.items():
            value.copy_(torch_npu.npu_format_cast(d[name], 29))
        owner.refresh_weights()
        stream.synchronize()
        assert owner.prepared_weight_info() == prepared
        for name, graph in graphs.items():
            reset(name)
            graph.replay()
            stream.synchronize()
            row = result['routes'][name]
            row['checks']['graph_restored'] = check(name, captured[name], expected, data)
            row['status'] = 'PASS' if all(v['pass'] for v in row['checks'].values()) else 'FAIL_GRAPH_UPDATE'
        result['inputs_immutable_after_restore'] = all(torch.equal(t.cpu(), data[n]) for n, t in d.items())
        result['nz_immutable_after_restore'] = all(torch.equal(t.cpu(), data[n]) for n, t in nz.items())
        valid = [n for n in routes if result['routes'][n]['status'] == 'PASS']

        if args.profile_target:
            name = args.profile_target
            assert name in valid
            phase('profile_replay', route=name)
            result['profile_marker'] = {'route': name, 'calls_per_graph': CALLS, 'replays': 2,
                                        'host_begin_ns': time.time_ns()}
            for _ in range(2):
                graphs[name].replay()
            stream.synchronize()
            result['profile_marker']['host_end_ns'] = time.time_ns()
        elif not args.no_timing:
            phase('interleaved_device_event_timing')
            for round_index in range(SAMPLES):
                offset = round_index % len(valid) if valid else 0
                order = valid[offset:] + valid[:offset]
                if round_index % 2:
                    order.reverse()
                for name in order:
                    reset(name)
                    start = torch.npu.Event(enable_timing=True)
                    stop = torch.npu.Event(enable_timing=True)
                    start.record(stream)
                    graphs[name].replay()
                    stop.record(stream)
                    stop.synchronize()
                    total_us = start.elapsed_time(stop) * 1000
                    result['raw_samples'].append({'round': round_index, 'route': name,
                        'graph_us': total_us, 'per_call_us': total_us / CALLS})
            for name in valid:
                values = [s['per_call_us'] for s in result['raw_samples'] if s['route'] == name]
                q = statistics.quantiles(values, n=10)
                result['routes'][name]['timing'] = {'median_us': statistics.median(values),
                    'p10_us': q[0], 'p90_us': q[8], 'min_us': min(values), 'max_us': max(values),
                    'sample_count': len(values)}

    result['runtime_identity'] = identity()
    result['host_end_ns'] = time.time_ns()
    result['status'] = 'PASS' if all(result['routes'][n]['status'] == 'PASS' for n in routes) else 'PARTIAL_FAILURE'
    result['grouping_status'] = 'REQUIRES_FULL_DEVICE_TRACE'
    phase('finished', status=result['status'])
    print(json.dumps({n: {'status': r['status'], 'timing': r.get('timing')} for n, r in result['routes'].items()}), flush=True)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
