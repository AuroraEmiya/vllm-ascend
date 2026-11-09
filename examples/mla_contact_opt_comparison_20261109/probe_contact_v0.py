"""Run the unchanged Contact V0, retaining checkpoints even on device stalls.

V0 precision uses its actual BF16 matrix outputs as independent FP64 inputs.
Full mode calls the unchanged Contact opt4 Python entry and the existing full
FP64 oracle. No SuperKernel, kernel edit, dtype promotion or V0 substitution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--sequence', type=int, default=1)
    parser.add_argument('--rank', type=int, choices=(2, 3), default=2)
    parser.add_argument('--mode', choices=('v0', 'full'), default='v0')
    parser.add_argument('--no-rope', action='store_true')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'args': vars(args) | {'output': str(out)}, 'started': now(),
              'status': 'STARTED', 'checkpoints': [], 'samples_us': []}

    def save():
        temp = out / 'result.tmp'
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(out / 'result.json')

    def checkpoint(phase):
        result['current_phase'] = phase
        item = {'phase': phase, 'time': now()}
        result['checkpoints'].append(item)
        save()
        print(json.dumps(item), flush=True)

    try:
        checkpoint('import')
        import torch
        import torch_npu
        from mla_superkernel.reference import build_case, check_outputs, compare, oracle, rms, rope_reference
        torch.set_num_threads(4)
        torch.npu.set_device(0)
        torch.npu.config.allow_internal_format = True
        result['environment'] = {
            'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
            'soc': torch_npu._C._npu_get_soc_version(),
            'properties': str(torch.npu.get_device_properties(0)),
            'cann': os.environ.get('ASCEND_HOME_PATH'),
            'custom_opp': os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
        }
        prefix = 'cann_ops_transformer_custom.ops.attention.mla_prolog_v3_contact.'
        base = importlib.import_module(prefix + 'mla_prolog_contact')
        opt4 = importlib.import_module(prefix + 'mla_prolog_contact_opt4')
        checkpoint('binding_load')
        broad = base._builder.load()
        opt4._builder._loaded_ops[opt4._builder.name] = broad
        result['python_identity'] = {
            n: {'path': m.__file__, 'sha256': hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
            for n, m in [('base', base), ('opt4', opt4)]}
        result['schema'] = str(torch.ops.cann_ops_transformer.mla_prolog_norm_rope_contact.default._schema)
        checkpoint('data_preparation')
        case = dict(batch=args.batch, sequence=args.sequence, hidden=7168,
                    heads=32, seed=20261109, block_size=16, index_mode='edges')
        cpu, initial_kv, initial_kr = build_case(case)
        dev = {k: v.npu() for k, v in cpu.items()}
        weights = {k: torch_npu.npu_format_cast(dev[k], 29) for k in ('wdq', 'wuq', 'wdkv')}
        rows = args.batch * args.sequence
        x = dev['x'].view(rows, 7168) if args.rank == 2 else dev['x']
        sine = dev['sin'].view(rows, 64) if args.rank == 2 else dev['sin']
        cosine = dev['cos'].view(rows, 64) if args.rank == 2 else dev['cos']
        index = dev['index'].view(rows) if args.rank == 2 else dev['index']
        kv, kr = initial_kv.npu(), initial_kr.npu()
        torch.npu.synchronize()
        if args.mode == 'v0':
            checkpoint('matmul_dq_before')
            dq = torch.matmul(x, weights['wdq'])
            torch.npu.synchronize()
            checkpoint('matmul_dq_completed')
            dkv = torch.matmul(x, weights['wdkv'])
            torch.npu.synchronize()
            checkpoint('matmul_dkv_completed')
            cq_ref = rms(dq.cpu().double(), cpu['gq'].double())
            dkv_cpu = dkv.cpu().double()
            kv_ref = rms(dkv_cpu[..., :512], cpu['gkv'].double())
            raw_kr = dkv_cpu[..., 512:]
            kr_ref = raw_kr if args.no_rope else rope_reference(raw_kr, sine.cpu().double(), cosine.cpu().double())
            expected_kv, expected_kr = initial_kv.double().clone(), initial_kr.double().clone()
            expected_kv.view(-1, 512)[cpu['index'].flatten()] = kv_ref.reshape(rows, 512)
            expected_kr.view(-1, 64)[cpu['index'].flatten()] = kr_ref.reshape(rows, 64)

            def call():
                return torch.ops.cann_ops_transformer.mla_prolog_norm_rope_contact(
                    dq, dkv, dev['gq'], dev['gkv'], sine, cosine, index, kv, kr,
                    epsilon_cq=1e-5, epsilon_ckv=1e-5, qc_qr_scale=1., kc_scale=1.,
                    cache_mode=0, do_rope=not args.no_rope)

            def validate(value):
                checks = {name: compare(got, want) for name, got, want in
                          [('cq', value, cq_ref), ('kv_cache', kv, expected_kv), ('kr_cache', kr, expected_kr)]}
                checks['finite'] = all(bool(torch.isfinite(t.cpu().float()).all()) for t in (value, kv, kr))
                touched = torch.zeros(initial_kv.numel() // 512, dtype=torch.bool)
                touched[cpu['index'].flatten()] = True
                checks['untouched_kv_exact'] = torch.equal(kv.cpu().view(-1, 512)[~touched], initial_kv.view(-1, 512)[~touched])
                checks['untouched_kr_exact'] = torch.equal(kr.cpu().view(-1, 64)[~touched], initial_kr.view(-1, 64)[~touched])
                checks['pass'] = all(checks[k] for k in ('finite', 'untouched_kv_exact', 'untouched_kr_exact')) and all(
                    checks[k]['matched_ratio'] >= .99 and checks[k]['max_abs_error'] <= 1 for k in ('cq', 'kv_cache', 'kr_cache'))
                return checks
        else:
            if args.no_rope:
                raise ValueError('Full precision currently uses the RoPE-enabled public oracle')
            expected = oracle(cpu, 32, initial_kv, initial_kr)

            def call():
                return opt4.mla_prolog(
                    x, weights['wdq'], weights['wuq'], dev['wuk'], weights['wdkv'],
                    dev['gq'], dev['gkv'], kv, kr, rope_sin=sine, rope_cos=cosine,
                    cache_index=index, cache_mode='PA_BSND', query_norm_flag=False,
                    enable_superkernel=False)

            def validate(value):
                canonical = (value[0].view(args.batch, args.sequence, 32, 512),
                             value[1].view(args.batch, args.sequence, 32, 64), kv, kr)
                return check_outputs(canonical, (kv, kr), expected, cpu, initial_kv, initial_kr)

        result['loaded_binaries'] = {}
        for line in Path('/proc/self/maps').read_text().splitlines():
            path = Path(line.split()[-1])
            if path.is_file() and path.suffix == '.so' and any(s in str(path) for s in ('contact', 'cust_opapi', 'optiling', 'nnopbase', 'opapi')):
                result['loaded_binaries'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint('operator_call_before')
        started = time.perf_counter_ns()
        value = call()
        result['first_call_enqueue_wall_us'] = (time.perf_counter_ns() - started) / 1000
        checkpoint('operator_enqueued_waiting_for_sync')
        torch.npu.synchronize()
        result['first_call_and_sync_wall_us'] = (time.perf_counter_ns() - started) / 1000
        checkpoint('operator_first_sync_completed')
        result['eager_precision'] = validate(value)
        if not result['eager_precision']['pass']:
            raise AssertionError('Original operator failed independent precision')
        checkpoint('eager_precision_passed')
        if args.samples:
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                for _ in range(10):
                    value = call()
            stream.synchronize()
            checkpoint('warmup_completed')
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = call()
            torch.npu.synchronize()
            checkpoint('graph_capture_completed')
            for sample in range(args.samples):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                with torch.npu.stream(stream):
                    begin.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                end.synchronize()
                result['samples_us'].append(begin.elapsed_time(end) * 1000 / 10)
                save()
            result['graph_precision'] = validate(captured)
            if not result['graph_precision']['pass']:
                raise AssertionError('Graph replay failed independent precision')
            result['median_us'] = statistics.median(result['samples_us'])
        result['status'] = 'PASS'
        checkpoint('completed')
    except Exception as exc:
        result.update(status='ERROR', error=str(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    result['finished'] = now()
    save()
    print(json.dumps({'status': result['status'], 'phase': result['current_phase'],
                      'median_us': result.get('median_us'), 'output': str(out)}), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
