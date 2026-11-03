#!/usr/bin/env python3
"""D19 stateful S adapter for the official msprof application protocol.

Explicit extension: setup may launch target-named kernels. We retain the whole
CSV and require a reviewed setup-aware map; no name-only collector is used.
No Python performance timing. Only Task Duration from the separate mapper.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import uuid
from common import SOC, load_case, load_json, now, require, sha, task_root, verify_bundle, verify_sources, write_json
from fixture import IMMUTABLE, INPUT_NAMES, cpu_inputs, load_module, tensor_hash
import resource_guard_a3 as guard

WARMUP_SHAPE = (128, 1024, 1024)
WARMUP_BYTES = 256 * 1024**2


def proc_identity():
    stat = Path('/proc/self/stat').read_text()
    return {'pid_in_container': os.getpid(), 'start_ticks': stat.rsplit(')', 1)[1].split()[19],
            'pid_namespace': os.readlink('/proc/self/ns/pid'), 'cgroup': Path('/proc/self/cgroup').read_text(),
            'status': Path('/proc/self/status').read_text()}


def ownership_handshake(out, receipt_path, timeout=180):
    """Outer sole executor verifies actual host PID/ns/cgroup; snapshot != lock."""
    request = {'nonce': uuid.uuid4().hex, 'created_at': now(), 'process': proc_identity(),
               'selected': {k: os.environ.get(k) for k in ('PERF_HOST', 'PERF_NPU_ID', 'PERF_CHIP_ID', 'PERF_PHYSICAL_DEVICE', 'ASCEND_RT_VISIBLE_DEVICES')}}
    write_json(out / 'ownership-request.json', request)
    deadline = time.monotonic() + timeout
    while not Path(receipt_path).is_file():
        require(time.monotonic() < deadline, 'Outer host-PID ownership receipt timed out; no custom operator executed')
        time.sleep(0.25)
    receipt = load_json(receipt_path)
    require(receipt.get('nonce') == request['nonce'], 'Stale/foreign ownership receipt')
    require(receipt.get('outer_executor_verified') is True, 'Actual hostPID namespace/cgroup ownership not verified')
    require(receipt.get('process') == request['process'], 'Receipt does not bind actual runner process')
    require(receipt.get('selected') == request['selected'], 'Receipt device mapping differs')
    host_pid = receipt.get('host_pid')
    require(type(host_pid) is int and host_pid > 0, 'Actual host PID missing')
    require(receipt.get('host_NSpid') and receipt['host_NSpid'][-1] == os.getpid(), 'Host NSpid does not map to runner')
    require(receipt.get('evidence_files'), 'Outer host /proc namespace/cgroup/starttime evidence required')
    # Every raw evidence file is checked; interpreting actual host PID ownership
    # remains the outer executor's explicit responsibility, not len(PIDs)==1.
    for item in receipt['evidence_files']:
        require(sha(item['path']) == item['sha256'], 'Ownership evidence bytes changed')
    write_json(out / 'ownership-receipt-consumed.json', receipt)
    guard.gate(out, 'ownership-confirmed', expected={str(host_pid)})
    return host_pid


def warmup_call(torch, tensor, selection, root):
    if selection['mode'] == 'compiled':
        sys.path.insert(0, str(root / 'official/compiled_warmup'))
        from ops_profiling_warmup import compiled_warmup
        compiled_warmup(tensor)
    elif selection['logical_op'] == 'ArgMax':
        torch.argmax(tensor)
    elif selection['logical_op'] == 'Max':
        torch.max(tensor)
    else:
        raise RuntimeError('Unknown actual warmup selection')


def output_state_digest(torch, output, state):
    require(output.dtype == torch.bfloat16 and output.ndim == 2 and output.shape[1] == 512, 'Output dtype/shape differs')
    require(state.dtype == torch.float32 and state.ndim == 3 and tuple(state.shape[1:]) == (32, 1024), 'State contract differs')
    # Check on bounded CPU chunks: no extra finite-check compute kernels in target.
    for value in (output, state):
        flat = value.detach().reshape(-1)
        for begin in range(0, flat.numel(), 4 * 1024**2):
            require(bool(torch.isfinite(flat[begin:begin + 4 * 1024**2].cpu()).all()), 'Nonfinite actual output/state')
    return {'output_sha256': tensor_hash(output), 'state_sha256': tensor_hash(state)}


def verify_loaded(root, build, out):
    expected = [root / 'source/dependencies/original_a3/n_impl/build/libcv60n.so',
                root / 'source/operators/compressor_c2_a3_bf16_c_only/build/libc2_a3_bf16_c_only.so']
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line}
    records = []
    for path in expected:
        require(str(path) in mapped, 'Fresh task DSO not actually mapped: ' + str(path))
        require({p for p in mapped if Path(p).name == path.name} == {str(path)}, 'Competing DSO with same basename loaded')
        row = next(r for r in build['artifacts'] if r['path'] == path.relative_to(root).as_posix())
        require(sha(path) == row['sha256'], 'Loaded DSO bytes differ from fresh build')
        records.append({'path': str(path), 'sha256': sha(path)})
    (out / 'proc-self-maps.txt').write_text(Path('/proc/self/maps').read_text())
    write_json(out / 'actual-loaded-identity.json', {'DSOs': records, 'maps_sha256': sha(out / 'proc-self-maps.txt')})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--ownership-receipt', required=True)
    parser.add_argument('--epoch', required=True)
    parser.add_argument('--soc', required=True, choices=[SOC])
    parser.add_argument('--case-ids')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--execution-manifest')
    parser.add_argument('--warmup-selection')
    parser.add_argument('--warmup-mode', choices=['framework_ops', 'compiled'])
    parser.add_argument('--warmup-op')
    parser.add_argument('--probe-warmup-op', choices=['ArgMax', 'Max'])
    parser.add_argument('--hbm-budget-bytes', type=int)
    parser.add_argument('--cpu-budget-bytes', type=int)
    args = parser.parse_args()
    root = task_root(args.task_root)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / 'runner-observation.json').exists(), 'Never overwrite an existing attempt')
    record = {'schema_version': 'c2-observation-v1', 'status': 'STARTED', 'classification': 'DIAGNOSTIC',
              'accepted_correctness': False, 'side': 'S', 'epoch': args.epoch, 'started_at': now(),
              'C_us': None, 'V_us': None, 'C_fraction': None, 'F_executed': False,
              'runner_sha256': sha(__file__), 'source_manifest_sha256': verify_sources(root),
              'bundle_manifest_sha256': verify_bundle(root), 'samples_per_case': args.samples,
              'command_argv': sys.argv, 'groups': [], 'checks': {}}
    manifest_path = Path(args.execution_manifest) if args.execution_manifest else out / 'runner-observation.json'
    try:
        guard.gate(out, 'before-runtime', expected=set())
        import torch
        import torch_npu
        guard.verify_runtime_selection(torch, out)
        host_pid = ownership_handshake(out, args.ownership_receipt)
        record['actual_process'] = proc_identity()
        if args.probe_warmup_op:
            tensor = torch.empty(WARMUP_SHAPE, dtype=torch.float16, device='npu:0')
            require(tensor.numel() * tensor.element_size() == WARMUP_BYTES, 'Warmup tensor is not exact 256MiB')
            torch.npu.synchronize()
            selection = {'mode': 'framework_ops', 'logical_op': args.probe_warmup_op}
            record['probe_operation_invoked'] = True
            write_json(manifest_path, record)
            warmup_call(torch, tensor, selection, root)
            torch.npu.synchronize()
            record['probe_operation_passed'] = True
            guard.gate(out, 'after-probe', expected={str(host_pid)})
            record.update(status='PROBE_PASS', probe={'actual_op': args.probe_warmup_op, 'tensor_bytes': WARMUP_BYTES})
            print('OPS_WARMUP_PROBE_RESULT: ' + __import__('json').dumps({'logical_op': args.probe_warmup_op,
                'implementation': 'torch_' + args.probe_warmup_op.lower() + '_256mib_fp16_v1'}), flush=True)
            return 0
        require(args.samples == 10, '--samples must be exactly10')
        require(args.case_ids and ',' not in args.case_ids, 'One case per process/session; release all large buffers on exit')
        case = load_case(root, int(args.case_ids))
        record.update(case_id=str(case['original_index']), uid=case['uid'])
        require(args.execution_manifest and args.warmup_selection and args.warmup_mode and args.warmup_op, 'Missing application contract args')
        selection_record = load_json(args.warmup_selection)
        require(selection_record.get('status') == 'SELECTED', 'Unverified warmup selection')
        selection = selection_record['selected']
        require(selection_record['epoch'] == args.epoch and selection_record['runner_sha256'] == sha(__file__), 'Warmup probe identity differs')
        require(selection['mode'] == args.warmup_mode and selection['logical_op'] == args.warmup_op, 'Warmup selection differs')
        require(selection.get('tensor_bytes') == WARMUP_BYTES and selection.get('tensor_allocation') ==
                {'shape': list(WARMUP_SHAPE), 'dtype': 'float16', 'elements': WARMUP_BYTES // 2}, 'Warmup allocation contract differs')
        if selection['mode'] == 'compiled':
            require(selection_record['probe']['status'] != 'success' and selection_record['build']['status'] == 'success', 'Compiled warmup without actual failed ops probe and successful build')
            require(sha(selection['binary']) == selection['binary_sha256'], 'Compiled warmup identity changed')
        else:
            require(selection_record['probe']['status'] == 'success', 'Framework warmup has no real device probe')
        record['warmup'] = selection
        build_path = root / 'build-evidence/build-identity.json'
        build = load_json(build_path)
        require(build['status'] == 'PASS_CPU_BUILD_ONLY', 'No complete fresh-build evidence')
        require(build['environment']['source_manifest_sha256'] == record['source_manifest_sha256'], 'Build source changed')
        require(build['environment']['bundle_manifest_sha256'] == record['bundle_manifest_sha256'], 'Published bundle changed since build')
        require(build['environment']['soc_explicit_build_target'] == args.soc, 'Build target differs')
        require(build['environment']['torch'] == torch.__version__ and build['environment']['torch_npu'] == torch_npu.__version__ and
                build['environment']['torch_CXX11_ABI'] == bool(torch._C._GLIBCXX_USE_CXX11_ABI), 'Runtime framework/ABI differs from fresh build')
        for artifact in build['artifacts']:
            require(sha(root / artifact['path']) == artifact['sha256'], 'Fresh build artifact changed')
        record['build_identity_sha256'] = sha(build_path)
        require(selection_record['bundle_manifest_sha256'] == record['bundle_manifest_sha256'], 'Probe bundle changed')
        require(selection_record['build_identity_sha256'] == record['build_identity_sha256'], 'Probe used different build identity')
        require(args.hbm_budget_bytes and args.cpu_budget_bytes, 'Executor-reviewed actual HBM/CPU budget required')
        lower = case['memory_lower_bound_bytes']
        # This is a known allocation lower bound, explicitly not a peak promise.
        require(args.hbm_budget_bytes > lower + WARMUP_BYTES, 'Actual HBM budget below known case lower bound+warmup')
        require(args.cpu_budget_bytes > lower, 'Actual cgroup-aware CPU budget below known lower bound')
        snap = guard.read_snapshot(out, 'ownership-confirmed')
        free_hbm = (snap['HBM_total_MiB'] - snap['HBM_used_MiB']) * 1024**2
        require(args.hbm_budget_bytes <= free_hbm, 'Requested HBM budget exceeds fresh actual free HBM')
        record['memory_budget'] = {'executor_CPU_bytes': args.cpu_budget_bytes, 'executor_HBM_bytes': args.hbm_budget_bytes,
                                   'known_lower_bound_bytes': lower, 'peak_proven': False,
                                   'on_OOM': 'stop and retain attempt; never crop T/P/metadata'}
        cpu, evidence = cpu_inputs(root, case, torch)
        record['input_evidence'] = evidence
        write_json(manifest_path, record)
        inputs = {n: value.to('npu:0') if isinstance(value, torch.Tensor) else value for n, value in cpu.items()}
        del cpu
        gc.collect()
        state = inputs['state_cache']
        pristine = state.clone()
        model = load_module(root / 'source/operators/compressor_c2_a3_bf16_c_only/model.py', 'd19_c2_current_S')
        path = model.prepare(*(inputs[name] for name in INPUT_NAMES), soc=args.soc)
        record['loaded_DSOs'] = verify_loaded(root, build, out)
        record['actual_route'] = {'mode': path.mode, 'chunk': path.chunk, 'legacy_route': path.route, 'tiling': path.info,
                                  'C_layout': 'ND forced layout=0', 'actual_soc_strict_prepare_pass': args.soc}
        # Historical capture recipe: two eager calls, capture, poisoned replay.
        for _ in range(2):
            state.copy_(pristine)
            output = path()
        torch.npu.synchronize()
        anchor = output_state_digest(torch, output, state)
        state.copy_(pristine)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = path()
        torch.npu.synchronize()

        def reset():
            # All are before warmup. Packed weights and C pair must be rewritten
            # by the complete target, not retained from preparation.
            state.copy_(pristine)
            output.fill_(float('nan'))
            path.w.fill_(float('nan'))
            path.pair.fill_(float('nan'))

        reset()
        graph.replay()
        torch.npu.synchronize()
        require(output_state_digest(torch, output, state) == anchor, 'Graph replay output/state differs from eager')
        record['checks']['graph_eager_output_state_match'] = True
        record['anchor'] = anchor
        warmup_tensor = torch.empty(WARMUP_SHAPE, dtype=torch.float16, device='npu:0')
        require(warmup_tensor.numel() * warmup_tensor.element_size() == WARMUP_BYTES, 'Warmup tensor size differs')
        torch.npu.synchronize()
        for group in range(10):
            guard.gate(out, f'before-group-{group:02d}', expected={str(host_pid)})
            reset()
            torch.npu.synchronize()
            warmup_call(torch, warmup_tensor, selection, root)
            torch.npu.synchronize()
            graph.replay()  # Complete per-call cat + new BF16 C + reused V.
            torch.npu.synchronize()
            record['groups'].append({'group': group, 'reset_before_warmup': True, 'warmup_then_complete_S': True,
                                     'same_stream': True, 'C_us': None, 'V_us': None, 'C_fraction': None})
            write_json(manifest_path, record)
        require(output_state_digest(torch, output, state) == anchor, 'Final replay output/state mismatch')
        record['checks']['final_replay_output_state_match'] = True
        after = {n: tensor_hash(inputs[n]) for n in IMMUTABLE}
        require(after == evidence['hashes'], 'Immutable inputs changed on device')
        record['checks']['immutable_inputs_after_match'] = True
        record['immutable_after'] = after
        guard.gate(out, 'after-groups', expected={str(host_pid)})
        record['status'] = 'OBSERVED_10_GROUPS_DIAGNOSTIC'
        record['mapping_status'] = 'PENDING_REVIEWED_ACTUAL_TASK_TOPOLOGY'
        return 0
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['finished_at'] = now()
        write_json(manifest_path, record)


if __name__ == '__main__':
    raise SystemExit(main())
