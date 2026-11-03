"""Reuse official selection helpers; preserve actual probe/build failure logs.

Resource/ownership/setup failure is never reinterpreted as missing ops. The
probe command must run this bundle runner's real 256MiB probe entry point.
"""
import argparse
import os
from pathlib import Path
import shlex
import sys
from common import load_json, now, require, sha, task_root, verify_sources, write_json
from fixture import load_module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-root', required=True)
    p.add_argument('--epoch', required=True)
    p.add_argument('--probe-command', required=True, help='Exact runner probe command containing {warmup_op}')
    p.add_argument('--probe-observation', required=True, help='runner-observation.json from that exact probe process')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--probe-timeout', type=int, default=600)
    p.add_argument('--build-timeout', type=int, default=900)
    a = p.parse_args()
    root = task_root(a.task_root)
    verify_sources(root)
    require('{warmup_op}' in a.probe_command, 'Missing actual op placeholder')
    require(str(root / 'runner.py') in a.probe_command, 'Probe must use current runner entry point')
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    official = load_module(root / 'official/collect_perf_application.py', 'd19_official_collector')
    build_path = root / 'build-evidence/build-identity.json'
    require(load_json(build_path)['status'] == 'PASS_CPU_BUILD_ONLY', 'Fresh build evidence missing')
    record = {'status': 'STARTED', 'epoch': a.epoch, 'created_at': now(), 'runner_sha256': sha(root / 'runner.py'),
              'build_identity_sha256': sha(build_path), 'source_manifest_sha256': sha(root / 'source-manifest.json'),
              'bundle_manifest_sha256': sha(root / 'publish-manifest.json'), 'selected': None}
    try:
        record['cann_environment'] = official.inspect_cann_environment()
        require(str(record['cann_environment'].get('version', '')).startswith('9.2'), 'Actual resolved CANN version is not9.2')
        probe = official.run_ops_warmup_probe(a.probe_command, 'CompressorC2A3CompleteS', out / 'ops-probe.log', a.probe_timeout)
        record['probe'] = probe
        actual = load_json(a.probe_observation)
        require(actual.get('epoch') == a.epoch and actual.get('runner_sha256') == record['runner_sha256'], 'Probe identity differs')
        if probe['status'] == 'success':
            require(actual.get('status') == 'PROBE_PASS' and actual.get('probe_operation_passed') is True, 'Process exit0 is not an actual warmup probe PASS')
            build = {'status': 'not_needed', 'reason': 'ops_warmup_probe_succeeded'}
        else:
            require(actual.get('probe_operation_invoked') is True and not actual.get('probe_operation_passed'),
                    'Probe failed before actual op or after passing op: resource/setup gate failure, no compiled fallback')
            script = root / 'official/compiled_warmup/build.sh'
            command = ' '.join(['env', 'PYTHON=' + shlex.quote(sys.executable), 'bash', shlex.quote(str(script)), '--soc=ascend910_93'])
            build = official.run_warmup_build(command, out / 'compiled-build.log', a.build_timeout)
        record['build'] = build
        selected = official.select_warmup('CompressorC2A3CompleteS', probe, build)
        selected = official.attach_compiled_warmup_binary(selected, root / 'official/compiled_warmup/ops_profiling_warmup/_C.so')
        record.update(status='SELECTED', selected=selected,
                      probe_observation={'path': str(Path(a.probe_observation).resolve()), 'sha256': sha(a.probe_observation)})
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['finished_at'] = now()
        write_json(out / 'warmup-selection.json', record)


if __name__ == '__main__':
    main()
