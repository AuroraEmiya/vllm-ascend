"""One case / one external msprof application session, no automatic retries.

Uses the original official profiler flags and warmup identity; every failure
retains raw data. Observation is not classification or correctness acceptance.
"""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys
from common import SOC, load_case, load_json, now, require, sha, task_root, verify_bundle, verify_sources, write_json
from fixture import load_module


def validate_pilot_approval(path, epoch, build_sha, *, root=None, warmup_selection_path=None):
    """Revalidate four concrete evidence chains against the LIVE task build.

    Approval/result headers are not authorities for their underlying evidence.
    Rehash raw/topology/execution, load their current bytes, then run the actual
    all-row mapper again and compare every reported group/metric/task record.
    """
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    require(warmup_selection_path is not None, 'Current warmup selection is required for pilot validation')
    build_path = root / 'build-evidence/build-identity.json'
    require(build_path.is_file() and sha(build_path) == build_sha, 'Actual current build identity missing or changed')
    build = load_json(build_path)
    require(build.get('status') == 'PASS_CPU_BUILD_ONLY', 'Actual current fresh build incomplete')
    current = {'epoch': epoch, 'build_identity_sha256': build_sha,
               'source_manifest_sha256': verify_sources(root), 'bundle_manifest_sha256': verify_bundle(root),
               'runner_sha256': sha(root / 'runner.py')}
    for key in ('source_manifest_sha256', 'bundle_manifest_sha256'):
        require(build['environment'].get(key) == current[key], 'Current build was made from different source/bundle')
    require(build['environment'].get('soc_explicit_build_target') == SOC, 'Current build target is not the reviewed9362')
    selection = load_json(warmup_selection_path)
    require(selection.get('status') == 'SELECTED', 'Actual current warmup selection not verified')
    require(all(selection.get(k) == v for k, v in current.items()), 'Warmup evidence is stale for current source/build/bundle/runner/epoch')
    approval = load_json(path)
    require(approval.get('status') == 'APPROVED_FOR_REMAINING_S' and approval.get('reviewer'), 'Independent four-route pilot approval missing')
    require(approval.get('epoch') == epoch and approval.get('build_identity_sha256') == build_sha, 'Pilot epoch/build differs')
    require(isinstance(approval.get('pilot_results'), list) and len(approval['pilot_results']) == 4, 'Exactly four pilot results required')
    required = {0: (0, 8), 14: (1, 8), 52: (0, 16), 56: (1, 16)}
    found, verified = set(), []
    from task_mapping import evidence_sha, map_rows, read_csv

    def verify_ref(ref, label):
        require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and
                re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', ''))) is not None, 'Malformed ' + label + ' reference')
        actual = Path(ref['path'])
        require(actual.is_file(), 'Missing underlying ' + label)
        require(sha(actual) == ref['sha256'], 'Changed underlying ' + label)
        return actual

    expected_DSOs = {}
    for rel in ('source/dependencies/original_a3/n_impl/build/libcv60n.so',
                'source/operators/compressor_c2_a3_bf16_c_only/build/libc2_a3_bf16_c_only.so'):
        artifact = [r for r in build['artifacts'] if r['path'] == rel]
        require(len(artifact) == 1, 'Current build DSO missing or duplicate: ' + rel)
        verify_ref({'path': str(root / rel), 'sha256': artifact[0]['sha256']}, 'current DSO')
        expected_DSOs[str(root / rel)] = artifact[0]['sha256']

    for item in approval['pilot_results']:
        result_path = verify_ref(item, 'pilot mapping')
        result = load_json(result_path)
        require(result['status'] == 'MAPPED_DIAGNOSTIC' and result['epoch'] == epoch, 'Pilot map not accepted')
        index = int(result['case_id'])
        require(str(index) == result['case_id'] and index in required and index not in found, 'Missing, foreign or duplicate pilot case')
        case = load_case(root, index)
        require(result.get('uid') == case['uid'], 'Pilot result UID differs from frozen case')
        require(isinstance(result.get('inputs'), dict) and set(result['inputs']) == {'csv', 'topology', 'execution_manifest'}, 'Pilot must reference actual raw/topology/execution evidence')
        paths = {key: verify_ref(ref, key) for key, ref in result['inputs'].items()}
        exe, topology = load_json(paths['execution_manifest']), load_json(paths['topology'])
        require(all(exe.get(k) == v for k, v in current.items()), 'Pilot execution is stale for current source/build/bundle/runner/epoch')
        require(exe.get('case_id') == str(index) and exe.get('uid') == case['uid'], 'Pilot execution UID/index differs')
        require(exe.get('classification') == 'DIAGNOSTIC' and exe.get('accepted_correctness') is False, 'Pilot cannot upgrade correctness')
        inputs = exe.get('input_evidence', {})
        require(inputs.get('hashes') == case['historical_immutable_input_hashes'] and inputs.get('historical_immutable_input_hashes_match') is True, 'Pilot actual five hashes differ or are missing')
        require(re.fullmatch(r'[0-9a-f]{64}', str(inputs.get('new_S0_sha256', ''))) is not None and inputs.get('historical_S0_sha256') is None, 'New S0 digest missing or historical S0 falsely claimed')
        for name, shape in case['shape'].items():
            actual = inputs.get('tensors', {}).get(name, {})
            require(actual.get('shape') == shape and actual.get('dtype') == case['dtypes'][name], 'Pilot actual input tensor contract differs: ' + name)
        require(exe.get('warmup') == selection['selected'], 'Pilot actual warmup differs from current selected implementation')
        require(exe['actual_route'].get('actual_soc_strict_prepare_pass') == SOC and
                (exe['actual_route'].get('mode'), exe['actual_route'].get('chunk')) == required[index], 'Pilot did not prove its actual four-route9362 coverage')
        loaded = exe.get('loaded_DSOs', [])
        require(len(loaded) == 2 and {r.get('path'): r.get('sha256') for r in loaded} == expected_DSOs, 'Pilot actual loaded DSOs differ from current fresh build')
        require(topology.get('observed_csv_sha256') == result['inputs']['csv']['sha256'], 'Topology does not bind actual raw CSV')
        _, rows = read_csv(paths['csv'])
        evidence_sha.cache_clear()  # do not reuse a prior validation's file hash
        rerun = map_rows(rows, topology, exe)
        for key, value in rerun.items():
            require(result.get(key) == value, 'Pilot claimed mapping/group/metric differs from underlying recomputation: ' + key)
        found.add(index)
        verified.append({'case_id': str(index), 'uid': case['uid'], 'mapping_sha256': item['sha256'],
                         'raw_sha256': result['inputs']['csv']['sha256'],
                         'topology_sha256': result['inputs']['topology']['sha256'],
                         'execution_sha256': result['inputs']['execution_manifest']['sha256']})
    require(found == set(required), 'Required four pilot cases not complete')
    return {'status': 'REVALIDATED_CURRENT_FOUR_PILOTS', 'current_identity': current,
            'approval_sha256': sha(path), 'pilots': verified}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-root', required=True)
    p.add_argument('--case-id', required=True, type=int)
    p.add_argument('--epoch', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--ownership-receipt', required=True)
    p.add_argument('--warmup-selection', required=True)
    p.add_argument('--hbm-budget-bytes', required=True, type=int)
    p.add_argument('--cpu-budget-bytes', required=True, type=int)
    p.add_argument('--stage', required=True, choices=['pilot', 'remaining'])
    p.add_argument('--pilot-approval')
    p.add_argument('--msprof-bin', default='msprof')
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    root = task_root(a.task_root)
    verify_sources(root)
    case = load_case(root, a.case_id)
    build_sha = sha(root / 'build-evidence/build-identity.json')
    pilot_validation = None
    if a.stage == 'pilot':
        require(a.case_id in (0, 14, 52, 56), 'Pilot is restricted to original cases0/14/52/56')
    else:
        require(a.pilot_approval, 'Remaining cases require four-route pilot review first')
        pilot_validation = validate_pilot_approval(a.pilot_approval, a.epoch, build_sha,
                                                   root=root, warmup_selection_path=a.warmup_selection)
    selection = load_json(a.warmup_selection)
    require(selection.get('status') == 'SELECTED', 'Warmup selection not verified')
    require(selection['build_identity_sha256'] == build_sha and selection['epoch'] == a.epoch, 'Warmup build/epoch differs')
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    warm = selection['selected']
    runner = [sys.executable, str(root / 'runner.py'), '--task-root', str(root), '--out-dir', str(out / 'runner'),
              '--ownership-receipt', str(Path(a.ownership_receipt).resolve()), '--epoch', a.epoch, '--soc', SOC,
              '--case-ids', str(a.case_id), '--samples', '10', '--execution-manifest', str(out / 'execution-observation.json'),
              '--warmup-selection', str(Path(a.warmup_selection).resolve()), '--warmup-mode', warm['mode'], '--warmup-op', warm['logical_op'],
              '--hbm-budget-bytes', str(a.hbm_budget_bytes), '--cpu-budget-bytes', str(a.cpu_budget_bytes)]
    application = shlex.join(runner)
    require(len(application) < 2048, 'Official --application length limit exceeded')
    argv = [a.msprof_bin, '--application=' + application, '--output=' + str(out / 'raw'),
            '--aic-mode=task-based', '--aic-metrics=PipeUtilization']
    record = {'status': 'STARTED', 'side': 'S', 'uid': case['uid'], 'case_id': str(a.case_id), 'epoch': a.epoch,
              'classification': 'DIAGNOSTIC', 'accepted_correctness': False, 'samples': 10,
              'argv': argv, 'application_argv': runner, 'started_at': now(), 'returncode': None,
              'timed_out': False, 'C_us': None, 'V_us': None, 'C_fraction': None,
              'source_manifest_sha256': sha(root / 'source-manifest.json'), 'build_identity_sha256': build_sha,
              'warmup_selection_sha256': sha(a.warmup_selection), 'bundle_manifest_sha256': sha(root / 'publish-manifest.json'),
              'automatic_retry': False, 'max_whole_session_rows': 540,
              'aic_mode': 'task-based', 'aic_mode_rationale': 'same official application collector CANN adaptation'}
    record['pilot_approval_revalidation'] = pilot_validation
    write_json(out / 'collection-manifest.json', record)
    try:
        official = load_module(root / 'official/collect_perf_application.py', 'd19_official_collector')
        record['cann_environment'] = official.inspect_cann_environment()
        require(str(record['cann_environment'].get('version', '')).startswith('9.2'), 'Actual CANN version is not9.2')
        with (out / 'msprof.log').open('wb') as log:
            completed = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout, check=False)
        record['returncode'] = completed.returncode
        require(completed.returncode == 0, 'msprof/application failed; preserve raw data and do not retry/delete')
        observed = load_json(out / 'execution-observation.json')
        require(observed['status'] == 'OBSERVED_10_GROUPS_DIAGNOSTIC', 'Application did not complete all10 groups and checks')
        csv_path = official.find_op_summary(out / 'raw')
        from task_mapping import read_csv
        header, rows = read_csv(csv_path)
        require(len(rows) <= 540, 'Whole observed session exceeds540; retain as invalid session')
        record.update(status='OBSERVED_PENDING_INDEPENDENT_TASK_MAPPING', op_summary={'path': str(csv_path), 'sha256': sha(csv_path), 'header': header, 'rows': len(rows)})
        write_json(out / 'ordered-task-observation.json', {'status': 'UNCLASSIFIED', 'raw_csv_sha256': sha(csv_path), 'rows': rows})
    except subprocess.TimeoutExpired as error:
        record.update(status='FAILED_TIMEOUT', timed_out=True, error=repr(error),
                      cleanup='Outer executor must verify and clean only owned descendant processes before any retry')
        raise
    except BaseException as error:
        record.update(status='FAILED', error=repr(error))
        raise
    finally:
        record['finished_at'] = now()
        record['msprof_log_sha256'] = sha(out / 'msprof.log') if (out / 'msprof.log').is_file() else None
        write_json(out / 'collection-manifest.json', record)


if __name__ == '__main__':
    main()
