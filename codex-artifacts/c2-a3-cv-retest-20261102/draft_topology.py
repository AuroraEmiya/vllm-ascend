"""Create an UNACCEPTED map draft from exact observed task rows.

Requires explicit reviewed sequence boundaries, not guesses from C/V names.
`sequence-spec.json`: {reset:[...], warmup:[...], target:[...]}; each task has
exact op_name/op_type/task_type + category/compute_kind/stage/evidence.
Prefix/suffix are copied as setup for reviewer inspection. No DRAFT is accepted.
"""
import argparse
from pathlib import Path
from common import PLAN_SHA, load_json, require, sha, write_json
from task_mapping import read_csv, signature


def draft(rows, seq, execution):
    warm = seq['warmup']
    require(warm and seq['target'], 'Explicit nonempty sequences required')
    starts = [i for i in range(len(rows) - len(warm) + 1)
              if [signature(r) for r in rows[i:i + len(warm)]] == [signature(r) for r in warm]]
    require(len(starts) == 10, 'Must observe exactly 10 unique complete warmup sequences')
    reset_n = len(seq.get('reset', []))
    target_n = len(seq['target'])
    prefix_n = starts[0] - reset_n
    require(prefix_n >= 0, 'Reset boundary precedes trace')
    cursor = prefix_n
    for start in starts:
        expected = seq.get('reset', []) + warm + seq['target']
        require(cursor + reset_n == start, 'Extra/missing inter-group task; inspect raw')
        require([signature(r) for r in rows[cursor:cursor + len(expected)]] == [signature(r) for r in expected], 'Full reset/warmup/target differs')
        cursor += len(expected)

    def setup(items):
        return [dict(signature(row), category='setup', compute_kind=None,
                     evidence=['REVIEW_REQUIRED: prove setup membership and actual compute kind; raw record ' + str(row['source_record'])]) for row in items]

    out = {key: execution.get(key) for key in ('source_manifest_sha256', 'build_identity_sha256', 'bundle_manifest_sha256',
                                               'runner_sha256', 'epoch', 'warmup', 'case_id', 'uid')}
    out.update(schema_version='c2-setup-aware-v1', status='DRAFT', reviewer=None, review_evidence=None,
               plan_sha256=PLAN_SHA, route={key: execution['actual_route'][key] for key in ('mode', 'chunk')},
               prefix=setup(rows[:prefix_n]), group=seq, suffix=setup(rows[cursor:]),
               observed_rows=len(rows), full_csv_mapping_required=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--execution-manifest', required=True)
    p.add_argument('--sequence-spec', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    require(not Path(a.output).exists(), 'Never overwrite map')
    _, rows = read_csv(a.csv)
    result = draft(rows, load_json(a.sequence_spec), load_json(a.execution_manifest))
    result['observed_csv_sha256'] = sha(a.csv)
    write_json(a.output, result)


if __name__ == '__main__':
    main()
