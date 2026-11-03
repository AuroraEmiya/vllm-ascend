"""Freeze explicit source/config-only publish allowlist. Never publishes anything."""
from pathlib import Path
from common import sha, write_json

ROOT = Path(__file__).resolve().parent
ROOT_FILES = [
    'README.md', 'NOTICE.md', 'LICENSE.CANN', 'SOC-ADAPTATION.diff', 'source-manifest.json',
    'IMPLEMENTATION_REVIEW.json', 'common.py', 'fixture.py', 'build.py', 'runner.py',
    'prepare_warmup.py', 'collect.py', 'task_mapping.py', 'draft_topology.py',
    'resource_guard_a3.py', 'test_candidate.py', 'freeze_candidate.py', 'summarize.py',
]


def main():
    paths = [ROOT / p for p in ROOT_FILES]
    for directory in ('source', 'frozen', 'official', 'provenance'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    records = []
    for path in sorted(set(paths)):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('Missing or linked whitelist entry: ' + str(path))
        if path.suffix.lower() in ('.so', '.o', '.a', '.csv', '.npz', '.tar', '.gz', '.zip', '.pyc') or any(p in ('build', 'results', 'raw', 'PROF') for p in path.relative_to(ROOT).parts):
            raise RuntimeError('Non-source artifact cannot be published: ' + str(path))
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    result = {'schema_version': 1, 'revision': 2, 'scope': 'D19_C2_A3_S60_CONFIG_SOURCE_ONLY',
              'status': 'LOCAL_REVIEW_CANDIDATE_NOT_PUBLISHED_NOT_EXECUTED',
              'supersedes_publish_manifest_sha256': 'fe91fc73dee929d5f17cc528c27d26b588b329e1276d0a22655069c72fb4d23d',
              'source_archive_sha256': '73a40269854a781ffca0b9e7a27ede0d91897984aae93b3ca988844e765fdec6',
              'authority_plan_sha256': '25775fe0785dd902117b3d1958d70e465cb9f93f7df160d7e186cd9d83110731',
              'files': records, 'manifest_self_in_allowlist': True,
              'exclude_all_other_files': True,
              'excluded': ['historical all.json/cases.csv/latency/speedup/environment and review logs',
                           'DSO, objects, generated build outputs, new profile/runtime evidence',
                           'credentials/SSH details, materializer, local validation logs and source audit helper'],
              'commands': 'root reviews; root alone publishes this exact allowlist plus publish-manifest.json'}
    write_json(ROOT / 'publish-manifest.json', result)
    (ROOT / 'PUBLISH_ALLOWLIST.txt').write_text('\n'.join(r['path'] for r in records) + '\npublish-manifest.json\n', encoding='utf-8')
    print(__import__('json').dumps({'files_including_manifest': len(records) + 1, 'manifest_sha256': sha(ROOT / 'publish-manifest.json')}))


if __name__ == '__main__':
    main()
