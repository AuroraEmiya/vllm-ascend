"""Shared evidence helpers. Standard-library only; importing never starts NPU work."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

PLAN_SHA = '25775fe0785dd902117b3d1958d70e465cb9f93f7df160d7e186cd9d83110731'
SOC = 'Ascend910_9362'


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def task_root(value):
    root = Path(value).resolve()
    require(root == Path(__file__).resolve().parent, '--task-root must identify this isolated bundle directory')
    return root


def verify_sources(root):
    manifest = load_json(root / 'source-manifest.json')
    require(load_json(root / 'frozen/case-config.json')['authority_plan_sha256'] == PLAN_SHA, 'Sole frozen plan identity changed')
    for row in manifest['files']:
        require(sha(root / row['path']) == row['sha256'], 'Source changed: ' + row['path'])
    return sha(root / 'source-manifest.json')


def verify_bundle(root):
    manifest = load_json(root / 'publish-manifest.json')
    require(manifest.get('scope') == 'D19_C2_A3_S60_CONFIG_SOURCE_ONLY', 'Wrong publish scope')
    for row in manifest['files']:
        require(sha(root / row['path']) == row['sha256'], 'Published bundle byte changed: ' + row['path'])
    return sha(root / 'publish-manifest.json')


def load_case(root, index):
    require(type(index) is int and 0 <= index < 60, 'Expected a full60 original index 0..59')
    plan = load_json(root / 'frozen/case-config.json')
    require(len(plan['cases']) == 60, 'Plan does not contain exactly 60 cases')
    cases = [c for c in plan['cases'] if c['original_index'] == index]
    require(len(cases) == 1, 'Ambiguous case index')
    return cases[0]
