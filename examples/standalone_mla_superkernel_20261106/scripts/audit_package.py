# SPDX-License-Identifier: Apache-2.0
"""Offline hash/path/syntax audit of the extracted delivery package."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
assert manifest['format']==1
for item in manifest['files']:
    relative=Path(item['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    p=(root/relative).resolve()
    assert p.is_relative_to(root) and p.is_file(),relative
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],relative
    assert p.stat().st_size==item['bytes'],relative
    if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8'),filename=str(relative))
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
expected={x['path'] for x in manifest['files']}
assert actual==expected,dict(extra=sorted(actual-expected),missing=sorted(expected-actual))
print(json.dumps(dict(status='PASS',files=len(expected),source_sha=manifest['source_local_commit']),indent=2))
