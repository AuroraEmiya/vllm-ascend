"""Bind each diagnostic replay to its actual specialization and compiled binary."""
import base64
import hashlib
import json
import shutil
from pathlib import Path

import torch
import next_benchmark as bench

out = Path('results/optimization/next/selected-ir-row-weights')
out.mkdir(exist_ok=True)
records = []
torch.npu.set_device(0)
for key in ('o144', 'c144', 'c114'):
    for impl in ('baseline', 'candidate'):
        c = bench.make_case(key + ':' + impl)
        bench.invoke(c)
        torch.npu.synchronize()
        assert bench.check(c)['passed']
        binary = c['compiled'][0]
        meta = dict(vars(binary.metadata)) if hasattr(binary.metadata, '__dict__') else binary.metadata._asdict()
        cache_key = base64.b32encode(bytes.fromhex(meta['hash'])).decode().rstrip('=')
        folder = Path('/work/.triton-cache') / cache_key
        target = out / (key + '_' + impl)
        target.mkdir(exist_ok=True)
        files = {}
        for f in folder.iterdir():
            if f.is_file():
                shutil.copy2(f, target / f.name)
                files[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        record = dict(case_id=key, impl=impl, spec=c['spec'], cache_key=cache_key,
                      files_sha256=files, metadata=meta)
        records.append(record)
        print(json.dumps(dict(case_id=key, impl=impl, cache_key=cache_key)), flush=True)
(out / 'selection.json').write_text(json.dumps(records, indent=2, default=str))
