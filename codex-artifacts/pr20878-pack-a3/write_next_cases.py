import csv
import json
from pathlib import Path
from next_benchmark import SPECS

p = Path('results/optimization/next')
v = json.loads((p / 'validation.json').read_text())
admitted = []
for key in SPECS:
    records = [r for r in v['cases'] if r['case_id'] == key]
    if len(records) == 4 and all(r['passed'] for r in records): admitted.append(key)
for name, keys in [('all', admitted), ('actual', [k for k in admitted if k in ('o144','o152','c144','c152','c114')])]:
    with (p / (name + '.csv')).open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case_id', 'attrs'])
        for key in keys:
            for impl in ('baseline', 'candidate'): w.writerow([key + '_' + impl, 'id=' + key + ':' + impl])
identity = json.loads(Path('results/perf-identity.json').read_text())
identity.update(source_sha256=v['source_sha256'], binary_sha256=list(v['binaries'].values()), comparison='Frozen owner/combine versus first parallel/cache candidates')
(p / 'identity.json').write_text(json.dumps(identity, indent=2))
print('ADMITTED', admitted)
