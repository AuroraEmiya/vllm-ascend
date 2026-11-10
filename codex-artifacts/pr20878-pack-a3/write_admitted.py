"""Build profiling CSV from explicit precision admissions, retaining excluded rows in JSON."""
from pathlib import Path
import argparse,csv,json
from general_benchmark import SPECS
p=argparse.ArgumentParser();p.add_argument('--validation',required=True);p.add_argument('--prefix',required=True);a=p.parse_args()
v=json.loads(Path(a.validation).read_text());groups={'sort':[],'dcp':[]};excluded=[]
for key,s in SPECS.items():
    records=[r for r in v['cases'] if r['case_id']==key]
    if records and all(r['passed'] for r in records) and {r['impl'] for r in records}=={'baseline','candidate'}:
        groups[s['kind']].append(key)
    else:excluded.append(dict(case_id=key,reason='precision or compilation not admitted',records=records))
for kind,keys in groups.items():
    with Path(a.prefix+'-'+kind+'.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','attrs'])
        for key in keys:
            for impl in ('baseline','candidate'):w.writerow([key+'_'+impl,'id='+key+':'+impl])
Path(a.prefix+'-admissions.json').write_text(json.dumps(dict(admitted=groups,excluded=excluded),indent=2))
print('ADMITTED', {k:len(v) for k,v in groups.items()},'EXCLUDED',len(excluded))
