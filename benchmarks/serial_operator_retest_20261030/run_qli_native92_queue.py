"""Build the published serial snapshot, then collect all 12 adjacent QLI pairs."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path.cwd()
OUT=ROOT/'outputs/qli-old-retest-20261030/paired-114-native92-001'
assert not OUT.exists(), 'Never overwrite an earlier queue'
OUT.mkdir(parents=True)
STATUS=OUT/'status.json'
state={'host':'141.61.45.114','physical_device':7,'cann_mode':'native92',
       'state':'starting','started_unix':time.time(),'expected_runs':24,'runs':[],
       'precision_status':'DEFERRED_BY_USER','frequency_gate':'UNVERIFIED'}

def save():
    tmp=STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(STATUS)

def bounded(command, logpath, seconds):
    with logpath.open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            return 124

try:
    state['state']='verifying_serial_snapshot'
    save()
    archive=ROOT/'inputs/a5-test-sources-20261027.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='b89b843ea181c0c7ac7260b3d3a875a0845fbcbb94d992477a2f9d5a72df4749'
    source_rows=[]
    with zipfile.ZipFile(archive) as z:
        for record in json.loads(z.read('SOURCE_MANIFEST.json'))['files']:
            rel=Path(record['path'])
            assert not rel.is_absolute() and '..' not in rel.parts
            if not record['path'].startswith('src/qli_a5/'): continue
            raw=z.read(record['path'])
            assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
            target=ROOT/rel
            before=target.read_bytes() if target.is_file() else None
            normalized=raw.replace(b'\r\n',b'\n') if rel.suffix=='.sh' or rel.name=='CMakeLists.txt' else raw
            if before is not None and before!=normalized:
                backup=OUT/'source-before'/rel
                backup.parent.mkdir(parents=True,exist_ok=True)
                backup.write_bytes(before)
            target.parent.mkdir(parents=True,exist_ok=True)
            # Touch all canonical sources so existing build objects cannot be reused silently.
            target.write_bytes(normalized)
            source_rows.append(dict(record,actual_sha256=hashlib.sha256(normalized).hexdigest(),
                                    before_sha256=None if before is None else hashlib.sha256(before).hexdigest(),
                                    adaptation='build-file CRLF normalization' if normalized!=raw else None))
    assert len(source_rows)==67
    (OUT/'serial-source-identity.json').write_text(json.dumps(source_rows,indent=2)+'\n')
    package=ROOT/'outputs/qli-old-retest-20261030/package/qli_old_dsl_baseline_20261029'
    assert json.loads((ROOT/'src/qli_a5/benchmark/case_manifest.json').read_text())==json.loads((package/'benchmark/case_manifest.json').read_text())
    state['state']='building_serial'
    save()
    rc=bounded(['bash','src/qli_a5/ascendc_quant_lightning_indexer/build.sh'],OUT/'serial-build.log',600)
    state['build_exit']=rc
    binaries={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/qli_a5').rglob('*.so')}
    state['serial_binaries']=binaries
    save()
    assert rc==0 and len(binaries)==5, 'Serial build failed'
    runner=ROOT/'paired_qli_old_measure.py'
    assert hashlib.sha256(runner.read_bytes()).hexdigest()=='715406f6d68a6728c56edfea1e8a4383bf011565567cfb9d137c967b75ddacbc'
    for case in range(12):
        for implementation in ('old-dsl','serial'):
            out=OUT/f'{implementation}-case{case:02d}'
            state.update(state='measuring',current_case=case,current_implementation=implementation,current_output=str(out))
            save()
            args=[sys.executable,str(runner),'--baseline-root',str(package),'--implementation',implementation,
                  '--cann-mode','native92','--case',str(case),'--device','0','--physical-device','7','--smi-device','7','--out',str(out)]
            rc=bounded(args,OUT/f'{implementation}-case{case:02d}.log',900)
            execution=out/'execution.json'
            row=json.loads(execution.read_text()) if execution.is_file() else {}
            measurement=row.get('measurement_summary',{})
            result={'case':case,'implementation':implementation,'exit_code':rc,'completed':row.get('completed',False),
                    'execution':str(execution),'input_identity_sha256':row.get('input_identity_sha256'),
                    'total_ms':measurement.get('total_ms'),'C_fraction':measurement.get('C_fraction'),
                    'spread_over_median':measurement.get('spread_over_median'),'stability_passed':measurement.get('stability_passed')}
            if row.get('error'): result['error_final_line']=row['error'].strip().splitlines()[-1]
            state['runs'].append(result)
            save()
            print(json.dumps(result),flush=True)
            assert rc==0 and result['completed'], 'Measurement stopped; preserve failed attempt for diagnosis'
        old,serial=state['runs'][-2:]
        assert old['input_identity_sha256']==serial['input_identity_sha256'], 'Pair input identity mismatch'
    state.update(state='complete',finished_unix=time.time(),paired_cases=12)
except Exception:
    state.update(state='needs_attention',error=traceback.format_exc(),finished_unix=time.time())
    print(state['error'],flush=True)
finally:
    save()
sys.exit(0 if state['state']=='complete' else 2)
