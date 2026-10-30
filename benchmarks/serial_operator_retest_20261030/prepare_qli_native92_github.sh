#!/usr/bin/env bash
# Run inside this task's device-free CANN 9.2 build container.
set -euo pipefail
cd /home/s30083169/perf-serial-graph
test -z "$(find /dev -maxdepth 1 -name 'davinci[0-9]*' -print -quit)"
out=outputs/qli-old-retest-20261030/serial-build-native92-001
mkdir "$out"
exec > >(tee "$out/build.log") 2>&1
mkdir -p inputs
curl --fail --location --connect-timeout 8 --max-time 60 \
  --proxy http://141.1.186.255:3128 --cacert host-ca-bundle.crt \
  --output inputs/a5-test-sources-20261027.zip \
  https://raw.githubusercontent.com/AuroraEmiya/vllm-ascend/48d34102c3605041bf6a1d7a7f93474aa64f515c/benchmarks/serial_operator_retest_20261027/a5-test-sources-20261027.zip
echo 'b89b843ea181c0c7ac7260b3d3a875a0845fbcbb94d992477a2f9d5a72df4749  inputs/a5-test-sources-20261027.zip' | sha256sum -c -
python3 - <<'PY'
from pathlib import Path
import hashlib,json,zipfile
root=Path.cwd()
changes=[]
records=[]
with zipfile.ZipFile('inputs/a5-test-sources-20261027.zip') as z:
    for row in json.loads(z.read('SOURCE_MANIFEST.json'))['files']:
        rel=Path(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        if not row['path'].startswith('src/qli_a5/'):
            continue
        raw=z.read(row['path'])
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        for dest in (root/rel,root/'inputs/pristine'/rel):
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                assert dest.read_bytes()==raw, f'Existing file differs: {dest}'
            else:
                dest.write_bytes(raw)
        records.append(row)
        if rel.suffix=='.sh' or rel.name=='CMakeLists.txt':
            normalized=raw.replace(b'\r\n',b'\n')
            if normalized!=raw:
                (root/rel).write_bytes(normalized)
                changes.append({'path':row['path'],'reason':'CRLF normalization in build script only',
                                'before_sha256':row['sha256'],'after_sha256':hashlib.sha256(normalized).hexdigest()})
out=root/'outputs/qli-old-retest-20261030/serial-build-native92-001'
(out/'source-download-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'build-adaptations.json').write_text(json.dumps(changes,indent=2)+'\n')
print('QLI_SOURCE_DOWNLOAD_VERIFIED',len(records),'BUILD_ONLY_NORMALIZATIONS',len(changes))
PY
set +u
source /usr/local/Ascend/cann/set_env.sh
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1
set +e
timeout --signal=TERM --kill-after=15 600 bash src/qli_a5/ascendc_quant_lightning_indexer/build.sh
status=$?
set -e
printf '%s\n' "$status" > "$out/build.exit"
find src/qli_a5 -name '*.so' -type f -exec sha256sum {} \; > "$out/binary-sha256.txt"
printf 'SERIAL_BUILD_EXIT %s\n' "$status"
exit "$status"
