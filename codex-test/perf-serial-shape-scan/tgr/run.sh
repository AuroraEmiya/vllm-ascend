set -euo pipefail
ulimit -c 0
root="$1"
id="$2"
h="$root/tgr/split-s21"
t="$root/tgr"
base="$t/baseline/vendors/custom_transformer"
src="$t/checkout/vllm_ascend"
cand="$src/_cann_ops_custom/vendors/serial_transformer"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LOCAL_WORLD_SIZE=1
export PYTHONPATH="$t/pydeps"
export TORCH_EXTENSIONS_DIR="$t/torch-extensions/split-s21" MAX_JOBS=4
export LD_LIBRARY_PATH="$src:$base/op_api/lib:$cand/op_api/lib:$LD_LIBRARY_PATH"
export TGR_CASE="$h/$id.json"
out="$root/results/$id"
mkdir -p "$out"
for route in baseline candidate; do
  if test "$route" = baseline; then export ASCEND_CUSTOM_OPP_PATH="$base"; else export ASCEND_CUSTOM_OPP_PATH="$cand"; fi
  timeout --signal=TERM --kill-after=8s 145s python3 -u "$h/driver.py" precision "$TGR_CASE" "$out/precision/$route" --route "$route" --physical 10 > "$out/$route-precision.log" 2>&1 || { tail -n 24 "$out/$route-precision.log"; exit 21; }
  echo "TGR_PRECISION_ROUTE_COMPLETE $id $route"
done
python3 "$h/driver.py" compare "$TGR_CASE" "$out/precision" || exit 24
export TGR_PRECISION="$out/precision/pair.json"
cd "$h"
for route in baseline candidate; do
  if test "$route" = baseline; then export ASCEND_CUSTOM_OPP_PATH="$base"; else export ASCEND_CUSTOM_OPP_PATH="$cand"; fi
  export TGR_ROUTE="$route" TGR_ROUTE_OUTPUT="$out/$route"
  mkdir -p "$out/profiles/$route"
  timeout --signal=TERM --kill-after=12s 175s msprof --application="python3 p.py" --output="$out/profiles/$route" --aic-mode=task-based --aic-metrics=PipeUtilization > "$out/$route-msprof.log" 2>&1 || { tail -n 24 "$out/$route-msprof.log"; exit 22; }
  if ! test -f "$TGR_ROUTE_OUTPUT/runtime-receipt.json"; then
    tail -n 28 "$out/$route-msprof.log"
    exit 23
  fi
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); assert r['protocol_id']=='tgr_direct_staged_eager_2p5_split_routes_v1' and len(r['groups'])==7 and r['route']==sys.argv[2]" "$TGR_ROUTE_OUTPUT/runtime-receipt.json" "$route"
  echo "TGR_PROFILE_CAPTURE_COMPLETE $id $route"
done
