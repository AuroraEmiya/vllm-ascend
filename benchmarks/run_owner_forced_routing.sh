#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Validation-only capture controller; contains no deployment configuration.
set -eu
main_source=$1
n4_source=$2
validation_source=$3
out=$4
runner="$validation_source/benchmarks/validate_owner_forced_routing.py"
oracle="$validation_source/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_owner_prefix_adaptive.py"
mkdir -p "$out"
export PYTHONDONTWRITEBYTECODE=1
for spec in main:6385299e417741fc576c30c95f7860fd69dbec24 n4:53b4021ab8a75ad132158119e664ce6cc6e9da87; do
 side=${spec%:*}; commit=${spec#*:}
 if test "$side" = main; then src=$main_source; else src=$n4_source; fi
 test "$(git -C "$src" rev-parse HEAD)" = "$commit"
 test -z "$(git -C "$src" status --porcelain)"
 sha256sum "$src/vllm_ascend/ops/triton/sharded_indexer.py"
done > "$out/source-identity.txt"
git -C "$validation_source" show --no-patch --format=fuller HEAD > "$out/validation-commit.txt"
# Compile and check every pilot specialization before any timed capture.
for spec in 2:2 4:32 8:2 8:64; do
 IFS=: read -r world rows <<< "$spec"
 owner=3; if test "$world" -lt 4; then owner=$((world-1)); fi
 shape="r${rows}-w${world}-o${owner}-i768-random"
 for mode in main serial parallel; do
  src=$n4_source; if test "$mode" = main; then src=$main_source; fi
  cd "$src"
  export PYTHONPATH="$src" TRITON_CACHE_DIR="$out/cache-$mode-$shape"
  timeout --signal=TERM --kill-after=10 600 python "$runner" --mode "$mode" --oracle-file "$oracle" --case "$rows" "$world" "$owner" 768 random --output "$out/precheck-$mode-$shape.json" > "$out/precheck-$mode-$shape.log" 2>&1
  echo "PRECISION_PASS $mode $shape"
 done
done
for round in 1 2 3; do
 case "$round" in 1) order="main serial parallel" ;; 2) order="serial parallel main" ;; 3) order="parallel main serial" ;; esac
 for spec in 2:2 4:32 8:2 8:64; do
  IFS=: read -r world rows <<< "$spec"
  owner=3; if test "$world" -lt 4; then owner=$((world-1)); fi
  shape="r${rows}-w${world}-o${owner}-i768-random"
  for mode in $order; do
   src=$n4_source; if test "$mode" = main; then src=$main_source; fi
   cd "$src"
   export PYTHONPATH="$src" TRITON_CACHE_DIR="$out/cache-$mode-$shape"
   capture="$mode-$shape-round$round"
   date -Is > "$out/$capture-start.txt"
   timeout --signal=TERM --kill-after=10 600 msprof op --kernel-name=_merge_owner_prefix --warm-up=5 --launch-count=1 --aic-metrics=BasicInfo,PipeUtilization --output="$out/$capture" python "$runner" --mode "$mode" --oracle-file "$oracle" --case "$rows" "$world" "$owner" 768 random --output "$out/$capture-identity.json" > "$out/$capture.log" 2>&1
   test "$(find "$out/$capture" -name OpBasicInfo.csv | wc -l)" = 1
   date -Is > "$out/$capture-end.txt"
   echo "COMPLETE $capture"
  done
 done
 echo "ROUND_COMPLETE $round"
done
