#!/bin/bash
# Validation-only adapter: keep torch_npu's normal parser pipeline while using
# CANN's installed official Python offline commands without device enumeration.
set -euo pipefail
entry=/usr/local/Ascend/cann-9.2.1/tools/profiler/profiler_tool/analysis/msprof/msprof.py
mode= output= type=text
for arg in "$@";do
 case "$arg" in
  --export=on) test -z "$mode";mode=export;;
  --analyze=on) test -z "$mode";mode=analyze;;
  --output=*) output=${arg#--output=};;
  --type=text) type=text;;
  *) printf 'Unsupported offline argument: %s\n' "$arg" >&2;exit 2;;
 esac
 done
 test -f "$entry"
 test -n "$mode"
 test -n "${MSPROF_OFFLINE_ALLOWED_ROOT:-}"
 allowed=$(readlink -f "$MSPROF_OFFLINE_ALLOWED_ROOT")
 output=$(readlink -f "$output")
 case "$output" in
  "$allowed"/*/profiles/*/PROF_*) ;;
  *) echo 'Refusing a path outside the configured profiling result directory' >&2;exit 2;;
 esac
 test -d "$output"
 if [ "$mode" = export ];then
  python "$entry" export summary -dir "$output" --format csv
  python "$entry" export timeline -dir "$output"
 else
  python "$entry" analyze -dir "$output" --rule communication,communication_matrix --type "$type"
 fi
