#!/usr/bin/env bash
# Source after CANN set_env.sh. No global Python or Git configuration changes.
mlask_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ASCEND_CUSTOM_OPP_PATH="$mlask_root/install/vendors/mlask_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$mlask_root/install/vendors/mlask_transformer/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$mlask_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MAX_JOBS=${MAX_JOBS:-4}
export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}
export PYTHONDONTWRITEBYTECODE=1
