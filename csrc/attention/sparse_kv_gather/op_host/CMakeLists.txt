# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

# Current measured default on Ascend 910B4/A2.
# Override with either:
#   cmake ... -DSPARSE_KV_GATHER_PLAN_ROWS=512
#   export SPARSE_KV_GATHER_PLAN_ROWS=512
set(SPARSE_KV_GATHER_PLAN_ROWS "256" CACHE STRING
    "SparseKvGather metadata planning rows (multiple of 16, 16..2048)")

if(DEFINED ENV{SPARSE_KV_GATHER_PLAN_ROWS}
   AND NOT "$ENV{SPARSE_KV_GATHER_PLAN_ROWS}" STREQUAL "")
    set(SPARSE_KV_GATHER_PLAN_ROWS "$ENV{SPARSE_KV_GATHER_PLAN_ROWS}"
        CACHE STRING "SparseKvGather metadata planning rows" FORCE)
endif()

if(SPARSE_KV_GATHER_PLAN_ROWS LESS 16 OR SPARSE_KV_GATHER_PLAN_ROWS GREATER 2048)
    message(FATAL_ERROR
        "SPARSE_KV_GATHER_PLAN_ROWS must be in [16, 2048], got ${SPARSE_KV_GATHER_PLAN_ROWS}")
endif()

math(EXPR _SKG_PLAN_ROWS_MOD "${SPARSE_KV_GATHER_PLAN_ROWS} % 16")
if(NOT _SKG_PLAN_ROWS_MOD EQUAL 0)
    message(FATAL_ERROR
        "SPARSE_KV_GATHER_PLAN_ROWS must be a multiple of 16, got ${SPARSE_KV_GATHER_PLAN_ROWS}")
endif()

message(STATUS "SparseKvGather plan rows: ${SPARSE_KV_GATHER_PLAN_ROWS}")

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE sparse_kv_gather_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME SparseKvGather
    OPTIONS -DSKG_METADATA_PLAN_ROWS=${SPARSE_KV_GATHER_PLAN_ROWS}
            --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_kv_gather ACLNNTYPE aclnn)
endif()
