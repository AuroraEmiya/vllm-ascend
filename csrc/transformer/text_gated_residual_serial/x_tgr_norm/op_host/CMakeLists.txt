# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
if(BUILD_OPEN_PROJECT)
  target_sources(op_host_aclnn PRIVATE x_tgr_norm_def.cpp)
endif()
add_ops_compile_options(OP_NAME XTgrNorm OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
if(NOT BUILD_OPS_RTY_KERNEL)
  add_modules_sources(OPTYPE x_tgr_norm ACLNNTYPE aclnn)
endif()
