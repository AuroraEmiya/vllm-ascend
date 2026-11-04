# Licensed under CANN Open Software License Agreement Version 2.0.
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE x_mla_vector_def.cpp)
endif()
add_ops_compile_options(OP_NAME XMlaVector OPTIONS --cce-auto-sync=off)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE x_mla_vector ACLNNTYPE aclnn)
endif()
add_op_to_compiled_list()
