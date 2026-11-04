# Version-pinned optional matrix adapter; no device compute kernel is defined here.
option(VLLM_ASCEND_ENABLE_SERIAL_V3 "Build the version-pinned CANN 9.2 serial V3 adapter" OFF)
if(VLLM_ASCEND_ENABLE_SERIAL_V3)
  if(NOT EXISTS "${ASCEND_HOME_PATH}/include/aclnn/opdev/make_op_executor.h")
    message(FATAL_ERROR "Serial V3 requires the selected CANN opdev headers")
  endif()
  target_compile_definitions(vllm_ascend_C PRIVATE VLLM_ASCEND_SERIAL_V3)
  target_include_directories(vllm_ascend_C PRIVATE ${ASCEND_HOME_PATH}/include/aclnn)
  # CANN 9.2 opdev is built with the old libstdc++ ABI. Keep the Torch
  # extension's ABI and expose only ACL/C types across this small boundary.
  set(CANN_OPDEV_CXX11_ABI 0 CACHE STRING "libstdc++ ABI used by CANN opdev")
  add_library(serial_matmul_opdev STATIC csrc/aclnn_torch_adapter/serial_matmul_v3/opdev.cpp)
  set_target_properties(serial_matmul_opdev PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(serial_matmul_opdev PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/include/aclnn)
  target_compile_definitions(serial_matmul_opdev PRIVATE _GLIBCXX_USE_CXX11_ABI=${CANN_OPDEV_CXX11_ABI})
  target_link_directories(serial_matmul_opdev PRIVATE ${ASCEND_HOME_PATH}/lib64)
  target_link_libraries(serial_matmul_opdev PUBLIC opapi nnopbase)
  target_link_libraries(vllm_ascend_C PRIVATE serial_matmul_opdev)
endif()
