# Build the same registered HC sources without rebuilding unrelated CANN operators.
if(NOT SOC_VERSION MATCHES "ascend910_93.*")
  message(FATAL_ERROR "The HC-only target currently requires Ascend A3 (910_93)")
endif()
ascendc_library(vllm_ascend_hc_kernels SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/qwen_hc/op_kernel/qwen_hc.cpp)
pybind11_add_module(vllm_ascend_hc_C
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/qwen_hc/op_host/qwen_hc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/qwen_hc/op_host/qwen_hc_tiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/qwen_hc/op_host/qwen_hc_platform.cpp)
target_compile_definitions(vllm_ascend_hc_C PRIVATE VLLM_ASCEND_HC_STANDALONE)
target_include_directories(vllm_ascend_hc_C PRIVATE
  ${ASCEND_HOME_PATH}/include ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_PATH}/include)
target_link_directories(vllm_ascend_hc_C PRIVATE
  ${TORCH_LIBRARY_DIRS} ${TORCH_NPU_PATH}/lib ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(vllm_ascend_hc_C PRIVATE vllm_ascend_hc_kernels
  ${TORCH_LIBRARIES} torch_npu ascendcl tiling_api platform)
set_target_properties(vllm_ascend_hc_C PROPERTIES
  BUILD_RPATH "$ORIGIN;$ORIGIN/lib;${TORCH_NPU_PATH}/lib" INSTALL_RPATH "$ORIGIN")
install(TARGETS vllm_ascend_hc_C vllm_ascend_hc_kernels
  DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
