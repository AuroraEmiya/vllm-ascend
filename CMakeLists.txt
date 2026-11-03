cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_C)

# include(CheckCXXcompilerFlag)
# check_cxx_compiler_flag("-std=c++17", COMPILER_SUPPORTS_CXX17)
set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

find_package(pybind11 REQUIRED)

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
set(VLLM_ASCEND_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION
  "import torch; print(torch.__version__)" "Failed to locate torch path")
# check torch version is 2.10.0
if(NOT ${TORCH_VERSION} VERSION_EQUAL "2.10.0")
  message(FATAL_ERROR "Expected PyTorch version 2.10.0, but found ${TORCH_VERSION}")
endif()

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION ${SOC_VERSION})
message(STATUS "Detected SOC version: ${SOC_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRINGS "Build type Release/Debug (default Release)" FORCE)
endif()

if (CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRINGS "path to install()")
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})
if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

file(GLOB KERNEL_FILES
${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/*.cpp)

set(VLLM_ASCEND_CUSTOM_OP
    ${KERNEL_FILES}
)

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
    message(STATUS "Hardware ${SOC_VERSION} detected: skip vllm_ascend_kernels compile")
else()
    ascendc_library(vllm_ascend_kernels SHARED
        ${VLLM_ASCEND_CUSTOM_OP}
    )
endif()

message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

if(SOC_VERSION MATCHES "ascend310p.*")
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
else()
    file(GLOB VLLM_ASCEND_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/aclnn_torch_adapter/*.cpp)
endif()

# Sparse KV cache offloading is currently supported on Atlas A3 only.
if(SOC_VERSION MATCHES "ascend910_93.*")
  find_package(OpenMP REQUIRED)
endif()

include_directories(
  ${pybind11_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${TORCH_INCLUDE_DIRS}
  # Prefer the selected CANN toolkit headers over the ACL snapshot bundled in
  # some torch_npu wheels. Feature checks below use ASCEND_HOME_PATH as well.
  ${ASCEND_HOME_PATH}/include
  ${TORCH_NPU_PATH}/include
)

set(
  INCLUDES
  ${TORCH_INCLUDE_DIRS}
  ${TORCH_NPU_INCLUDE_DIRS}
  ${ASCEND_HOME_PATH}/include
)

pybind11_add_module(vllm_ascend_C ${VLLM_ASCEND_SRC})
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
  add_library(serial_matmul_opdev STATIC csrc/serial_matmul/serial_matmul_opdev.cpp)
  set_target_properties(serial_matmul_opdev PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(serial_matmul_opdev PRIVATE
    ${ASCEND_HOME_PATH}/include ${ASCEND_HOME_PATH}/include/aclnn)
  target_compile_definitions(serial_matmul_opdev PRIVATE _GLIBCXX_USE_CXX11_ABI=${CANN_OPDEV_CXX11_ABI})
  target_link_directories(serial_matmul_opdev PRIVATE ${ASCEND_HOME_PATH}/lib64)
  target_link_libraries(serial_matmul_opdev PUBLIC opapi nnopbase)
  target_link_libraries(vllm_ascend_C PRIVATE serial_matmul_opdev)
endif()
if(SOC_VERSION MATCHES "ascend910_93.*")
  target_compile_definitions(vllm_ascend_C PRIVATE VLLM_ASCEND_ENABLE_SPARSE_KV_OFFLOAD)
  target_link_libraries(vllm_ascend_C PRIVATE OpenMP::OpenMP_CXX)
endif()

# Detect aclrtMemcpyBatchAsync availability (CANN 8.5+)
# Can be overridden via VLLM_ASCEND_ENABLE_BATCH_MEMCPY env var (registered
# in vllm_ascend/envs.py, forwarded by setup.py as a CMake variable):
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1  -> force enable
#   VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0  -> force disable
#   unset                               -> auto-detect from CANN headers
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${ASCEND_HOME_PATH}/include)
set(CMAKE_REQUIRED_LIBRARIES ascendcl)
set(CMAKE_REQUIRED_LINK_OPTIONS "-L${ASCEND_HOME_PATH}/lib64")

# Keep builds on older CANN versions working; endpoint selection must fail
# explicitly if the installed CANN lacks the user-to-physical conversion.
check_cxx_source_compiles("
  #include <acl/acl_rt.h>
  int main() {
    int32_t physical_id = 0;
    return aclrtGetPhyDevIdByUserDevId(0, &physical_id);
  }
" HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
if(HAVE_ACLRT_GET_PHY_DEV_ID_BY_USER_DEV_ID)
  target_compile_definitions(vllm_ascend_C PRIVATE CANN_DEVICE_ID_MAPPING)
endif()

if(DEFINED VLLM_ASCEND_ENABLE_BATCH_MEMCPY)
  if("${VLLM_ASCEND_ENABLE_BATCH_MEMCPY}" STREQUAL "1")
    message(STATUS "aclrtMemcpyBatchAsync: force enabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=1")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: force disabled via VLLM_ASCEND_ENABLE_BATCH_MEMCPY=0")
  endif()
else()
  # Test the full code pattern we actually use, including struct member access.
  # This ensures the macro is only defined when the API is fully compatible.
  check_cxx_source_compiles("
    #include <acl/acl_rt.h>
    int main() {
      aclrtMemLocation loc = {};
      loc.type = ACL_MEM_LOCATION_TYPE_HOST;
      loc.id = 0;
      aclrtMemcpyBatchAttr attr = {};
      attr.srcLoc = loc;
      attr.dstLoc = loc;
      (void)aclrtMemcpyBatchAsync;
      return 0;
    }
  " HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
  if(HAVE_ACLRT_MEMCPY_BATCH_ASYNC)
    message(STATUS "aclrtMemcpyBatchAsync: detected in CANN headers, enabling batch memcpy path")
    target_compile_definitions(vllm_ascend_C PRIVATE CANN_MEMCPY_BATCH_ASYNC)
  else()
    message(STATUS "aclrtMemcpyBatchAsync: not found in CANN headers, using fallback aclrtMemcpyAsync loop")
  endif()
endif()

if(SOC_VERSION MATCHES "ascend310p.*")
    target_compile_definitions(vllm_ascend_C PRIVATE -DASCEND_PLATFORM_310P)
endif()

if(NOT (SOC_VERSION MATCHES "ascend310p.*|ascend950"))
    target_compile_definitions(vllm_ascend_C PRIVATE -DVLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
endif()

target_link_directories(
  vllm_ascend_C
  PRIVATE
  ${TORCH_LIBRARY_DIRS}
  ${TORCH_NPU_PATH}/lib/
  ${ASCEND_HOME_PATH}/lib64
)

set(VLLM_ASCEND_C_COMMON_LIBS
  ${TORCH_LIBRARIES}
  torch_npu
  ascendcl
  tiling_api
  register
  platform
  ascendalog
  dl
  opapi
)

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
else()
  target_link_libraries(
    vllm_ascend_C
    PUBLIC
    vllm_ascend_kernels
    ${VLLM_ASCEND_C_COMMON_LIBS}
  )
endif()

target_link_options(vllm_ascend_C PRIVATE "-Wl,-rpath,$ORIGIN:$ORIGIN/lib:$ORIGIN/_cann_ops_custom/vendors/custom_transformer/op_api/lib")

if(SOC_VERSION MATCHES "ascend310p.*|ascend950")
  install(TARGETS vllm_ascend_C DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
else()
  install(TARGETS vllm_ascend_C vllm_ascend_kernels DESTINATION ${VLLM_ASCEND_INSTALL_PATH})
endif()
