# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT DEFINED NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT)
  set(NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT $ENV{NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT})
endif()
if(NOT NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT or env NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT when NNADAPTER_WITH_HUAWEI_ASCEND_NPU=ON")
endif()
message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Huawei Ascend NPU SDK.")
endif()

add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_OF_MDC=${NNADAPTER_HUAWEI_ASCEND_NPU_OF_MDC})
# For Huawei MDC
if (NNADAPTER_HUAWEI_ASCEND_NPU_OF_MDC)
  include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/local/Ascend/runtime/include")
  include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/include/")
  # libascendcl.so 
  find_library(HUAWEI_ASCEND_NPU_SDK_ASCENDCL_HAL_FILE NAMES ascend_hal
    PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/lib/driver
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT HUAWEI_ASCEND_NPU_SDK_ASCENDCL_HAL_FILE)
    message(FATAL_ERROR "Missing libascend_hal.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/include/driver")
  endif()
  add_library(ascend_hal SHARED IMPORTED GLOBAL)
  set_property(TARGET ascend_hal PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ASCENDCL_HAL_FILE})

  # libascend_hal.so 
  find_library(HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE NAMES ascendcl
    PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/local/Ascend/runtime/lib64
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE)
    message(FATAL_ERROR "Missing libascendcl.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/sysroot/usr/local/Ascend/runtime/lib64")
  endif()
  add_library(acl_ascendcl SHARED IMPORTED GLOBAL)
  set_property(TARGET acl_ascendcl PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE})
  set(DEPS ${DEPS} ascend_hal acl_ascendcl)
else()
  if(NOT NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION)
    # Extract CANN version from NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT, such as NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION=3.3.0
    # when NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT=/usr/local/Ascend/ascend-toolkit/3.3.0
    get_filename_component(NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT "${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}" REALPATH)
    message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}")
    string(REGEX MATCH "[0-9]+\\.[0-9]+[0-9a-zA-Z\\.]*" NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION "${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}")
    if(NOT NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION)
      message(FATAL_ERROR "Failed to extract the CANN version, please set the correct NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT or manually set NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION as NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION=3.3.0")
    endif()
  endif()
  message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION}")
  string(REGEX MATCHALL "[0-9]+" CANN_VERSION_NUM_LIST "${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION}")
  list(LENGTH CANN_VERSION_NUM_LIST CANN_VERSION_NUM_LIST_LENGTH)
  set(CANN_PATCH_VERSION 0)
  if(CANN_VERSION_NUM_LIST_LENGTH GREATER_EQUAL 2)
    list(GET CANN_VERSION_NUM_LIST 0 CANN_MAJOR_VERSION)
    list(GET CANN_VERSION_NUM_LIST 1 CANN_MINOR_VERSION)
    if (CANN_VERSION_NUM_LIST_LENGTH GREATER_EQUAL 3)
      list(GET CANN_VERSION_NUM_LIST 2 CANN_PATCH_VERSION)
    endif()
  else()
    message(FATAL_ERROR "Failed to extract the CANN version, please set the correct NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT or manually set NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION as NNADAPTER_HUAWEI_ASCEND_NPU_SDK_VERSION=3.3.0")
  endif()
  message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION: ${CANN_MAJOR_VERSION}")
  message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION: ${CANN_MINOR_VERSION}")
  message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION: ${CANN_PATCH_VERSION}")
  add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION=${CANN_MAJOR_VERSION})
  add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION=${CANN_MINOR_VERSION})
  add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION=${CANN_PATCH_VERSION})

  include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/include")
  include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/include")
  include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/opp")
  # ACL libraries
  # libascendcl.so 
  find_library(HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE NAMES ascendcl
    PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE)
    message(FATAL_ERROR "Missing libascendcl.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64")
  endif()
  add_library(acl_ascendcl SHARED IMPORTED GLOBAL)
  set_property(TARGET acl_ascendcl PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE})
  # ATC libraries
  # libge_compiler.so
  find_library(HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE NAMES ge_compiler
    PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
    CMAKE_FIND_ROOT_PATH_BOTH)
  if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE)
    message(FATAL_ERROR "Missing libge_compiler.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
  endif()
  add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
  set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE})
  # ACL libs should before ATC libs
  set(DEPS ${DEPS} acl_ascendcl atc_ge_compiler)
endif()

