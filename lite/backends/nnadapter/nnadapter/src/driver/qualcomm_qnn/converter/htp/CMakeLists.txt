# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

set(OP_PACKAGE_NAME qualcomm_qnn_custom_htp_op_package)

if(NOT DEFINED NNADAPTER_HEXAGON_TOOLS_ROOT)
  set(NNADAPTER_HEXAGON_TOOLS_ROOT $ENV{NNADAPTER_HEXAGON_TOOLS_ROOT})
  set(NNADAPTER_HEXAGON_TOOLS_ROOT "/Work/qnn/Hexagon_SDK/4.3.0.0/tools/HEXAGON_Tools/8.4.11/Tools")
  if(NOT NNADAPTER_HEXAGON_TOOLS_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_HEXAGON_TOOLS_ROOT or env NNADAPTER_HEXAGON_TOOLS_ROOT when NNADAPTER_WITH_QUALCOMM_QNN=ON")
  endif()
endif()
message(STATUS "NNADAPTER_HEXAGON_TOOLS_ROOT: ${NNADAPTER_HEXAGON_TOOLS_ROOT}")

set(PKG_NAME "Custom.Htp.OpPackage")

# Resolve the compilation error caused by the introduction of HTP header files
set(CMAKE_CXX_STANDARD 17)
string(REPLACE "-std=c++11" "-std=c++17" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wnon-virtual-dtor" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include -Wreorder -Wno-missing-braces -Wno-format -Wno-unused-command-line-argument -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTHIS_PKG_NAME=${PKG_NAME} -MMD -D__HVXDBL__ -ffast-math -DUSE_OS_LINUX -fomit-frame-pointer -Wno-invalid-offsetof")

include_directories(.)
include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include/HTP/)
include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include/HTP/core/)
include_directories(${NNADAPTER_HEXAGON_TOOLS_ROOT}/libnative/include)

aux_source_directory(. SRCS)

add_library(${OP_PACKAGE_NAME} SHARED ${SRCS})
target_link_libraries(${OP_PACKAGE_NAME} "-Wl,--whole-archive -L${NNADAPTER_HEXAGON_TOOLS_ROOT}/libnative/lib -lnative -Wl,--no-whole-archive -lpthread")
