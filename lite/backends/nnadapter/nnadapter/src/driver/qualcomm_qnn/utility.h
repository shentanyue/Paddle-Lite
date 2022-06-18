// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <dlfcn.h>
#include <cstdlib>
#include "QnnInterface.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

/**
 * Qualcomm-Qnn device types.
 *
 * Available since version 1.
 */
typedef enum {
  CPU_DEVICE = 0,
  HTP_DEVICE = 1,
  GPU_DEVICE = 2,
  DSP_DEVICE = 3,
} QualcommQnnDeviceType;

// Use which lib at runtime, for example: "libQnnCpu.so", "libQnnHtp.so"
#define QUALCOMM_QNN_RUNTIME_LIB "QUALCOMM_QNN_RUNTIME_LIB"
// Use which lib at runtime to load op package, for example:
// "libqualcomm_qnn.so"
#define QUALCOMM_QNN_OP_PACKAGE_LIB "QUALCOMM_QNN_OP_PACKAGE_LIB"

#define QNN_CHECK(a) NNADAPTER_CHECK_EQ((a), QNN_SUCCESS);

#define QNN_CHECK_VALUE(value, return_error) \
  do {                                       \
    if (!(value)) {                          \
      return return_error;                   \
    }                                        \
  } while (0);

#define QNN_CHECK_STATUS(status)   \
  do {                             \
    if ((status) != QNN_SUCCESS) { \
      return status;               \
    }                              \
  } while (0);

#define QNN_CHECK_EQ(a, b, return_error) \
  do {                                   \
    if ((a) != (b)) {                    \
      return return_error;               \
    }                                    \
  } while (0);

void LogStdoutCallback(const char* fmt,
                       QnnLog_Level_t level,
                       uint64_t timestamp,
                       va_list argp);

QNN_INTERFACE_VER_TYPE GetQnnInterface(void* lib_backend_handle);

Qnn_DataType_t ConvertToQnnDatatype(
    const NNAdapterOperandPrecisionCode precision);

template <typename T>
Qnn_DataType_t GetQnnDatatype();

}  // namespace qualcomm_qnn
}  // namespace nnadapter
