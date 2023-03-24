// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/im2col.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

inline int CalcOutputSize(int input_size,
                          int kernel_size,
                          int dilation,
                          int padding0,
                          int padding1,
                          int stride) {
  const int dkernel = dilation * (kernel_size - 1) + 1;
  int output_size = (input_size + padding0 + padding1 - dkernel) / stride + 1;
  return output_size;
}

NNADAPTER_EXPORT bool ValidateIm2col(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareIm2col(core::Operation* operation) {
  IM2COL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);
  output_type.dimensions.count = 3;
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    int output_height = CalcOutputSize(input_dimensions[2],
                                       kernel_sizes[0],
                                       dilations[0],
                                       paddings[0],
                                       paddings[2],
                                       strides[0]);
    int output_width = CalcOutputSize(input_dimensions[3],
                                      kernel_sizes[1],
                                      dilations[1],
                                      paddings[1],
                                      paddings[3],
                                      strides[1]);
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] =
        input_dimensions[1] * kernel_sizes[0] * kernel_sizes[1];
    output_dimensions[2] = output_height * output_width;
  };
  infer_output_shape(input_type.dimensions.data, output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteIm2col(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
