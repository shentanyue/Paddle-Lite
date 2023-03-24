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

#pragma once

#include <vector>

namespace nnadapter {
namespace operation {

#define IM2COL_OPERATION_EXTRACT_INPUTS_OUTPUTS                                \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 5);                                          \
  NNADAPTER_CHECK_EQ(output_count, 1);                                         \
  /* Input */                                                                  \
  auto input_operand = input_operands[0];                                      \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);            \
  /* Kernel Size */                                                            \
  auto kernel_sizes_operand = input_operands[1];                               \
  NNADAPTER_VLOG(5) << "kernel sizes: "                                        \
                    << OperandToString(kernel_sizes_operand);                  \
  auto kernel_sizes_count = kernel_sizes_operand->length / sizeof(int32_t);    \
  auto kernel_sizes_data =                                                     \
      reinterpret_cast<int32_t*>(kernel_sizes_operand->buffer);                \
  auto kernel_sizes = std::vector<int32_t>(                                    \
      kernel_sizes_data, kernel_sizes_data + kernel_sizes_count);              \
  for (size_t i = 0; i < kernel_sizes.size(); i++) {                           \
    NNADAPTER_VLOG(5) << "kernel_sizes[" << i << "]: " << kernel_sizes[i];     \
  }                                                                            \
  /* Stride */                                                                 \
  auto strides_operand = input_operands[2];                                    \
  NNADAPTER_VLOG(5) << "strides: " << OperandToString(strides_operand);        \
  auto strides_count = strides_operand->length / sizeof(int32_t);              \
  auto strides_data = reinterpret_cast<int32_t*>(strides_operand->buffer);     \
  auto strides =                                                               \
      std::vector<int32_t>(strides_data, strides_data + strides_count);        \
  for (size_t i = 0; i < strides.size(); i++) {                                \
    NNADAPTER_VLOG(5) << "strides[" << i << "]: " << strides[i];               \
  }                                                                            \
  /* Paddings */                                                               \
  auto paddings_operand = input_operands[3];                                   \
  NNADAPTER_VLOG(5) << "paddings: " << OperandToString(paddings_operand);      \
  auto paddings_count = paddings_operand->length / sizeof(int32_t);            \
  auto paddings_data = reinterpret_cast<int32_t*>(paddings_operand->buffer);   \
  auto paddings =                                                              \
      std::vector<int32_t>(paddings_data, paddings_data + paddings_count);     \
  for (size_t i = 0; i < paddings.size(); i++) {                               \
    NNADAPTER_VLOG(5) << "paddings[" << i << "]: " << paddings[i];             \
  }                                                                            \
  /* Dilations */                                                              \
  auto dilations_operand = input_operands[4];                                  \
  NNADAPTER_VLOG(5) << "dilations: " << OperandToString(dilations_operand);    \
  auto dilations_count = dilations_operand->length / sizeof(int32_t);          \
  auto dilations_data = reinterpret_cast<int32_t*>(dilations_operand->buffer); \
  auto dilations =                                                             \
      std::vector<int32_t>(dilations_data, dilations_data + dilations_count);  \
  for (size_t i = 0; i < dilations.size(); i++) {                              \
    NNADAPTER_VLOG(5) << "dilations[" << i << "]: " << dilations[i];           \
  }                                                                            \
  /* Output */                                                                 \
  auto output_operand = output_operands[0];                                    \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
