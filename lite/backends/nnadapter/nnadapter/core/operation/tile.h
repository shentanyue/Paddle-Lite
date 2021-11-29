// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

namespace nnadapter {
namespace operation {

#define TILE_OPERATION_EXTRACT_INPUTS_OUTPUTS                                 \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 2);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                        \
  /* Input */                                                                 \
  auto input_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);           \
  /* Repeat times */                                                          \
  auto repeat_times_operand = input_operands[1];                              \
  NNADAPTER_VLOG(5) << "repeat_times: "                                       \
                    << OperandToString(repeat_times_operand);                 \
  uint32_t repeat_times_count;                                                     \
  int32_t* repeat_times_data;                                                      \
  std::vector<int32_t> repeat_times;                                          \
  if (repeat_times_operand) {                                                 \
    repeat_times_count = repeat_times_operand->length / sizeof(int32_t); \
    repeat_times_data =                                                  \
        reinterpret_cast<int32_t*>(repeat_times_operand->buffer);             \
    repeat_times = std::vector<int32_t>(                                      \
        repeat_times_data, repeat_times_data + repeat_times_count);           \
  }                                                                           \
  for (size_t i = 0; i < repeat_times.size(); i++) {                          \
    NNADAPTER_VLOG(5) << "repeat_times[" << i << "]: " << repeat_times[i];    \
  }                                                                           \
  /* Output */                                                                \
  auto output_operand = output_operands[0];                                   \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
