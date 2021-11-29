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

#include "core/operation/tile.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareTile(hal::Operation* operation) {
  TILE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);

  auto& repeat_times_type = repeat_times_operand->type;
  if (repeat_times_type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    auto repeat_times_operand_dimension =
        *reinterpret_cast<NNAdapterOperandDimensionType*>(
            repeat_times_operand->buffer);
    repeat_times_count = repeat_times_operand_dimension.count;
    repeat_times_data = repeat_times_operand_dimension.data;
    repeat_times =  std::vector<int32_t>(repeat_times_data, repeat_times_count + repeat_times_data);
  } else if (!IsConstantOperand(repeat_times_operand)) {
    NNADAPTER_LOG(FATAL) << "Unsupported repeat_times lifetime: "
                         << static_cast<int32_t>(repeat_times_type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }

  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                              uint32_t input_dimensions_count,
                              int32_t* output_dimensions_data) {
                              
    // broadcast for vec_in_dims.size() equal to repeat_times.size()
    std::vector<int> input_dims_vec;
    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      input_dims_vec.push_back(input_dimensions_data[i]);
    }

    if (repeat_times_count < input_dimensions_count) {
      int diff = input_dimensions_count - repeat_times_count;
      repeat_times.insert(repeat_times.begin(), diff, 1);
      output_type.dimensions.count = input_dimensions_count;
    } else {
      int diff = repeat_times_count - input_dimensions_count;
      input_dims_vec.insert(input_dims_vec.begin(), diff, 1);
      output_type.dimensions.count = repeat_times_count;
    }

    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      output_dimensions_data[i] = input_dims_vec[i] * repeat_times_data[i];
    }
  };

  infer_output_shape(input_type.dimensions.data,
                    input_type.dimensions.count,
                    output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i]);
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
