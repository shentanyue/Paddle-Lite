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

#include "core/operation/range.h"

#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

void GetRangeOperandValue(hal::Operand* operand, int64_t& data) {
  if (IsConstantOperand(operand)) {
    data = reinterpret_cast<int64_t*>(operand->buffer)[0];
    NNADAPTER_VLOG(5) << "GetRangeOperandValue: " << data;
  } else if (IsTemporaryShapeOperand(operand)) {
    auto& temporary_shape = *(GetTemporaryShape(operand));
    NNADAPTER_CHECK_EQ(temporary_shape.count, 1);
    data = temporary_shape.data[0];
  } else {
    data = NNADAPTER_UNKNOWN;
  }
}

int PrepareRange(hal::Operation* operation) {
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto& output_type = output_operand->type;
  NNADAPTER_CHECK_EQ(start_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_EQ(limit_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_EQ(delta_operand->type.dimensions.count, 1);

  int64_t start_data, limit_data, delta_data;
  start_data = limit_data = delta_data = -1;
  GetRangeOperandValue(start_operand, start_data);
  GetRangeOperandValue(limit_operand, limit_data);
  GetRangeOperandValue(delta_operand, delta_data);

  output_type.dimensions.count = 1;
  if (start_data == NNADAPTER_UNKNOWN || limit_data == NNADAPTER_UNKNOWN ||
      delta_data == NNADAPTER_UNKNOWN) {
    output_type.dimensions.data[0] = NNADAPTER_UNKNOWN;
  } else {
    output_type.dimensions.data[0] =
        GetSpanCount(start_data, limit_data, delta_data);
  }
  output_type.precision = start_operand->type.precision;
  output_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
