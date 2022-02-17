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

#include "core/operation/layer_normalization.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertLayerNormalization(Converter* converter, hal::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);

  // Cast input op
  auto cast_input_op = converter->AddOperator<ge::op::Cast>(output_operand, "cast_input");
  cast_input_op->set_attr_dst_type(ge::DT_FLOAT);
  SET_INPUT(cast_input_op, x, input_operator);
  auto cast_input_operator = MAP_OUTPUT(cast_input_op, y, output_operand);

  // Cast scale op
  auto cast_scale_op = converter->AddOperator<ge::op::Cast>(output_operand, "cast_scale");
  cast_scale_op->set_attr_dst_type(ge::DT_FLOAT);
  SET_INPUT(cast_scale_op, x, scale_operator);
  auto cast_scale_operator = MAP_OUTPUT(cast_scale_op, y, output_operand);

  // Cast bias op
  auto cast_bias_op = converter->AddOperator<ge::op::Cast>(output_operand, "cast_bias");
  cast_bias_op->set_attr_dst_type(ge::DT_FLOAT);
  SET_INPUT(cast_bias_op, x, bias_operator);
  auto cast_bias_operator = MAP_OUTPUT(cast_bias_op, y, output_operand);

  // Layer normalization
  auto layer_norm_op =
      converter->AddOperator<ge::op::LayerNorm>(output_operand);
  layer_norm_op->set_attr_epsilon(epsilon);
  layer_norm_op->set_attr_begin_norm_axis(begin_norm_axis);
  layer_norm_op->set_attr_begin_params_axis(begin_norm_axis);
  SET_INPUT(layer_norm_op, x, cast_input_operator);
  SET_INPUT(layer_norm_op, beta, cast_bias_operator);
  SET_INPUT(layer_norm_op, gamma, cast_scale_operator);
  MAP_OUTPUT(layer_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
