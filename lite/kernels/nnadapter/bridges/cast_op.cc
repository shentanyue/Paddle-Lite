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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/nnadapter/bridges/converter.h"
#include "lite/kernels/nnadapter/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nnadapter {

// Tensor Precision
NNAdapterOperandPrecisionCode GetPrecisionCode(int dtype_code) {
  NNAdapterOperandPrecisionCode precision_code = NNADAPTER_FLOAT32;
  switch (dtype_code) {
    case 0:  // BOOL = 0;
      precision_code = NNADAPTER_TENSOR_BOOL8;
      break;
    case 1:  // INT16 = 1
      precision_code = NNADAPTER_TENSOR_INT16;
      break;
    case 2:  // INT32 = 2
      precision_code = NNADAPTER_TENSOR_INT32;
      break;
    case 3:  // INT64 = 3
      precision_code = NNADAPTER_TENSOR_INT64;
      break;
    case 4:  // FP16 = 4
      precision_code = NNADAPTER_TENSOR_FLOAT16;
      break;
    case 5:  // FP32 = 5
      precision_code = NNADAPTER_TENSOR_FLOAT32;
      break;
    case 6:  // FP64 = 6
      precision_code = NNADAPTER_TENSOR_FLOAT64;
      break;
    case 20:  // UINT8 = 20
      precision_code = NNADAPTER_TENSOR_UINT8;
      break;
    case 21:  // INT8 = 21
      precision_code = NNADAPTER_TENSOR_INT8;
      break;
    default:
      LOG(FATAL) << "unsupported data type: " << dtype_code;
      break;
  }
  return precision_code;
}

int CastConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto converter = static_cast<Converter*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "Converting " << op_type << " ...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto has_x_scale = op_info->HasInputScale(x_scale_name, true);
  auto x_scale =
      has_x_scale ? op_info->GetInputScale(x_scale_name, true)[0] : 0.f;
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";
  auto has_out_scale = op_info->HasOutputScale(out_scale_name, true);
  auto out_scale =
      has_out_scale ? op_info->GetOutputScale(out_scale_name, true)[0] : 0.f;
  auto out = scope->FindMutableTensor(out_name);
  auto out_dims = out->dims();

  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  auto in_dtype = op_info->GetAttr<int>("in_dtype");
  auto out_dtype = op_info->GetAttr<int>("out_dtype");

  // Input&Output dtype
  NNAdapterOperandPrecisionCode itype = GetPrecisionCode(in_dtype);
  NNAdapterOperandPrecisionCode otype = GetPrecisionCode(out_dtype);

  // Input operand
  NNAdapterOperand* input_operand = nullptr;
  if (converter->HasOperand(x_name)) {
    input_operand = converter->GetOperand(x_name);
  } else {
    if (has_x_scale) {
      input_operand =
          converter->AddQuant8VariableOperand(x_dims, x_scale, x_name);
    } else {
      input_operand = converter->AddVariableOperand(x_dims, x_name, itype);
    }
  }

  // scalar dtype operand
  int32_t dtype = NNADAPTER_FLOAT32;
  switch (out_dtype) {
    case 0:                     // BOOL = 0;
      dtype = NNADAPTER_BOOL8;  // 0
      break;
    case 1:                     // INT16 = 1
      dtype = NNADAPTER_INT16;  // 3
      break;
    case 2:                     // INT32 = 2
      dtype = NNADAPTER_INT32;  // 6
      break;
    case 3:                     // INT64 = 3
      dtype = NNADAPTER_INT64;  // 7
      break;
    case 4:                       // FP16 = 4
      dtype = NNADAPTER_FLOAT16;  // 9
      break;
    case 5:                       // FP32 = 5
      dtype = NNADAPTER_FLOAT32;  // 10
      break;
    case 6:                       // FP64 = 6
      dtype = NNADAPTER_FLOAT64;  // 11
      break;
    case 20:                    // UINT8 = 20
      dtype = NNADAPTER_UINT8;  // 2
      break;
    case 21:                   // INT8 = 21
      dtype = NNADAPTER_INT8;  // 1
      break;
    default:
      LOG(FATAL) << "unsupported data type: " << out_dtype;
      break;
  }

  auto dtype_operand = converter->AddInt32ConstantOperand(dtype);

  // Output operand
  NNAdapterOperand* output_operand = nullptr;
  if (has_out_scale) {
    output_operand =
        converter->AddQuant8VariableOperand(out_dims, out_scale, out_name);
  } else {
    output_operand = converter->AddVariableOperand(out_dims, out_name, otype);
  }

  // Cast operation
  std::vector<NNAdapterOperand*> input_operands = {input_operand,
                                                   dtype_operand};
  std::vector<NNAdapterOperand*> output_operands = {output_operand};
  auto cast_operation = converter->AddOperation(NNADAPTER_CAST);
  converter->SetOperation(cast_operation, &input_operands, &output_operands);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nnadapter
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(cast,
                         kNNAdapter,
                         paddle::lite::subgraph::nnadapter::CastConverter);
