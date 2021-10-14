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

#include "core/operation/conv2d.h"
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertConv2D(Converter* converter, hal::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // Check depthwise mode, and decide whether use ConvolutionDepthwise
  bool use_depthwise_conv = false;
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.data[0],
                     filter_width);
                     // output_channel_size);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  if (use_depthwise_conv && is_depthwise_mode) {
    NNADAPTER_VLOG(5) << " DepthwiseConv is unimpleted.";
    return NNADAPTER_DEVICE_INTERNAL_ERROR;
  } else {
    auto conv_node =
        converter->network()->AddIConvNode(input_tensor, filter_tensor, bias_tensor);
    if (conv_node == nullptr) {
      NNADAPTER_VLOG(5) << "Failed to add convolution node.";
      return NNADAPTER_DEVICE_INTERNAL_ERROR;
    }
    auto pre_h = pads_buffer[0];
    auto post_h = pads_buffer[1];
    auto pre_w = pads_buffer[2];
    auto post_w = pads_buffer[3];
    conv_node->SetPad(static_cast<int64_t>(pre_h), static_cast<int64_t>(post_h),
                      static_cast<int64_t>(pre_w), static_cast<int64_t>(post_w));
    conv_node->SetStride(static_cast<int64_t>(stride_height), static_cast<int64_t>(stride_width));
    conv_node->SetDilation(static_cast<int64_t>(dilation_height), static_cast<int64_t>(dilation_width));
    conv_node->SetGroup(static_cast<int64_t>(group));
    magicmind::Layout input_layout = ConvertToMagicMindDataLayout(input_operand->type.layout);
    conv_node->SetLayout(input_layout, magicmind::Layout::HWCN, input_layout);
    auto output_tensor = conv_node->GetOutput(0);
    // fuse activations
    switch (fuse_code) {
      case NNADAPTER_FUSED_RELU:
        {
          auto activation_node = converter->network()->AddIActivationNode(output_tensor, magicmind::IActivation::RELU);
          auto fuse_out_tensor = activation_node->GetOutput(0);
          converter->UpdateTensorMap(output_operand, fuse_out_tensor);
          break;
        }
      case NNADAPTER_FUSED_RELU6:
        {
          auto activation_node = converter->network()->AddIActivationNode(output_tensor, magicmind::IActivation::RELU6);
          auto fuse_out_tensor = activation_node->GetOutput(0);
          converter->UpdateTensorMap(output_operand, fuse_out_tensor);
          break;
        }
      case NNADAPTER_FUSED_NONE:
        converter->UpdateTensorMap(output_operand, output_tensor);
        break;
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
