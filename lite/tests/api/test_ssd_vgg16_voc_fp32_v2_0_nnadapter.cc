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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/test/lite_api_test_helper.h"
#include "lite/api/test/test_helper.h"
#include "lite/tests/api/PASCALVOC2012_utility.h"
#include "lite/tests/api/detection_model_utility.h"

DEFINE_string(data_dir, "", "data dir");
DEFINE_int32(iteration, 1, "iteration times to run");
DEFINE_int32(batch, 1, "batch of image");
DEFINE_int32(channel, 3, "image channel");
DEFINE_int32(height, 300, "image height");
DEFINE_int32(width, 300, "image width");
DEFINE_int32(model_version, 2, "model version");

namespace paddle {
namespace lite {

TEST(SSD_VGG16_VOC_V2, test_ssd_vgg16_voc_fp32_v2_0_nnadapter) {
  std::vector<std::string> nnadapter_device_names;
  std::string nnadapter_context_properties;
  std::vector<paddle::lite_api::Place> valid_places;
  valid_places.push_back(
      lite_api::Place{TARGET(kNNAdapter), PRECISION(kFloat)});
#if defined(LITE_WITH_ARM)
  valid_places.push_back(lite_api::Place{TARGET(kARM), PRECISION(kFloat)});
#elif defined(LITE_WITH_X86)
  valid_places.push_back(lite_api::Place{TARGET(kX86), PRECISION(kFloat)});
#else
  LOG(INFO) << "Unsupported host arch!";
  return;
#endif
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  nnadapter_device_names.emplace_back("huawei_ascend_npu");
  nnadapter_context_properties = "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0";
#else
  LOG(INFO) << "Unsupported NNAdapter device!";
  return;
#endif

  TestSSDDetectionModel(FLAGS_model_dir,
                        FLAGS_data_dir,
                        FLAGS_batch,
                        FLAGS_channel,
                        FLAGS_height,
                        FLAGS_width,
                        FLAGS_iteration,
                        FLAGS_model_version,
                        nnadapter_device_names,
                        nnadapter_context_properties,
                        valid_places);
}

}  // namespace lite
}  // namespace paddle
