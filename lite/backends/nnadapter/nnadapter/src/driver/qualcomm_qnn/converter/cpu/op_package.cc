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

#include <cstring>
#include <unordered_map>

#include "CPU/QnnCpuOpPackage.h"
#include "QnnSdkBuildId.h"

#include "driver/qualcomm_qnn/converter/cpu/op_package.h"
#include "driver/qualcomm_qnn/converter/cpu/relu.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

std::mutex CpuOpPackage::s_mtx;
std::shared_ptr<CpuOpPackage> CpuOpPackage::s_op_package;
bool CpuOpPackage::s_is_initialized;

static Qnn_ErrorHandle_t OpImplFunc(void* node_data) {
  auto op_pkg = CpuOpPackage::GetInstance();
  if (!op_pkg) {
    return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  }

  QNN_CHECK_STATUS(op_pkg->ExecuteNode(node_data));

  return QNN_SUCCESS;
}

bool CpuOpPackage::GetIsInitialized() {
  std::lock_guard<std::mutex> locker(s_mtx);
  return s_is_initialized;
}

void CpuOpPackage::DestroyInstance() {
  SetIsInitialized(false);
  s_op_package.reset();
}

void CpuOpPackage::SetIsInitialized(bool is_initialized) {
  std::lock_guard<std::mutex> locker(s_mtx);
  s_is_initialized = is_initialized;
}

std::shared_ptr<CpuOpPackage> CpuOpPackage::GetInstance() {
  std::lock_guard<std::mutex> locker(s_mtx);
  if (!s_op_package) {
    s_op_package.reset(new (std::nothrow) CpuOpPackage());
  }

  return s_op_package;
}

Qnn_ErrorHandle_t CpuOpPackage::SetPackageInfo(const char* package_name) {
  op_names_ = {"Relu"};
  package_name_.assign(package_name);
  sdk_api_version_ = QNN_CPU_API_VERSION_INIT;
  package_info_ = {package_name_.c_str(),
                   op_names_.data(),                         // Operations
                   nullptr,                                  // Operation info
                   static_cast<uint32_t>(op_names_.size()),  // Num Operations
                   nullptr,                                  // Optimizations
                   0,                                        // numOptimizations
                   QNN_SDK_BUILD_ID,                         // sdkBuildId
                   &sdk_api_version_,                        // sdkApiVersion
                   nullptr,                                  // packageInfo
                   {0}};                                     // reserved

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CpuOpPackage::GetPackageInfo(
    const QnnOpPackage_Info_t** info) {
  *info = &package_info_;

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CpuOpPackage::CreateOpImpl(
    QnnCpuOpPackage_GraphInfrastructure_t* graph_infrastructure,
    QnnCpuOpPackage_Node_t* node,
    QnnCpuOpPackage_OpImpl_t** op_impl_ptr) {
  std::shared_ptr<OpBase> op;

  std::string node_type(node->typeName);
  if (node_type == "Relu") {
    op = std::make_shared<Relu>(node);
    QNN_CHECK_STATUS(op->SetOpNode(node));
  } else {
    return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  }

  // Finalize
  QNN_CHECK_STATUS(op->Finalize());

  // Update op reference
  auto op_impl = std::make_shared<QnnCpuOpPackage_OpImpl_t>();
  op_impl->opImplFn = OpImplFunc;
  op_impl->userData = reinterpret_cast<void*>(GetHandle(op));

  // Update out opImpl param
  *op_impl_ptr = op_impl.get();

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CpuOpPackage::ExecuteNode(void* kernelHandle) {
  auto op = GetObject((size_t)kernelHandle);
  QNN_CHECK_STATUS(op->Execute());

  return QNN_SUCCESS;
}

Qnn_ErrorHandle_t CpuOpPackage::FreeOpImpl(QnnCpuOpPackage_OpImpl_t* op_impl) {
  return (RemoveObject((size_t)op_impl->userData));
}

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
