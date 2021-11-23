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
#include <string>
#include <vector>
#include "core/hal/types.h"

namespace nnadapter {
namespace optimizer {

class NetOptimizer {
 public:
  // Some appoint here, one pass should be only one of the following kinds.
  enum class Kind {
    // Will modify the entire graph topology. e.g. data_format, eliminate,
    // constant folding.
    kGraphPass = 0,
    // Will fuse the operation. e.g. matmul+add->fullyconnected.
    kFusionPass,
    // Will not modify the NNAdapter IR, just collect information or
    // visualization.
    kDebugPass,
  };

  explicit NetOptimizer(Kind kind) : kind_(kind) {}

  void set_name(const std::string& name) { name_ = name; }

  const std::string& name() const { return name_; }

  Kind kind() const { return kind_; }
  bool is_graph_pass() const { return kind_ == Kind::kGraphPass; }
  bool is_fuse_pass() const { return kind_ == Kind::kFusePass; }
  bool is_debug_pass() const { return kind_ == Kind::kDebugPass; }

  virtual void Apply(const hal::Model* model) = 0;

  virtual bool isEnable(const char* properties) = 0;

  virtual ~NetOptimizer() = default;

 private:
  const Kind kind_;
  std::string name_;
};

// Different kinds.
class GraphPass : public NetOptimizer {
 public:
  GraphPass() : NetOptimizer(Kind::kGraphPass) {}

  ~GraphPass() override = default;

  void Apply(const hal::Model* model) override = 0;
};

class FusionPass : public NetOptimizer {
 public:
  FusionPass() : NetOptimizer(Kind::FusionPass) {}

  ~FusionPass() override = default;

  virtual void BuildPattern() = 0;
  virtual bool MatchPatterns() = 0;
  void Apply(const hal::Model* model) override = 0;
};

class DebugPass : public NetOptimizer {
 public:
  DebugPass() : NetOptimizer(Kind::kDebug) {}

  ~DebugPass() override = default;
};

}  // namespace optimizer
}  // namespace nnadapter
