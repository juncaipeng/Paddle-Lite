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

#include <Eigen/Core>
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/relu_op.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename T>
class ReluCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::ActivationParam;

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    auto n = param.X->dims().production();
    const float* input = param.X->data<float>();
    float* output = param.Out->mutable_data<float>();
    for (int i = 0; i < n; i++) {
      output[i] = std::max(0.f, input[i]);
    }
  }

  virtual ~ReluCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
