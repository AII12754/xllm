/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "core/framework/model/rec_model.h"

namespace xllm {

// Recommendation models built on decoder-only language models.
class CausalRec : public RecModel {
 public:
  ~CausalRec() override = default;

  RecModelKind model_kind() const override final {
    return RecModelKind::kLlmRec;
  }
};

template <typename ModelType>
class CausalRecImpl final : public detail::RecModelImpl<ModelType, CausalRec> {
 public:
  using detail::RecModelImpl<ModelType, CausalRec>::RecModelImpl;
};

}  // namespace xllm
