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

#include <torch/types.h>

#include <variant>

namespace xllm {

// Value-semantic strategy for q. Representation details stay outside the
// sampler. Normalizers belong to one verify call: producers populate logits
// after creation.
class DraftDistribution final {
 public:
  DraftDistribution() = default;
  explicit DraftDistribution(torch::Tensor logits);
  static DraftDistribution from_sparse(torch::Tensor candidate_ids,
                                       torch::Tensor logits);

  // FP32 [batch, steps] normalizers, undefined for a deterministic draft.
  torch::Tensor logsumexp() const;
  // Subtract normalized log q(token_ids) from log_p in place.
  void subtract_log_prob(torch::Tensor& log_p,
                         const torch::Tensor& token_ids,
                         const torch::Tensor& lse) const;
  // Replace log_p [batch, vocab] with log((p - q)+) at the selected steps.
  // A negative rejected token removes q entirely, recovering directly from p.
  void apply_residual(torch::Tensor& log_p,
                      const torch::Tensor& steps,
                      const torch::Tensor& rejected_tokens,
                      const torch::Tensor& lse) const;

  void validate(const torch::Tensor& token_ids) const;
  // Producer view for preallocated dense buffers; undefined for other forms.
  torch::Tensor dense_logits() const;

 private:
  struct Deterministic {};
  struct Dense {
    torch::Tensor logits;
  };
  struct Sparse {
    torch::Tensor token_ids;
    torch::Tensor logits;
  };
  explicit DraftDistribution(Sparse sparse);
  std::variant<Deterministic, Dense, Sparse> storage_;
};

}  // namespace xllm
