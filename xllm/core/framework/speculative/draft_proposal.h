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

#include <glog/logging.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/speculative/draft_distribution.h"

namespace xllm {

// Rejection-sampling input: drafted tokens plus the draft distribution q
// for probabilistic drafts (greedy drafts carry tokens only, q = 1). -1
// sentinels in token_ids are force-rejected.
class DraftProposal final {
 public:
  DraftProposal() = default;

  explicit DraftProposal(torch::Tensor token_ids,
                         std::optional<torch::Tensor> logits = std::nullopt)
      : token_ids_(std::move(token_ids)),
        distribution_(logits.has_value() && logits->defined()
                          ? DraftDistribution(std::move(*logits))
                          : DraftDistribution()) {}

  const torch::Tensor& token_ids() const { return token_ids_; }
  const DraftDistribution& distribution() const { return distribution_; }
  torch::Tensor logits() const { return distribution_.dense_logits(); }

  static DraftProposal from_sparse(torch::Tensor token_ids,
                                   torch::Tensor candidate_ids,
                                   torch::Tensor candidate_logits);

  void set_token_ids(torch::Tensor token_ids) {
    CHECK(token_ids.defined());
    CHECK_EQ(token_ids.sizes(), token_ids_.sizes());
    CHECK_EQ(token_ids.device(), token_ids_.device());
    CHECK_EQ(token_ids.scalar_type(), token_ids_.scalar_type());
    token_ids_ = std::move(token_ids);
  }

  // Lifetime owner for no-sync draft inputs that must outlive the verify
  // step's stream sync; never read by the rejection sampler.
  std::vector<std::shared_ptr<const void>>& keep_alive() { return keep_alive_; }

  void validate(int64_t expected_batch_size,
                int64_t expected_num_speculative_tokens) const {
    CHECK(token_ids_.defined()) << "draft proposal token_ids must be defined";
    CHECK_EQ(token_ids_.dim(), 2)
        << "draft proposal token_ids must be [batch, n_spec]";
    CHECK_EQ(token_ids_.size(0), expected_batch_size)
        << "draft proposal batch size mismatch";
    CHECK_EQ(token_ids_.size(1), expected_num_speculative_tokens)
        << "draft proposal speculative length mismatch";

    distribution_.validate(token_ids_);
  }

  // Preallocates token_ids [batch, n_spec] plus the q buffer when
  // probabilistic; write via tokens_at()/logits_at().
  static DraftProposal preallocate(int64_t batch,
                                   int64_t num_speculative_tokens,
                                   int64_t vocab_size,
                                   bool is_probabilistic_draft,
                                   const torch::TensorOptions& token_options,
                                   const torch::TensorOptions& logits_options);

  static DraftProposal from_token_buffer(
      torch::Tensor token_ids,
      int64_t vocab_size,
      bool is_probabilistic_draft,
      const torch::TensorOptions& logits_options);

  torch::Tensor tokens_at(int64_t step) {
    return token_ids_.select(/*dim=*/1, /*index=*/step);
  }
  torch::Tensor logits_at(int64_t step) {
    torch::Tensor logits = distribution_.dense_logits();
    CHECK(logits.defined()) << "proposal carries no dense logits buffer";
    return logits.select(/*dim=*/1, /*index=*/step);
  }

 private:
  torch::Tensor token_ids_;
  DraftDistribution distribution_;
  std::vector<std::shared_ptr<const void>> keep_alive_;
};

}  // namespace xllm
