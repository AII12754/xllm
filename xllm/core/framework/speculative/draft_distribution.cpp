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

#include "core/framework/speculative/draft_distribution.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <limits>
#include <type_traits>
#include <utility>

#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

void validate_logits(const torch::Tensor& logits) {
  CHECK(logits.defined());
  CHECK_EQ(logits.dim(), 3) << "draft logits must be [batch, steps, support]";
  CHECK_GT(logits.size(-1), 0);
  CHECK(logits.is_floating_point());
}

void subtract_log_mass(torch::Tensor& log_q, const torch::Tensor& log_p) {
  log_q.sub_(log_p)
      .nan_to_num_(/*nan=*/0.0)
      .clamp_max_(0)
      .exp_()
      .neg_()
      .log1p_();
}

}  // namespace

DraftDistribution::DraftDistribution(torch::Tensor logits)
    : storage_(Dense{std::move(logits)}) {
  validate_logits(std::get<Dense>(storage_).logits);
}

DraftDistribution::DraftDistribution(Sparse sparse)
    : storage_(std::move(sparse)) {
  const Sparse& values = std::get<Sparse>(storage_);
  validate_logits(values.logits);
  CHECK(values.token_ids.defined());
  CHECK_EQ(values.token_ids.sizes(), values.logits.sizes());
  CHECK_EQ(values.token_ids.scalar_type(), torch::kInt64);
  CHECK_EQ(values.token_ids.device(), values.logits.device());
}

DraftDistribution DraftDistribution::from_sparse(torch::Tensor candidate_ids,
                                                 torch::Tensor logits) {
  // IDs are unique per support row; producers retain storage through verify.
  return DraftDistribution(Sparse{std::move(candidate_ids), std::move(logits)});
}

void DraftDistribution::validate(const torch::Tensor& token_ids) const {
  CHECK(token_ids.defined());
  CHECK_EQ(token_ids.dim(), 2);
  std::visit(
      [&](const auto& values) {
        using Representation = std::decay_t<decltype(values)>;
        if constexpr (!std::is_same_v<Representation, Deterministic>) {
          CHECK_EQ(values.logits.size(0), token_ids.size(0));
          CHECK_EQ(values.logits.size(1), token_ids.size(1));
          CHECK_EQ(values.logits.device(), token_ids.device());
        }
      },
      storage_);
}

torch::Tensor DraftDistribution::dense_logits() const {
  const Dense* values = std::get_if<Dense>(&storage_);
  return values == nullptr ? torch::Tensor() : values->logits;
}

torch::Tensor DraftDistribution::logsumexp() const {
  return std::visit(
      [](const auto& values) -> torch::Tensor {
        using Representation = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<Representation, Deterministic>) {
          return {};
        } else {
          return torch::logsumexp(values.logits.to(torch::kFloat32),
                                  /*dim=*/-1);
        }
      },
      storage_);
}

void DraftDistribution::subtract_log_prob(torch::Tensor& log_p,
                                          const torch::Tensor& token_ids,
                                          const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& values) {
        using Representation = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<Representation, Dense>) {
          log_p.sub_(index_select_2d(values.logits, /*dim=*/-1, token_ids)
                         .to(torch::kFloat32)
                         .sub(lse));
        } else if constexpr (std::is_same_v<Representation, Sparse>) {
          const torch::Tensor matches =
              values.token_ids.eq(token_ids.unsqueeze(-1));
          const torch::Tensor selected =
              values.logits.to(torch::kFloat32)
                  .masked_fill(matches.logical_not(),
                               -std::numeric_limits<float>::infinity());
          log_p.sub_(torch::logsumexp(selected, /*dim=*/-1).sub(lse));
        }
        // A deterministic draft has q(draft_token) = 1, so log q = 0.
      },
      storage_);
}

void DraftDistribution::apply_residual(torch::Tensor& log_p,
                                       const torch::Tensor& steps,
                                       const torch::Tensor& rejected_tokens,
                                       const torch::Tensor& lse) const {
  std::visit(
      [&](const auto& values) {
        using Representation = std::decay_t<decltype(values)>;
        if constexpr (std::is_same_v<Representation, Deterministic>) {
          const torch::Tensor ids = rejected_tokens.clamp_min(0);
          const torch::Tensor remaining = log_p.gather(/*dim=*/-1, ids);
          remaining.masked_fill_(rejected_tokens.ge(0),
                                 -std::numeric_limits<float>::infinity());
          log_p.scatter_(/*dim=*/-1, ids, remaining);
        } else {
          const torch::Tensor rows =
              steps.expand({log_p.size(0), values.logits.size(-1)});
          torch::Tensor ids;
          if constexpr (std::is_same_v<Representation, Sparse>) {
            ids = index_select_2d(values.token_ids, /*dim=*/1, rows);
          }
          torch::Tensor log_q = index_select_2d(values.logits, /*dim=*/1, rows)
                                    .to(torch::kFloat32)
                                    .sub_(lse.gather(/*dim=*/1, steps));
          log_q.masked_fill_(rejected_tokens.lt(0),
                             -std::numeric_limits<float>::infinity());
          if constexpr (std::is_same_v<Representation, Dense>) {
            subtract_log_mass(log_q, log_p);
            log_p.add_(log_q);
          } else {
            const torch::Tensor support_log_p = log_p.gather(/*dim=*/-1, ids);
            subtract_log_mass(log_q, support_log_p);
            log_q.add_(support_log_p);
            // Only q support is corrected. Outside it the residual remains p.
            log_p.scatter_(/*dim=*/-1, ids, log_q);
          }
        }
      },
      storage_);
}

}  // namespace xllm
