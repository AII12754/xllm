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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/layers/npu_torch/qwen3_5_gated_delta_net.h"
#include "tests/core/layers/npu_torch/tests_utils.h"

namespace xllm::layer {
namespace {

const torch::Device kDevice(torch::kPrivateUse1, 0);
constexpr int64_t kHidden = 128;
constexpr int64_t kSlots = 8;
using WeightMap = std::unordered_map<std::string, torch::Tensor>;
WeightMap make_checkpoint() {
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  auto random = [&](std::vector<int64_t> shape) {
    return (torch::randn(shape) * 0.02).to(options);
  };
  return {{"in_proj_qkv.weight", random({10240, kHidden})},
          {"in_proj_z.weight", random({6144, kHidden})},
          {"in_proj_b.weight", random({48, kHidden})},
          {"in_proj_a.weight", random({48, kHidden})},
          {"conv1d.weight", random({10240, 1, 4})},
          {"out_proj.weight", random({kHidden, 6144})},
          {"norm.weight", torch::ones({128}, options)},
          {"dt_bias", torch::full({48}, -1.0f)},
          {"A_log", torch::zeros({48})}};
}

ModelArgs make_args(int64_t hidden = kHidden) {
  ModelArgs args;
  args.hidden_size() = hidden;
  args.linear_num_key_heads() = 16;
  args.linear_num_value_heads() = 48;
  args.linear_key_head_dim() = 128;
  args.linear_value_head_dim() = 128;
  args.linear_conv_kernel_dim() = 4;
  args.rms_norm_eps() = 1e-6;
  return args;
}
// Accumulate BF16 input/weight products on CPU in FP32 and round once to
// BF16, then run the unchanged production GDN and output path.
class GdnFp32ProjectionReference final : public Qwen3_5GatedDeltaNetImpl {
 public:
  using Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl;

 protected:
  std::optional<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
  project_split_inputs(const torch::Tensor& hidden_states,
                       const AttentionMetadata& metadata) override {
    const auto input = hidden_states.to(torch::kCPU, torch::kFloat32);
    const auto parameters = named_parameters();
    auto project = [&](const std::string& name) {
      auto weight = parameters[name].to(torch::kCPU, torch::kFloat32);
      auto projected = torch::nn::functional::linear(input, weight)
                           .to(hidden_states.device(), torch::kBFloat16);
      return reshape_projected_tokens_with_pad(metadata, projected);
    };
    auto qkv = project("in_proj_qkv.weight");
    auto z = project("in_proj_z.weight");
    auto b = project("in_proj_b.weight");
    auto a = project("in_proj_a.weight");
    z = z.view({z.size(0), z.size(1), b.size(2), 128});
    return std::make_tuple(qkv, z, b, a);
  }
};
using GdnFp32Reference = torch::nn::ModuleHolder<GdnFp32ProjectionReference>;

template <typename Layer = Qwen3_5GatedDeltaNet>
Layer make_layer(const ParallelArgs& parallel,
                 const WeightMap& weights,
                 bool packed,
                 const std::string& strategy) {
  auto& config = KernelConfig::get_instance();
  const KernelConfig saved = config;
  config.enable_qwen35_gdn_packed_prefill() = packed;
  config.qwen35_gdn_prefill_projection() = strategy;
  Layer layer(make_args(weights.at("in_proj_qkv.weight").size(1)),
              QuantArgs{},
              parallel,
              torch::TensorOptions().device(kDevice).dtype(torch::kBFloat16));
  config = saved;
  // Simulate checkpoint shards with QKV arriving after the other projections.
  WeightMap first = weights;
  first.erase("in_proj_qkv.weight");
  layer->load_state_dict(StateDict(first));
  layer->load_state_dict(
      StateDict({{"in_proj_qkv.weight", weights.at("in_proj_qkv.weight")}}));
  layer->verify_loaded_weights("");
  return layer;
}

AttentionMetadata make_metadata(const std::vector<int32_t>& lengths,
                                bool chunked = false,
                                bool decode = false) {
  AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_prefill = !chunked && !decode;
  metadata.is_chunked_prefill = chunked;
  metadata.q_seq_lens_vec = lengths;
  metadata.max_query_len = *std::max_element(lengths.begin(), lengths.end());
  std::vector<int32_t> cumulative{0};
  cumulative.reserve(lengths.size() + 1);
  for (int32_t length : lengths) {
    cumulative.emplace_back(cumulative.back() + length);
  }
  metadata.q_cu_seq_lens = torch::tensor(cumulative, torch::kInt32).to(kDevice);
  metadata.q_seq_lens = torch::tensor(lengths, torch::kInt32).to(kDevice);
  return metadata;
}

KVCache make_cache(int64_t tp, int64_t stride = 1) {
  const auto options = torch::TensorOptions().device(kDevice);
  return KVCache(LinearAttentionKVCacheTensors{
      torch::zeros({kSlots, stride + 2, 10240 / tp},
                   options.dtype(torch::kBFloat16)),
      torch::zeros({kSlots * stride, 48 / tp, 128, 128},
                   options.dtype(torch::kFloat32))});
}

void expect_equivalent(const torch::Tensor& actual,
                       const torch::Tensor& expected,
                       bool exact) {
  ASSERT_EQ(actual.sizes(), expected.sizes());
  auto result = actual.to(torch::kCPU, torch::kFloat32);
  auto reference = expected.to(torch::kCPU, torch::kFloat32);
  ASSERT_TRUE(torch::isfinite(result).all().item<bool>());
  ASSERT_GT(reference.abs().max().item<double>(), 0.0);
  if (exact) {
    EXPECT_TRUE(torch::equal(result, reference));
  } else {
    const double error =
        (result - reference).square().mean().sqrt().item<double>();
    const double scale = reference.square().mean().sqrt().item<double>();
    const double relative_rmse = error / std::max(scale, 1e-8);
    ::testing::Test::RecordProperty("last_relative_rmse", relative_rmse);
    ::testing::Test::RecordProperty(
        "last_max_absolute_error",
        (result - reference).abs().max().item<double>());
    auto outside_tolerance =
        (result - reference).abs() > (reference.abs() * 0.02 + 0.002);
    const double failure_fraction =
        outside_tolerance.to(torch::kFloat32).mean().item<double>();
    EXPECT_LE(relative_rmse, 0.01);
    EXPECT_EQ(failure_fraction, 0.0);
  }
}

class Qwen35GdnPrefillTest : public ::testing::TestWithParam<std::string> {};

TEST(Qwen35GdnPrefillDefaultsTest, MatchesExplicitGroupedPackedThroughDecode) {
  torch::InferenceMode inference;
  torch::manual_seed(20261025);
  const auto weights = make_checkpoint();
  for (const int32_t tp : {1, 2, 4}) {
    SCOPED_TRACE(tp);
    test::MockProcessGroup group(kDevice, tp - 1, tp);
    ParallelArgs parallel(tp - 1, tp, &group);
    parallel.tp_group_ = &group;
    parallel.single_rank_group_ = &group;
    auto& config = KernelConfig::get_instance();
    const KernelConfig saved = config;
    config = KernelConfig{};
    Qwen3_5GatedDeltaNet candidate(
        make_args(),
        QuantArgs{},
        parallel,
        torch::TensorOptions().device(kDevice).dtype(torch::kBFloat16));
    config = saved;
    candidate->load_state_dict(StateDict(weights));
    candidate->verify_loaded_weights("");
    auto reference = make_layer(parallel, weights, true, "grouped");
    auto reference_cache = make_cache(tp);
    auto candidate_cache = make_cache(tp);
    LlmModelParams params;
    params.embedding.linear_state_ids = {1, 3, 5};
    params.linear_state_validity_mask = {0, 0, 0};
    const std::vector<std::vector<int32_t>> steps = {
        {17, 33, 65}, {13, 7, 9}, {1, 1, 1}};
    for (size_t step = 0; step < steps.size(); ++step) {
      auto metadata = make_metadata(steps[step], step == 1, step == 2);
      const int64_t tokens =
          std::accumulate(steps[step].begin(), steps[step].end(), int64_t{0});
      auto input =
          (torch::randn({tokens, kHidden}) * 0.2).to(kDevice, torch::kBFloat16);
      const auto expected = reference->forward(
          input, metadata, reference_cache, ModelParamsView(params));
      const auto actual = candidate->forward(
          input, metadata, candidate_cache, ModelParamsView(params));
      expect_equivalent(actual, expected, true);
      expect_equivalent(candidate_cache.get_conv_cache(),
                        reference_cache.get_conv_cache(),
                        true);
      expect_equivalent(candidate_cache.get_ssm_cache(),
                        reference_cache.get_ssm_cache(),
                        true);
      params.linear_state_validity_mask = {1, 1, 1};
    }
  }
}

TEST_P(Qwen35GdnPrefillTest, PreservesColdWarmRaggedAndDecodeOutputsAndCaches) {
  torch::InferenceMode inference;
  torch::manual_seed(20261024);
  const auto weights = make_checkpoint();
  for (const bool packed : {false, true}) {
    SCOPED_TRACE(packed);
    for (const int32_t tp : {1, 2, 4, 8}) {
      SCOPED_TRACE(tp);
      test::MockProcessGroup group(kDevice, tp - 1, tp);
      ParallelArgs parallel(tp - 1, tp, &group);
      parallel.tp_group_ = &group;
      parallel.single_rank_group_ = &group;
      auto reference = make_layer(parallel, weights, false, "separate");
      auto candidate = make_layer(parallel, weights, packed, GetParam());
      auto reference_cache = make_cache(tp);
      auto candidate_cache = make_cache(tp);
      LlmModelParams params;
      params.embedding.linear_state_ids = {1, 3, 5};
      params.linear_state_validity_mask = {0, 0, 0};
      const std::vector<std::vector<int32_t>> steps = {
          {17, 33, 65}, {13, 7, 9}, {1, 1, 1}};
      for (size_t step = 0; step < steps.size(); ++step) {
        SCOPED_TRACE(step);
        auto metadata = make_metadata(steps[step], step == 1, step == 2);
        const int64_t tokens =
            std::accumulate(steps[step].begin(), steps[step].end(), int64_t{0});
        auto input = (torch::randn({tokens, kHidden}) * 0.2)
                         .to(kDevice, torch::kBFloat16);
        auto expected = reference->forward(
            input, metadata, reference_cache, ModelParamsView(params));
        auto actual = candidate->forward(
            input, metadata, candidate_cache, ModelParamsView(params));
        const bool exact = GetParam() == "separate";
        expect_equivalent(actual, expected, exact);
        expect_equivalent(candidate_cache.get_conv_cache(),
                          reference_cache.get_conv_cache(),
                          exact);
        expect_equivalent(candidate_cache.get_ssm_cache(),
                          reference_cache.get_ssm_cache(),
                          exact);
        params.linear_state_validity_mask = {1, 1, 1};
      }
    }
  }
}

TEST_P(Qwen35GdnPrefillTest,
       PreservesSingleAndEqualLengthPrefillWithEachLayoutSwitch) {
  torch::InferenceMode inference;
  torch::manual_seed(871);
  const auto weights = make_checkpoint();
  test::MockProcessGroup group(kDevice);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  const std::vector<std::vector<int32_t>> batches = {{65}, {17, 17, 17, 17}};
  for (const bool packed : {false, true}) {
    SCOPED_TRACE(packed);
    auto reference = make_layer(parallel, weights, false, "separate");
    auto candidate = make_layer(parallel, weights, packed, GetParam());
    for (const auto& lengths : batches) {
      auto reference_cache = make_cache(1);
      auto candidate_cache = make_cache(1);
      auto metadata = make_metadata(lengths);
      const int64_t tokens =
          std::accumulate(lengths.begin(), lengths.end(), int64_t{0});
      auto input =
          (torch::randn({tokens, kHidden}) * 0.2).to(kDevice, torch::kBFloat16);
      LlmModelParams params;
      for (size_t i = 0; i < lengths.size(); ++i) {
        params.embedding.linear_state_ids.emplace_back(
            static_cast<int32_t>(i + 1));
      }
      params.linear_state_validity_mask.resize(lengths.size(), 0);
      auto expected = reference->forward(
          input, metadata, reference_cache, ModelParamsView(params));
      auto actual = candidate->forward(
          input, metadata, candidate_cache, ModelParamsView(params));
      const bool exact = GetParam() == "separate";
      expect_equivalent(actual, expected, exact);
      expect_equivalent(candidate_cache.get_conv_cache(),
                        reference_cache.get_conv_cache(),
                        exact);
      expect_equivalent(candidate_cache.get_ssm_cache(),
                        reference_cache.get_ssm_cache(),
                        exact);
    }
  }
}

TEST_P(Qwen35GdnPrefillTest, PreservesCheckpointRowsAcrossEveryTpRank) {
  torch::InferenceMode inference;
  torch::manual_seed(713);
  const auto weights = make_checkpoint();
  for (const int32_t tp : {1, 2, 4, 8}) {
    for (int32_t rank = 0; rank < tp; ++rank) {
      SCOPED_TRACE(tp);
      SCOPED_TRACE(rank);
      test::MockProcessGroup group(kDevice, rank, tp);
      ParallelArgs parallel(rank, tp, &group);
      parallel.tp_group_ = &group;
      auto candidate = make_layer(parallel, weights, true, GetParam());
      const auto parameters = candidate->named_parameters();
      std::vector<torch::Tensor> qkv_parts;
      qkv_parts.reserve(3);
      int64_t offset = 0;
      for (const int64_t width : {2048, 2048, 6144}) {
        const int64_t local_width = width / tp;
        qkv_parts.emplace_back(weights.at("in_proj_qkv.weight")
                                   .slice(0,
                                          offset + rank * local_width,
                                          offset + (rank + 1) * local_width));
        offset += width;
      }
      EXPECT_TRUE(torch::equal(parameters["in_proj_qkv.weight"].cpu(),
                               torch::cat(qkv_parts, 0)));
      for (const std::string name :
           {"in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight"}) {
        const auto& global = weights.at(name);
        const int64_t width = global.size(0) / tp;
        EXPECT_TRUE(
            torch::equal(parameters[name].cpu(),
                         global.slice(0, rank * width, (rank + 1) * width)));
        EXPECT_TRUE(parameters[name].is_contiguous());
      }
    }
  }
}

TEST_P(Qwen35GdnPrefillTest, LeavesRaggedSpeculativeVerificationUnchanged) {
  torch::InferenceMode inference;
  torch::manual_seed(516);
  const auto weights = make_checkpoint();
  test::MockProcessGroup group(kDevice);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  parallel.single_rank_group_ = &group;
  auto reference = make_layer(parallel, weights, false, "separate");
  auto candidate = make_layer(parallel, weights, true, GetParam());
  auto reference_cache = make_cache(1, 2);
  auto candidate_cache = make_cache(1, 2);
  LlmModelParams params;
  params.embedding.linear_state_ids = {1, 3, 5};
  params.linear_state_validity_mask = {1, 1, 1};
  params.is_spec_verify = true;
  params.num_accepted_tokens_host = {1, 2, 1};
  auto metadata = make_metadata({2, 1, 2}, true);
  auto input = (torch::randn({5, kHidden}) * 0.2).to(kDevice, torch::kBFloat16);
  auto expected = reference->forward(
      input, metadata, reference_cache, ModelParamsView(params));
  auto actual = candidate->forward(
      input, metadata, candidate_cache, ModelParamsView(params));
  expect_equivalent(actual, expected, true);
  expect_equivalent(
      candidate_cache.get_conv_cache(), reference_cache.get_conv_cache(), true);
  expect_equivalent(
      candidate_cache.get_ssm_cache(), reference_cache.get_ssm_cache(), true);
}

TEST_P(Qwen35GdnPrefillTest, MatchesFp32ProjectionOracle) {
  torch::InferenceMode inference;
  torch::manual_seed(20261025);
  const auto weights = make_checkpoint();
  test::MockProcessGroup group(kDevice);
  ParallelArgs parallel(0, 1, &group);
  parallel.tp_group_ = &group;
  auto oracle =
      make_layer<GdnFp32Reference>(parallel, weights, false, "separate");
  auto candidate = make_layer(parallel, weights, true, GetParam());
  auto oracle_cache = make_cache(1);
  auto candidate_cache = make_cache(1);
  LlmModelParams params;
  params.embedding.linear_state_ids = {1, 3};
  params.linear_state_validity_mask = {0, 0};
  for (const bool warm : {false, true}) {
    auto metadata = make_metadata({17, 33}, warm);
    auto input =
        (torch::randn({50, kHidden}) * 0.2).to(kDevice, torch::kBFloat16);
    const auto expected =
        oracle->forward(input, metadata, oracle_cache, ModelParamsView(params));
    const auto actual = candidate->forward(
        input, metadata, candidate_cache, ModelParamsView(params));
    expect_equivalent(actual, expected, false);
    expect_equivalent(
        candidate_cache.get_conv_cache(), oracle_cache.get_conv_cache(), false);
    expect_equivalent(
        candidate_cache.get_ssm_cache(), oracle_cache.get_ssm_cache(), false);
    params.linear_state_validity_mask = {1, 1};
  }
}

INSTANTIATE_TEST_SUITE_P(Strategies,
                         Qwen35GdnPrefillTest,
                         ::testing::Values("separate", "grouped"));

}  // namespace
}  // namespace xllm::layer
