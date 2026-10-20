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

#include <vector>

#include "core/framework/model/rec_model_params.h"

namespace xllm {
namespace detail {

inline bool rec_graph_tensor_matches(const torch::Tensor& captured,
                                     const torch::Tensor& current) {
  if (!captured.defined() || !current.defined()) {
    return captured.defined() == current.defined();
  }
  return captured.device() == current.device() &&
         captured.scalar_type() == current.scalar_type() &&
         captured.sizes() == current.sizes() &&
         captured.strides() == current.strides() &&
         captured.data_ptr() == current.data_ptr();
}

inline bool rec_graph_tensors_match(const std::vector<torch::Tensor>& captured,
                                    const std::vector<torch::Tensor>& current) {
  if (captured.size() != current.size()) {
    return false;
  }
  for (size_t i = 0; i < captured.size(); ++i) {
    if (!rec_graph_tensor_matches(captured[i], current[i])) {
      return false;
    }
  }
  return true;
}

inline bool rec_graph_strategy_matches(const OneRecModelInputParams& captured,
                                       const OneRecModelInputParams& current) {
  // seq_len and generated_tokens only drive worker-side decoding. Keep
  // is_first_prefill because it selects the model's cross-attention path.
  return captured.rec_stage == current.rec_stage &&
         captured.is_hybrid_mode == current.is_hybrid_mode &&
         captured.is_encoder_forward == current.is_encoder_forward &&
         captured.has_encoder_output == current.has_encoder_output &&
         captured.encoder_seq_lens == current.encoder_seq_lens &&
         captured.encoder_max_seq_len == current.encoder_max_seq_len &&
         captured.is_first_prefill == current.is_first_prefill &&
         captured.bs == current.bs &&
         captured.group_width == current.group_width &&
         captured.cross_attn_kv_cu_seq_lens_vec ==
             current.cross_attn_kv_cu_seq_lens_vec &&
         rec_graph_tensor_matches(captured.encoder_seq_lens_tensor,
                                  current.encoder_seq_lens_tensor) &&
         rec_graph_tensor_matches(captured.encoder_sparse_embedding,
                                  current.encoder_sparse_embedding) &&
         rec_graph_tensor_matches(captured.decoder_context_embedding,
                                  current.decoder_context_embedding) &&
         rec_graph_tensor_matches(captured.cross_attn_kv_cu_seq_lens,
                                  current.cross_attn_kv_cu_seq_lens) &&
         rec_graph_tensor_matches(captured.cross_attn_new_cache_slots,
                                  current.cross_attn_new_cache_slots) &&
         rec_graph_tensor_matches(captured.cross_attn_block_tables,
                                  current.cross_attn_block_tables) &&
         rec_graph_tensor_matches(captured.encoder_token_ids,
                                  current.encoder_token_ids) &&
         rec_graph_tensor_matches(captured.encoder_positions,
                                  current.encoder_positions);
}

inline bool rec_graph_strategy_matches(const OneRecXAttentionParams& captured,
                                       const OneRecXAttentionParams& current) {
  return rec_graph_strategy_matches(
             static_cast<const OneRecModelInputParams&>(captured),
             static_cast<const OneRecModelInputParams&>(current)) &&
         rec_graph_tensors_match(captured.unshared_k_caches,
                                 current.unshared_k_caches) &&
         rec_graph_tensors_match(captured.unshared_v_caches,
                                 current.unshared_v_caches) &&
         rec_graph_tensors_match(captured.shared_k_caches,
                                 current.shared_k_caches) &&
         rec_graph_tensors_match(captured.shared_v_caches,
                                 current.shared_v_caches) &&
         rec_graph_tensor_matches(captured.beam_width_tensor,
                                  current.beam_width_tensor) &&
         rec_graph_tensor_matches(captured.current_round_tensor,
                                  current.current_round_tensor) &&
         rec_graph_tensor_matches(captured.debug_selected_token_idxes,
                                  current.debug_selected_token_idxes) &&
         captured.debug_selected_token_idxes_expected ==
             current.debug_selected_token_idxes_expected;
}

inline bool rec_graph_strategy_matches(const LlmRecMultiRoundParams& captured,
                                       const LlmRecMultiRoundParams& current) {
  // The worker selects decode_positions_tensor_list into input.positions,
  // which replay copies into the graph's persistent position buffer.
  return captured.batch_size == current.batch_size &&
         captured.beam_width == current.beam_width &&
         captured.total_round == current.total_round &&
         rec_graph_tensors_match(captured.full_k_caches,
                                 current.full_k_caches) &&
         rec_graph_tensors_match(captured.full_v_caches,
                                 current.full_v_caches) &&
         rec_graph_tensors_match(captured.unshared_k_caches,
                                 current.unshared_k_caches) &&
         rec_graph_tensors_match(captured.unshared_v_caches,
                                 current.unshared_v_caches) &&
         rec_graph_tensors_match(captured.shared_k_caches,
                                 current.shared_k_caches) &&
         rec_graph_tensors_match(captured.shared_v_caches,
                                 current.shared_v_caches) &&
         rec_graph_tensor_matches(captured.beam_width_tensor,
                                  current.beam_width_tensor) &&
         rec_graph_tensor_matches(captured.current_round_tensor,
                                  current.current_round_tensor) &&
         rec_graph_tensor_matches(captured.two_stage_shared_lse,
                                  current.two_stage_shared_lse) &&
         rec_graph_tensor_matches(captured.two_stage_shared_o,
                                  current.two_stage_shared_o) &&
         rec_graph_tensor_matches(captured.two_stage_unshared_lse,
                                  current.two_stage_unshared_lse) &&
         rec_graph_tensor_matches(captured.two_stage_unshared_o,
                                  current.two_stage_unshared_o) &&
         rec_graph_tensor_matches(captured.two_stage_q_cu_seq_lens_shared,
                                  current.two_stage_q_cu_seq_lens_shared) &&
         rec_graph_tensor_matches(captured.two_stage_qo_indptr_expanded,
                                  current.two_stage_qo_indptr_expanded) &&
         rec_graph_tensor_matches(captured.two_stage_paged_kv_indptr_expanded,
                                  current.two_stage_paged_kv_indptr_expanded) &&
         rec_graph_tensor_matches(
             captured.two_stage_paged_kv_indices_expanded,
             current.two_stage_paged_kv_indices_expanded) &&
         rec_graph_tensor_matches(
             captured.two_stage_paged_kv_last_page_len_expanded,
             current.two_stage_paged_kv_last_page_len_expanded);
}

}  // namespace detail

// Graph replay writes into the captured KV addresses. Re-capture when those
// bindings change, so subsequent beam operations observe the same cache
// storage. Tensor contents (including round/beam scalars) can change in place.
inline bool rec_graph_bindings_match(const RecModelParams& captured,
                                     const RecModelParams& current) {
  if (captured.rec_params.index() != current.rec_params.index()) {
    return false;
  }
  if (const auto* params = captured.onerec_xattention_params()) {
    return detail::rec_graph_strategy_matches(
        *params, *current.onerec_xattention_params());
  }
  if (const auto* params = captured.onerec_params()) {
    return detail::rec_graph_strategy_matches(*params,
                                              *current.onerec_params());
  }
  if (const auto* params = captured.llmrec_params()) {
    return detail::rec_graph_strategy_matches(*params,
                                              *current.llmrec_params());
  }
  return true;
}

}  // namespace xllm
