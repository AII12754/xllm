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

#include <atomic>
#include <future>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "core/kv_cache/events/kv_cache_event_journal.h"

namespace xllm {

// vLLM's three-frame msgpack wire protocol. The optional pool socket confirms
// residency of xLLM's local Host copies to Motor; it is not a KV data store.
class KVCacheEventPublisher final {
 public:
  KVCacheEventPublisher(std::shared_ptr<KVCacheEventJournal> journal,
                        const nlohmann::json& config,
                        const std::string& model_id,
                        int32_t dp_size);
  ~KVCacheEventPublisher();
  KVCacheEventPublisher(const KVCacheEventPublisher&) = delete;
  KVCacheEventPublisher& operator=(const KVCacheEventPublisher&) = delete;
  bool healthy() const;

  static uint64_t wire_hash(const std::string& hash);
  static nlohmann::json engine_event(const nlohmann::json& event);
  static nlohmann::json pool_event(const nlohmann::json& event);
  static std::string rank_endpoint(const std::string& endpoint, int32_t rank);

 private:
  void run(const nlohmann::json& config,
           const std::string& model_id,
           int32_t dp_size,
           const std::shared_ptr<std::promise<void>>& ready);
  bool run_impl(const nlohmann::json& config,
                const std::string& model_id,
                int32_t dp_size,
                const std::shared_ptr<std::promise<void>>& ready,
                bool& ready_sent);

  std::shared_ptr<KVCacheEventJournal> journal_;
  std::atomic<bool> stopped_{false};
  std::atomic<bool> healthy_{false};
  std::thread thread_;
};

}  // namespace xllm
