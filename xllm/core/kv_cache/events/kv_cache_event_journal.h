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
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace xllm {

struct KVCacheEventScope {
  std::string model_id;
  int32_t dp_rank = 0;
  std::string medium = "gpu";
};

// The cache owns residency; exporters own transport and protocol conversion.
// Producers only enqueue native records. The consumer owns residency, JSON
// conversion and snapshots, and never holds the producer lock while doing them.
class KVCacheEventJournal final {
 public:
  explicit KVCacheEventJournal(size_t capacity = 4096,
                               size_t pending_capacity = 65536);
  static std::shared_ptr<KVCacheEventJournal> get_instance();
  static std::string encode_hash(const uint8_t* hash);
  bool enabled() const;
  void disable();

  void store(KVCacheEventScope scope,
             std::string hash,
             std::string parent_hash,
             std::vector<int32_t> tokens,
             uint32_t block_index);
  void store_copy(KVCacheEventScope scope, std::string hash);
  void remove(KVCacheEventScope scope, std::string hash);
  // Filter before copying residency data; cursors still cover all updates.
  nlohmann::json read(const std::string& epoch,
                      uint64_t cursor,
                      size_t limit = 1024,
                      const std::string& medium = "");

 private:
  using Key = std::tuple<std::string, int32_t, std::string>;
  struct Entry {
    nlohmann::json data;
    std::set<std::string> media;
  };
  enum class Operation : uint8_t { STORE, COPY, REMOVE };
  struct Update {
    Operation operation;
    KVCacheEventScope scope;
    std::string hash;
    std::string parent_hash;
    std::vector<int32_t> tokens;
    uint32_t block_index = 0;
  };

  void enqueue(Update update);
  bool apply(Update update);
  void append(nlohmann::json event);

  const size_t capacity_;
  const size_t pending_capacity_;
  std::atomic<bool> enabled_{true};
  std::mutex producer_mutex_;
  std::vector<Update> pending_;
  std::mutex consumer_mutex_;
  std::vector<Update> draining_;
  bool invalidated_ = false;
  std::string epoch_;
  uint64_t sequence_ = 0;
  std::map<Key, Entry> residents_;
  std::deque<nlohmann::json> events_;
};

}  // namespace xllm
