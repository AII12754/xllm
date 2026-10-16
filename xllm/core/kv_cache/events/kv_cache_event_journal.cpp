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

#include "core/kv_cache/events/kv_cache_event_journal.h"

#include <glog/logging.h>

#include <algorithm>
#include <random>
#include <utility>

namespace xllm {
namespace {
std::string create_epoch() {
  std::random_device random;
  return std::to_string(random()) + "-" + std::to_string(random()) + "-" +
         std::to_string(random()) + "-" + std::to_string(random());
}
}  // namespace

KVCacheEventJournal::KVCacheEventJournal(size_t capacity,
                                         size_t pending_capacity)
    : capacity_(capacity),
      pending_capacity_(pending_capacity),
      epoch_(create_epoch()) {
  CHECK_GT(capacity_, 0);
  CHECK_GT(pending_capacity_, 0);
  pending_.reserve(pending_capacity_);
  draining_.reserve(pending_capacity_);
}

std::shared_ptr<KVCacheEventJournal> KVCacheEventJournal::get_instance() {
  static const auto journal = std::make_shared<KVCacheEventJournal>();
  return journal;
}

std::string KVCacheEventJournal::encode_hash(const uint8_t* hash) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result(32, '0');
  for (size_t i = 0; i < 16; ++i) {
    result[i * 2] = kHex[hash[i] >> 4];
    result[i * 2 + 1] = kHex[hash[i] & 15];
  }
  return result;
}

void KVCacheEventJournal::append(nlohmann::json event) {
  event["sequence"] = ++sequence_;
  events_.emplace_back(std::move(event));
  if (events_.size() > capacity_) {
    events_.pop_front();
  }
}

bool KVCacheEventJournal::enabled() const {
  return enabled_.load(std::memory_order_relaxed);
}

void KVCacheEventJournal::disable() {
  enabled_.store(false, std::memory_order_relaxed);
}

void KVCacheEventJournal::enqueue(Update update) {
  if (!enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(producer_mutex_);
  if (!enabled()) {
    return;
  }
  if (pending_.size() == pending_capacity_) {
    // Losing a removal must not leave a stale residency claim. Stop collecting
    // until engine restart; the consumer publishes an empty directory reset.
    disable();
    return;
  }
  pending_.emplace_back(std::move(update));
}

void KVCacheEventJournal::store(KVCacheEventScope scope,
                                std::string hash,
                                std::string parent_hash,
                                std::vector<int32_t> tokens,
                                uint32_t block_index) {
  enqueue({Operation::STORE,
           std::move(scope),
           std::move(hash),
           std::move(parent_hash),
           std::move(tokens),
           block_index});
}

void KVCacheEventJournal::store_copy(KVCacheEventScope scope,
                                     std::string hash) {
  enqueue({Operation::COPY, std::move(scope), std::move(hash), {}, {}, 0});
}

void KVCacheEventJournal::remove(KVCacheEventScope scope, std::string hash) {
  enqueue({Operation::REMOVE, std::move(scope), std::move(hash), {}, {}, 0});
}

bool KVCacheEventJournal::apply(Update update) {
  const Key key{update.scope.model_id, update.scope.dp_rank, update.hash};
  if (update.operation == Operation::REMOVE) {
    auto iter = residents_.find(key);
    if (iter == residents_.end() ||
        !iter->second.media.erase(update.scope.medium)) {
      return true;
    }
    nlohmann::json event = iter->second.data;
    event["type"] = "removed";
    event["medium"] = update.scope.medium;
    append(std::move(event));
    if (iter->second.media.empty()) {
      residents_.erase(iter);
    }
    return true;
  }
  if (update.operation == Operation::COPY) {
    auto iter = residents_.find(key);
    if (iter == residents_.end()) {
      LOG(ERROR) << "KV event collection disabled: copy has no source metadata";
      return false;
    }
    if (iter->second.media.insert(update.scope.medium).second) {
      nlohmann::json event = iter->second.data;
      event["medium"] = update.scope.medium;
      append(std::move(event));
    }
    return true;
  }
  if (update.tokens.empty()) {
    LOG(ERROR) << "KV event collection disabled: store has no tokens";
    return false;
  }
  if (update.hash.size() != 32 ||
      (!update.parent_hash.empty() && update.parent_hash.size() != 32)) {
    LOG(ERROR) << "KV event collection disabled: invalid chained hash";
    return false;
  }
  Entry& entry = residents_[key];
  if (entry.data.is_null()) {
    entry.data = {{"type", "stored"},
                  {"model_id", update.scope.model_id},
                  {"dp_rank", update.scope.dp_rank},
                  {"block_hash", update.hash},
                  {"parent_hash", update.parent_hash},
                  {"token_ids", update.tokens},
                  {"block_size", update.tokens.size()},
                  {"block_index", update.block_index}};
  } else {
    if (entry.data["token_ids"] != update.tokens ||
        entry.data["parent_hash"] != update.parent_hash) {
      LOG(ERROR) << "KV event collection disabled: conflicting chained hash "
                    "content";
      return false;
    }
  }
  if (entry.media.insert(update.scope.medium).second) {
    nlohmann::json event = entry.data;
    event["medium"] = update.scope.medium;
    append(std::move(event));
  }
  return true;
}

nlohmann::json KVCacheEventJournal::read(const std::string& epoch,
                                         uint64_t cursor,
                                         size_t limit,
                                         const std::string& medium) {
  CHECK_GT(limit, 0);
  std::lock_guard<std::mutex> consumer_lock(consumer_mutex_);
  {
    std::lock_guard<std::mutex> producer_lock(producer_mutex_);
    pending_.swap(draining_);
  }
  if (enabled()) {
    for (auto& update : draining_) {
      if (!apply(std::move(update))) {
        disable();
        break;
      }
    }
  }
  draining_.clear();
  if (!enabled() && !invalidated_) {
    residents_.clear();
    events_.clear();
    sequence_ = 0;
    epoch_ = create_epoch();
    invalidated_ = true;
  }
  const bool reset = epoch != epoch_ || cursor > sequence_ ||
                     (!events_.empty() &&
                      cursor < events_.front()["sequence"].get<uint64_t>() - 1);
  nlohmann::json output = {{"schema_version", 1},
                           {"epoch", epoch_},
                           {"reset", reset},
                           {"cursor", cursor},
                           {"events", nlohmann::json::array()}};
  auto& result = output["events"];
  if (reset) {
    std::vector<nlohmann::json> snapshot;
    snapshot.reserve(residents_.size() * (medium.empty() ? 2 : 1));
    for (const auto& [key, entry] : residents_) {
      for (const std::string& resident_medium : entry.media) {
        if (!medium.empty() && resident_medium != medium) {
          continue;
        }
        nlohmann::json event = entry.data;
        event["medium"] = resident_medium;
        snapshot.emplace_back(std::move(event));
      }
    }
    std::sort(
        snapshot.begin(), snapshot.end(), [](const auto& a, const auto& b) {
          return a["block_index"].template get<uint32_t>() <
                 b["block_index"].template get<uint32_t>();
        });
    result = std::move(snapshot);
    output["cursor"] = sequence_;
  } else {
    for (const auto& event : events_) {
      if (event["sequence"].get<uint64_t>() > cursor) {
        output["cursor"] = event["sequence"];
        if (!medium.empty() && event["medium"] != medium) {
          continue;
        }
        result.push_back(event);
        if (result.size() == limit) {
          break;
        }
      }
    }
  }
  return output;
}

}  // namespace xllm
