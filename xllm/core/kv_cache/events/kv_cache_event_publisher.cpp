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

#include "core/kv_cache/events/kv_cache_event_publisher.h"

#include <glog/logging.h>
#include <zmq.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <set>
#include <vector>

namespace xllm {

uint64_t KVCacheEventPublisher::wire_hash(const std::string& hash) {
  CHECK_EQ(hash.size(), 32) << "KV event journal must validate hash length";
  // Fold the full chained 128-bit identity, including its prefix, to an opaque
  // 64-bit sequence identity. Routers derive their content hashes from tokens.
  uint64_t result = 14695981039346656037ULL;
  for (char digit : hash) {
    result ^= static_cast<uint8_t>(digit);
    result *= 1099511628211ULL;
  }
  return result;
}

nlohmann::json KVCacheEventPublisher::engine_event(
    const nlohmann::json& event) {
  const bool stored = event["type"] == "stored";
  nlohmann::json result = {
      {"type", stored ? "BlockStored" : "BlockRemoved"},
      {"block_hashes", {wire_hash(event["block_hash"].get<std::string>())}},
      {"medium", event["medium"]}};
  if (stored) {
    const std::string parent = event["parent_hash"].get<std::string>();
    result["parent_block_hash"] = parent.empty()
                                      ? nlohmann::json(nullptr)
                                      : nlohmann::json(wire_hash(parent));
    result["token_ids"] = event["token_ids"];
    result["block_size"] = event["block_size"];
    result["lora_id"] = nullptr;
  }
  return result;
}

nlohmann::json KVCacheEventPublisher::pool_event(const nlohmann::json& event) {
  return {{"event_type", event["type"]},
          {"medium", "cpu"},
          {"dp_rank", event["dp_rank"]},
          {"seq_hashes", {wire_hash(event["block_hash"].get<std::string>())}}};
}

std::string KVCacheEventPublisher::rank_endpoint(const std::string& endpoint,
                                                 int32_t rank) {
  if (endpoint.substr(0, 6) != "tcp://") {
    LOG(ERROR) << "KV events require a TCP endpoint";
    return "";
  }
  const size_t delimiter = endpoint.rfind(':');
  if (delimiter == std::string::npos) {
    LOG(ERROR) << "KV events require a TCP port";
    return "";
  }
  const std::string port_string = endpoint.substr(delimiter + 1);
  if (port_string.empty() || port_string.size() > 5 ||
      port_string.find_first_not_of("0123456789") != std::string::npos) {
    LOG(ERROR) << "KV event port must be numeric";
    return "";
  }
  int32_t base_port = 0;
  const auto parsed = std::from_chars(
      port_string.data(), port_string.data() + port_string.size(), base_port);
  const int64_t port = static_cast<int64_t>(base_port) + rank;
  if (parsed.ec != std::errc{} || rank < 0 || port <= 0 || port > 65535) {
    LOG(ERROR) << "KV event port must be within 1-65535";
    return "";
  }
  return endpoint.substr(0, delimiter + 1) + std::to_string(port);
}

KVCacheEventPublisher::KVCacheEventPublisher(
    std::shared_ptr<KVCacheEventJournal> journal,
    const nlohmann::json& config,
    const std::string& model_id,
    int32_t dp_size)
    : journal_(std::move(journal)) {
  auto ready = std::make_shared<std::promise<void>>();
  auto future = ready->get_future();
  thread_ = std::thread(
      &KVCacheEventPublisher::run, this, config, model_id, dp_size, ready);
  future.get();
}

bool KVCacheEventPublisher::healthy() const {
  return healthy_.load() && journal_->enabled();
}

KVCacheEventPublisher::~KVCacheEventPublisher() {
  stopped_.store(true);
  if (thread_.joinable()) {
    thread_.join();
  }
}

namespace {

double timestamp() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::array<uint8_t, 8> sequence_bytes(uint64_t sequence) {
  std::array<uint8_t, 8> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[bytes.size() - i - 1] = static_cast<uint8_t>(sequence >> (i * 8));
  }
  return bytes;
}

bool check_zmq(int32_t result, const std::string& operation) {
  if (result < 0) {
    const int32_t error = zmq_errno();
    LOG(ERROR) << operation << " failed: " << zmq_strerror(error);
    return false;
  }
  return true;
}

bool send_frame(void* socket, const void* data, size_t size, bool more) {
  return check_zmq(
      zmq_send(socket, data, size, ZMQ_DONTWAIT | (more ? ZMQ_SNDMORE : 0)),
      "KV event send");
}

using ReplayPayload = std::shared_ptr<std::vector<uint8_t>>;

struct ReplaySnapshot {
  ReplayPayload payload;
  uint64_t sequence = 0;
};

void release_replay_payload(void* /*data*/, void* hint) {
  std::unique_ptr<ReplayPayload> owner(static_cast<ReplayPayload*>(hint));
}

class ZmqMessage final {
 public:
  ZmqMessage()
      : valid_(check_zmq(zmq_msg_init(&message_), "KV replay frame init")) {}
  explicit ZmqMessage(ReplayPayload payload) : valid_(false) {
    // ZMQ keeps this shared owner until its last queued copy is released,
    // even after the publisher's replay batch has gone out of scope.
    auto owner = std::make_unique<ReplayPayload>(std::move(payload));
    valid_ = check_zmq(zmq_msg_init_data(&message_,
                                         (*owner)->data(),
                                         (*owner)->size(),
                                         &release_replay_payload,
                                         owner.get()),
                       "KV replay shared payload init");
    if (valid_) {
      owner.release();
    }
  }
  ~ZmqMessage() {
    if (valid_) {
      zmq_msg_close(&message_);
    }
  }
  ZmqMessage(const ZmqMessage&) = delete;
  ZmqMessage& operator=(const ZmqMessage&) = delete;
  zmq_msg_t* get() { return &message_; }
  bool valid() const { return valid_; }

 private:
  zmq_msg_t message_;
  bool valid_;
};

constexpr int32_t kMaxReplayFrames = 16;
constexpr int32_t kMaxReplayRequestsPerBatch = 16;
constexpr int32_t kMaxSubscriptionCommands = 64;
constexpr std::chrono::seconds kSubscriptionSnapshotInterval{1};
constexpr std::chrono::seconds kReplaySnapshotInterval{1};
constexpr int64_t kMaxReplayRequestBytes = 4096;
constexpr int32_t kShutdownLingerMilliseconds = 200;

class EventSocket final {
 public:
  EventSocket(void* context,
              int32_t type,
              const std::string& endpoint,
              int32_t hwm,
              const std::string& topic)
      : socket_(zmq_socket(context, type), &zmq_close),
        topic_(topic),
        subscription_buffer_(topic.size() + 1) {
    if (!socket_) {
      check_zmq(-1, "KV event socket");
      return;
    }
    const int32_t linger = 0;
    if (!check_zmq(
            zmq_setsockopt(socket_.get(), ZMQ_LINGER, &linger, sizeof(linger)),
            "KV event linger") ||
        !check_zmq(zmq_setsockopt(socket_.get(), ZMQ_SNDHWM, &hwm, sizeof(hwm)),
                   "KV event HWM")) {
      return;
    }
    const int32_t verbose = 1;
    if (type == ZMQ_XPUB &&
        !check_zmq(
            zmq_setsockopt(
                socket_.get(), ZMQ_XPUB_VERBOSE, &verbose, sizeof(verbose)),
            "KV event verbose subscriptions")) {
      return;
    }
    if (type == ZMQ_ROUTER &&
        !check_zmq(zmq_setsockopt(socket_.get(),
                                  ZMQ_MAXMSGSIZE,
                                  &kMaxReplayRequestBytes,
                                  sizeof(kMaxReplayRequestBytes)),
                   "KV replay maximum frame size")) {
      return;
    }
    const int32_t send_timeout = 0;
    if (!check_zmq(zmq_setsockopt(socket_.get(),
                                  ZMQ_SNDTIMEO,
                                  &send_timeout,
                                  sizeof(send_timeout)),
                   "KV event send timeout") ||
        !check_zmq(zmq_bind(socket_.get(), endpoint.c_str()),
                   "KV event bind " + endpoint)) {
      return;
    }
    valid_ = true;
  }

  bool valid() const { return valid_; }
  uint64_t sequence() const { return sequence_; }

  bool set_linger(int32_t milliseconds) {
    return check_zmq(
        zmq_setsockopt(
            socket_.get(), ZMQ_LINGER, &milliseconds, sizeof(milliseconds)),
        "KV event shutdown linger");
  }

  bool publish(const nlohmann::json& payload) {
    const auto bytes = sequence_bytes(++sequence_);
    const auto packed = nlohmann::json::to_msgpack(payload);
    return send_frame(socket_.get(), topic_.data(), topic_.size(), true) &&
           send_frame(socket_.get(), bytes.data(), bytes.size(), true) &&
           send_frame(socket_.get(), packed.data(), packed.size(), false);
  }

  bool has_subscription(bool& subscribed) {
    for (int32_t count = 0; count < kMaxSubscriptionCommands; ++count) {
      const int32_t size = zmq_recv(socket_.get(),
                                    subscription_buffer_.data(),
                                    subscription_buffer_.size(),
                                    ZMQ_DONTWAIT);
      if (size < 0) {
        return zmq_errno() == EAGAIN ||
               check_zmq(-1, "KV event subscription receive");
      }
      if (size <= 0 || subscription_buffer_[0] != 1) {
        continue;
      }
      // ZMQ filters are binary prefixes of the published topic. recv reports
      // the full frame size even when truncated; longer filters cannot match.
      const size_t filter_size = static_cast<size_t>(size - 1);
      subscribed = subscribed || (filter_size <= topic_.size() &&
                                  std::memcmp(subscription_buffer_.data() + 1,
                                              topic_.data(),
                                              filter_size) == 0);
    }
    // Yield even while commands remain, so publication and shutdown progress.
    return true;
  }

  template <typename Snapshot>
  bool replay(const Snapshot& snapshot, bool& received) {
    received = false;
    if (discarding_request_) {
      return discard_request();
    }
    ZmqMessage identity;
    if (!identity.valid()) {
      return false;
    }
    if (zmq_msg_recv(identity.get(), socket_.get(), ZMQ_DONTWAIT) < 0) {
      if (zmq_errno() != EAGAIN) {
        return check_zmq(-1, "KV replay receive");
      }
      return true;
    }
    received = true;
    // vLLM replay: ROUTER receives [identity, empty, start_sequence]. A full
    // current snapshot also covers requests older than the bounded journal.
    ZmqMessage frame;
    if (!frame.valid()) {
      return false;
    }
    bool more = zmq_msg_more(identity.get()) != 0;
    int32_t frames = 0;
    int64_t request_bytes = 0;
    bool valid_request = true;
    while (more && frames < kMaxReplayFrames) {
      if (!check_zmq(zmq_msg_recv(frame.get(), socket_.get(), ZMQ_DONTWAIT),
                     "KV replay frame receive")) {
        return false;
      }
      more = zmq_msg_more(frame.get()) != 0;
      const size_t frame_size = zmq_msg_size(frame.get());
      ++frames;
      request_bytes += static_cast<int64_t>(frame_size);
      valid_request =
          valid_request && ((frames == 1 && frame_size == 0) ||
                            (frames == 2 && frame_size == sizeof(uint64_t)));
      if (request_bytes > kMaxReplayRequestBytes) {
        discarding_request_ = more;
        received = !more;
        LOG(WARNING) << "Ignoring oversized KV replay request";
        return true;
      }
    }
    if (more) {
      discarding_request_ = true;
      received = false;
      LOG(WARNING) << "Ignoring KV replay request with too many frames";
      return true;
    }
    if (!valid_request || frames != 2) {
      LOG(WARNING) << "Ignoring malformed KV replay request";
      return true;
    }
    const std::optional<ReplaySnapshot> response = snapshot();
    if (!response) {
      return false;
    }
    ZmqMessage payload(response->payload);
    if (!payload.valid()) {
      return false;
    }
    // Snapshot recovery uses the live engine watermark, not the ROUTER's
    // independent counter. The next live batch follows this watermark.
    const auto bytes = sequence_bytes(response->sequence);
    if (!send_frame(socket_.get(),
                    zmq_msg_data(identity.get()),
                    zmq_msg_size(identity.get()),
                    true) ||
        !send_frame(socket_.get(), nullptr, 0, true) ||
        !send_frame(socket_.get(), bytes.data(), bytes.size(), true) ||
        !check_zmq(zmq_msg_send(payload.get(), socket_.get(), ZMQ_DONTWAIT),
                   "KV replay payload send")) {
      return false;
    }
    const auto end = sequence_bytes(UINT64_MAX);
    return send_frame(socket_.get(),
                      zmq_msg_data(identity.get()),
                      zmq_msg_size(identity.get()),
                      true) &&
           send_frame(socket_.get(), nullptr, 0, true) &&
           send_frame(socket_.get(), end.data(), end.size(), true) &&
           send_frame(socket_.get(), nullptr, 0, false);
  }

 private:
  bool discard_request() {
    ZmqMessage frame;
    if (!frame.valid()) {
      return false;
    }
    // Drain across iterations so peer input cannot monopolize publication.
    for (int32_t count = 0; discarding_request_ && count < kMaxReplayFrames;
         ++count) {
      if (zmq_msg_recv(frame.get(), socket_.get(), ZMQ_DONTWAIT) < 0) {
        if (zmq_errno() == EAGAIN) {
          return true;
        }
        return check_zmq(-1, "KV replay discard receive");
      }
      discarding_request_ = zmq_msg_more(frame.get()) != 0;
    }
    return true;
  }

  std::unique_ptr<void, decltype(&zmq_close)> socket_;
  std::string topic_;
  std::vector<uint8_t> subscription_buffer_;
  uint64_t sequence_ = 0;
  bool valid_ = false;
  bool discarding_request_ = false;
};

nlohmann::json engine_batch(const nlohmann::json& events, int32_t rank) {
  return nlohmann::json::array({timestamp(), events, rank});
}

// Borrow the snapshot owned by this publisher iteration. Group once and lazily
// serialize each requested DP; no cached state survives the replay batch.
class ReplaySnapshotBatch final {
 public:
  ReplaySnapshotBatch(const nlohmann::json& state,
                      const std::string& model_id,
                      int32_t dp_size)
      : events_(state["events"]), indices_(dp_size), payloads_(dp_size) {
    for (auto& indices : indices_) {
      indices.reserve(events_.size() / static_cast<size_t>(dp_size) + 1);
    }
    for (size_t index = 0; index < events_.size(); ++index) {
      const auto& event = events_[index];
      if (event["model_id"] != model_id) {
        continue;
      }
      const int32_t rank = event["dp_rank"].get<int32_t>();
      if (rank < 0 || rank >= dp_size) {
        continue;
      }
      indices_[rank].emplace_back(index);
    }
  }

  const ReplayPayload& payload(int32_t rank) {
    auto& packed = payloads_[rank];
    if (!packed) {
      nlohmann::json events = nlohmann::json::array();
      events.push_back({{"type", "AllBlocksCleared"}});
      for (size_t index : indices_[rank]) {
        events.push_back(KVCacheEventPublisher::engine_event(events_[index]));
      }
      packed = std::make_shared<std::vector<uint8_t>>(
          nlohmann::json::to_msgpack(engine_batch(events, rank)));
    }
    return packed;
  }

 private:
  const nlohmann::json& events_;
  std::vector<std::vector<size_t>> indices_;
  std::vector<ReplayPayload> payloads_;
};

}  // namespace

void KVCacheEventPublisher::run(
    const nlohmann::json& config,
    const std::string& model_id,
    int32_t dp_size,
    const std::shared_ptr<std::promise<void>>& ready) {
  bool ready_sent = false;
  if (!run_impl(config, model_id, dp_size, ready, ready_sent)) {
    journal_->disable();
    LOG(ERROR) << "KV event publisher disabled; inference continues";
  }
  healthy_.store(false);
  if (!ready_sent) {
    ready->set_value();
  }
}

bool KVCacheEventPublisher::run_impl(
    const nlohmann::json& config,
    const std::string& model_id,
    int32_t dp_size,
    const std::shared_ptr<std::promise<void>>& ready,
    bool& ready_sent) {
  if (dp_size <= 0 || !config.is_object() || !config.contains("endpoint")) {
    LOG(ERROR) << "Invalid KV event publisher configuration or DP size";
    return false;
  }
  for (const std::string& key : {"publisher",
                                 "endpoint",
                                 "motor_pool_endpoint",
                                 "replay_endpoint",
                                 "topic"}) {
    if (config.contains(key) && !config[key].is_string()) {
      LOG(ERROR) << "KV event " << key << " must be a string";
      return false;
    }
  }
  for (const std::string& key : {"hwm", "snapshot_interval_seconds"}) {
    if (config.contains(key) &&
        (!config[key].is_number_integer() || config[key] <= 0 ||
         config[key] > std::numeric_limits<int32_t>::max())) {
      LOG(ERROR) << "KV event " << key << " must be a positive int32";
      return false;
    }
  }
  if (config.value("publisher", std::string("zmq")) != "zmq") {
    LOG(ERROR) << "KV events require the zmq publisher";
    return false;
  }
  std::unique_ptr<void, decltype(&zmq_ctx_term)> context(zmq_ctx_new(),
                                                         &zmq_ctx_term);
  if (!context) {
    return check_zmq(-1, "KV event context");
  }
  const std::string endpoint = config.at("endpoint").get<std::string>();
  if (endpoint.empty()) {
    LOG(ERROR) << "KV event publisher requires an endpoint";
    return false;
  }
  const std::string pool_endpoint =
      config.value("motor_pool_endpoint", std::string());
  const std::string replay_endpoint =
      config.value("replay_endpoint", std::string());
  const std::string topic = config.value("topic", std::string());
  const int32_t hwm = config.value("hwm", 10000);
  const int32_t snapshot_seconds =
      config.value("snapshot_interval_seconds", 60);
  std::vector<std::unique_ptr<EventSocket>> engines;
  std::vector<std::unique_ptr<EventSocket>> pools;
  std::vector<std::unique_ptr<EventSocket>> replays;
  engines.reserve(dp_size);
  pools.reserve(dp_size);
  replays.reserve(dp_size);
  std::set<std::string> endpoints;
  for (int32_t rank = 0; rank < dp_size; ++rank) {
    auto bind = [&](const std::string& base,
                    int32_t type,
                    std::unique_ptr<EventSocket>& output) {
      if (base.empty()) {
        return true;
      }
      const std::string address = rank_endpoint(base, rank);
      if (address.empty()) {
        return false;
      }
      if (!endpoints.insert(address).second) {
        LOG(ERROR) << "KV event ports overlap: " << address;
        return false;
      }
      output = std::make_unique<EventSocket>(
          context.get(), type, address, hwm, topic);
      return output->valid();
    };
    engines.emplace_back();
    pools.emplace_back();
    replays.emplace_back();
    if (!bind(endpoint, ZMQ_XPUB, engines.back()) ||
        !bind(pool_endpoint, ZMQ_XPUB, pools.back()) ||
        !bind(replay_endpoint, ZMQ_ROUTER, replays.back())) {
      return false;
    }
  }
  healthy_.store(true);
  ready_sent = true;
  ready->set_value();
  std::string epoch;
  uint64_t cursor = 0;
  auto next_snapshot = std::chrono::steady_clock::now();
  auto next_subscription_snapshot = next_snapshot;
  auto next_replay_snapshot = next_snapshot;
  auto repair_until = next_snapshot;
  auto next_repair = next_snapshot;
  bool disabled_reported = false;
  bool subscription_pending = false;
  const auto publish_page = [&](const nlohmann::json& page) -> bool {
    const bool reset = page["reset"].get<bool>();
    for (int32_t rank = 0; rank < dp_size; ++rank) {
      nlohmann::json engine_events = nlohmann::json::array();
      nlohmann::json pool_events = nlohmann::json::array();
      if (reset) {
        engine_events.push_back({{"type", "AllBlocksCleared"}});
        pool_events.push_back(
            {{"event_type", "cleared"}, {"medium", "cpu"}, {"dp_rank", rank}});
      }
      for (const auto& event : page["events"]) {
        if (event["model_id"] != model_id || event["dp_rank"] != rank) {
          continue;
        }
        engine_events.push_back(engine_event(event));
        if (event["medium"] == "cpu") {
          pool_events.push_back(pool_event(event));
        }
      }
      if (!engine_events.empty() &&
          !engines[rank]->publish(engine_batch(engine_events, rank))) {
        return false;
      }
      if (pools[rank] && !pool_events.empty() &&
          !pools[rank]->publish({{"events", pool_events}})) {
        return false;
      }
    }
    epoch = page["epoch"].get<std::string>();
    cursor = page["cursor"].get<uint64_t>();
    if (reset) {
      const auto published_at = std::chrono::steady_clock::now();
      subscription_pending = false;
      next_snapshot = published_at + std::chrono::seconds(snapshot_seconds);
      next_subscription_snapshot = published_at + kSubscriptionSnapshotInterval;
      repair_until = published_at + std::chrono::seconds(2);
      next_repair = published_at + std::chrono::milliseconds(250);
    }
    return true;
  };
  const auto publish_loop = [&]() -> bool {
    while (!stopped_.load()) {
      bool subscription = false;
      for (int32_t rank = 0; rank < dp_size; ++rank) {
        if (!engines[rank]->has_subscription(subscription) ||
            (pools[rank] && !pools[rank]->has_subscription(subscription))) {
          return false;
        }
      }
      if (stopped_.load()) {
        break;
      }
      const auto now = std::chrono::steady_clock::now();
      subscription_pending = subscription_pending || subscription;
      const bool snapshot =
          now >= next_snapshot ||
          (subscription_pending && now >= next_subscription_snapshot);
      const nlohmann::json page = journal_->read(snapshot ? "" : epoch, cursor);
      if (!journal_->enabled() && !disabled_reported) {
        LOG(WARNING)
            << "KV event collection disabled; publishing an empty directory "
               "until engine restart";
        disabled_reported = true;
      }
      const bool reset = page["reset"].get<bool>();
      if (!publish_page(page)) {
        return false;
      }
      if (std::chrono::steady_clock::now() >= next_replay_snapshot) {
        nlohmann::json replay_state;
        std::optional<ReplaySnapshotBatch> replay_batch;
        for (int32_t rank = 0; rank < dp_size; ++rank) {
          if (!replays[rank]) {
            continue;
          }
          // ROUTER fair-queues peers. Leave requests queued during cooldown;
          // a bounded batch serves multiple peers without starving live work.
          for (int32_t request = 0; request < kMaxReplayRequestsPerBatch;
               ++request) {
            bool received = false;
            if (!replays[rank]->replay(
                    [&]() -> std::optional<ReplaySnapshot> {
                      if (!replay_batch) {
                        if (!reset) {
                          replay_state = journal_->read("", 0);
                          // The fresh read may include updates that arrived
                          // after the live page. Publish this exact state and
                          // advance the cursor before capturing its watermark.
                          const bool unpublished =
                              replay_state["epoch"] != epoch ||
                              replay_state["cursor"] != cursor;
                          if (unpublished && !publish_page(replay_state)) {
                            return std::nullopt;
                          }
                        }
                        replay_batch.emplace(
                            reset ? page : replay_state, model_id, dp_size);
                      }
                      return ReplaySnapshot{replay_batch->payload(rank),
                                            engines[rank]->sequence()};
                    },
                    received)) {
              return false;
            }
            if (!received) {
              break;
            }
          }
        }
        if (replay_batch) {
          // Payloads expire with this batch. Rebuild fresh state next time,
          // rather than returning stale bytes with a newer live watermark.
          next_replay_snapshot =
              std::chrono::steady_clock::now() + kReplaySnapshotInterval;
        }
      }
      // Replay replies precede Host repair so a reused reset page cannot be
      // paired with the sequence of a newer Host-only publication.
      const auto repair_now = std::chrono::steady_clock::now();
      if (!pool_endpoint.empty() && repair_now >= next_repair &&
          repair_now < repair_until) {
        const nlohmann::json state =
            journal_->read("", 0, /*limit=*/1024, /*medium=*/"cpu");
        for (int32_t rank = 0; rank < dp_size; ++rank) {
          nlohmann::json host_engine = nlohmann::json::array();
          nlohmann::json host_pool = nlohmann::json::array();
          for (const auto& event : state["events"]) {
            if (event["model_id"] == model_id && event["dp_rank"] == rank &&
                event["medium"] == "cpu") {
              host_engine.push_back(engine_event(event));
              host_pool.push_back(pool_event(event));
            }
          }
          if (!host_engine.empty()) {
            if (!engines[rank]->publish(engine_batch(host_engine, rank)) ||
                (pools[rank] &&
                 !pools[rank]->publish({{"events", host_pool}}))) {
              return false;
            }
          }
        }
        next_repair = repair_now + std::chrono::milliseconds(250);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  };
  const bool success = publish_loop();
  if (!success) {
    journal_->disable();
    healthy_.store(false);
  }
  // Normal stop and transport failure share bounded, best-effort invalidation.
  // Keep the sockets alive while clearing all independent engine/pool streams.
  for (int32_t rank = 0; rank < dp_size; ++rank) {
    engines[rank]->set_linger(kShutdownLingerMilliseconds);
    engines[rank]->publish(engine_batch(
        nlohmann::json::array({{{"type", "AllBlocksCleared"}}}), rank));
    if (pools[rank]) {
      pools[rank]->set_linger(kShutdownLingerMilliseconds);
      pools[rank]->publish({{"events",
                             nlohmann::json::array({{{"event_type", "cleared"},
                                                     {"medium", "cpu"},
                                                     {"dp_rank", rank}}})}});
    }
  }
  replays.clear();
  pools.clear();
  engines.clear();
  return success;
}

}  // namespace xllm
