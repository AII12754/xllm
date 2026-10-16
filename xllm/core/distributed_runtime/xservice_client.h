/* Copyright 2025-2026 The xLLM Authors.

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

#include <brpc/channel.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/etcd_client.h"
#include "common/types.h"
#include "core/distributed_runtime/heartbeat_callback_registry.h"
#include "core/framework/request/request_output.h"
#include "xservice.pb.h"
namespace xllm {

class XServiceClient {
 public:
  using HeartbeatCallback = detail::HeartbeatCallbackRegistry<
      xllm_service::proto::HeartbeatRequest>::Callback;
  using HeartbeatCallbackRegistration = detail::HeartbeatCallbackRegistry<
      xllm_service::proto::HeartbeatRequest>::RegistrationId;

  static XServiceClient* get_instance() {
    static XServiceClient xservice_client;
    return &xservice_client;
  }

  ~XServiceClient();
  bool init(const std::string& etcd_addr,
            const std::string& instance_name = "",
            const std::string& etcd_namespace = "");
  HeartbeatCallbackRegistration set_heartbeat_callback(
      HeartbeatCallback callback);
  void clear_heartbeat_callback(HeartbeatCallbackRegistration registration);
  bool initialize_done() { return initialize_done_; }

  std::string get_instance_name();
  void register_instance(const InstanceInfo& instance_info);
  void heartbeat();
  InstanceInfo get_instance_info(const std::string& instance_name);
  std::vector<std::string> get_static_decode_list();

  // get all xllm_service addrs
  std::vector<std::string> get_all_xservice_addrs();

  // response generation tokens to xllm service
  bool generation(const RequestOutput& output);
  std::vector<bool> generations(const std::vector<RequestOutput>& outputs);

 private:
  std::vector<bool> send_generations(std::span<const RequestOutput> outputs);

  bool register_instance_with_retry(const std::string& key,
                                    const std::string& value);
  bool reconcile_registration();
  void reconcile_registration_loop();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);
  void handle_xservices_watch(const etcd::Response& response,
                              const uint64_t& prefix_len);

  // connect to specific xllm_service
  bool connect_to_xservice(const std::string& xservice_addr);

  // remove the connection to specific xllm_service
  void disconnect_xservice(const std::string& xservice_addr);

  // call rpc with current master stub atomically.
  bool with_master_stub(
      const std::function<void(xllm_service::proto::XllmRpcService_Stub*)>& fn,
      std::string* master_addr);

  // find stub by address, caller should hold mutex_
  xllm_service::proto::XllmRpcService_Stub* find_stub_locked(
      const std::string& xservice_addr);

 private:
  XServiceClient() = default;

  std::atomic_bool exited_{false};
  std::atomic_bool register_done_{false};
  bool initialize_done_ = false;
  std::string instance_name_;
  std::string incarnation_id_;
  std::string registration_key_;
  std::string registration_value_;

  std::string master_xservice_addr_;
  std::unordered_map<std::string, std::unique_ptr<brpc::Channel>>
      xservice_channels_;
  std::unordered_map<std::string,
                     std::unique_ptr<xllm_service::proto::XllmRpcService_Stub>>
      xservice_stubs_;
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::unique_ptr<std::thread> reconcile_thread_;

  std::shared_mutex mutex_;
  std::mutex registration_mutex_;
  brpc::ChannelOptions chan_options_;
  std::unique_ptr<EtcdClient> etcd_client_;
  detail::HeartbeatCallbackRegistry<xllm_service::proto::HeartbeatRequest>
      heartbeat_callback_registry_;
};

}  // namespace xllm
