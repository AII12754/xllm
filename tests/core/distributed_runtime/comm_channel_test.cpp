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

#include "core/distributed_runtime/comm_channel.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <gtest/gtest.h>

#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"

namespace xllm {
namespace {

class ModelRoutingService final : public proto::DistributeWorker {
 public:
  void ExecuteLLMModel(google::protobuf::RpcController* controller,
                       const proto::LLMForwardRequest* request,
                       proto::LLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 1) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(1);
  }

  void ExecuteVLMModel(google::protobuf::RpcController* controller,
                       const proto::VLMForwardRequest* request,
                       proto::VLMForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 2) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(2);
  }

  void ExecuteRECModel(google::protobuf::RpcController* controller,
                       const proto::RECForwardRequest* request,
                       proto::RECForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 3) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->add_outputs()->add_tokens()->set_id(3);
  }

  void ExecuteDITModel(google::protobuf::RpcController* controller,
                       const proto::DITForwardRequest* request,
                       proto::DITForwardResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    if (!request->has_packed_input() ||
        request->packed_input().payload().size() < 11 ||
        static_cast<uint8_t>(request->packed_input().payload()[10]) != 4) {
      controller->SetFailed("Unexpected packed input domain");
      return;
    }
    response->mutable_dit_forward_output()->add_text_output("diffusion");
  }

  void GetLastLLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::LLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }

  void GetLastVLMStepResult(google::protobuf::RpcController* controller,
                            const proto::Empty* /*request*/,
                            proto::VLMForwardResponse* /*response*/,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    controller->SetFailed("result unavailable");
  }
};

class CommChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    ASSERT_TRUE(channel_.init_brpc(
        butil::endpoint2str(server_.listen_address()).c_str()));
  }

  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  ModelRoutingService service_;
  brpc::Server server_;
  CommChannel channel_;
};

TEST_F(CommChannelTest, RoutesLLMInputToItsOwnRpc) {
  LlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_llm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 1);
}

TEST_F(CommChannelTest, RoutesVLMInputToItsOwnRpc) {
  VlmForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_vlm_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 2);
}

TEST_F(CommChannelTest, RoutesRECInputToItsOwnRpc) {
  RecForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_rec_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->outputs.size(), 1);
  ASSERT_EQ(output->outputs[0].tokens.size(), 1);
  EXPECT_EQ(output->outputs[0].tokens[0].id, 3);
}

TEST_F(CommChannelTest, RoutesDITInputToItsOwnRpc) {
  DiTForwardInput input;
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  channel_.execute_dit_model_async(input, promise);
  const auto output = std::move(future).get();
  ASSERT_TRUE(output.has_value());
  ASSERT_EQ(output->dit_forward_output.text_output.size(), 1);
  EXPECT_EQ(output->dit_forward_output.text_output[0], "diffusion");
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastLLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_llm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

TEST_F(CommChannelTest, CompletesPromiseWhenLastVLMResultRpcFails) {
  folly::Promise<std::optional<RawForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  EXPECT_FALSE(channel_.get_last_vlm_step_result_async(promise));
  ASSERT_TRUE(future.isReady());
  EXPECT_FALSE(std::move(future).get().has_value());
}

}  // namespace
}  // namespace xllm
