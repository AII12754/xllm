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

#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/distributed_runtime/worker_memory_rpc_client.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/kv_cache/storage/vmm_kv_page_allocator_impl.h"

namespace xllm {

class DistributedVirtualMemoryCoordinatorTestPeer final {
 public:
  static void connect(DistributedVirtualMemoryCoordinator& coordinator,
                      const std::vector<std::string>& addresses) {
    const int32_t count = static_cast<int32_t>(addresses.size());
    coordinator.configure_workers(count, /*dp_size=*/1, /*tp_size=*/count);
    for (int32_t worker = 0; worker < count; ++worker) {
      coordinator.clients_.emplace_back(std::make_shared<WorkerMemoryRpcClient>(
          worker, addresses[worker], torch::Device(torch::kCPU)));
    }
  }

  static std::shared_ptr<DistributedVirtualMemoryCoordinator> create() {
    return std::shared_ptr<DistributedVirtualMemoryCoordinator>(
        new DistributedVirtualMemoryCoordinator(),
        [](DistributedVirtualMemoryCoordinator* coordinator) {
          delete coordinator;
        });
  }

  static void refresh_worker_capacity_snapshot(
      DistributedVirtualMemoryCoordinator& coordinator) {
    coordinator.refresh_worker_capacity_snapshot();
  }
};

namespace {

enum class MappingFailure { NONE, PREPARE, PREPARE_REPLY, COMMIT, WEIGHT };

// Exercise the real brpc transport without requiring device allocations.
class TransactionTestService final : public proto::ModelMemoryDist {
 public:
  explicit TransactionTestService(MappingFailure failure = MappingFailure::NONE,
                                  size_t allocatable_pages = 10,
                                  int64_t available_memory = 0,
                                  int64_t total_memory = 0)
      : failure_(failure),
        allocatable_pages_(allocatable_pages),
        available_memory_(available_memory),
        total_memory_(total_memory) {}

  void Hello(google::protobuf::RpcController* /*controller*/,
             const proto::Status* /*request*/,
             proto::Status* response,
             google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
  }

  void GetMemoryInfo(google::protobuf::RpcController* /*controller*/,
                     const proto::Status* /*request*/,
                     proto::MemoryInfoResponse* response,
                     google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    response->set_available_memory(available_memory_);
    response->set_total_memory(total_memory_);
    response->set_pool_initialized(true);
    response->set_pool_allocatable_pages(allocatable_pages_);
    ++memory_info_calls_;
  }

  void InitPhysicalPagePool(google::protobuf::RpcController* /*controller*/,
                            const proto::InitPhysicalPagePoolRequest* request,
                            proto::Status* response,
                            google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_pages_ = request->num_pages();
    lazy_ = request->lazy();
    response->set_ok(true);
  }

  void PrepareKvMapping(google::protobuf::RpcController* controller,
                        const proto::PrepareKvMappingRequest* /*request*/,
                        proto::Status* response,
                        google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_ == MappingFailure::PREPARE) {
      response->set_ok(false);
      return;
    }
    mapped_ = true;
    if (failure_ == MappingFailure::PREPARE_REPLY) {
      controller->SetFailed("injected prepare reply loss after mapping");
      return;
    }
    response->set_ok(true);
  }

  void CommitKvMapping(google::protobuf::RpcController* /*controller*/,
                       const proto::KvMappingTransactionRequest* /*request*/,
                       proto::Status* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(failure_ != MappingFailure::COMMIT);
  }

  void AbortKvMapping(google::protobuf::RpcController* /*controller*/,
                      const proto::KvMappingTransactionRequest* /*request*/,
                      proto::Status* response,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    mapped_ = false;
    ++abort_calls_;
    response->set_ok(true);
  }

  void PrepareWeightPages(google::protobuf::RpcController* /*controller*/,
                          const proto::PrepareWeightPagesRequest* /*request*/,
                          proto::Status* response,
                          google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    weights_ = failure_ != MappingFailure::WEIGHT;
    response->set_ok(weights_);
  }

  void CommitWeightPages(google::protobuf::RpcController* /*controller*/,
                         const proto::KvMappingTransactionRequest* /*request*/,
                         proto::Status* response,
                         google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
  }

  void AbortWeightPages(google::protobuf::RpcController* /*controller*/,
                        const proto::KvMappingTransactionRequest* /*request*/,
                        proto::Status* response,
                        google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    std::lock_guard<std::mutex> lock(mutex_);
    weights_ = false;
    ++abort_calls_;
    response->set_ok(true);
  }

  bool has_mappings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mapped_ || weights_;
  }
  int32_t abort_calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return abort_calls_;
  }
  int64_t initialized_pages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_pages_;
  }
  bool lazy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return lazy_;
  }
  int32_t memory_info_calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_info_calls_;
  }

 private:
  const MappingFailure failure_;
  const size_t allocatable_pages_;
  const int64_t available_memory_;
  const int64_t total_memory_;
  mutable std::mutex mutex_;
  bool mapped_ = false;
  bool weights_ = false;
  int32_t abort_calls_ = 0;
  int64_t initialized_pages_ = 0;
  bool lazy_ = false;
  int32_t memory_info_calls_ = 0;
};

class TransactionTestServer final {
 public:
  explicit TransactionTestServer(MappingFailure failure = MappingFailure::NONE,
                                 size_t allocatable_pages = 10,
                                 int64_t available_memory = 0,
                                 int64_t total_memory = 0)
      : service_(failure, allocatable_pages, available_memory, total_memory) {}

  ~TransactionTestServer() {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }

  int32_t start() {
    const int32_t result =
        server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE);
    if (result != 0) {
      return result;
    }
    return server_.Start("127.0.0.1:0", nullptr);
  }

  std::string address() const {
    return "127.0.0.1:" + std::to_string(server_.listen_address().port);
  }
  const TransactionTestService& service() const { return service_; }

 private:
  TransactionTestService service_;
  brpc::Server server_;
};

class ElasticPoolInitializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& config = KVCacheConfig::get_instance();
    previous_elastic_ = config.vmm_elastic();
    config.vmm_elastic(true);
  }
  void TearDown() override {
    KVCacheConfig::get_instance().vmm_elastic(previous_elastic_);
  }

 private:
  bool previous_elastic_ = false;
};

TEST_F(ElasticPoolInitializationTest,
       HeterogeneousWorkersApplyTheirOwnRatioCap) {
  const int64_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  TransactionTestServer first(
      MappingFailure::NONE, 10, 20 * page_size, 80 * page_size);
  TransactionTestServer second(
      MappingFailure::NONE, 10, 36 * page_size, 40 * page_size);
  ASSERT_EQ(first.start(), 0);
  ASSERT_EQ(second.start(), 0);
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  DistributedVirtualMemoryCoordinatorTestPeer::connect(
      *coordinator, {first.address(), second.address()});

  EXPECT_EQ(coordinator->init_physical_page_pools(0.8, 0), 4);
  EXPECT_EQ(first.service().initialized_pages(), 4);
  EXPECT_EQ(second.service().initialized_pages(), 4);
}

TEST_F(ElasticPoolInitializationTest, ZeroRatioPreservesLazyAddressCapacity) {
  const int64_t page_size =
      KVCacheConfig::get_instance().phy_page_granularity_size();
  TransactionTestServer first(
      MappingFailure::NONE, 10, 20 * page_size, 80 * page_size);
  TransactionTestServer second(
      MappingFailure::NONE, 10, 36 * page_size, 40 * page_size);
  ASSERT_EQ(first.start(), 0);
  ASSERT_EQ(second.start(), 0);
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  DistributedVirtualMemoryCoordinatorTestPeer::connect(
      *coordinator, {first.address(), second.address()});

  EXPECT_EQ(coordinator->init_physical_page_pools(0.0, 0), 20);
  EXPECT_EQ(first.service().initialized_pages(), 20);
  EXPECT_EQ(second.service().initialized_pages(), 20);
  EXPECT_TRUE(first.service().lazy());
  EXPECT_TRUE(second.service().lazy());
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     ReservesKvAndWeightsOnEveryTargetWorker) {
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("model",
                                           /*dp_size=*/2,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("model", /*num_pages=*/3);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}, {0, 1}}};

  EXPECT_TRUE(coordinator->try_reserve_model_memory("model", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({5, 5, 3, 3}));
  EXPECT_EQ(coordinator->get_free_pages_for_model("model"), 3);
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     FailedJointReservationLeavesAllWorkersUntouched) {
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("model",
                                           /*dp_size=*/2,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("model", /*num_pages=*/3);
  ASSERT_TRUE(coordinator->try_reserve_pages("model",
                                             /*dp_rank=*/1,
                                             /*physical_pages=*/5));
  const auto before = coordinator->get_worker_free_page_counts();
  // KV alone fits on worker 2; adding weights exceeds its five free pages.
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}, {0, 1}}};

  EXPECT_FALSE(coordinator->try_reserve_model_memory("model", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(), before);
  EXPECT_EQ(coordinator->get_weight_page_count("model"), 3);
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     FailedResumeReleasesWholeReservationAndPreservesWeightDemand) {
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy(
      "model", /*dp_size=*/2, /*tp_size=*/2);
  coordinator->set_weight_page_count("model", /*num_pages=*/3);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}, {0, 1}}};
  const auto before = coordinator->get_worker_free_page_counts();
  ASSERT_TRUE(coordinator->try_reserve_model_memory("model", plan));
  coordinator->release_model_memory_budget("model", plan);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(), before);
  EXPECT_EQ(coordinator->get_weight_page_count("model"), 3);
  EXPECT_TRUE(coordinator->try_reserve_model_memory("model", plan));
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     ForkModelUsesOnlyItsOwnDpAndTpWorkers) {
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/2,
                                /*worker_count=*/4,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_model_parallel_strategy("fork",
                                           /*dp_size=*/1,
                                           /*tp_size=*/2);
  coordinator->set_weight_page_count("fork", /*num_pages=*/3);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0, 1}, {}}};

  EXPECT_TRUE(coordinator->try_reserve_model_memory("fork", plan));
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({3, 3, 10, 10}));
  EXPECT_EQ(coordinator->available_pages("fork", /*dp_rank=*/1), 0);
  EXPECT_FALSE(coordinator->try_reserve_pages("fork",
                                              /*dp_rank=*/1,
                                              /*physical_pages=*/1));
  coordinator->release_kv_mapping_budget("fork", plan);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({7, 7, 10, 10}));
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     ConcurrentJointReservationsCannotOvercommit) {
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/1,
                                /*worker_count=*/2,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  coordinator->set_weight_page_count("first", /*num_pages=*/4);
  coordinator->set_weight_page_count("second", /*num_pages=*/4);
  const KVCacheMappingPlan plan{/*physical_pages_per_virtual_page=*/2,
                                /*dp_group_page_ids=*/{{0}}};
  std::atomic<int32_t> successes{0};
  auto reserve = [&](const std::string& model_id) {
    if (coordinator->try_reserve_model_memory(model_id, plan)) {
      successes.fetch_add(1);
    }
  };
  std::thread first(reserve, "first");
  std::thread second(reserve, "second");
  first.join();
  second.join();

  EXPECT_EQ(successes.load(), 1);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({4, 4}));
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     MappingFailureRollsBackEveryWorkerIncludingUnknownResults) {
  for (MappingFailure failure : {MappingFailure::PREPARE,
                                 MappingFailure::PREPARE_REPLY,
                                 MappingFailure::COMMIT}) {
    SCOPED_TRACE(static_cast<int32_t>(failure));
    TransactionTestServer first;
    TransactionTestServer second(failure);
    ASSERT_EQ(first.start(), 0);
    ASSERT_EQ(second.start(), 0);
    auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
    coordinator->init_page_budget(/*total_pages=*/10,
                                  /*dp_size=*/1,
                                  /*worker_count=*/2,
                                  /*page_size=*/4096,
                                  /*enable_prealloc=*/false);
    DistributedVirtualMemoryCoordinatorTestPeer::connect(
        *coordinator, {first.address(), second.address()});
    ASSERT_TRUE(coordinator->try_reserve_pages(
        "model", /*dp_rank=*/0, /*physical_pages=*/2));
    EXPECT_FALSE(coordinator->map_pages("model", /*dp_rank=*/0, {0}));
    EXPECT_FALSE(first.service().has_mappings());
    EXPECT_FALSE(second.service().has_mappings());
    EXPECT_GT(first.service().abort_calls(), 0);
    EXPECT_GT(second.service().abort_calls(), 0);
    // The allocator releases accounting only after the coordinator confirms
    // every physical rollback, including the worker with an unknown reply.
    coordinator->release_pages("model", /*dp_rank=*/0, /*physical_pages=*/2);
    EXPECT_EQ(coordinator->get_worker_free_page_counts(),
              std::vector<size_t>({10, 10}));
  }
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     DynamicCapacityUsesTheMostConstrainedWorker) {
  TransactionTestServer first(MappingFailure::NONE, /*allocatable_pages=*/8);
  TransactionTestServer second(MappingFailure::NONE, /*allocatable_pages=*/3);
  ASSERT_EQ(first.start(), 0);
  ASSERT_EQ(second.start(), 0);
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/1,
                                /*worker_count=*/2,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  DistributedVirtualMemoryCoordinatorTestPeer::connect(
      *coordinator, {first.address(), second.address()});
  auto& config = KVCacheConfig::get_instance();
  const bool previous_elastic = config.vmm_elastic();
  config.vmm_elastic(true);
  struct ElasticConfigGuard final {
    KVCacheConfig& config;
    bool previous_elastic;
    ~ElasticConfigGuard() { config.vmm_elastic(previous_elastic); }
  } config_guard{config, previous_elastic};
  DistributedVirtualMemoryCoordinatorTestPeer::refresh_worker_capacity_snapshot(
      *coordinator);
  EXPECT_EQ(coordinator->available_pages("model", /*dp_rank=*/0), 3);
  const int32_t first_calls = first.service().memory_info_calls();
  const int32_t second_calls = second.service().memory_info_calls();
  ASSERT_TRUE(coordinator->try_reserve_pages(
      "model", /*dp_rank=*/0, /*physical_pages=*/8));
  EXPECT_EQ(coordinator->available_pages("model", /*dp_rank=*/0), 2);
  EXPECT_EQ(first.service().memory_info_calls(), first_calls);
  EXPECT_EQ(second.service().memory_info_calls(), second_calls);
}

TEST(DistributedVirtualMemoryCoordinatorTest,
     WeightShortageRollsBackWorkersAndRestoresLedger) {
  TransactionTestServer first;
  TransactionTestServer second(MappingFailure::WEIGHT);
  ASSERT_EQ(first.start(), 0);
  ASSERT_EQ(second.start(), 0);
  auto coordinator = DistributedVirtualMemoryCoordinatorTestPeer::create();
  coordinator->init_page_budget(/*total_pages=*/10,
                                /*dp_size=*/1,
                                /*worker_count=*/2,
                                /*page_size=*/4096,
                                /*enable_prealloc=*/false);
  DistributedVirtualMemoryCoordinatorTestPeer::connect(
      *coordinator, {first.address(), second.address()});
  EXPECT_FALSE(coordinator->reserve_weight_pages("model", /*num_pages=*/3));
  EXPECT_FALSE(first.service().has_mappings());
  EXPECT_FALSE(second.service().has_mappings());
  EXPECT_GT(first.service().abort_calls(), 0);
  EXPECT_GT(second.service().abort_calls(), 0);
  EXPECT_EQ(coordinator->get_worker_free_page_counts(),
            std::vector<size_t>({10, 10}));
  EXPECT_EQ(coordinator->get_weight_page_count("model"), 0);
}

}  // namespace
}  // namespace xllm
