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

#include "core/runtime/worker_memory_resources.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/framework/allocator/global_virtual_memory_region.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "tests/core/framework/allocator/virtual_memory_test_utils.h"
#include "tests/npu_test_environment.h"

namespace xllm {

class WorkerMemoryResourcesTestPeer final {
 public:
  static void release_resources() {
    WorkerMemoryResources::get_instance().destroy();
  }
};

namespace {

constexpr char kModelId[] = "virtual-memory-kv-regression";
constexpr size_t kPoolPages = 10;

class WorkerMemoryEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    testing::init_npu_test_runtime();
    google::InitGoogleLogging("worker_memory_resources_test");
  }

  void TearDown() override {
    testing::finalize_npu_test_runtime();
    google::ShutdownGoogleLogging();
  }
};

::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new WorkerMemoryEnvironment);

class WorkerMemoryResourcesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint32_t device_count = 0;
    ASSERT_EQ(aclrtGetDeviceCount(&device_count), ACL_SUCCESS);
    if (device_count == 0) {
      GTEST_SKIP() << "No NPU device available";
    }
    ASSERT_EQ(aclrtSetDevice(/*device_id=*/0), ACL_SUCCESS);

    const torch::Device device("npu:0");
    WorkerMemoryResources::get_instance().init(device);
    auto& pool = PhysicalPagePool::get_instance();
    const size_t page_size = vmm::get_recommended_granularity(device.index());
    pool.init(device, kPoolPages, page_size);
    GlobalVirtualMemoryRegion::get_instance().init(device);
    resources_initialized_ = true;
    ASSERT_TRUE(GlobalVirtualMemoryRegion::get_instance().is_initialized());
    ASSERT_EQ(pool.num_available(), kPoolPages);
  }

  void TearDown() override {
    if (!resources_initialized_) {
      return;
    }
    auto& pool = PhysicalPagePool::get_instance();
    pool.release_reserved_pages(blocking_page_ids_);
    WorkerMemoryResourcesTestPeer::release_resources();
    VirtualMemoryTestPeer::release_resources();
  }

  bool resources_initialized_ = false;
  std::vector<page_id_t> blocking_page_ids_;
};

TEST_F(WorkerMemoryResourcesTest, KvShortageLeavesEveryLayerUnmapped) {
  auto& pool = PhysicalPagePool::get_instance();
  auto& allocator = WorkerMemoryResources::get_instance();
  const size_t page_size =
      GlobalVirtualMemoryRegion::get_instance().page_size();
  const std::vector<int64_t> dims = {static_cast<int64_t>(page_size)};
  const auto k_tensors = allocator.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  const auto v_tensors = allocator.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  ASSERT_EQ(k_tensors.size(), 2);
  ASSERT_EQ(v_tensors.size(), 2);
  blocking_page_ids_ = pool.allocate_pages_from_right(kPoolPages - 3);
  ASSERT_EQ(pool.num_available(), 3);

  EXPECT_FALSE(allocator.map_to_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), 3);
  for (int64_t layer_id = 0; layer_id < 2; ++layer_id) {
    const auto offsets = allocator.get_global_offsets_for_block(
        kModelId, layer_id, /*block_id=*/0, page_size);
    EXPECT_EQ(offsets.first, std::numeric_limits<uint64_t>::max());
    EXPECT_EQ(offsets.second, std::numeric_limits<uint64_t>::max());
  }

  pool.release_reserved_pages(blocking_page_ids_);
  blocking_page_ids_.clear();
  ASSERT_TRUE(allocator.map_to_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages - 4);
  ASSERT_TRUE(allocator.unmap_from_kv_tensors(kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

TEST_F(WorkerMemoryResourcesTest,
       CommittedMappingRemainsReversibleUntilRetired) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const size_t page_size = pool.page_size();
  const std::vector<int64_t> dims = {static_cast<int64_t>(2 * page_size)};
  const auto keys = resources.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  const auto values = resources.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  ASSERT_TRUE(resources.commit_kv_mapping(/*transaction_id=*/1));
  EXPECT_TRUE(resources.commit_kv_mapping(/*transaction_id=*/1));

  // A lost commit reply must still permit complete distributed rollback.
  ASSERT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), kPoolPages);
  EXPECT_FALSE(resources.commit_kv_mapping(/*transaction_id=*/1));
  EXPECT_FALSE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {/*offset=*/0}));
}

TEST_F(WorkerMemoryResourcesTest, AbortBeforePrepareBlocksDelayedRpc) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const std::vector<int64_t> dims = {static_cast<int64_t>(pool.page_size())};
  const auto keys = resources.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  const auto values = resources.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/42));
  EXPECT_FALSE(resources.prepare_kv_mapping(
      /*transaction_id=*/42, kModelId, {/*offset=*/0}));
  EXPECT_FALSE(resources.prepare_kv_mapping(
      /*transaction_id=*/7, kModelId, {/*offset=*/0}));
  EXPECT_EQ(pool.num_available(), kPoolPages);

  // DP routing may skip transaction numbers for this worker.
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/99, kModelId, {/*offset=*/0}));
  ASSERT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/99));
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

TEST_F(WorkerMemoryResourcesTest, ExistingMappingsAreNeverOwnedByRollback) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const size_t page_size = pool.page_size();
  const std::vector<int64_t> dims = {static_cast<int64_t>(2 * page_size)};
  const auto keys = resources.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  const auto values = resources.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(resources.map_to_kv_tensors(kModelId, {/*offset=*/0}));
  const auto before = resources.get_global_offsets_for_block(
      kModelId, /*layer_id=*/0, /*block_id=*/0, page_size);
  ASSERT_NE(before.first, std::numeric_limits<uint64_t>::max());

  EXPECT_FALSE(resources.prepare_kv_mapping(
      /*transaction_id=*/1,
      kModelId,
      {/*offset=*/0, static_cast<offset_t>(page_size)}));
  ASSERT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  EXPECT_EQ(resources.get_global_offsets_for_block(
                kModelId, /*layer_id=*/0, /*block_id=*/0, page_size),
            before);
  ASSERT_TRUE(resources.unmap_from_kv_tensors(kModelId, {/*offset=*/0}));
}

TEST_F(WorkerMemoryResourcesTest, LateAbortCannotTouchNewerMappings) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const size_t page_size = pool.page_size();
  const std::vector<int64_t> dims = {static_cast<int64_t>(2 * page_size)};
  const auto keys = resources.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  const auto values = resources.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {/*offset=*/0}));
  ASSERT_TRUE(resources.commit_kv_mapping(/*transaction_id=*/1));
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/2, kModelId, {static_cast<offset_t>(page_size)}));
  ASSERT_TRUE(resources.commit_kv_mapping(/*transaction_id=*/2));
  EXPECT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), kPoolPages - 4);
  EXPECT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/2));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  ASSERT_TRUE(resources.unmap_from_kv_tensors(kModelId, {/*offset=*/0}));
}

TEST_F(WorkerMemoryResourcesTest, FailedPrepareDoesNotRetainPhysicalPages) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const std::vector<int64_t> dims = {static_cast<int64_t>(pool.page_size())};
  const auto keys = resources.create_k_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  const auto values = resources.create_v_tensors(
      kModelId, dims, torch::kUInt8, /*num_layers=*/2);
  blocking_page_ids_ = pool.allocate_pages_from_right(kPoolPages - 3);
  EXPECT_FALSE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {/*offset=*/0}));
  EXPECT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), 3);
}

TEST_F(WorkerMemoryResourcesTest, LateAbortPreservesNormallyRemappedKvPages) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  const size_t page_size = pool.page_size();
  const offset_t second_offset = static_cast<offset_t>(page_size);
  const std::vector<int64_t> dims{static_cast<int64_t>(2 * page_size)};
  resources.create_k_tensors(kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  resources.create_v_tensors(kModelId, dims, torch::kUInt8, /*num_layers=*/1);
  ASSERT_TRUE(resources.prepare_kv_mapping(
      /*transaction_id=*/1, kModelId, {0, second_offset}));
  ASSERT_TRUE(resources.commit_kv_mapping(/*transaction_id=*/1));
  ASSERT_TRUE(resources.unmap_from_kv_tensors(kModelId, {0}));
  ASSERT_TRUE(resources.map_to_kv_tensors(kModelId, {0}));
  const auto reused = resources.get_global_offsets_for_block(
      kModelId, /*layer_id=*/0, /*block_id=*/0, page_size);

  ASSERT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_TRUE(resources.abort_kv_mapping(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  EXPECT_EQ(resources.get_global_offsets_for_block(
                kModelId, /*layer_id=*/0, /*block_id=*/0, page_size),
            reused);
  EXPECT_EQ(resources.get_global_offsets_for_block(
                kModelId, /*layer_id=*/0, /*block_id=*/1, page_size),
            std::make_pair(UINT64_MAX, UINT64_MAX));
  EXPECT_TRUE(resources.unmap_from_kv_tensors(kModelId, {0}));
}

TEST_F(WorkerMemoryResourcesTest, LateAbortPreservesReusedWeightModelName) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_TRUE(resources.prepare_weight_pages(
      /*transaction_id=*/1, kModelId, /*num_pages=*/2));
  ASSERT_TRUE(resources.commit_weight_pages(/*transaction_id=*/1));
  ASSERT_EQ(resources.free_weight(kModelId), 2);
  ASSERT_TRUE(resources.alloc_weight_pages(kModelId, /*num_pages=*/2));
  const auto reused = resources.get_weight_allocation_info(kModelId);
  ASSERT_TRUE(reused.has_value());

  ASSERT_TRUE(resources.abort_weight_pages(/*transaction_id=*/1));
  EXPECT_FALSE(resources.prepare_weight_pages(
      /*transaction_id=*/1, kModelId, /*num_pages=*/2));
  const auto actual = resources.get_weight_allocation_info(kModelId);
  ASSERT_TRUE(actual.has_value());
  EXPECT_EQ(actual->base_ptr, reused->base_ptr);
  EXPECT_EQ(actual->num_pages, reused->num_pages);
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  EXPECT_EQ(resources.free_weight(kModelId), 2);
}

TEST_F(WorkerMemoryResourcesTest, ExecutionHeadroomNeverShrinksAcrossProfiles) {
  auto& resources = WorkerMemoryResources::get_instance();
  const size_t page_size = PhysicalPagePool::get_instance().page_size();
  resources.set_execution_memory_reserve(2 * page_size);
  resources.set_execution_memory_reserve(page_size);
  const WorkerMemoryCapacity capacity = resources.memory_capacity_snapshot();
  EXPECT_TRUE(capacity.pool_initialized);
  EXPECT_EQ(capacity.pool_capacity_pages, kPoolPages);
  EXPECT_EQ(capacity.execution_memory_reserve_bytes, 2 * page_size);
}

TEST_F(WorkerMemoryResourcesTest, WeightPrepareRollbackReleasesOnlyFreshPages) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_TRUE(resources.alloc_weight_pages(kModelId, /*num_pages=*/2));
  const auto before = resources.get_weight_allocation_info(kModelId);
  ASSERT_TRUE(before.has_value());
  EXPECT_FALSE(resources.prepare_weight_pages(
      /*transaction_id=*/1, kModelId, /*num_pages=*/2));
  ASSERT_TRUE(resources.abort_weight_pages(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), kPoolPages - 2);
  const auto after = resources.get_weight_allocation_info(kModelId);
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->base_ptr, before->base_ptr);
  EXPECT_EQ(after->num_pages, before->num_pages);
  EXPECT_EQ(resources.free_weight(kModelId), 2);

  ASSERT_TRUE(resources.prepare_weight_pages(
      /*transaction_id=*/2, kModelId, /*num_pages=*/3));
  EXPECT_EQ(pool.num_available(), kPoolPages - 3);
  ASSERT_TRUE(resources.commit_weight_pages(/*transaction_id=*/2));
  ASSERT_TRUE(resources.abort_weight_pages(/*transaction_id=*/2));
  EXPECT_TRUE(resources.abort_weight_pages(/*transaction_id=*/2));
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

TEST_F(WorkerMemoryResourcesTest, WeightAbortPreventsDelayedAllocation) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  ASSERT_TRUE(resources.abort_weight_pages(/*transaction_id=*/17));
  EXPECT_FALSE(resources.prepare_weight_pages(
      /*transaction_id=*/17, kModelId, /*num_pages=*/2));
  EXPECT_FALSE(resources.commit_weight_pages(/*transaction_id=*/17));
  EXPECT_EQ(pool.num_available(), kPoolPages);
}

TEST_F(WorkerMemoryResourcesTest, WeightShortageIsAnAtomicRejection) {
  auto& resources = WorkerMemoryResources::get_instance();
  auto& pool = PhysicalPagePool::get_instance();
  blocking_page_ids_ = pool.allocate_pages_from_right(kPoolPages - 1);
  EXPECT_FALSE(resources.prepare_weight_pages(
      /*transaction_id=*/1, kModelId, /*num_pages=*/2));
  EXPECT_TRUE(resources.abort_weight_pages(/*transaction_id=*/1));
  EXPECT_EQ(pool.num_available(), 1);
  const auto allocation = resources.get_weight_allocation_info(kModelId);
  EXPECT_TRUE(!allocation.has_value() || allocation->num_pages == 0);
}

}  // namespace
}  // namespace xllm
