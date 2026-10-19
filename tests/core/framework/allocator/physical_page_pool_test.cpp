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

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace xllm {

// The production growth boundary is exercised without initializing a device
// runtime. Stub pages own no driver handles, but use the real pool state
// machine.
class PhysicalPagePoolTestPeer final {
 public:
  static void inject(
      PhysicalPagePool& pool,
      std::function<int64_t()> memory_query,
      std::function<std::unique_ptr<PhysicalPage>(page_id_t)> factory) {
    std::lock_guard<std::mutex> lock(pool.mtx_);
    pool.free_memory_query_ = std::move(memory_query);
    pool.page_factory_ = std::move(factory);
  }

  static std::unique_ptr<PhysicalPage> make_page(page_id_t page_id) {
    auto& pool = PhysicalPagePool::get_instance();
    return std::unique_ptr<PhysicalPage>(
        new PhysicalPage(pool.device_, pool.page_size_, page_id, nullptr));
  }
};

namespace {

constexpr size_t kCapacity = 6;
constexpr size_t kPageSize = 4096;

class PhysicalPagePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto& pool = PhysicalPagePool::get_instance();
    ASSERT_TRUE(pool.reset());
    pool.init(torch::Device(torch::kCPU), kCapacity, kPageSize, /*lazy=*/true);
    inject_factory();
  }

  void TearDown() override {
    EXPECT_TRUE(PhysicalPagePool::get_instance().reset());
  }

  void inject_factory() {
    PhysicalPagePoolTestPeer::inject(
        PhysicalPagePool::get_instance(),
        [this]() { return free_bytes_; },
        [this](page_id_t page_id) {
          ++allocation_attempts_;
          if (fail_on_attempt_ == allocation_attempts_) {
            return std::unique_ptr<PhysicalPage>{};
          }
          return PhysicalPagePoolTestPeer::make_page(page_id);
        });
  }

  int64_t free_bytes_ = kCapacity * kPageSize;
  size_t allocation_attempts_ = 0;
  size_t fail_on_attempt_ = 0;
};

TEST_F(PhysicalPagePoolTest, LazyInitializationSeparatesCapacityAndBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  EXPECT_EQ(pool.num_total(), kCapacity);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_allocatable(), kCapacity);
  EXPECT_EQ(allocation_attempts_, 0);
  for (const PhysicalPage* page : pool.get_all_pages()) {
    EXPECT_EQ(page, nullptr);
  }
  EXPECT_TRUE(pool.batch_get(/*count=*/0).empty());
  EXPECT_TRUE(pool.batch_get(kCapacity + 1).empty());
  EXPECT_EQ(allocation_attempts_, 0);
}

TEST_F(PhysicalPagePoolTest, PartialAllocationFailurePublishesNoBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  fail_on_attempt_ = 2;
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(allocation_attempts_, 2);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 0);
  for (const PhysicalPage* page : pool.get_all_pages()) {
    EXPECT_EQ(page, nullptr);
  }
  fail_on_attempt_ = 0;
  auto pages = pool.batch_get(/*count=*/3);
  ASSERT_EQ(pages.size(), 3);
  EXPECT_EQ(pages.front()->page_id(), 0);
  EXPECT_EQ(pool.num_committed(), 3);
  pool.batch_put(pages);
}

TEST_F(PhysicalPagePoolTest, FailedGrowthPreservesResidentBackingAndFreeSlots) {
  auto& pool = PhysicalPagePool::get_instance();
  auto page = pool.get();
  ASSERT_NE(page, nullptr);
  const PhysicalPage* original = page.get();
  pool.put(std::move(page));
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 1);
  EXPECT_EQ(pool.num_available(), kCapacity);
  EXPECT_EQ(pool.get_all_pages()[0], original);
  EXPECT_EQ(pool.get_all_pages()[1], nullptr);
}

TEST_F(PhysicalPagePoolTest,
       HeadroomChecksGrowthWithoutChargingIdlePagesAgain) {
  auto& pool = PhysicalPagePool::get_instance();
  pool.set_allocation_headroom(kPageSize);
  free_bytes_ = 3 * kPageSize - 1;
  EXPECT_EQ(pool.num_allocatable(), 1);
  EXPECT_TRUE(pool.batch_get(/*count=*/2).empty());
  EXPECT_EQ(allocation_attempts_, 0);
  free_bytes_ = 3 * kPageSize;
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 2);
  pages = pool.batch_get(/*count=*/2);
  EXPECT_EQ(pages.size(), 2);
  pool.batch_put(pages);
  pool.set_allocation_headroom(std::numeric_limits<size_t>::max());
  EXPECT_TRUE(pool.batch_get(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 2);
}

TEST_F(PhysicalPagePoolTest,
       ExecutionWatermarkAvoidsChargingResidentCachesTwice) {
  auto& pool = PhysicalPagePool::get_instance();
  free_bytes_ = 3 * kPageSize;
  pool.observe_execution_memory();
  free_bytes_ = 6 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 3);
  EXPECT_TRUE(pool.batch_get(/*count=*/4).empty());
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  // KV growth reduces free bytes but does not increase execution reserve.
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 1);
  // Resident workspace already reduces free bytes; do not charge it twice.
  free_bytes_ = kPageSize;
  pool.observe_execution_memory();
  EXPECT_EQ(pool.num_allocatable(), 1);
  pool.batch_put(pages);
  EXPECT_EQ(pool.num_allocatable(), 3);
  free_bytes_ = 4 * kPageSize;
  EXPECT_EQ(pool.num_allocatable(), 3);
  pool.set_allocation_headroom(kPageSize);
  EXPECT_EQ(pool.num_allocatable(), 2);
}

TEST_F(PhysicalPagePoolTest, TrimmingKeepsLiveIdsAndConfiguredIdleBacking) {
  auto& pool = PhysicalPagePool::get_instance();
  auto pages = pool.batch_get(/*count=*/4);
  ASSERT_EQ(pages.size(), 4);
  const page_id_t active_id = pages.front()->page_id();
  const PhysicalPage* active_ptr = pages.front().get();
  for (size_t i = 1; i < pages.size(); ++i) {
    pool.put(std::move(pages[i]));
  }
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/1), 2);
  EXPECT_EQ(pool.num_total(), kCapacity);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pages.front()->page_id(), active_id);
  EXPECT_EQ(pages.front().get(), active_ptr);
  EXPECT_FALSE(pool.reset());
  pool.batch_put(pages);
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 2);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
}

TEST_F(PhysicalPagePoolTest, WeightReservationsPinBackingUntilReleased) {
  auto& pool = PhysicalPagePool::get_instance();
  const page_id_t start = pool.allocate_contiguous_from_right(/*count=*/2);
  ASSERT_EQ(start, static_cast<page_id_t>(kCapacity - 2));
  auto kv_page = pool.get();
  ASSERT_NE(kv_page, nullptr);
  pool.put(std::move(kv_page));
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 1);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pool.num_available(), kCapacity - 2);
  pool.release_reserved_pages({start, start + 1});
  free_bytes_ = 0;
  EXPECT_EQ(pool.num_allocatable(), 2);
  // Returned resident weight slots must be reused before empty logical slots.
  auto pages = pool.batch_get(/*count=*/2);
  ASSERT_EQ(pages.size(), 2);
  EXPECT_GE(pages.front()->page_id(), start);
  pool.batch_put(pages);
  EXPECT_EQ(pool.trim_idle(/*keep_pages=*/0), 2);
}

TEST_F(PhysicalPagePoolTest, FailedWeightReservationLeavesNoPinnedSlots) {
  auto& pool = PhysicalPagePool::get_instance();
  fail_on_attempt_ = 2;
  EXPECT_EQ(pool.allocate_contiguous_from_right(/*count=*/3), -1);
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
  fail_on_attempt_ = allocation_attempts_ + 2;
  EXPECT_TRUE(pool.allocate_pages_from_right(/*count=*/3).empty());
  EXPECT_EQ(pool.num_committed(), 0);
  EXPECT_EQ(pool.num_available(), kCapacity);
}

TEST_F(PhysicalPagePoolTest, ConcurrentGrowthRechecksMemoryUnderOnePoolLock) {
  auto& pool = PhysicalPagePool::get_instance();
  std::atomic<int64_t> free_bytes(3 * kPageSize);
  PhysicalPagePoolTestPeer::inject(
      pool,
      [&free_bytes]() { return free_bytes.load(); },
      [&free_bytes](page_id_t page_id) {
        free_bytes.fetch_sub(kPageSize);
        return PhysicalPagePoolTestPeer::make_page(page_id);
      });
  std::vector<std::unique_ptr<PhysicalPage>> first;
  std::vector<std::unique_ptr<PhysicalPage>> second;
  std::thread first_thread([&]() { first = pool.batch_get(/*count=*/2); });
  std::thread second_thread([&]() { second = pool.batch_get(/*count=*/2); });
  first_thread.join();
  second_thread.join();
  EXPECT_EQ(first.size() + second.size(), 2);
  EXPECT_EQ(pool.num_committed(), 2);
  EXPECT_EQ(pool.num_available(), kCapacity - 2);
  pool.batch_put(first);
  pool.batch_put(second);
}

}  // namespace
}  // namespace xllm
