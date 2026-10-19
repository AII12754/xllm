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

#include "core/framework/allocator/vmm_memory_budget.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

TEST(VmmMemoryBudgetTest, PreserveConfiguredReserveBeforeWarmup) {
  VmmMemoryBudget budget;
  EXPECT_EQ(budget.headroom(/*free_bytes=*/600,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, ExcludePhysicalPoolGrowthFromExecutionWatermark) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/600, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/400, /*committed_bytes=*/400);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/200,
                            /*committed_bytes=*/600,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, DoNotCountResidentExecutionCachesTwice) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/300,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, ReserveObservedExecutionPeakAfterCacheRelease) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            250);
}

TEST(VmmMemoryBudgetTest, RetainLargestObservedExecutionFootprint) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/600, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  budget.observe(/*free_bytes=*/450, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/0),
            200);
}

TEST(VmmMemoryBudgetTest, PreserveWatermarkWhenIdlePhysicalPagesAreTrimmed) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  EXPECT_EQ(budget.headroom(/*free_bytes=*/500,
                            /*committed_bytes=*/0,
                            /*configured_reserve=*/50),
            50);
}

TEST(VmmMemoryBudgetTest, SaturateReserveWithoutUnsignedOverflow) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/0, /*committed_bytes=*/0);
  const size_t maximum = std::numeric_limits<size_t>::max();
  EXPECT_EQ(budget.headroom(maximum,
                            /*committed_bytes=*/1,
                            /*configured_reserve=*/1),
            maximum);
}

TEST(VmmMemoryBudgetTest, ResetDropsPreviousDeviceWatermark) {
  VmmMemoryBudget budget;
  budget.observe(/*free_bytes=*/300, /*committed_bytes=*/200);
  budget.reset();
  EXPECT_EQ(budget.headroom(/*free_bytes=*/600,
                            /*committed_bytes=*/200,
                            /*configured_reserve=*/50),
            50);
}

}  // namespace
}  // namespace xllm
