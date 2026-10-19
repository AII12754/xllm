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

#include "core/framework/allocator/global_virtual_memory_region.h"

#include <glog/logging.h>

#include "core/framework/allocator/virtual_memory/physical_page_pool.h"

namespace xllm {

GlobalVirtualMemoryRegion::GlobalVirtualMemoryRegion() {
  // The page owner must outlive this shared address space.
  PhysicalPagePool::get_instance();
}

void GlobalVirtualMemoryRegion::init(const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (mapping_->is_initialized()) {
    LOG(WARNING) << "GlobalVirtualMemoryRegion already initialized";
    return;
  }

  auto& pool = PhysicalPagePool::get_instance();
  CHECK(pool.is_initialized()) << "PhysicalPagePool must be initialized first";
  CHECK_EQ(device, pool.device())
      << "GlobalVirtualMemoryRegion device mismatch";
  pool.attach_shared_mapping(mapping_);
}

void GlobalVirtualMemoryRegion::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!mapping_->is_initialized()) {
    return;
  }

  CHECK(PhysicalPagePool::get_instance().detach_shared_mapping(mapping_.get()))
      << "GlobalVirtualMemoryRegion has active mapping leases";
  mapping_->reset();
}

std::shared_ptr<void> GlobalVirtualMemoryRegion::acquire_mapping_lease() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(mapping_->is_initialized())
      << "GlobalVirtualMemoryRegion is not initialized";
  return PhysicalPagePool::get_instance().acquire_shared_mapping_lease();
}

std::shared_ptr<void>
GlobalVirtualMemoryRegion::try_acquire_contiguous_mapping_lease() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!mapping_->is_initialized()) {
    return nullptr;
  }
  return PhysicalPagePool::get_instance().acquire_shared_mapping_lease(
      /*require_contiguous=*/true);
}

}  // namespace xllm
