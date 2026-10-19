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

#include "core/framework/allocator/virtual_memory/physical_page.h"

namespace xllm {

PhysicalPage::PhysicalPage(torch::Device device,
                           size_t page_size,
                           page_id_t page_id,
                           std::nullptr_t)
    : device_(device), page_size_(page_size), page_id_(page_id) {}

PhysicalPage::PhysicalPage(torch::Device device,
                           size_t page_size,
                           page_id_t page_id)
    : PhysicalPage(device, page_size, page_id, nullptr) {
  vmm::create_phy_mem_handle(phy_handle_, device_.index(), page_size_);
  owns_handle_ = true;
}

std::unique_ptr<PhysicalPage> PhysicalPage::try_create(torch::Device device,
                                                       size_t page_size,
                                                       page_id_t page_id) {
  auto page = std::unique_ptr<PhysicalPage>(
      new PhysicalPage(device, page_size, page_id, nullptr));
  if (!vmm::try_create_phy_mem_handle(
          page->phy_handle_, device.index(), page_size)) {
    return nullptr;
  }
  page->owns_handle_ = true;
  return page;
}

PhysicalPage::~PhysicalPage() {
  if (owns_handle_) {
    vmm::release_phy_mem_handle(phy_handle_);
  }
}

}  // namespace xllm
