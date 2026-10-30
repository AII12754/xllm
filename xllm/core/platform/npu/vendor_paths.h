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

#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace xllm {

// External OPP paths precede config.ini in CANN's search. Include the other
// backend vendors here so adding the bundled math ops preserves that order.
inline std::vector<std::string> bundled_npu_vendor_paths(
    const std::filesystem::path& bundle,
    const std::filesystem::path& system_vendors,
    const std::vector<std::string>& existing,
    bool is_python) {
  namespace fs = std::filesystem;
  const auto vendor_name = [](const std::string& value) {
    const fs::path path = fs::path(value).lexically_normal();
    return path.filename().empty() ? path.parent_path().filename()
                                   : path.filename();
  };
  const std::array<std::string, 3> order =
      is_python ? std::array<std::string, 3>{"glm_next_transformer",
                                             "custom_transformer",
                                             "custom_xllm_math"}
                : std::array<std::string, 3>{"custom_xllm_math",
                                             "custom_transformer",
                                             "glm_next_transformer"};
  std::vector<std::string> paths;
  paths.reserve(order.size() + existing.size());
  for (const std::string& vendor : order) {
    if (vendor == "custom_xllm_math") {
      paths.emplace_back(bundle.string());
      continue;
    }
    const auto external = std::find_if(
        existing.begin(), existing.end(), [&](const std::string& path) {
          return vendor_name(path) == vendor &&
                 fs::is_regular_file(fs::path(path) /
                                     "op_api/lib/libcust_opapi.so");
        });
    if (external != existing.end()) {
      paths.emplace_back(*external);
      continue;
    }
    if (system_vendors.empty()) {
      continue;
    }
    const fs::path installed = system_vendors / vendor;
    if (fs::is_regular_file(installed / "op_api/lib/libcust_opapi.so")) {
      paths.emplace_back(installed.string());
    }
  }
  for (const std::string& path : existing) {
    if (path.empty() || vendor_name(path) == "custom_xllm_math" ||
        std::find(paths.begin(), paths.end(), path) != paths.end()) {
      continue;
    }
    paths.emplace_back(path);
  }
  return paths;
}

}  // namespace xllm
