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

#include <dlfcn.h>
#include <link.h>

#include <filesystem>
#include <system_error>

namespace xllm {

// The anchor must belong to the native host whose adjacent bundle is needed.
inline std::filesystem::path native_host_path(const void* anchor,
                                              std::error_code& error) {
  Dl_info location{};
  void* extra = nullptr;
  if (::dladdr1(anchor, &location, &extra, RTLD_DL_LINKMAP) == 0 ||
      extra == nullptr || location.dli_fname == nullptr) {
    error = std::make_error_code(std::errc::no_such_file_or_directory);
    return {};
  }
  const auto* mapping = static_cast<const link_map*>(extra);
  // glibc leaves the main executable's link-map name empty, even when
  // dladdr reports argv[0] as a bare name or a path relative to an old cwd.
  // Shared hosts must keep their own location, not the Python executable's.
  const char* path =
      mapping->l_name[0] == '\0' ? "/proc/self/exe" : location.dli_fname;
  return std::filesystem::canonical(path, error);
}

}  // namespace xllm
