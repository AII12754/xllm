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
#include <cstddef>
#include <limits>
#include <optional>

namespace xllm {

// Tracks execution residency independently of elastic physical backing. Adding
// committed pool bytes back to driver free memory removes KV/weight growth from
// the sample. Warmup establishes the lowest observed non-pool availability.
class VmmMemoryBudget final {
 public:
  void observe(size_t free_bytes, size_t committed_bytes) {
    const size_t available =
        outside_pool_available(free_bytes, committed_bytes);
    low_watermark_ = low_watermark_.has_value()
                         ? std::min(*low_watermark_, available)
                         : available;
  }

  size_t headroom(size_t free_bytes,
                  size_t committed_bytes,
                  size_t configured_reserve) const {
    if (!low_watermark_.has_value()) {
      return configured_reserve;
    }
    const size_t available =
        outside_pool_available(free_bytes, committed_bytes);
    const size_t recovery =
        available > *low_watermark_ ? available - *low_watermark_ : 0;
    // Resident allocator caches already reduce driver free memory. Reserve
    // only the additional bytes needed if those caches have been released.
    return saturating_add(configured_reserve, recovery);
  }

  void reset() { low_watermark_.reset(); }

 private:
  static size_t saturating_add(size_t lhs, size_t rhs) {
    const size_t maximum = std::numeric_limits<size_t>::max();
    return lhs > maximum - rhs ? maximum : lhs + rhs;
  }

  static size_t outside_pool_available(size_t free_bytes,
                                       size_t committed_bytes) {
    return saturating_add(free_bytes, committed_bytes);
  }

  std::optional<size_t> low_watermark_;
};

}  // namespace xllm
