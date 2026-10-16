# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_library)

cc_library(
  NAME
    kv_cache_events
  HDRS
    kv_cache_event_journal.h
    kv_cache_event_publisher.h
  SRCS
    kv_cache_event_journal.cpp
    kv_cache_event_publisher.cpp
  DEPS
    glog::glog
    nlohmann_json::nlohmann_json
    Threads::Threads
)
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  # Explicit development-library paths remain available for offline builds.
  target_include_directories(kv_cache_events SYSTEM PRIVATE ${ZMQ_INCLUDE_DIR})
  target_link_libraries(kv_cache_events PRIVATE ${ZMQ_LIBRARY})
else()
  find_package(ZeroMQ CONFIG REQUIRED)
  if(TARGET libzmq-static)
    target_link_libraries(kv_cache_events PRIVATE libzmq-static)
  else()
    target_link_libraries(kv_cache_events PRIVATE libzmq)
  endif()
endif()
