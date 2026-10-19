# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint adapter shared by Phi-3 loader and CUDA tests."""

from __future__ import annotations

import torch


class StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def keys(self) -> list[str]:
        return list(self.tensors)

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]
