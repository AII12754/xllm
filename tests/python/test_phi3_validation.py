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

"""Acceptance must reject truncated inputs and either direction of mismatch."""

from typing import Any

import pytest

from tools.validate_phi3 import _check_topk, _validate_candidate


@pytest.mark.parametrize(
    ("actual", "reference", "passed"),
    [
        ([3, 4, 5, 6, 7], [4, 3, 2, 1, 0], True),
        ([3, 4, 5, 6, 7], [4, 2, 1, 0, 8], False),
        ([3, 5, 6, 7, 8], [4, 3, 2, 1, 0], False),
    ],
)
def test_mutual_topk_requires_both_predictions(actual: list[int], reference: list[int], passed: bool) -> None:
    assert _check_topk(actual, reference) == passed


@pytest.mark.parametrize("failure", ["incomplete", "input", "steps", "ids", "nonfinite"])
def test_candidate_validation_rejects_incomplete_or_corrupt_evidence(failure: str) -> None:
    expected = [{"length": 2, "synthetic": False, "token_ids": [1, 2]}]
    case = {**expected[0], "steps": [{"ids": [3, 4, 5, 6, 7], "logprobs": [-0.5] * 5} for _ in range(8)]}
    candidate: dict[str, Any] = {"protocol": "phi3-bf16-mutual-top5-v1", "seed": 20261019, "cases": [case]}
    _validate_candidate(candidate, expected)
    if failure == "incomplete":
        candidate["cases"] = []
    elif failure == "input":
        case["token_ids"] = [1, 3]
    elif failure == "steps":
        case["steps"].pop()
    elif failure == "ids":
        case["steps"][-1]["ids"] = [3] * 5
    else:
        case["steps"][-1]["logprobs"][-1] = float("nan")
    with pytest.raises(ValueError):
        _validate_candidate(candidate, expected)
