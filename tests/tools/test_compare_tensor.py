# Copyright 2026 The xLLM Authors.
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

"""Regression tests for standalone CPU tensor comparison and verbose output."""

from unittest.mock import patch

import pytest
import torch

from tools import compare_tensor


@pytest.mark.parametrize(
    ("shape", "coordinate", "expected_coordinate"),
    [
        pytest.param((3,), (2,), "2", id="1d"),
        pytest.param((2, 3), (1, 2), "1,2", id="2d"),
        pytest.param((2, 3, 4), (1, 2, 3), "1,2,3", id="3d"),
        pytest.param((2, 2, 3, 4), (1, 1, 2, 3), "1,1,2,3", id="4d"),
    ],
)
def test_verbose_reports_complete_coordinates_and_values(
    shape: tuple[int, ...], coordinate: tuple[int, ...], expected_coordinate: str
) -> None:
    a = torch.zeros(shape, dtype=torch.float32)
    b = a.clone()
    a[coordinate] = 1.5
    b[coordinate] = 7.25

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 1
    log_info.assert_called_once_with(f"diff at {expected_coordinate}: 1.500000 - 7.250000 = -5.750000")


def test_verbose_reports_every_difference_with_its_own_values() -> None:
    a = torch.zeros((2, 3, 4))
    b = a.clone()
    a[0, 1, 2] = 2.0
    b[0, 1, 2] = 0.5
    a[1, 2, 3] = -1.0
    b[1, 2, 3] = 3.0

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 2
    assert log_info.call_count == 2
    assert {entry.args[0] for entry in log_info.call_args_list} == {
        "diff at 0,1,2: 2.000000 - 0.500000 = 1.500000",
        "diff at 1,2,3: -1.000000 - 3.000000 = -4.000000",
    }


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_does_not_change_difference_count(verbose: bool) -> None:
    a = torch.tensor([0.0, 1.0, 2.0, 3.0])
    b = torch.tensor([0.0, 4.0, 2.0, 8.0])

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=verbose)

    assert type(count) is int
    assert count == 2
    assert log_info.call_count == (2 if verbose else 0)


def test_default_comparison_does_not_log_differences() -> None:
    a = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    b = torch.tensor([[4.0, 1.0], [2.0, 5.0]])

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b)

    assert type(count) is int
    assert count == 2
    log_info.assert_not_called()


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)], ids=["1d", "2d", "3d"])
def test_equal_tensors_return_zero_without_logging(shape: tuple[int, ...]) -> None:
    a = torch.ones(shape)
    b = a.clone()

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 0
    log_info.assert_not_called()


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        pytest.param((2,), (3,), id="different-sizes"),
        pytest.param((2, 3), (3,), id="broadcastable-shapes"),
    ],
)
def test_shape_mismatch_is_rejected(shape_a: tuple[int, ...], shape_b: tuple[int, ...], verbose: bool) -> None:
    a = torch.zeros(shape_a)
    b = torch.zeros(shape_b)

    with (
        patch.object(compare_tensor.logger, "info") as log_info,
        pytest.raises(ValueError, match="Shape mismatch") as error,
    ):
        compare_tensor.compare_tensors(a, b, verbose=verbose)

    assert str(a.shape) in str(error.value)
    assert str(b.shape) in str(error.value)
    log_info.assert_not_called()


def test_only_differences_strictly_above_tolerance_are_logged() -> None:
    a = torch.zeros(3, dtype=torch.float64)
    b = torch.tensor([0.0625, 0.125, 0.25], dtype=torch.float64)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, tol=0.125, verbose=True)

    assert count == 1
    log_info.assert_called_once_with("diff at 2: 0.000000 - 0.250000 = -0.250000")


def test_default_tolerance_is_preserved() -> None:
    a = torch.zeros(3, dtype=torch.float64)
    b = torch.tensor([0.0, 0.5e-6, 2e-6], dtype=torch.float64)

    assert compare_tensor.compare_tensors(a, b) == 1


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)], ids=["empty-1d", "empty-3d"])
def test_empty_tensors_return_zero_without_logging(shape: tuple[int, ...]) -> None:
    a = torch.empty(shape)
    b = torch.empty(shape)

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert type(count) is int
    assert count == 0
    log_info.assert_not_called()


def test_verbose_supports_non_contiguous_tensors_without_modifying_inputs() -> None:
    a = torch.zeros((2, 3, 4)).transpose(0, 2)
    b = a.clone()
    a[3, 2, 1] = 1.5
    b[3, 2, 1] = 7.25
    original_a = a.clone()
    original_b = b.clone()
    assert not a.is_contiguous()
    assert not b.is_contiguous()

    with patch.object(compare_tensor.logger, "info") as log_info:
        count = compare_tensor.compare_tensors(a, b, verbose=True)

    assert count == 1
    log_info.assert_called_once_with("diff at 3,2,1: 1.500000 - 7.250000 = -5.750000")
    assert torch.equal(a, original_a)
    assert torch.equal(b, original_b)
