# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2025 sgl-project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Upstream implementation revision: 6fba476a4c4387fdbb4aec09f0619415b9495f0d.
"""Same-shape sigmoid/multiply fusion with FP32 intermediates."""

from __future__ import annotations

from functools import cache

import torch
import triton
import triton.language as tl

_ELEM_BLOCK_SIZE = 2048


@triton.jit
def _fused_sigmoid_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    block_start = pid * BLOCK_SIZE

    for offset in tl.range(block_start, n_elements, num_programs * BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask).to(tl.float32)
        g = tl.load(gate_ptr + idx, mask=mask).to(tl.float32)
        out = x * tl.sigmoid(g)
        tl.store(out_ptr + idx, out.to(x_ptr.dtype.element_ty), mask=mask)


def fused_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` in one fused AIV kernel (same-shape).

    Args:
        x: input tensor, any shape.
        gate: gating tensor, must have the same shape as ``x``.

    Returns:
        A tensor with the same shape as ``x``.
    """
    if x.shape != gate.shape:
        raise ValueError(f"x and gate must have the same shape, got {x.shape} vs {gate.shape}")

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    _, num_vectorcore = _get_device_properties(torch.npu.current_device())
    num_blocks = min(triton.cdiv(n, _ELEM_BLOCK_SIZE), num_vectorcore)
    num_blocks = max(num_blocks, 1)

    _fused_sigmoid_mul_kernel[(num_blocks,)](
        x,
        gate,
        out,
        n,
        BLOCK_SIZE=_ELEM_BLOCK_SIZE,
    )
    return out


@cache
def _get_device_properties(device: int) -> tuple[int, int]:
    properties = triton.runtime.driver.active.utils.get_device_properties(device)
    num_aicore = properties.get("num_aicore", -1)
    num_vectorcore = properties.get("num_vectorcore", -1)
    if num_aicore <= 0 or num_vectorcore <= 0:
        raise RuntimeError("Failed to detect NPU device properties")
    return num_aicore, num_vectorcore
