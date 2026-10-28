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

"""Request-seeded categorical selection from nonnegative row weights."""

import tilelang.language as T
from tilelang import tvm

from .utils import DEFAULT_ASCEND_PASS_CONFIGS

MAX_ROWS = 4096
MAX_VOCAB = 262144
TILE_SIZE = 4096
MAX_CHUNKS = MAX_VOCAB // TILE_SIZE
SEEDED_SAMPLE_PASS_CONFIGS = {
    **DEFAULT_ASCEND_PASS_CONFIGS,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


@T.macro
def select_categorical_half(
    half: int,
    tile: tvm.tir.Buffer,
    reduced: tvm.tir.Buffer,
    right: tvm.tir.Buffer,
    scratch: tvm.tir.Buffer,
    state: tvm.tir.Buffer,
    choice: tvm.tir.Buffer,
) -> None:
    T.tile.fill(scratch, 0.0)
    T.copy(tile[choice[1] : choice[1] + half], scratch[:half])
    T.reduce_sum(scratch, reduced, dim=0)
    T.set_flag("v", "s", 7)
    T.wait_flag("v", "s", 7)
    T.tile.fill(scratch, 0.0)
    T.copy(tile[choice[1] + half : choice[1] + 2 * half], scratch[:half])
    T.reduce_sum(scratch, right, dim=0)
    T.set_flag("v", "s", 7)
    T.wait_flag("v", "s", 7)
    if right[0] > 0.0 and (state[3] >= reduced[0] or reduced[0] == 0.0):
        state[3] = T.min(T.max(state[3] - reduced[0], 0.0), right[0] * (1.0 - 2.0**-24))
        choice[1] = choice[1] + half
    else:
        state[3] = T.min(state[3], reduced[0] * (1.0 - 2.0**-24))


def build_seeded_sample_kernel(task_count: int = 48) -> tvm.tir.PrimFunc:
    """Select a tile and then a token without a vocabulary-sized CDF buffer.

    Each vector core owns complete rows, so reduction order and random draws
    are independent of row order and batch size. The caller supplies uniform
    draws only for rows whose seed is -1. Weights must be finite, nonnegative,
    and contain at least one positive value in each sampled row.
    """
    if task_count <= 0 or task_count % 2:
        raise ValueError("task_count must be a positive even integer")

    @T.prim_func
    def seeded_sample(
        weights: T.Tensor((MAX_ROWS * MAX_VOCAB,), "float32"),
        seeds: T.Tensor((MAX_ROWS,), "int64"),
        offsets: T.Tensor((MAX_ROWS,), "int64"),
        uniforms: T.Tensor((MAX_ROWS,), "float32"),
        output: T.Tensor((MAX_ROWS,), "int64"),
        rows: T.int32,
        vocab: T.int32,
        row_stride: T.int32,
        rng_flags: T.int32,
    ) -> None:
        with T.Kernel(task_count // 2, is_npu=True) as (cid, vid):
            task = cid * 2 + vid
            with T.Scope("V"):
                tile = T.alloc_ub((TILE_SIZE,), "float32")
                chunks = T.alloc_ub((MAX_CHUNKS,), "float32")
                reduced = T.alloc_ub((1,), "float32")
                right = T.alloc_ub((1,), "float32")
                scratch = T.alloc_ub((TILE_SIZE // 2,), "float32")
                state = T.alloc_ub((8,), "float32")
                choice = T.alloc_ub((8,), "int32")
                result = T.alloc_ub((4,), "int64")
                for row_index in T.serial(T.ceildiv(rows, task_count)):
                    row = row_index * task_count + task
                    if row < rows:
                        n_chunks = T.ceildiv(vocab, TILE_SIZE)
                        state[0] = 0.0
                        for chunk in T.serial(n_chunks):
                            count = T.min(TILE_SIZE, vocab - chunk * TILE_SIZE)
                            T.tile.fill(tile, 0.0)
                            T.copy(
                                weights[
                                    row * row_stride + chunk * TILE_SIZE : row * row_stride + chunk * TILE_SIZE + count
                                ],
                                tile,
                            )
                            T.reduce_sum(tile, reduced, dim=0)
                            T.set_flag("v", "s", 7)
                            T.wait_flag("v", "s", 7)
                            chunks[chunk] = reduced[0]
                            state[0] = state[0] + reduced[0]

                        uniform = T.alloc_var("float32", 0.0)
                        if (rng_flags & 1) == 0 and (rng_flags & 2) != 0:
                            uniform = uniforms[row]
                        if (rng_flags & 1) != 0:
                            if seeds[row] == -1 and (rng_flags & 2) != 0:
                                uniform = uniforms[row]
                            if seeds[row] != -1:
                                value = T.alloc_var(
                                    "uint64",
                                    T.Cast("uint64", seeds[row])
                                    + T.Cast("uint64", offsets[row])
                                    * ((T.uint64(0x9E3779B9) << 32) | T.uint64(0x7F4A7C15)),
                                )
                                value = (value ^ (value >> 30)) * ((T.uint64(0xBF58476D) << 32) | T.uint64(0x1CE4E5B9))
                                value = (value ^ (value >> 27)) * ((T.uint64(0x94D049BB) << 32) | T.uint64(0x133111EB))
                                value = value ^ (value >> 31)
                                uniform = T.Cast("float32", T.Cast("int32", value >> 40)) * (1.0 / (1 << 24))

                        # Half-open intervals; clamp rounding at the last row
                        # endpoint. Positive lower bounds give one owner even
                        # when independently rounded interval ends disagree.
                        state[1] = T.min(uniform * state[0], state[0] * (1.0 - 2.0**-24))
                        state[2] = 0.0
                        choice[0] = 0
                        state[3] = 0.0
                        for chunk in T.serial(n_chunks):
                            if chunks[chunk] > 0.0 and state[2] <= state[1]:
                                choice[0] = chunk
                                state[3] = T.min(T.max(state[1] - state[2], 0.0), chunks[chunk] * (1.0 - 2.0**-24))
                            state[2] = state[2] + chunks[chunk]

                        count = T.min(TILE_SIZE, vocab - choice[0] * TILE_SIZE)
                        T.tile.fill(tile, 0.0)
                        T.copy(
                            weights[
                                row * row_stride + choice[0] * TILE_SIZE : row * row_stride
                                + choice[0] * TILE_SIZE
                                + count
                            ],
                            tile,
                        )
                        # Reduce a binary tree in UB. The final 32 values are
                        # scalar to avoid unsupported sub-block vector slices.
                        choice[1] = 0
                        select_categorical_half(2048, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(1024, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(512, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(256, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(128, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(64, tile, reduced, right, scratch, state, choice)
                        select_categorical_half(32, tile, reduced, right, scratch, state, choice)
                        choice[2] = choice[1]
                        state[2] = 0.0
                        for column in T.serial(32):
                            weight = tile[choice[1] + column]
                            if weight > 0.0 and state[2] <= state[3]:
                                choice[2] = choice[1] + column
                            state[2] = state[2] + weight
                        result[0] = T.Cast("int64", choice[0] * TILE_SIZE + choice[2])
                        T.copy(result[:1], output[row : row + 1])

    return seeded_sample
