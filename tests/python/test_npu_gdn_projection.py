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

"""Real NPU and ACL graph coverage for the grouped GDN input projection."""

from __future__ import annotations

import pytest
import torch

from tests.python.qwen3_5_test_utils import make_linear_config
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_grouped_gdn_projection_supports_meta_tensors(tp_size: int) -> None:
    cfg = make_linear_config(
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        tp_size=tp_size,
        world_size=tp_size,
        moe_tp_size=tp_size,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("meta"))
    hidden = torch.empty(17, cfg.hidden_size, dtype=torch.bfloat16, device="meta")

    qkv, a, b, z = layer._project_prefill_inputs(hidden)

    assert qkv.shape == (17, layer.conv_dim)
    assert a.shape == b.shape == (17, layer.num_v_heads)
    assert z.shape == (17, layer.num_v_heads, layer.value_head_dim)
    for value in (qkv, a, b, z):
        assert value.device.type == "meta"
        assert value.dtype == torch.bfloat16
        assert value.is_contiguous()


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("value_head_ratio", [1, 3])
@pytest.mark.parametrize("num_tokens", [1, 17, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "aclgraph"])
@pytest.mark.parametrize("strided_input", [False, True], ids=["contiguous", "strided"])
@torch.inference_mode()
def test_grouped_gdn_projection_matches_separate_on_npu(
    tp_size: int,
    value_head_ratio: int,
    num_tokens: int,
    dtype: torch.dtype,
    use_graph: bool,
    strided_input: bool,
) -> None:
    torch.manual_seed(2031)
    cfg = make_linear_config(
        hidden_size=512,
        linear_num_key_heads=4,
        linear_num_value_heads=4 * value_head_ratio,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        tp_size=tp_size,
        world_size=tp_size,
        moe_tp_size=tp_size,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, dtype, torch.device("npu:0"))
    layer.in_proj_qkvz.weight.normal_(std=0.02)
    layer.in_proj_ba.weight.normal_(std=0.02)
    storage = torch.randn(num_tokens, cfg.hidden_size * 2, dtype=dtype, device="npu:0") * 0.1
    hidden = storage[:, ::2] if strided_input else storage[:, : cfg.hidden_size].contiguous()
    stream = torch.npu.Stream(device="npu:0")
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            layer._project_prefill_inputs(hidden)
        if use_graph:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                outputs = layer._project_prefill_inputs(hidden)
        for scale in (0.0, 0.2, 1.0):
            hidden.normal_(std=scale)
            expected = layer._project_verify_inputs(hidden)
            if use_graph:
                graph.replay()
            else:
                outputs = layer._project_prefill_inputs(hidden)
            torch.npu.synchronize()
            for actual, reference in zip(outputs, expected):
                assert actual.is_contiguous()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=2e-4)
                error = (actual.float() - reference.float()).square().mean()
                signal = reference.float().square().mean().clamp_min(1e-16)
                assert (error / signal).sqrt().item() <= 0.005
    torch.npu.current_stream().wait_stream(stream)
