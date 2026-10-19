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

"""Phi-3 production configuration and fused checkpoint loading contracts."""

from __future__ import annotations

import copy
import re
from unittest.mock import Mock

import pytest
import torch

from tests.python.phi3_test_utils import StateDict as _StateDict
from xllm.python import registry
from xllm.python.models.phi3 import Phi3Config, Phi3ForCausalLM
from xllm.python.registry import get_model_class

_CONFIG = {
    "model_type": "phi3",
    "hidden_size": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 12,
    "num_hidden_layers": 2,
    "vocab_size": 16,
    "max_position_embeddings": 32,
    "sliding_window": 7,
    "device": "cpu",
    "dtype": "float32",
}


def _checkpoint() -> dict[str, torch.Tensor]:
    # Derive only destination names from the model, not the shard expectations.
    model = Phi3ForCausalLM(_CONFIG)
    return {
        name: torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        for name, parameter in model.named_parameters()
    }


def test_registry_resolves_phi3_architecture_and_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "cuda")
    assert get_model_class("Phi3ForCausalLM") is Phi3ForCausalLM
    assert get_model_class("phi3") is Phi3ForCausalLM


def test_registry_rejects_unvalidated_npu_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import_model = Mock()
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    monkeypatch.setattr(registry, "import_module", import_model)
    with pytest.raises(NotImplementedError, match="phi3.*npu"):
        get_model_class("phi3")
    import_model.assert_not_called()


def test_normalized_engine_dimensions_override_generic_defaults() -> None:
    cfg = Phi3Config.from_dict({**_CONFIG, "n_heads": 2, "n_kv_heads": 2, "n_layers": 3, "head_dim": 4})
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.head_dim) == (2, 2, 3, 4)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rope_scaling": {"type": "longrope"}}, "scaled RoPE"),
        ({"rope_scaling_rope_type": "longrope"}, "scaled RoPE"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"tie_word_embeddings": True}, "untied"),
        ({"attention_bias": True}, "bias-free"),
        ({"cp_size": 2}, "context parallelism"),
        ({"cp_size": 1.5}, "context parallelism"),
        ({"cp_size": "1"}, "context parallelism"),
        ({"cp_size": True}, "context parallelism"),
        ({"tp_size": 3}, "divisible"),
        ({"tp_size": 0}, "positive integer"),
        ({"dp_rank": 1}, "parallel rank"),
        ({"tp_rank": True}, "parallel rank"),
        ({"world_size": 2}, "world_size"),
        ({"world_size": 1.0}, "world_size"),
        ({"world_size": True}, "world_size"),
        ({"head_dim": 4}, "head_dim"),
    ],
)
def test_rejects_unsupported_configuration(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Phi3Config.from_dict({**_CONFIG, **changes})


def test_window_includes_current_token() -> None:
    model = Phi3ForCausalLM(_CONFIG)
    assert model.model.layers[0].self_attn.attn.sliding_window == 8


@pytest.mark.parametrize("rank", [0, 1])
def test_production_loader_shards_each_fused_group_and_endpoints(rank: int) -> None:
    tensors = _checkpoint()
    before = {name: value.clone() for name, value in tensors.items()}
    # Reverse order and split across shards: loading must not depend on file order.
    items = list(reversed(tensors.items()))
    shards = [_StateDict(dict(items[::2])), _StateDict(dict(items[1::2]))]
    model = Phi3ForCausalLM({**_CONFIG, "tp_size": 2, "tp_rank": rank})
    model.load_weights(shards, rank, 2)
    for name, parameter in model.named_parameters():
        source = tensors[name]
        if name.endswith("qkv_proj.weight"):
            # Source Q has 8 rows, K/V 4 each; rank gets 4Q + 2K + 2V.
            rows = ([0, 1, 2, 3, 8, 9, 12, 13], [4, 5, 6, 7, 10, 11, 14, 15])[rank]
            expected = source[rows]
        elif name.endswith("gate_up_proj.weight"):
            rows = (list(range(6)) + list(range(12, 18)), list(range(6, 12)) + list(range(18, 24)))[rank]
            expected = source[rows]
        elif name == "model.embed_tokens.weight" or name.endswith(("o_proj.weight", "down_proj.weight")):
            width = source.shape[1] // 2
            expected = source[:, rank * width : (rank + 1) * width]
        elif name == "lm_head.weight":
            expected = source[rank * 8 : (rank + 1) * 8]
        else:
            expected = source
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors[name], before[name], rtol=0, atol=0)


def test_tp1_consumes_every_weight_exactly() -> None:
    tensors = _checkpoint()
    model = Phi3ForCausalLM(_CONFIG)
    model.load_weights([_StateDict(tensors)], 0, 1)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, tensors[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "duplicate", "topology"])
def test_loader_rejects_bad_checkpoint_before_modifying_parameters(failure: str) -> None:
    tensors = _checkpoint()
    model = Phi3ForCausalLM(_CONFIG)
    for parameter in model.parameters():
        parameter.data.fill_(42)
    shards = [_StateDict(copy.copy(tensors))]
    key = "model.layers.1.mlp.down_proj.weight"
    if failure == "missing":
        del shards[0].tensors[key]
        message = "missing.*" + re.escape(key)
    elif failure == "extra":
        shards[0].tensors["unexpected.weight"] = torch.ones(1)
        message = "unexpected.weight"
    elif failure == "shape":
        shards[0].tensors[key] = torch.ones(1)
        message = re.escape(key) + ": expected shape"
    elif failure == "dtype":
        shards[0].tensors[key] = tensors[key].to(torch.int32)
        message = re.escape(key) + ": expected floating-point"
    elif failure == "duplicate":
        shards.append(_StateDict({key: tensors[key]}))
        message = re.escape(key) + ": duplicate"
    else:
        message = "topology"
    with pytest.raises(ValueError, match=message):
        model.load_weights(shards, 0, 2 if failure == "topology" else 1)
    for parameter in model.parameters():
        assert torch.all(parameter == 42)


@pytest.mark.parametrize("window", [None, -1])
def test_null_or_disabled_window_matches_reference(window: int | None) -> None:
    cfg = Phi3Config.from_dict({**_CONFIG, "sliding_window": window})
    assert cfg.sliding_window == -1


@pytest.mark.parametrize("field", ["rms_norm_eps", "rope_theta"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "invalid", True])
def test_rejects_invalid_numerical_parameters(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Phi3Config.from_dict({**_CONFIG, field: value})
