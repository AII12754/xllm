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

"""Phi-3 mini 4K decoder with native fused checkpoint loading.

Each layer applies input RMSNorm, QKV, full rotate-half RoPE, causal window
attention, output projection and a residual, then RMSNorm, SiLU(gate) * up,
down projection and a residual. The final RMSNorm consumes the last residual.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    apply_rotary_half,
    gather_half_rope_cos_sin,
)
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, load_causal_lm_weights
from xllm.python.models.base import PyModelBase
from xllm.python.models.phi3_checkpoint import (
    Phi3CheckpointConfig,
    expected_phi3_shapes,
    parse_phi3_config,
    shard_phi3_fused_weight,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


@dataclass(frozen=True)
class Phi3Config(Phi3CheckpointConfig):
    vocab_size: int = 32064
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096
    sliding_window: int = -1
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("vocab_size", "max_position_embeddings", "tp_size", "dp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "hidden_size"):
            if getattr(self, name) % self.tp_size:
                raise ValueError(f"{name} must be divisible by tp_size={self.tp_size}")
        for rank, size in ((self.tp_rank, self.tp_size), (self.dp_rank, self.dp_size)):
            if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
                raise ValueError(f"parallel rank {rank!r} must be an integer in [0, {size})")
        if self.head_dim % 2:
            raise ValueError("Phi-3 full rotary embedding requires an even head_dim")
        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(self.sliding_window, bool) or not isinstance(self.sliding_window, int):
            raise ValueError("sliding_window must be an integer")
        if self.sliding_window == 0 or self.sliding_window < -1:
            raise ValueError("sliding_window must be positive, or -1 to disable it")

    @classmethod
    def from_dict(cls, config: dict) -> Phi3Config:
        normalized = dict(config)
        for target, source in (
            ("num_hidden_layers", "n_layers"),
            ("num_attention_heads", "n_heads"),
            ("num_key_value_heads", "n_kv_heads"),
        ):
            if config.get(source) is not None:
                normalized[target] = config[source]
        dimensions = parse_phi3_config(normalized)
        if config.get("rope_scaling") not in (None, -1) or config.get("rope_scaling_rope_type"):
            raise ValueError("Phi-3 mini 4K does not support scaled RoPE")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Phi-3 requires hidden_act='silu'")
        if config.get("attention_bias", False) or config.get("tie_word_embeddings", False):
            raise ValueError("Phi-3 mini 4K requires bias-free attention and untied embeddings")
        cp_size = config.get("cp_size", 1)
        if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size != 1:
            raise ValueError("Phi-3 context parallelism requires integer cp_size=1")
        kwargs = {field.name: getattr(dimensions, field.name) for field in fields(Phi3CheckpointConfig)}
        kwargs.update(
            {
                field.name: config[field.name]
                for field in fields(cls)
                if config.get(field.name) is not None and field.name not in kwargs
            }
        )
        cfg = cls(**kwargs)
        world_size = config.get("world_size", cfg.tp_size * cfg.dp_size)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size != cfg.tp_size * cfg.dp_size:
            raise ValueError("world_size must be an integer equal to tp_size * dp_size")
        return cfg


class Phi3MLP(GatedMLP):
    def __init__(self, cfg: Phi3Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)
        self.cfg = cfg
        self.layer_id = layer_id

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        name = f"model.layers.{self.layer_id}.mlp.gate_up_proj.weight"
        value = shard_phi3_fused_weight(
            weights.get_tensor("gate_up_proj.weight"),
            name=name,
            config=self.cfg,
            tp_rank=context.tp_rank,
            tp_size=context.tp_size,
        )
        weights.copy(self.gate_up_proj.weight, value, "gate_up_proj.weight")
        weights.load_tensor(
            self.down_proj.weight, "down_proj.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size
        )


class Phi3Attention(nn.Module):
    def __init__(self, cfg: Phi3Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        num_heads = cfg.num_attention_heads // cfg.tp_size
        num_kv_heads = cfg.num_key_value_heads // cfg.tp_size
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            # The HF mask counts preceding tokens; xLLM counts the current token too.
            sliding_window=cfg.sliding_window + 1 if cfg.sliding_window > 0 else -1,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        half_cos: torch.Tensor,
        half_sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        num_tokens = qkv.size(0)
        q = qkv[:, : self.q_size].view(num_tokens, self.num_heads, self.head_dim)
        k = qkv[:, self.q_size : self.q_size + self.kv_size].view(num_tokens, self.num_kv_heads, self.head_dim)
        v = qkv[:, self.q_size + self.kv_size :]

        q = apply_rotary_half(q, half_cos, half_sin)
        k = apply_rotary_half(k, half_cos, half_sin)

        attn_out = self.attn(q.reshape(num_tokens, self.q_size), k.reshape(num_tokens, self.kv_size), v)
        return self.o_proj(attn_out)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        name = f"model.layers.{self.layer_id}.self_attn.qkv_proj.weight"
        value = shard_phi3_fused_weight(
            weights.get_tensor("qkv_proj.weight"),
            name=name,
            config=self.cfg,
            tp_rank=context.tp_rank,
            tp_size=context.tp_size,
        )
        weights.copy(self.qkv_proj.weight, value, "qkv_proj.weight")
        weights.load_tensor(
            self.o_proj.weight, "o_proj.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size
        )
        self.o_proj.process_weights_after_loading()


class Phi3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Phi3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Phi3Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = Phi3MLP(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        half_cos: torch.Tensor,
        half_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(hidden, half_cos, half_sin)

        residual_out: torch.Tensor
        hidden, residual_out = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual_out

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()


class Phi3Model(nn.Module):
    def __init__(self, cfg: Phi3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = cast(
            "Sequence[Phi3DecoderLayer]",
            nn.ModuleList(Phi3DecoderLayer(cfg, i, dtype, device) for i in range(cfg.num_hidden_layers)),
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids) if input_embedding is None else input_embedding

        positions = positions.to(torch.int64).contiguous()
        half_cos, half_sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, half_cos, half_sin)
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class Phi3ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Phi3Config.from_dict(config)
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "cuda"))
        self.model = Phi3Model(self.cfg, self.dtype, self.device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        if (tp_rank, tp_size) != (self.cfg.tp_rank, self.cfg.tp_size):
            raise ValueError("weight-loading topology must match the model topology")
        self._validate_checkpoint(state_dicts)
        weights = ScopedWeightLoader(state_dicts, src_prefixes=("model.", ""))
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            weights,
            ParallelLoadContext(tp_rank, tp_size),
            tie_word_embeddings=False,
        )

    def _validate_checkpoint(self, state_dicts: list[StateDict]) -> None:
        """Reject missing, extra, duplicate and malformed tensors before copying."""
        cfg = self.cfg
        shapes: dict[str, tuple[int, ...]] = dict(expected_phi3_shapes(cfg))
        shapes.update(
            {
                "model.embed_tokens.weight": (cfg.vocab_size, cfg.hidden_size),
                "model.norm.weight": (cfg.hidden_size,),
                "lm_head.weight": (cfg.vocab_size, cfg.hidden_size),
            }
        )
        for layer in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (cfg.hidden_size,),
                    f"{prefix}.post_attention_layernorm.weight": (cfg.hidden_size,),
                    f"{prefix}.self_attn.o_proj.weight": (cfg.hidden_size, cfg.hidden_size),
                    f"{prefix}.mlp.down_proj.weight": (cfg.hidden_size, cfg.intermediate_size),
                }
            )
        seen: set[str] = set()
        for state in state_dicts:
            for name in state.keys():
                if name in seen:
                    raise ValueError(f"{name}: duplicate checkpoint weight")
                if name not in shapes:
                    raise ValueError(f"{name}: unexpected checkpoint weight")
                tensor = state.get_tensor(name)
                if not tensor.is_floating_point():
                    raise ValueError(f"{name}: expected floating-point checkpoint weight, got {tensor.dtype}")
                shape = tuple(tensor.shape)
                if shape != shapes[name]:
                    raise ValueError(f"{name}: expected shape {shapes[name]}, got {shape}")
                seen.add(name)
        missing = shapes.keys() - seen
        if missing:
            raise ValueError(f"missing checkpoint weights: {sorted(missing)}")
