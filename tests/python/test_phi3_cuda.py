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

"""Real CUDA regressions for Phi-3's logical D=96 and padded paged cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from xllm.python.platform import current_platform

if not current_platform.is_cuda() or not torch.cuda.is_available():
    pytest.skip("requires a CUDA device", allow_module_level=True)
pytest.importorskip("flashinfer")

from tests.python.phi3_test_utils import StateDict as _StateDict
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.flashinfer import FlashInferBackend
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner
from xllm.python.models.phi3 import Phi3ForCausalLM

_CONFIG = {
    "model_type": "phi3",
    "hidden_size": 384,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "sliding_window": 15,
    "dtype": "bfloat16",
    "device": "cuda",
}
_BLOCK_SIZE = 16
_BLOCKS_PER_REQUEST = 8


def _reference(tokens: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Independent unfused HF equations, including the explicit causal window."""
    hidden = F.embedding(tokens, weights["model.embed_tokens.weight"])
    length = tokens.numel()
    positions = torch.arange(length, device=tokens.device)
    angles = positions.float()[:, None] * (10000.0 ** (-torch.arange(0, 96, 2, device=tokens.device).float() / 96))
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(hidden.dtype)[:, None]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(hidden.dtype)[:, None]
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance <= 15)

    def _norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-5)
        return normalized.to(value.dtype) * weight

    def _rotate(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return value * cos + torch.cat((-second, first), dim=-1) * sin

    for layer in range(2):
        prefix = f"model.layers.{layer}"
        normalized = _norm(hidden, weights[f"{prefix}.input_layernorm.weight"])
        q, k, v = F.linear(normalized, weights[f"{prefix}.self_attn.qkv_proj.weight"]).view(length, 3, 4, 96).unbind(1)
        q, k = _rotate(q), _rotate(k)
        attention = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], attn_mask=allowed
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(length, 384)
        )
        hidden = hidden + F.linear(attention, weights[f"{prefix}.self_attn.o_proj.weight"])
        normalized = _norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"])
        gate, up = F.linear(normalized, weights[f"{prefix}.mlp.gate_up_proj.weight"]).chunk(2, dim=-1)
        hidden = hidden + F.linear(F.silu(gate) * up, weights[f"{prefix}.mlp.down_proj.weight"])
    return F.linear(_norm(hidden, weights["model.norm.weight"]), weights["lm_head.weight"])


class _PagedRunner:
    def __init__(self, *, graph: bool = False) -> None:
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.model = Phi3ForCausalLM(_CONFIG).eval()
        generator = torch.Generator().manual_seed(2026)
        self.weights = {
            name: (
                torch.ones_like(parameter, device="cpu")
                if "norm.weight" in name
                else torch.randn(parameter.shape, generator=generator) * 0.02
            ).to(self.device, torch.bfloat16)
            for name, parameter in self.model.named_parameters()
        }
        self.model.load_weights([_StateDict(self.weights)], 0, 1)
        self.caches = [
            LayerCache(
                torch.full((16, _BLOCK_SIZE, 4, 128), float("nan"), dtype=torch.bfloat16, device=self.device),
                torch.full((16, _BLOCK_SIZE, 4, 128), float("nan"), dtype=torch.bfloat16, device=self.device),
            )
            for _ in range(2)
        ]
        self.backend = FlashInferBackend(
            4, 4, 96, 96**-0.5, _BLOCK_SIZE, self.device, torch.bfloat16, kv_cache_head_dim=128
        )
        self.backend.bind_kv_caches(self.caches)
        self.decode_graph: DecodeCudaGraphRunner | None = None
        if graph:
            self.decode_graph = DecodeCudaGraphRunner(self.model.model, self.backend, self.device, 2, 128)
            self.decode_graph.bind_layer_caches(self.caches)
            self.decode_graph.warmup(self.device, torch.bfloat16)
        self.history: list[torch.Tensor] = []

    def _integer(self, values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=self.device)

    def run(self, additions: list[torch.Tensor], *, prefill: bool = False) -> list[torch.Tensor]:
        previous = [0] * len(additions) if prefill else [tokens.numel() for tokens in self.history]
        self.history = additions if prefill else [torch.cat((old, new)) for old, new in zip(self.history, additions)]
        query_lengths = [tokens.numel() for tokens in additions]
        totals = [tokens.numel() for tokens in self.history]
        assert max(totals) <= 128
        # Reverse physical block IDs so logical positions cannot double as cache slots.
        blocks = [
            [
                2 * _BLOCKS_PER_REQUEST - 1 - request * _BLOCKS_PER_REQUEST - page
                for page in range((total + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
            ]
            for request, total in enumerate(totals)
        ]
        slots = [
            blocks[request][position // _BLOCK_SIZE] * _BLOCK_SIZE + position % _BLOCK_SIZE
            for request, (start, end) in enumerate(zip(previous, totals))
            for position in range(start, end)
        ]
        cumulative_queries = [0]
        cumulative_blocks = [0]
        cumulative_kv = [0]
        for query_length, total, pages in zip(query_lengths, totals, blocks):
            cumulative_queries.append(cumulative_queries[-1] + query_length)
            cumulative_blocks.append(cumulative_blocks[-1] + len(pages))
            cumulative_kv.append(cumulative_kv[-1] + total)
        metadata = SimpleNamespace(
            is_prefill=prefill,
            is_chunked_prefill=not prefill and max(query_lengths) > 1,
            slot_mapping=self._integer(slots),
            q_cu_seq_lens=self._integer(cumulative_queries),
            kv_cu_seq_lens=self._integer(cumulative_kv),
            kv_seq_lens_host=torch.tensor(cumulative_kv, dtype=torch.int32),
            qo_indptr=self._integer(cumulative_queries),
            paged_kv_indptr=self._integer(cumulative_blocks),
            paged_kv_indices=self._integer([page for pages in blocks for page in pages]),
            paged_kv_last_page_len=self._integer([(total - 1) % _BLOCK_SIZE + 1 for total in totals]),
        )
        positions = torch.cat([torch.arange(start, end, device=self.device) for start, end in zip(previous, totals)])
        # The native decode metadata kernel follows the executor's int32 IDs.
        input_ids = torch.cat(additions).to(torch.int32)
        if self.decode_graph is not None and not prefill and max(query_lengths) == 1:
            assert self.decode_graph.can_execute(input_ids, metadata)
            hidden = self.decode_graph.execute(input_ids, positions, metadata)
        else:
            self.backend.prepare(metadata)
            with forward_context(ForwardContext(self.backend, self.device, metadata, self.caches)):
                hidden = self.model.model(input_ids, positions)
        logits = self.model.compute_logits(hidden, None)
        return list(logits.split(query_lengths))


@pytest.mark.parametrize("lengths", [(15, 16), (17, 31), (63, 111)])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "cuda_graph"])
@torch.inference_mode()
def test_prefill_mixed_append_and_decode_match_independent_full_history(lengths: tuple[int, int], graph: bool) -> None:
    runner = _PagedRunner(graph=graph)
    initial = [
        (torch.arange(length, device=runner.device) * (request + 3) % 256).long()
        for request, length in enumerate(lengths)
    ]
    steps = [
        initial,
        [torch.tensor([7], device=runner.device), torch.tensor([11, 13, 17], device=runner.device)],
        [torch.tensor([19], device=runner.device), torch.tensor([23], device=runner.device)],
        [torch.tensor([29], device=runner.device), torch.tensor([31], device=runner.device)],
    ]
    for step, additions in enumerate(steps):
        actual = runner.run(additions, prefill=step == 0)
        for logits, tokens in zip(actual, runner.history):
            reference = _reference(tokens, runner.weights)[-logits.shape[0] :]
            assert torch.isfinite(logits).all()
            # BF16 fused and unfused reductions differ; bound the aggregate error.
            relative_l2 = torch.linalg.vector_norm(logits.float() - reference.float()) / torch.linalg.vector_norm(
                reference.float()
            )
            assert relative_l2 < 0.02
    # The unused physical padding must be zero for every written cache slot.
    for cache in runner.caches:
        for value in (cache.key, cache.value):
            rows = value.reshape(-1, 4, 128)
            written = torch.isfinite(rows[..., 0]).all(-1)
            assert torch.count_nonzero(rows[written, :, 96:]) == 0


@torch.inference_mode()
def test_padding_geometry_is_checked_before_cache_execution() -> None:
    backend = FlashInferBackend(4, 4, 96, 96**-0.5, 16, torch.device("cuda"), torch.bfloat16, kv_cache_head_dim=128)
    malformed = torch.empty(2, 16, 4, 96, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="physical head_dim=128"):
        backend.bind_kv_caches([LayerCache(malformed, malformed)])


@pytest.mark.parametrize("window", [16, 2048])
@torch.inference_mode()
def test_window_boundary_excludes_an_attention_sink(window: int) -> None:
    """A dominant first key makes an off-by-one window observable exactly."""
    device = torch.device("cuda", torch.cuda.current_device())
    max_length = window + 4
    num_blocks = (max_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    shape = (num_blocks, _BLOCK_SIZE, 4, 128)
    cache = LayerCache(
        torch.zeros(shape, device=device, dtype=torch.bfloat16),
        torch.zeros(shape, device=device, dtype=torch.bfloat16),
    )
    backend = FlashInferBackend(4, 4, 96, 96**-0.5, window, device, torch.bfloat16, kv_cache_head_dim=128)
    backend.bind_kv_caches([cache])
    layer = SimpleNamespace(layer_id=0, num_heads=4, num_kv_heads=4, head_dim=96)

    def _integer(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.int32)

    total = window - 1
    q = torch.zeros(total, 4, 96, device=device, dtype=torch.bfloat16)
    q[..., 0] = 1
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    k[0, :, 0] = 1000
    v[0] = 1
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        slot_mapping=_integer(list(range(total))),
        q_cu_seq_lens=_integer([0, total]),
        kv_cu_seq_lens=_integer([0, total]),
    )
    backend.prepare(metadata)
    backend.execute(q.flatten(1), k.flatten(1), v.flatten(1), layer)
    for query_length in (1, 1, 2, 1):
        previous, total = total, total + query_length
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=query_length > 1,
            slot_mapping=_integer(list(range(previous, total))),
            qo_indptr=_integer([0, query_length]),
            paged_kv_indptr=_integer([0, (total + 15) // 16]),
            paged_kv_indices=_integer(list(range((total + 15) // 16))),
            paged_kv_last_page_len=_integer([(total - 1) % 16 + 1]),
        )
        q = torch.zeros(query_length, 4, 96, device=device, dtype=torch.bfloat16)
        q[..., 0] = 1
        backend.prepare(metadata)
        actual = backend.execute(q.flatten(1), torch.zeros_like(q).flatten(1), torch.zeros_like(q).flatten(1), layer)
        expected = actual.new_tensor([float(position < window) for position in range(previous, total)])
        torch.testing.assert_close(actual, expected[:, None].expand_as(actual), rtol=0, atol=0)
