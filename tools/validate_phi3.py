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

"""Real Phi-3 BF16 checkpoint validation; see tools/phi3_validation.md.

Run candidate under Python/torchrun, then reference in a separate process.
Raw IDs avoid the ambiguous decoded labels exposed by HTTP top-logprob maps.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

_LENGTH_PAIRS = ((1, 17), (127, 128), (129, 257), (2047, 2048), (2049, 3072), (4087, 4088))
_NEW_TOKENS = 8
_TOP_K = 5
_SEED = 20261019
_BLOCK_SIZE = 128
_MAX_LENGTH = 4096
_TEXT = (
    "A library stores books on numbered shelves. Each reader returns a book before borrowing another. "
    "The librarian records the title, shelf number, and return date in a table. "
    "When the library closes, she checks the table and counts the available books. "
    "Write a clear explanation of how the records help her find a missing book.\n"
)


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def _fingerprints(checkpoint: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in ("config.json", "modeling_phi3.py", "configuration_phi3.py", "tokenizer.json")
    }


def _environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "argv": sys.argv,
        "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "safetensors")},
    }


def _cases(tokenizer: Any) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(_SEED)
    text_ids = tokenizer.encode(_TEXT, add_special_tokens=False)
    cases = []
    for pair in _LENGTH_PAIRS:
        for request, length in enumerate(pair):
            synthetic = request == 1
            tokens = (
                torch.randint(3, 32000, (length,), generator=generator).tolist()
                if synthetic
                else ([1] + (text_ids * (length // len(text_ids) + 1)))[:length]
            )
            cases.append({"length": length, "synthetic": synthetic, "token_ids": tokens})
    return cases


class _StateDict:
    """The read-only subset of the native state-dict interface used by loaders."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def keys(self) -> list[str]:
        return list(self._tensors)

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _Runner:
    def __init__(self, checkpoint: Path, rank: int, world: int) -> None:
        from safetensors import safe_open

        from xllm.python.attention.backend import LayerCache
        from xllm.python.attention.flashinfer import FlashInferBackend
        from xllm.python.models.phi3 import Phi3ForCausalLM

        self.device = torch.device("cuda", torch.cuda.current_device())
        config = json.loads((checkpoint / "config.json").read_text())
        config.update(device=str(self.device), dtype="bfloat16", tp_rank=rank, tp_size=world, world_size=world)
        self.model = Phi3ForCausalLM(config).eval()
        weights = {}
        for path in sorted(checkpoint.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as state:
                weights.update({key: state.get_tensor(key) for key in state.keys()})
        self.model.load_weights([_StateDict(weights)], rank, world)
        del weights
        gc.collect()
        self._pages = math.ceil(_MAX_LENGTH / _BLOCK_SIZE)
        # Match the backend layout selected by the formal engine. C++ regression
        # tests and formal service startup separately verify its configuration.
        physical_dim = 128
        shape = (2 * self._pages, _BLOCK_SIZE, 32 // world, physical_dim)
        self.caches = [
            LayerCache(
                torch.zeros(shape, device=self.device, dtype=torch.bfloat16),
                torch.zeros(shape, device=self.device, dtype=torch.bfloat16),
            )
            for _ in range(32)
        ]
        self.backend = FlashInferBackend(
            32 // world,
            32 // world,
            96,
            96**-0.5,
            2048,
            self.device,
            torch.bfloat16,
            kv_cache_head_dim=physical_dim,
        )
        self.backend.bind_kv_caches(self.caches)
        self._lengths: list[int] = []

    def step(self, additions: list[list[int]], *, prefill: bool) -> torch.Tensor:
        from xllm.python.model_executor.forward_context import ForwardContext, forward_context

        previous = [0] * len(additions) if prefill else self._lengths
        self._lengths = [old + len(new) for old, new in zip(previous, additions)]
        pages = [
            [2 * self._pages - 1 - request * self._pages - page for page in range(math.ceil(total / _BLOCK_SIZE))]
            for request, total in enumerate(self._lengths)
        ]
        slots = [
            pages[r][position // _BLOCK_SIZE] * _BLOCK_SIZE + position % _BLOCK_SIZE
            for r, (start, end) in enumerate(zip(previous, self._lengths))
            for position in range(start, end)
        ]
        qo, po, kv = [0], [0], [0]
        for addition, total, blocks in zip(additions, self._lengths, pages):
            qo.append(qo[-1] + len(addition))
            po.append(po[-1] + len(blocks))
            kv.append(kv[-1] + total)

        def _integer(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, device=self.device, dtype=torch.int32)

        metadata = SimpleNamespace(
            is_prefill=prefill,
            is_chunked_prefill=False,
            slot_mapping=_integer(slots),
            q_cu_seq_lens=_integer(qo),
            kv_cu_seq_lens=_integer(kv),
            qo_indptr=_integer(qo),
            paged_kv_indptr=_integer(po),
            paged_kv_indices=_integer([page for blocks in pages for page in blocks]),
            paged_kv_last_page_len=_integer([(total - 1) % _BLOCK_SIZE + 1 for total in self._lengths]),
        )
        self.backend.prepare(metadata)
        positions = torch.cat(
            [torch.arange(start, end, device=self.device) for start, end in zip(previous, self._lengths)]
        )
        tokens = torch.tensor([token for addition in additions for token in addition], device=self.device)
        with forward_context(ForwardContext(self.backend, self.device, metadata, self.caches)):
            hidden = self.model.model(tokens, positions)
            last_hidden = hidden[torch.tensor(qo[1:], device=self.device) - 1]
            return self.model.compute_logits(last_hidden, None).float()


def _check_topk(actual_ids: list[int], reference_ids: list[int]) -> bool:
    return actual_ids[0] in reference_ids and reference_ids[0] in actual_ids


def _validate_candidate(candidate: dict[str, Any], expected_cases: list[dict[str, Any]]) -> None:
    if candidate.get("protocol") != "phi3-bf16-mutual-top5-v1" or candidate.get("seed") != _SEED:
        raise ValueError("Unexpected candidate protocol or seed")
    cases = candidate["cases"]
    if len(cases) != len(expected_cases):
        raise ValueError("Incomplete candidate case matrix")
    for case, expected in zip(cases, expected_cases):
        if any(case.get(key) != value for key, value in expected.items()):
            raise ValueError("Candidate inputs differ from the fixed protocol")
        if len(case["steps"]) != _NEW_TOKENS:
            raise ValueError(f"Expected {_NEW_TOKENS} output positions")
        for step in case["steps"]:
            ids, logprobs = step["ids"], step["logprobs"]
            if len(ids) != _TOP_K or len(set(ids)) != _TOP_K or len(logprobs) != _TOP_K:
                raise ValueError("Expected five distinct candidate IDs and logprobs")
            if not all(isinstance(token, int) and 0 <= token < 32064 for token in ids):
                raise ValueError("Candidate token ID outside the model vocabulary")
            if not all(math.isfinite(value) for value in logprobs):
                raise ValueError("Non-finite candidate logprobs")


@torch.inference_mode()
def _candidate(args: argparse.Namespace) -> None:
    # Load the native extension before importing kernels/FlashInfer.
    from xllm import xllm_export  # noqa: F401
    from xllm.python import distributed, initialize_runtime

    initialize_runtime()
    from transformers import AutoTokenizer

    rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (1, 2):
        raise ValueError("The protocol covers TP=1 or TP=2")
    torch.cuda.set_device(rank)
    if world > 1:
        distributed.init_tp_group("127.0.0.1", args.tp_port, rank, world, f"cuda:{rank}", rank, world, 0)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    cases = _cases(tokenizer)
    runner = _Runner(args.checkpoint, rank, world)
    report = {
        "protocol": "phi3-bf16-mutual-top5-v1",
        "seed": _SEED,
        "checkpoint": str(args.checkpoint.resolve()),
        "world_size": world,
        "environment": _environment(),
        "reference_files_sha256": _fingerprints(args.checkpoint),
        "cases": [],
    }
    for start in range(0, len(cases), 2):
        pair = cases[start : start + 2]
        records = [{**case, "steps": []} for case in pair]
        additions = [case["token_ids"] for case in pair]
        for step in range(_NEW_TOKENS):
            logits = runner.step(additions, prefill=step == 0)
            if not torch.isfinite(logits).all():
                raise ValueError("Candidate produced non-finite logits")
            values, indices = logits.log_softmax(-1).topk(_TOP_K, dim=-1)
            for request, record in enumerate(records):
                record["steps"].append({"ids": indices[request].tolist(), "logprobs": values[request].tolist()})
            additions = [[int(token)] for token in indices[:, 0].tolist()]
        report["cases"].extend(records)
        if rank == 0:
            _write(args.output, report)
            logger.info("Recorded TP=%s lengths=%s", world, [case["length"] for case in pair])
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


@torch.inference_mode()
def _reference(args: argparse.Namespace) -> bool:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != "4.40.2":
        raise ValueError("The pinned remote reference requires transformers==4.40.2")
    candidate = json.loads(args.candidate.read_text())
    if candidate["reference_files_sha256"] != _fingerprints(args.checkpoint):
        raise ValueError("Reference files differ from the candidate's checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    _validate_candidate(candidate, _cases(tokenizer))
    torch.cuda.set_device(args.device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=True,
        )
        .to(args.device)
        .eval()
    )
    report = {
        "candidate": str(args.candidate),
        "protocol": candidate["protocol"],
        "environment": _environment(),
        "reference_files_sha256": _fingerprints(args.checkpoint),
        "results": [],
        "passed": False,
    }
    for case in candidate["cases"]:
        generated = [step["ids"][0] for step in case["steps"]]
        tokens = torch.tensor([case["token_ids"] + generated[:-1]], device=args.device)
        logits = model(tokens, use_cache=False).logits[0, case["length"] - 1 :].float()
        if not torch.isfinite(logits).all():
            raise ValueError("HF reference produced non-finite logits")
        values, indices = logits.log_softmax(-1).topk(_TOP_K, dim=-1)
        checks = [
            {
                "passed": _check_topk(actual["ids"], expected),
                "actual": actual,
                "reference_ids": expected,
                "reference_logprobs": probabilities,
            }
            for actual, expected, probabilities in zip(case["steps"], indices.tolist(), values.tolist())
        ]
        result = {
            "length": case["length"],
            "synthetic": case["synthetic"],
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }
        report["results"].append(result)
        _write(args.output, report)
        logger.info("length=%s synthetic=%s passed=%s", case["length"], case["synthetic"], result["passed"])
    report["passed"] = all(result["passed"] for result in report["results"])
    _write(args.output, report)
    return report["passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("candidate", "reference"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", type=Path, help="Candidate JSON for reference mode")
    parser.add_argument("--device", default="cuda:0", help="HF reference device")
    parser.add_argument("--tp-port", default=29643, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config.get("model_type") != "phi3" or config.get("num_hidden_layers") != 32 or config.get("head_dim", 96) != 96:
        raise ValueError("The protocol requires the real Phi-3 mini 4K checkpoint")
    if args.mode == "candidate":
        _candidate(args)
        return 0
    if args.candidate is None:
        parser.error("reference mode requires --candidate")
    return 0 if _reference(args) else 1


if __name__ == "__main__":
    sys.exit(main())
