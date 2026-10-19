<!-- Copyright 2026 The xLLM Authors. -->

# Phi-3 mini 4K: CUDA integration and validation

The implementation uses the Python model executor (`--model_impl=python`).
The reference checkpoint and remote implementation are
[`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/f39ac1d28e925b323eae81227eaba4464caced4e),
revision `f39ac1d28e925b323eae81227eaba4464caced4e`. Download the complete
checkpoint, tokenizer, `configuration_phi3.py`, and `modeling_phi3.py` into
`/root/share/models/` or another local model directory. Validation never
downloads missing files.

The tested path is CUDA BF16 weights, activations and KV cache, TP=1/2, eager
execution, block size 128, prefix caching, and a 4096-token context. The model
has 32 layers, hidden size 3072, 32 query/KV heads, logical head width 96,
intermediate size 8192 and a causal window of 2048 including the current token.
Scaled RoPE, Phi-3 128K, quantization, NPU and speculative decoding are outside
this validation. Focused small-model graph tests do not certify full-checkpoint
graph performance or TP=2 graph execution.

## Design and cost

`models/phi3.py` composes existing embeddings, parallel linear layers,
normalization, rotary embeddings and attention. `models/phi3_checkpoint.py`
owns the fused QKV/gate-up contract used by both loading and the offline tool;
the generic loader has no Phi-3 dependency. Each fused weight's shape is
validated directly, without rebuilding all layers' shapes for each tensor.

The C++ model parser records logical model dimensions. After parsing, the
CUDA Python backend's cache-layout policy resolves D=96 to physical D=128
before ModelArgs reaches cache estimation, allocation or the Python executor.
The policy depends on backend and geometry, not model type. Allocation,
budgeting and FlashInfer all consume the same resolved width. Projection,
RoPE and the attention scale still use D=96. This FA2 adaptation increases
KV storage by 33.3%, pads Q/K/V and materializes the cropped output on each
layer. Small-batch TP=2 may be slower than TP=1; no speedup is promised.

The common stopping checker excludes prompt tokens and overlap placeholders.
Phi-3's model stop IDs are 32000, 32001 and 32007. `ignore_eos` disables these
model stops; explicit request stop strings/IDs retain their own semantics.

## Environment and launch

Build xLLM for CUDA with the Python interpreter whose packages will run the
tests. The validation environment uses Python 3.12.3, PyTorch 2.10.0+cu130,
FlashInfer 0.7.2, TVM FFI 0.1.15 and RTX 4090 (SM89). HF reference execution
requires **Transformers 4.40.2 and tokenizers 0.19.1** for the pinned remote
implementation. Keep reference dependencies in a separate environment if the
serving environment needs a different Transformers version.

From the repository root, with the serving environment activated:

```bash
export PHI3_MODEL=/root/share/models/Phi-3-mini-4k-instruct/f39ac1d28e925b323eae81227eaba4464caced4e
export XLLM_BINARY="$PWD/build/lib.linux-x86_64-cpython-312/xllm/xllm"
export XLLM_RUNTIME_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="$PWD:$XLLM_RUNTIME_SITE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$XLLM_RUNTIME_SITE/tvm_ffi/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_CUDA_ARCH_LIST=8.9
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2

CUDA_VISIBLE_DEVICES=0 "$XLLM_BINARY" --model="$PHI3_MODEL" --model_id=Phi-3-mini-4k-instruct \
  --backend=llm --model_impl=python --port=18914 \
  --communication_backend=nccl --enable_graph=false --block_size=128 \
  --max_cache_size=4294967296 --max_memory_utilization=0.9 \
  --max_tokens_per_batch=512 --max_seqs_per_batch=4
```

For TP=2, run two processes in separate terminals with the same environment.
Set `CUDA_VISIBLE_DEVICES=0,1` for both processes and add
`--nnodes=2 --master_node_addr=127.0.0.1:19893` to both commands; rank 0
uses `--node_rank=0 --port=18918`, rank 1 uses
`--node_rank=1 --port=18919`. The existing master selects one visible GPU
according to node rank. Use the rank-0 HTTP port for
validation. Confirm startup logs show `world_size=2`, `tp_size=2`, logical
`head_dim=96` and physical `kv_cache_head_dim=128`. Stop both processes when
finished. Merely exposing two GPUs does not establish TP=2 execution.

The memory-utilization flag caps the cache budget; it does not reserve GPU
memory against other users. The 4 GiB cache cap is a reproducible test setting.

## Fixed numerical protocol

`validate_phi3.py` exercises the real Python model and FlashInfer execution path,
with the pinned independent HF eager BF16 model as reference. It runs explicit
batches of two with initial lengths `(1,17)`, `(127,128)`, `(129,257)`,
`(2047,2048)`, `(2049,3072)` and `(4087,4088)`. Each pair has repeated natural
prose and seeded synthetic tokens. Seed `20261019` and text are fixed in the
tool. Every request generates eight greedy tokens regardless of model EOS.
This checks 12 requests / 96 output positions per run. Prefill populates the
real cache; seven subsequent decode steps consume it. Physical page IDs are
reversed. This entrypoint bypasses scheduling and HTTP serialization; the
formal server tests separately validate those paths and prefix caching.

For each output position, HF recomputes logits using the **same actual token
history** as xLLM. Both first-choice token IDs must occur in the other
implementation's top five. **All positions must pass**, all reported values
must be finite, and lengths must match. Five matches the service API's maximum
top-logprob count and was fixed before running this protocol. The method
follows [vLLM's model correctness approach](https://docs.vllm.ai/en/latest/contributing/model/tests/),
but checks every teacher-forced output position, rather than stopping at the
first generation mismatch. This is a project-local acceptance protocol, not
an official universal error threshold.

The HTTP top-logprob maps expose decoded labels, which can represent multiple
token IDs; output IDs are currently omitted by the OpenAI response formatter.
Consequently numerical validation records raw IDs directly from the executor.
Reports retain input IDs, generated histories, top IDs/logprobs, reference
source hashes, environment and every pass/fail. Partial candidate matrices,
changed inputs, missing steps, duplicate IDs and non-finite values are rejected.

Run the candidate in the serving environment against matching native artifacts:

```bash
python tools/validate_phi3.py candidate --checkpoint "$PHI3_MODEL" \
  --output /tmp/phi3-tp1-candidate.json
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/validate_phi3.py candidate --checkpoint "$PHI3_MODEL" \
  --output /tmp/phi3-tp2-candidate.json
```

Then use the reference environment's Python, with access to the checkout:

```bash
python tools/validate_phi3.py reference --checkpoint "$PHI3_MODEL" \
  --candidate /tmp/phi3-tp1-candidate.json --output /tmp/phi3-tp1-acceptance.json
python tools/validate_phi3.py reference --checkpoint "$PHI3_MODEL" \
  --candidate /tmp/phi3-tp2-candidate.json --output /tmp/phi3-tp2-acceptance.json
```

Candidate and reference run in separate processes, releasing model memory
between phases. Stop other validation services when resources are limited.
Exit 0 in reference mode requires every check to pass; nonzero exits and
intermediate reports must be retained for failures.

Earlier exploratory raw-logit relative-L2 <=2% and strict KL-versus-HF-BF16
gates failed on some Phi-3 and existing-model comparisons. Those were local
diagnostics, not published framework requirements. Their failures are not
erased or converted to passes by this output-focused protocol. The 96-position
check is bounded integration evidence, not task accuracy, exhaustive input
coverage, or equivalence of every vocabulary logit. GSM8K/task evaluation and
broader performance characterization are separate work.

## Recorded validation

On 2026-10-19, the environment above produced the following results using the
final native build and Python sources:

| Path | Mutual top-5 | Identical top-1 | Initial lengths | Cache execution |
| --- | --- | --- | --- | --- |
| BF16 TP=1 | 96 / 96 | 93 / 96 | 1–4088 | batch=2 prefill + 7 decode steps |
| BF16 TP=2 | 96 / 96 | 92 / 96 | 1–4088 | two CUDA devices, real NCCL collectives |

The shared C++ regressions passed 102 cases. All six registered CUDA Python
CTest targets passed, including the focused D96/cache/graph regressions and
negative tests for the validation tool. Final service checks cover both
ordinary and streaming requests; raw logs/reports are retained outside Git.
These counts describe the bounded protocol above, including its non-identical
top-1 predictions, rather than exact output equivalence.

## Regression and service checks

```bash
python -m pytest -q tests/python/test_phi3_checkpoint.py tests/python/test_phi3.py \
  tests/python/test_registry.py tests/python/test_phi3_cuda.py \
  tests/python/test_flashinfer_attention.py tests/python/test_phi3_validation.py
ctest --test-dir build/cmake.linux-x86_64-cpython-312 \
  -R '^python_tests\.(phi3|phi3_checkpoint|phi3_cuda|phi3_validation|flashinfer_attention|registry)$' \
  --output-on-failure
```

CMake registers the CUDA modules only for CUDA; NPU retains its own modules
and distributed NZ entrypoints. Empty test collection is not treated as a
successful run. C++ loader, cache layout/budget, request-factory and stopping
checker tests cover the affected common code.

On the **final** TP=1 and TP=2 servers check ordinary/stream consistency, chat
with its prompt-ending assistant token, forced 32000/32001/32007 stops,
`ignore_eos`, output length, explicit stops, prefix reuse, concurrent unequal
lengths, and invalid/out-of-context requests. A successful numerical protocol
does not replace these API semantics checks. Full model files, tensor dumps
and machine-specific logs belong outside Git. Offline unit tests keep their
small fixed config and shape values inline in the checkpoint test module.

For example, replay the streaming model-stop check against either server:

```bash
curl --noproxy '*' -N http://127.0.0.1:18918/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Phi-3-mini-4k-instruct","token_ids":[1,32001],"allowed_token_ids":[32007],"temperature":0,"max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}'
```

Expect one generated token and `finish_reason="stop"`. Adding
`"ignore_eos":true` must yield four generated tokens and `finish_reason="length"`.
Repeat for allowed IDs 32000 and 32001. Keeping the prompt-ending 32001 while
forcing ordinary ID 306 must yield four tokens, proving the prompt itself
cannot stop generation.
