# NPU Timeline Generation Guide
## Prerequisites
- Python environment
- Chrome browser (for visualization)
## Implementation Steps
### 1. Code Modification
#### Register the subscriber
Add the following at the beginning of your program:
```cpp
MsptiMetrics::register_subscriber();
```
#### Add tracing to ACLNN functions (work for msprof as well)
Insert the following macro in your ACLNN functions where you want to measure performance:
```cpp
LLM_MSTX_RANGE();
```
#### Release the subscriber
Add this at the end of your program:
```cpp
MsptiMetrics::release_subscriber();
```
### 2. Log Processing
After running your program, process the generated log file using the timeline script:
```bash
python npu_timeline.py -i custom_log.log -o custom_output.json
```
### 3. Visualization
Open Chrome browser
Navigate to: chrome://tracing
Load the generated JSON file: custom_output.json

<!-- Copyright 2026 The xLLM Authors. -->

## Phi-3 checkpoint contract checker

For the complete CUDA model launch, pinned reference, numerical protocol and
known limitations, see [Phi-3 integration validation](phi3_validation.md).

`check_phi3_checkpoint.py` validates local config and tensor shape manifest
JSON files for the already fused QKV and gate-up matrices in Phi-3 checkpoints.
The reference is `microsoft/Phi-3-mini-4k-instruct` at revision
`f39ac1d28e925b323eae81227eaba4464caced4e`:
[config](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/f39ac1d28e925b323eae81227eaba4464caced4e/config.json),
[implementation](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/f39ac1d28e925b323eae81227eaba4464caced4e/modeling_phi3.py).

Run from a checkout with its Python dependencies installed:

```bash
python tools/check_phi3_checkpoint.py \
  --config /path/to/config.json \
  --manifest /path/to/manifest.json \
  --tp-size 2
```

Success emits a JSON dimension/count report on stdout and exits 0. Invalid
input exits nonzero with a diagnostic on stderr. This is an offline check;
it does not initialize native operators or devices, download weights, or run
inference. It validates shapes and TP arithmetic, without certifying dtype,
tensor values, attention/RoPE semantics, or platform support.

The manifest is an object keyed by complete checkpoint weight names. Each
target descriptor contains a `shape` array of two positive integers; for
example, `"model.layers.0.self_attn.qkv_proj.weight": {"shape": [9216, 3072]}`.
Every layer's `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight` must
be present. Additional weights and descriptor fields are allowed and are not
checked. A safetensors header already uses this descriptor format; combine the
headers of all shards into one manifest. A safetensors **index** maps names to
filenames and does not contain the required shapes.

Required config fields are `model_type="phi3"`, `hidden_size`,
`num_attention_heads`, `intermediate_size`, and `num_hidden_layers`. Dimensions
must be positive integers (booleans, strings and fractional values are
rejected). Missing/null `num_key_value_heads` defaults to query heads, matching
the fixed upstream config class. Hidden size must be divisible by query heads,
and query heads must be divisible by KV heads. `head_dim` is derived from hidden
size and query heads; an explicit value must agree. For the fixed reference,
hidden=3072, query/KV heads=32, head_dim=96, intermediate=8192, and layers=32.
The formulas yield QKV `[9216, 3072]` and gate-up `[16384, 3072]`.

Both matrices use `[output, input]` order. Their source output groups are
`[Q; K; V]` and `[gate; up]`, as defined by `Phi3Attention.forward` and
`Phi3MLP.forward`. TP takes rows on axis 0 **within each logical group**,
then concatenates the local groups in that same order. Every rank retains all
input columns. The following global source row ranges are half-open:

| TP | Rank | Q | K | V | gate | up |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | [0,3072) | [3072,6144) | [6144,9216) | [0,8192) | [8192,16384) |
| 2 | 0 | [0,1536) | [3072,4608) | [6144,7680) | [0,4096) | [8192,12288) |
| 2 | 1 | [1536,3072) | [4608,6144) | [7680,9216) | [4096,8192) | [12288,16384) |

Query heads, KV heads and intermediate channels must each divide evenly by
TP. Splitting a head or replicating KV heads is not supported by this contract.
These are mathematical constraints, not claims of device validation at any
particular TP size.

The reusable interfaces live in
`xllm.python.models.phi3_checkpoint`: `parse_phi3_config`,
`expected_phi3_shapes`, `validate_phi3_manifest`, and
`shard_phi3_fused_weight`. The sharder returns a contiguous tensor without
changing the source. TP=1 reuses an already contiguous source to avoid a
full-weight copy. It consumes a single **already fused**
source weight. The existing `load_gqa_fused_attention` instead uses
`ScopedWeightLoader.load_fused` to combine separate `q_proj/k_proj/v_proj`
sources. Its source lookup cannot be applied unchanged to this checkpoint.
The production Phi-3 loader reuses this sharder; device numerical and engine
validation remain separate from the offline contract checks.

The fixed config and shape metadata are defined directly in
`tests/python/test_phi3_checkpoint.py`, with their upstream revision recorded
beside the fixture. Run these checks and the independent synthetic rank tests
with the repository's built native extension available:

```bash
python -m pytest -q tests/python/test_phi3_checkpoint.py
```

Tests check all real layer shapes without allocating real-sized weights,
specific Q/K/V and gate/up rank rows, reconstruction, projection equality,
unequal GQA groups, malformed input diagnostics, and CLI success/failure.
`tests/python/conftest.py` initializes the native runtime for test collection;
the standalone checker itself does not require that initialization. Direct
pytest runs this file on CUDA; CMake registers it under both `USE_CUDA` and
`USE_NPU`.
