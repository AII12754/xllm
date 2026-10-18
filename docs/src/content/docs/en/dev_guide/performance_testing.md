---
title: "Inference Benchmarking with EvalScope"
description: "Measure xLLM LLM and VLM latency and throughput with streaming workloads."
---

Use `evalscope perf` to measure a running xLLM service's latency and throughput.
This guide includes an LLM script with random text and a VLM script with real
image-text inputs. For answer correctness, use
[Accuracy Evaluation with EvalScope](/en/dev_guide/accuracy_testing/).

## Prepare the environment and service

Install the latest stable EvalScope release with the `perf` dependencies by following
the [official installation guide](https://evalscope.readthedocs.io/en/latest/get_started/installation.html).

Start xLLM following [Launch xLLM](/en/getting_started/launch_xllm/) and verify a
text or image request using [Online Service](/en/getting_started/online_service/).

```bash
export HOST=127.0.0.1
export PORT=18000
export API_KEY=EMPTY
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" "http://${HOST}:${PORT}/v1/models"
```

Set `HOST`, `PORT`, and `API_KEY` to the deployment's values. `MODEL` must
match a returned model ID. Select the appropriate model service for each script;
the examples can reuse one port at different times. Unlike `evalscope eval
--api-url`, the performance scripts default to the complete `/v1/chat/completions`
endpoint to `--url`.

For a baseline without prefix-cache reuse, start xLLM with
`--enable_prefix_cache=false`. If testing with caching enabled, keep that setting
consistent and report it, including any repeated prompts or images.

## LLM: random text

Save the following as `eval_llm_performance.sh`. Set `TOKENIZER_PATH` to the local
tokenizer directory matching the deployed weights. The client loads tokenizer
files rather than model weights.

```bash
#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen3-8B}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the local tokenizer directory matching the deployed model}"
INPUT_TOKENS="${INPUT_TOKENS:-1024}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
if [[ ! "$INPUT_TOKENS" =~ ^[1-9][0-9]*$ || ! "$PREFIX_TOKENS" =~ ^(0|[1-9][0-9]*)$ ]] ||
   (( PREFIX_TOKENS >= INPUT_TOKENS )); then
  echo "Require integer lengths: 0 <= PREFIX_TOKENS < INPUT_TOKENS" >&2
  exit 1
fi
RANDOM_TOKENS=$((INPUT_TOKENS - PREFIX_TOKENS))
TOKENIZE_PROMPT="${TOKENIZE_PROMPT:-false}"
API_ARGS=(--url "http://${HOST}:${PORT}/v1/chat/completions")
if [[ "$TOKENIZE_PROMPT" == true ]]; then
  API_ARGS=(--url "http://${HOST}:${PORT}/v1/completions" --tokenize-prompt)
elif [[ "$TOKENIZE_PROMPT" != false ]]; then
  echo "TOKENIZE_PROMPT must be true or false" >&2
  exit 1
fi
RUN_DIR="${RUN_DIR:-outputs/performance/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

evalscope perf \
  --model "$MODEL" \
  "${API_ARGS[@]}" \
  --api-key "$API_KEY" \
  --api openai \
  --dataset random \
  --tokenizer-path "$TOKENIZER_PATH" \
  --prefix-length "$PREFIX_TOKENS" \
  --min-prompt-length "$RANDOM_TOKENS" \
  --max-prompt-length "$RANDOM_TOKENS" \
  --max-tokens "$OUTPUT_TOKENS" \
  --extra-args '{"ignore_eos": true}' \
  --temperature 0 \
  --parallel 1 4 8 \
  --number 64 128 256 \
  --warmup-num 8 \
  --stream \
  --outputs-dir "$RUN_DIR"
```

```bash
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B \
  bash eval_llm_performance.sh

# A longer input with the same output target.
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B INPUT_TOKENS=8192 \
  bash eval_llm_performance.sh
```

The script uses three closed-loop concurrency levels: `1`, `4`, and `8`, with
`64`, `128`, and `256` measured requests respectively. Each worker sends its next
request after the previous one completes. Each level also sends eight warmup
requests, excluded from performance statistics. If you raise concurrency, increase
`--warmup-num` to at least the largest concurrency and use enough measured requests
to observe steady behavior.

The default workload targets 1024 tokens of random prompt text and 256 output
tokens. Chat-template tokens can change the actual input length. `ignore_eos=true`
suppresses early EOS stopping to help control output length; verify actual lengths
in response `usage` and the report. Ensure the server context limit accommodates
both input and output. For production-like stopping behavior, remove `ignore_eos`
and report the observed output-length distribution.

These parameters define a synthetic workload, not a representative accuracy test.
For comparisons requiring identical prompts, use a fixed local dataset with
EvalScope's `line_by_line` dataset mode and the same `--dataset-path` in every run.
Keep concurrency, request count, and data order identical as well.

### Fix input and output lengths exactly

The default Chat Completions path decodes random tokens to text, which the server
then tokenizes again. This can cause small input-length differences. To send a
fixed number of input token IDs directly:

```bash
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B \
  INPUT_TOKENS=1024 OUTPUT_TOKENS=256 TOKENIZE_PROMPT=true \
  bash eval_llm_performance.sh
```

This option switches to `/v1/completions` and adds `--tokenize-prompt`, without
applying a chat template. It benchmarks the token-ID input path. `INPUT_TOKENS`
includes the shared prefix; `OUTPUT_TOKENS` and `ignore_eos=true` control output
length. The context window must accommodate both. Verify the actual `usage`.

### Test prefix caching

Start xLLM with `--enable_prefix_cache=true`, then set a shared prefix length:

```bash
# Target 1024 total input tokens, including a 512-token shared prefix.
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B \
  INPUT_TOKENS=1024 PREFIX_TOKENS=512 OUTPUT_TOKENS=256 \
  bash eval_llm_performance.sh
```

`PREFIX_TOKENS` defaults to 0. EvalScope's `random` dataset reuses one random
prefix within each run. Since `--prefix-length` adds tokens to the prompt length,
the script subtracts it from the total input budget first. The remaining budget
must also accommodate the chat template. With matching tokenizers and templates,
input length should be close to `INPUT_TOKENS`; use the server's
`usage.prompt_tokens` as the actual count.

Warmup requests can populate the cache, but hits also depend on identical prefix
tokens, cache block size, and eviction. Check
`usage.prompt_tokens_details.cached_tokens` in responses or
`num_prefix_cache_tokens` in server logs. The configured prefix ratio is not a
measured cache hit rate. To compare caching on and off, use the same fixed request
data and report cold-cache and warmed-cache results separately.

The current xLLM streaming `/v1/completions` response returns total token counts
but omits `prompt_tokens_details.cached_tokens`. With `TOKENIZE_PROMPT=true`,
check server logs for cache hits; EvalScope cache statistics without this
field cannot establish whether a hit occurred.

## VLM: Flickr8k image-text inputs

Save the following as `eval_vlm_performance.sh`. EvalScope's `flickr8k` plugin loads
the `test` split of `clip-benchmark/wds_flickr8k` and builds requests containing
both image captions and Base64-encoded images. This exercises the image-input path
of the deployed VLM.

Image batches can exceed brpc's default 64 MiB message limit. For higher
concurrency, add `--max_body_size=536870912` (512 MiB) to the VLM server
startup command. If logs still report `body_size ... is too large`, adjust the
limit for the actual image batch size or reduce concurrency.

```bash
#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
RUN_DIR="${RUN_DIR:-outputs/performance/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

evalscope perf \
  --model "$MODEL" \
  --url "http://${HOST}:${PORT}/v1/chat/completions" \
  --api-key "$API_KEY" \
  --api openai \
  --dataset flickr8k \
  --max-tokens "$OUTPUT_TOKENS" \
  --extra-args '{"ignore_eos": true}' \
  --temperature 0 \
  --parallel 1 4 8 \
  --number 64 128 256 \
  --warmup-num 8 \
  --stream \
  --outputs-dir "$RUN_DIR"
```

```bash
MODEL=Qwen3-VL-8B-Instruct bash eval_vlm_performance.sh
```

The warmup and concurrency settings have the same meaning as in the LLM script.
Flickr8k uses real images with varying sizes, rather than a fixed visual token
count. Keep the dataset revision, image sample order, processor settings,
resolution limits, and image-cache behavior consistent across comparisons.
For offline use, add `--dataset-path /path/to/local/flickr8k` pointing to a local
dataset directory that EvalScope can load, rather than an arbitrary image folder.

This script controls the output target with `OUTPUT_TOKENS`; it does not fix the
total image-text input length. `--prefix-length` only applies to `random`, not
`flickr8k`. To test VLM prefix caching, prepare repeated requests with identical
leading text and images, keep message order and image processing settings fixed,
and inspect actual cached token counts. Equal image dimensions alone do not
guarantee prefix hits.

The script relies on the server's streamed `usage` for token counts. Verify that
the final usage chunk includes `prompt_tokens` and `completion_tokens`. A text
tokenizer cannot reliably count a VLM's visual tokens; missing usage makes token
throughput unreliable. Across VLM architectures, compare request throughput,
TTFT, and output throughput under the same image workload, and document how input
tokens are counted.

## Metrics and result files

| Metric | Meaning |
| --- | --- |
| Request throughput (req/s) | Successful requests divided by benchmark duration. |
| Output throughput (token/s) | Total generated output tokens divided by benchmark duration. |
| Total throughput (token/s) | Input plus output tokens divided by benchmark duration; depends on input-token accounting. |
| TTFT | Time from sending a request to receiving the first output token, including queuing and network overhead. |
| TPOT | Per-request average time per output token after the first: `(latency - TTFT) / (output_tokens - 1)`, when more than one output token is produced. |
| ITL | Inter-output arrival intervals observed by the streaming client; chunks containing multiple tokens affect interpretation. |
| Latency and percentiles | End-to-end request duration and P50/P95/P99 distributions. |
| Success / failure counts | Whether the intended request workload actually completed. |

Keep `--stream` enabled to observe TTFT and token arrival intervals. These are
client-observed service metrics; use
[Profiling a Running Service](/en/dev_guide/online_profiling/) for device execution analysis.

EvalScope prints the actual output path under `RUN_DIR`, writes `benchmark.log`,
and creates a `performance_summary.txt` for the multiple-concurrency run. Preserve
the per-run request database (`benchmark_data.db`) to inspect payloads, failures,
lengths, and timing when a result is unexpected.

### xLLM parameter compatibility

The current xLLM API rejects non-null request fields `seed` and `min_tokens`.
EvalScope `perf --seed` forwards a generation seed to the server as well as seeding
the client; omit it for these scripts. Likewise, omit `--min-tokens`. The examples
use supported `max_tokens` and `ignore_eos` fields to control output length.

After upgrading EvalScope, check `evalscope perf --help` and send a small request
before a full run. A `400` response, missing streamed usage, or a request with no
image should be resolved before interpreting performance results.

## References

- [EvalScope installation](https://evalscope.readthedocs.io/en/latest/get_started/installation.html)
- [Performance quick start and metric definitions](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html)
- [Performance parameters and dataset configuration](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html)
- [Workload and warmup examples](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html)
