---
title: "Accuracy Evaluation with EvalScope"
description: "Evaluate xLLM LLM and VLM accuracy through the OpenAI-compatible API."
---

Use `evalscope eval` to measure answer accuracy through a running xLLM service.
This guide includes a text-only LLM script for GSM8K and an image-text VLM script
for AI2D. Both use EvalScope's native evaluation backend and rule-based scoring;
neither example requires a separate judge model or VLMEvalKit.


## Prepare the environment and service

Install the latest stable EvalScope release by following the
[official installation guide](https://evalscope.readthedocs.io/en/latest/get_started/installation.html).

The first run downloads data from ModelScope, so prepare dataset access and sufficient local cache space before evaluation.

1. Start a supported model following [Launch xLLM](/en/getting_started/launch_xllm/).
2. Confirm that the corresponding text or image request works using
   [Online Service](/en/getting_started/online_service/).
3. Set `HOST` and `PORT` to the service address. The scripts use the API root
   `http://${HOST}:${PORT}/v1`. `MODEL` must match a model ID returned by
   `/v1/models`, rather than the client's local weight path.
   The examples use port `18000`; replace it with the actual serving port.

```bash
export HOST=127.0.0.1
export PORT=18000
export API_KEY=EMPTY
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" "http://${HOST}:${PORT}/v1/models"
```

## LLM: GSM8K

Save the following as `eval_llm_accuracy.sh` and run it with Bash. It evaluates
GSM8K's `main` subset and `test` split with **four-shot** prompts and fixed
few-shot examples (`few_shot_random=false`). Generation uses `temperature=0.6`,
`top_p=0.95`, `top_k=20`, and a default limit of 1024 output tokens. The template
arguments request non-thinking mode for models whose templates support these
switches. The default API concurrency is 64.

```bash
#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen3-8B}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
RUN_DIR="${RUN_DIR:-outputs/accuracy/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

GENERATION_CONFIG="$(cat <<EOF
{
  "do_sample": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": ${MAX_TOKENS},
  "top_k": 20,
  "stream": false,
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false,
      "thinking": false
    }
  }
}
EOF
)"

eval_args=(--eval-backend Native)
if [[ "$LIMIT" != "all" ]]; then
  eval_args+=(--limit "$LIMIT")
fi

evalscope eval \
  --model "$MODEL" \
  --api-url "http://${HOST}:${PORT}/v1" \
  --api-key "$API_KEY" \
  --eval-type openai_api \
  --datasets gsm8k \
  --dataset-args '{"gsm8k": {"few_shot_num": 4, "few_shot_random": false}}' \
  --eval-batch-size "$BATCH_SIZE" \
  --generation-config "$GENERATION_CONFIG" \
  --work-dir "$RUN_DIR" \
  "${eval_args[@]}"
```

```bash
# Small regression run: 64 samples.
MODEL=Qwen3-8B bash eval_llm_accuracy.sh

# Full test split: omit EvalScope's sample limit.
MODEL=Qwen3-8B LIMIT=all bash eval_llm_accuracy.sh
```

`BATCH_SIZE=64` controls API concurrency; `LIMIT=64` independently limits the
number of evaluated samples. Set `LIMIT=all` to evaluate the full test split with
the same generation and four-shot settings.

The scorer extracts the final numerical answer and compares it with the reference.
Preserve the benchmark's prompt and answer format. For reasoning models, increase
`MAX_TOKENS` if responses are truncated before the final answer, and configure the
server context limit to fit both the prompt and generated output.

## VLM: AI2D

Save the following as `eval_vlm_accuracy.sh`. AI2D evaluates diagram understanding
using an image and a multiple-choice question for each sample. EvalScope loads the
images and sends image-text content through the Chat Completions API; a text-only
request to a VLM would not test its visual understanding.

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
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
RUN_DIR="${RUN_DIR:-outputs/accuracy/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

GENERATION_CONFIG="$(cat <<EOF
{
  "do_sample": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "max_tokens": ${MAX_TOKENS},
  "top_k": 20,
  "stream": false,
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false,
      "thinking": false
    }
  }
}
EOF
)"

eval_args=(--eval-backend Native)
if [[ "$LIMIT" != "all" ]]; then
  eval_args+=(--limit "$LIMIT")
fi

evalscope eval \
  --model "$MODEL" \
  --api-url "http://${HOST}:${PORT}/v1" \
  --api-key "$API_KEY" \
  --eval-type openai_api \
  --datasets ai2d \
  --eval-batch-size "$BATCH_SIZE" \
  --generation-config "$GENERATION_CONFIG" \
  --work-dir "$RUN_DIR" \
  "${eval_args[@]}"
```

```bash
MODEL=Qwen3-VL-8B-Instruct bash eval_vlm_accuracy.sh
MODEL=Qwen3-VL-8B-Instruct LIMIT=all bash eval_vlm_accuracy.sh
```

The script uses AI2D's `default` subset, `test` split, and zero-shot multiple-choice
accuracy. Keep image preprocessing, resolution limits, and model processor files
consistent across runs. Inspect saved inputs to confirm that images were included,
and inspect predictions for answer extraction failures or truncated responses.


## References

- [EvalScope installation](https://evalscope.readthedocs.io/en/latest/get_started/installation.html)
- [Evaluation parameters](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html)
- [GSM8K benchmark](https://evalscope.readthedocs.io/en/latest/benchmarks/gsm8k.html)
- [AI2D benchmark](https://evalscope.readthedocs.io/en/latest/benchmarks/ai2d.html)
