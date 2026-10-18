---
title: "使用 EvalScope 测试推理性能"
description: "通过流式负载测试 xLLM LLM 和 VLM 的延迟与吞吐量。"
---

使用 `evalscope perf` 测量已启动的 xLLM 服务的延迟和吞吐量。本文提供基于随机文本的
LLM 脚本，以及基于真实图文输入的 VLM 脚本。回答正确率评测请参见
[使用 EvalScope 评测模型精度](/zh/dev_guide/accuracy_testing/)。

## 准备环境与服务

请参考 [EvalScope 官方安装文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)，安装最新正式版及 `perf` 依赖。

按[启动 xLLM](/zh/getting_started/launch_xllm/) 部署服务，并参考
[在线服务](/zh/getting_started/online_service/) 验证文本或图片请求。

```bash
export HOST=127.0.0.1
export PORT=18000
export API_KEY=EMPTY
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" "http://${HOST}:${PORT}/v1/models"
```

若需要不复用前缀缓存的基线，启动 xLLM 时设置 `--enable_prefix_cache=false`。
若测试开启缓存的场景，应保持各次运行设置一致，并记录重复提示词或图片等缓存条件。

## LLM：随机文本

将以下内容保存为 `eval_llm_performance.sh`。`TOKENIZER_PATH` 指向与服务端权重匹配的
本地 tokenizer 目录；客户端只读取 tokenizer 文件，不加载模型权重。

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

# 增大输入长度，保持输出目标不变。
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B INPUT_TOKENS=8192 \
  bash eval_llm_performance.sh
```

脚本依次测试 `1`、`4`、`8` 三档闭环并发，分别发送 `64`、`128`、`256` 条正式请求。
每个客户端 worker 收到上一条请求的完整响应后才继续发送下一条。每档额外发送 8 条预热
请求，不计入性能统计。如果提高并发，应将 `--warmup-num` 提高到至少最大并发数，并增加
正式请求数量，以便观察稳定阶段的表现。

默认负载目标为 1024 token 的随机提示文本和 256 token 的输出。Chat template 会影响
实际输入长度。`ignore_eos=true` 抑制 EOS 提前停止，便于控制输出长度；仍应以响应 `usage`
和报告中的实际长度为准。服务端上下文限制需要同时容纳输入和输出。若测试接近业务的自然
停止行为，移除 `ignore_eos`，并报告实际输出长度分布。

这些参数定义的是合成负载，不能代表精度评测。需要各轮提示词完全相同时，可改用 EvalScope
的 `line_by_line` 数据集模式，并在每次运行中用相同的 `--dataset-path` 指向固定本地数据。
同时保持并发、请求数和数据顺序一致。

### 严格固定输入、输出长度

默认走 Chat Completions，随机 token 解码成文本后会被服务端重新分词，可能出现少量
输入长度偏差。需要严格固定输入 token 数时，直接发送 token ID：

```bash
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B \
  INPUT_TOKENS=1024 OUTPUT_TOKENS=256 TOKENIZE_PROMPT=true \
  bash eval_llm_performance.sh
```

此选项切换到 `/v1/completions` 并添加 `--tokenize-prompt`，不应用 chat template。
它测试的是 token ID 输入路径。`INPUT_TOKENS` 包含公共前缀，`OUTPUT_TOKENS`
结合 `ignore_eos=true` 控制输出长度；上下文窗口必须容纳两者。仍需核对实际 `usage`。

### 测试 prefix cache

启动 xLLM 时设置 `--enable_prefix_cache=true`，再指定公共前缀长度：

```bash
# 总输入目标仍为 1024 token，其中公共前缀目标为 512 token。
MODEL=Qwen3-8B TOKENIZER_PATH=/path/to/Qwen3-8B \
  INPUT_TOKENS=1024 PREFIX_TOKENS=512 OUTPUT_TOKENS=256 \
  bash eval_llm_performance.sh
```

`PREFIX_TOKENS` 默认为 0。EvalScope 的 `random` 数据集在每轮运行内复用同一个
随机前缀；`--prefix-length` 是额外添加的长度，因此脚本先从总输入预算中扣除它。
剩余预算还需要容纳 chat template。客户端和服务端使用相同 tokenizer 与模板时，
输入长度应接近 `INPUT_TOKENS`；以服务端 `usage.prompt_tokens` 为准。

预热请求可填充缓存，但实际命中还取决于前缀 token 是否相同、缓存块大小和淘汰行为。
检查响应的 `usage.prompt_tokens_details.cached_tokens` 或服务端日志的
`num_prefix_cache_tokens`，不要把配置的前缀比例直接当作实测命中率。比较缓存开关时，
使用相同的固定请求数据，并分别记录冷缓存与预热后结果。

当前 xLLM 的流式 `/v1/completions` 返回总 token 数，但不返回
`prompt_tokens_details.cached_tokens`。使用 `TOKENIZE_PROMPT=true` 时，
请从服务端日志核对缓存命中；EvalScope 缺少该字段时的缓存统计不能用来判断是否命中。

## VLM：Flickr8k 图文输入

将以下内容保存为 `eval_vlm_performance.sh`。EvalScope 的 `flickr8k` 插件读取
`clip-benchmark/wds_flickr8k` 的 `test` 划分，将图片说明文字与 Base64 编码的图片共同
放入请求，覆盖所部署 VLM 的图片输入处理路径。

图文批量处理可能超过 brpc 默认的 64 MiB 消息上限。运行较高并发时，可在启动 VLM
服务的命令中添加 `--max_body_size=536870912`（512 MiB）。如果日志仍提示
`body_size ... is too large`，需按实际图像批次大小调整上限或降低并发。

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

预热和并发配置的含义与 LLM 脚本一致。Flickr8k 使用尺寸不同的真实图片，视觉 token
数量并不固定。比较时应统一数据集版本、图片样本顺序、processor 配置、分辨率限制和图片
缓存行为。离线使用时，可添加 `--dataset-path /path/to/local/flickr8k`，指向 EvalScope
可以加载的本地数据集目录，而不是任意图片文件夹。

该脚本通过 `OUTPUT_TOKENS` 控制输出目标，不能固定图文输入的总 token 数。
`--prefix-length` 仅适用于 `random` 数据集，对 `flickr8k` 不生效。测试 VLM
prefix cache 时，需要准备具有相同起始文本和图片的重复请求，保持消息顺序和图片处理
参数一致，并检查实际 cached token 数；仅统一图片尺寸不能保证前缀命中。

脚本依赖服务端流式返回的 `usage` 统计 token 数。请确认最后的 usage chunk 包含
`prompt_tokens` 和 `completion_tokens`。文本 tokenizer 无法可靠统计 VLM 的视觉 token；
缺少 usage 时，token 吞吐量不可靠。跨 VLM 架构对比时，应在相同图片负载下比较请求吞吐、
TTFT 和输出吞吐，并说明输入 token 的统计口径。

## 指标与结果文件

| 指标 | 含义 |
| --- | --- |
| 请求吞吐量（req/s） | 成功请求数除以测试时长。 |
| 输出吞吐量（token/s） | 所有请求生成的输出 token 总数除以测试时长。 |
| 总吞吐量（token/s） | 输入与输出 token 总数除以测试时长，受输入 token 统计口径影响。 |
| TTFT | 从发送请求到收到首个输出 token 的时间，包含排队和网络开销。 |
| TPOT | 单请求首 token 后的平均每 token 耗时：`(latency - TTFT) / (output_tokens - 1)`，适用于输出多于一个 token 的请求。 |
| ITL | 流式客户端观测到的输出到达间隔；一个 chunk 携带多个 token 时，需要结合分块方式解读。 |
| Latency 与分位数 | 请求端到端耗时，以及 P50/P95/P99 分布。 |
| 成功与失败请求数 | 用于确认预定负载是否完整执行。 |

保留 `--stream`，才能观测 TTFT 和 token 到达间隔。这些指标反映客户端看到的服务性能；
设备执行分析请参见[在线服务性能分析](/zh/dev_guide/online_profiling/)。

EvalScope 会打印 `RUN_DIR`下的实际结果路径，写入 `benchmark.log`，并为多档并发运行生成
`performance_summary.txt`。保留各次运行的请求数据库 `benchmark_data.db`，便于在结果
异常时检查请求内容、失败原因、实际长度和耗时。

### xLLM 参数兼容性

xLLM 当前拒绝非空的请求字段 `seed` 和 `min_tokens`。EvalScope 的 `perf --seed`
除了设置客户端随机种子，还会把生成种子发送给服务端，因此这些脚本不使用该参数，也不使用
`--min-tokens`。示例通过受支持的 `max_tokens` 和 `ignore_eos` 控制输出长度。

升级 EvalScope 后，先检查 `evalscope perf --help`，并完成小规模请求验证，再运行完整测试。
遇到 `400` 响应、流式 usage 缺失或请求未包含图片时，应先解决这些问题，再解读性能结果。

## 参考资料

- [EvalScope 安装](https://evalscope.readthedocs.io/en/latest/get_started/installation.html)
- [性能测试快速开始与指标定义](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html)
- [性能测试参数与数据集配置](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html)
- [负载与预热示例](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/examples.html)
