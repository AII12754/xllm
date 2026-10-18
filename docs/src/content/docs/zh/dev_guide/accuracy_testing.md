---
title: "使用 EvalScope 评测模型精度"
description: "通过 OpenAI 兼容接口评测 xLLM 的 LLM 和 VLM 精度。"
---

使用 `evalscope eval`，通过已启动的 xLLM 服务评测回答的正确率。本文提供基于
GSM8K 的纯文本 LLM 脚本，以及基于 AI2D 的图文 VLM 脚本。两者均使用 EvalScope
原生评测后端和规则评分，无需单独部署裁判模型，也无需安装 VLMEvalKit。

延迟和吞吐量测试请参见[使用 EvalScope 测试推理性能](/zh/dev_guide/performance_testing/)。

## 准备环境与服务

请参考 [EvalScope 官方安装文档](https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html)，安装最新正式版。

首次运行会从 ModelScope 下载数据，请提前准备数据集访问条件和足够的本地缓存空间。

1. 按[启动 xLLM](/zh/getting_started/launch_xllm/) 部署支持的模型。
2. 参考[在线服务](/zh/getting_started/online_service/)，确认对应的文本或图片请求可以成功完成。
3. `HOST` 和 `PORT` 填写服务地址，脚本使用 `http://${HOST}:${PORT}/v1` 作为 API 根地址。
   `MODEL` 必须与 `/v1/models` 返回的模型 ID 一致，不是客户端的本地权重路径。
   示例使用 `18000` 端口，请替换为实际端口。

```bash
export HOST=127.0.0.1
export PORT=18000
export API_KEY=EMPTY
curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" "http://${HOST}:${PORT}/v1/models"
```

## LLM：GSM8K

将以下内容保存为 `eval_llm_accuracy.sh`，使用 Bash 执行。脚本评测 GSM8K 的 `main`
子集、`test` 划分，使用 **four-shot** 提示词和固定的 few-shot 样本
（`few_shot_random=false`）。生成参数为 `temperature=0.6`、`top_p=0.95`、`top_k=20`，
默认输出上限为 1024 token。模板参数请求关闭思考模式，适用于支持这些开关的模型模板。
默认 API 并发为 64。

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
# 小规模回归：64 条样本。
MODEL=Qwen3-8B bash eval_llm_accuracy.sh

# 完整 test 划分：不向 EvalScope 传入样本数量限制。
MODEL=Qwen3-8B LIMIT=all bash eval_llm_accuracy.sh
```

`BATCH_SIZE=64` 控制 API 并发，`LIMIT=64` 单独控制评测样本数。设置 `LIMIT=all`，
即可使用相同的生成和 four-shot 配置评测完整 test 划分。

评分器会提取最终数值答案，并与参考答案比较。请保留基准自带的提示词和答案格式。
对于推理模型，如果回答在最终答案前被截断，应提高 `MAX_TOKENS`，并确保服务端的上下文长度
能容纳输入和生成内容。

## VLM：AI2D

将以下内容保存为 `eval_vlm_accuracy.sh`。AI2D 的每条样本包含一张图片和一道选择题，
用于评测图表理解能力。EvalScope 会读取图片，通过 Chat Completions 接口发送图文内容；
只向 VLM 发送纯文本请求无法覆盖视觉理解能力。

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

脚本使用 AI2D 的 `default` 子集、`test` 划分和 zero-shot 选择题正确率。对比时应固定图片
预处理、分辨率限制和模型 processor 文件。检查保存的输入，确认包含图片，并查看预测结果中
是否存在答案提取失败或输出截断。

## 参考资料

- [EvalScope 安装](https://evalscope.readthedocs.io/en/latest/get_started/installation.html)
- [精度评测参数](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html)
- [GSM8K 基准](https://evalscope.readthedocs.io/en/latest/benchmarks/gsm8k.html)
- [AI2D 基准](https://evalscope.readthedocs.io/en/latest/benchmarks/ai2d.html)
