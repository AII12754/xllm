---
title: "KV Cache 与模型权重的虚拟内存"
sidebar:
  order: 52
---

xLLM 可以分别为 KV Cache 和模型权重启用虚拟内存管理（VMM）。两个开关默认为 `false`，仅 NPU 支持。

| 开关 | 用途 |
|:-----|:-----|
| `enable_vmm_kvcache` | 为 KV Cache 使用 VMM。关闭时使用物理 KV Cache 实现。 |
| `enable_vmm_weights` | 为模型权重使用 VMM，并隐式启用 manual loader。 |

两个开关可以单独启用，也可以同时启用，使 KV Cache 和模型权重共享 VMM 池。仅启用 KV VMM 时继续使用配置的权重加载器，不会隐式启用 `enable_manual_loader`；仅启用权重 VMM 时继续使用物理 KV Cache。权重 VMM 目前仅支持 LLM 后端。模型休眠、唤醒和 fork 需要设置 `enable_vmm_weights=true`。仅启用权重 VMM 时要求 `vmm_elastic=true`，避免固定权重页池占用物理 KV Cache 所需的显存。所有开关和默认值见 [CLI 参数参考](/zh/cli_reference/)。

## 弹性 KV Cache

设置 `enable_vmm_kvcache=true` 和 `vmm_elastic=true` 后，KV 页映射时按需增长物理 backing。页面上的最后一个 block 归还后，空闲 backing 会被释放。VMM 池会预留稳定的虚拟地址，并随需求激活逻辑槽位。

显式配置正数 `max_memory_utilization` 时，arena 不超过设备总显存的该比例，物理页增长也会在每个 worker 上保留剩余比例的空闲显存。不配置 ratio 或设为 `0` 时，arena 可覆盖设备总显存；`max_cache_size` 仍可作为字节上限。实际增长和请求准入始终依据实时可用显存并保留执行余量。支持原生 allocator 统计的后端可以将池外释放的显存用于新增 KV 容量，无需重启；不支持可靠统计的后端保留保守执行水位。未被预热覆盖的执行形状和直接分配的 SDK buffer 可通过 `vmm_memory_reserve_bytes` 预留空间。

可在现有启动命令中加入以下参数，取消两个显式容量上限，并为执行额外保留 1 GiB 空闲显存。弹性模式下，不配置 `max_memory_utilization` 与设为 `0` 效果相同：

```bash
--enable_vmm_kvcache=true \
--vmm_elastic=true \
--max_memory_utilization=0 \
--max_cache_size=0 \
--vmm_memory_reserve_bytes=1073741824
```

额外预留量应根据设备实测调整。`vmm_memory_reserve_bytes=0` 仍会保留运行时执行所需空间。弹性虚拟内存使用 graph 时必须执行启动预热（`disable_graph_warmup=false`）。使用 Mooncake 的 PD KV 传输及远端权重传输要求池完全驻留，应设置 `vmm_elastic=false` 并使用正数 `max_memory_utilization`。架构说明见[弹性虚拟 KV](/zh/dev_guide/code_arch/#弹性虚拟-kv)。
