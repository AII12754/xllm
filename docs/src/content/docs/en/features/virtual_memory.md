---
title: "Virtual Memory for KV Cache and Model Weights"
sidebar:
  order: 52
---

xLLM can use virtual memory management (VMM) independently for the KV Cache and model weights. Both options default to `false` and are supported on NPU.

| Flag | Purpose |
|:-----|:--------|
| `enable_vmm_kvcache` | Use VMM for the KV Cache. When disabled, xLLM uses the physical KV Cache implementation. |
| `enable_vmm_weights` | Use VMM for model weights and implicitly enable the manual loader. |

Enable either option on its own, or enable both to share the VMM pool between the KV Cache and model weights. KV-only VMM keeps the configured weight loader and does not imply `enable_manual_loader`. Weight-only VMM keeps the physical KV Cache implementation. Weight VMM currently supports the LLM backend. Model sleep, wakeup and fork require `enable_vmm_weights=true`. Weight-only VMM requires `vmm_elastic=true`; a fixed weight pool would consume memory needed by the physical KV Cache. See the [CLI Reference](/en/cli_reference/) for all flags and defaults.

## Elastic KV Cache

With `enable_vmm_kvcache=true` and `vmm_elastic=true`, physical backing grows as KV pages are mapped. Idle backing is released after the last block on a page is returned. The VMM pool reserves a stable virtual address arena and activates logical slots as demand grows.

An explicitly configured positive `max_memory_utilization` bounds the arena to that fraction of total device memory and leaves the remaining fraction free during physical growth on each worker. Omitting the ratio or setting it to `0` lets the arena cover total device memory. `max_cache_size` remains an optional byte limit. Actual growth and request admission use live available memory and preserve execution headroom. On backends with native allocator statistics, memory released outside the pool can provide additional KV capacity without restarting; backends without reliable statistics retain conservative execution watermarks. Use `vmm_memory_reserve_bytes` for direct SDK buffers and execution shapes not covered by warmup.

Append these flags to an existing launch command to remove both explicit capacity ceilings and keep an additional 1 GiB free for execution. In elastic mode, omitting `max_memory_utilization` has the same effect as setting it to `0`:

```bash
--enable_vmm_kvcache=true \
--vmm_elastic=true \
--max_memory_utilization=0 \
--max_cache_size=0 \
--vmm_memory_reserve_bytes=1073741824
```

Tune the additional reserve using device measurements. A value of `vmm_memory_reserve_bytes=0` still preserves runtime execution headroom. When using graphs with elastic virtual memory, startup warmup is required (`disable_graph_warmup=false`). PD KV transfer and remote weight transfer using Mooncake require a fully resident pool; use `vmm_elastic=false` and a positive `max_memory_utilization`. See [Elastic virtual KV](/en/dev_guide/code_arch/#elastic-virtual-kv) for architecture details.
