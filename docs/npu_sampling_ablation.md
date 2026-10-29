# NPU sampling: implementation and ablation

This report accompanies the working changes on `perf/npu-sampling-pipeline`,
based on xLLM `9dee07ef4ab6d8130794ceb538e2c6c049d49ed2`.
DeepInfer was inspected at `553741557a57df44a46fb3ab72b6aafeaf945337`.
Results below apply to Ascend 910B2, CANN 9.0.0, PyTorch 2.9.0 and
torch_npu 2.9.0.post2. Sustained tests use physical devices 4–7 only.

**Status: ordinary and non-task-pipeline DFlash2 implementation, native
ablations, same-process serving measurements and trace exports are complete.**
Real-weight task-pipeline E2E is excluded from this 910B2 acceptance scope
by user direction on 2026-10-28, because its current custom operator package
registers the required Varlen operator only for 910C. It is not a remaining
acceptance blocker for this task. Supported inputs select the optimizations
automatically; performance claims are scoped to the measured shapes and model pair.

## Changes and boundaries

Production has no sampling optimization switches: existing shape, batch and
request guards select the applicable paths automatically. These NPU changes
are independent of the CUDA changes in #2542 and #2550.

Only a dedicated `BUILD_NPU_SAMPLING_ABLATION=ON` build contains the offline
ablation control and native benchmark. Its mask defaults to 63 (all features);
mask 0 selects the legacy reference. Normal builds compile each feature check
to `true` and contain no mutable control symbol, CLI option or JSON setting.

The performance tables retain the original measured executable identities.
The complete 400-workload serving matrix has not been rerun after removing
the production switches; current validation adds native cross-checks and
real-model functional smoke tests. Those original executables used six
ExecutionConfig bools for ablation. Removing the
production options does not change the kernels or their input guards; those
measurements are not relabeled as timings of the new production executable.

| ID | Test bit | Optimization | Implementation |
| --- | ---: | --- | --- |
| A | 1 | Batched sampling | Batch seeded categorical draws on device; avoid copying seed/offset controls back to the host and constructing a generator per sampled request. |
| B | 2 | Random-row filtering | Compact random rows before temperature/min-p/top-k/top-p processing in supported ordinary mixed batches; scatter results back. Penalties and grammar still run for every row. |
| C | 4 | Fused logits sampling | Fuse normalization and categorical selection, avoiding a vocabulary-sized FP32 probability tensor. Requires A for ordinary sampling. Existing filtering stays separate. |
| D | 8 | Compact normalization | Row log-sum-exp plus selected-score gathering; reuse normalization in verification. Preserve the full log-softmax path for top-logprob requests. |
| E | 16 | Sparse DFlash2 | Preserve the DFlash2 selector's existing candidate support through proposal construction, transport, acceptance and exact residual recovery. |
| G | 32 | Verification sampling | Use the shared fused categorical kernel for bonus/recovery tokens in supported verifier batches. Acceptance/prefix handling stays in the existing verifier. |

The implementation follows the batching, on-chip normalization and compact
sampling-state ideas in DeepInfer's
[sampler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/sampler.py)
and [Ascend fused sampler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/layers/tilelang_ops/asc/fused_sample.py).
It uses the installed TileLang Ascend API and xLLM's existing sampling semantics.

Important boundaries:

- Keep xLLM's top-k-then-top-p order and native tie behavior. A probe replacing
  top-p-only filtering with a synthetic vocabulary-sized top-k changed one
  retained token on tied logits (`V=49152`, several p values). That candidate
  (F) is excluded. Full filtering fusion is not part of C.
- B applies to ordinary sampling with `B>=2`, `V>=32768`, at most half random
  rows, and no logprobs, returned logits or beam search. Verification retains its
  existing filter sequence.
- Ordinary D applies to selected logprobs only, BF16/FP16, `B>=128` and
  `V>=65536`; verification D requires at least 8 Mi logits elements.
  Subtracting LSE can round distinct logits into FP32 ties, so selecting raw-logit
  top-k would change reported token IDs. Top-logprob reporting keeps the
  original operation, including in the compact verifier.
- G requires at least eight verifier rows. Small-batch measurements did not
  justify this kernel there.
- Kernels support FP32/BF16/FP16 logits, unit inner stride, up to 4096 rows and
  262144 vocabulary entries, with validated address-span bounds. Unsupported
  shapes take the existing path before execution.
- E preserves existing, unique selector IDs and their exact q distribution.
  It does not truncate an arbitrary dense draft distribution. Sparse producer
  and consumer must be enabled together.
- A uses a stateless SplitMix64-derived uniform with request seed and offset.
  Seeded draws remain deterministic for fixed logits and independent of batch
  permutation/global RNG. The random sequence differs from the legacy
  multinomial generator. Unseeded draws use the normal NPU generator.
- Sampling metadata is propagated through device copies, reusable slots and
  packed inputs. The packed-input schema moves from 9 to 10; ranks must use the
  same version.
- A separate shutdown fix avoids synchronizing the TLS ACL top-k/top-p
  workspace after torch_npu has finalized the NPU runtime.

## Models and experiment method

Target: `/data/models/Qwen3.8-27B` (qwen3_5, 248320 vocabulary entries,
18 complete shards). The similarly named copy under `/home/lhp/models/Qwen`
is incomplete and was not used.

Draft: `/home/lhp/models/z-lab/Qwen3.8-27B-DFlash2`, public
[z-lab weights](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4),
revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
It supplies K=16 candidates and a seven-token draft block.

Native measurements restore fixed logits outside the timed region, warm up,
synchronize the compute stream, and measure host wall latency including launch
and completion. Variants are interleaved in one process. Memory is the
PyTorch allocator's peak above preallocated inputs; it excludes ACL raw
workspace allocations and is **not total device HBM usage**.

Serving comparisons keep the same executable, resident target/draft models,
four worker PIDs and graph settings within each run. The controller waits for
running/pending requests to drain, stops all owned ranks, writes and verifies
the ablation state, resumes the ranks, and warms the next workload. Variant
order is shuffled independently for each paired repeat. Historical datasets
used the six ExecutionConfig bools; the current test build uses one volatile
byte with the same feature bits. Production contains neither mechanism.

Serving uses TORCH backend, ACL graphs, chunked prefill, prefix cache disabled,
schedule overlap disabled, a 16 GiB KV cache, and explicit linear-state capacity
of 128 (ordinary) or 32 (DFlash2). The requested task-pipeline suite would
compare variants within its own resident group, but its baseline fails startup
and contributes no accepted timings.
Concurrency is a client setting; graph/kernel traces establish which batch
shapes actually reach the optimized paths.

Requests use one fixed 15-token prompt, temperature 0.7, top-k 20, top-p 0.95,
fixed requested completion length and ignore-EOS. Mixed batches have 25%
random requests (index modulo four is zero; concurrency one is entirely
random). Ordinary seeded requests use seed `100+request_index`.
DFlash2's API does not support request seeds; acceptance counters are recorded
before and after each measured workload. Warmup and profiling are outside
the timed window.

TPOT is `(last nonempty streamed chunk - first nonempty chunk)/(tokens-1)`.
A DFlash2 chunk can contain multiple tokens: this is mean streaming TPOT,
**not a per-token ITL distribution**. Throughput includes prefill and HTTP
completion time. Every request must return HTTP 200 and exactly its requested
completion-token count.

The model's raw logprobs vary even for repeated baseline requests. For example,
five identical single-request baseline top-5 runs gave first-token logprobs
between -1.661983 and -1.652640. Greedy baseline responses also vary in this
environment. Therefore serving text identity is not used as a substitute for
sampling correctness on fixed logits. The report does not claim model-level
bitwise determinism or a model-quality benchmark.

### Task-pipeline scope exclusion

The real-weight DFlash2 task-pipeline baseline fails during startup graph
capture before any measured request, with all six sampling options disabled:
`aclnnMegaGdnMtpDecodeVarlen[GetWorkspaceSize]` is unavailable. The installed
custom library exports `aclnnMegaGdnMtpDecode` but not its Varlen counterpart.
The checked-out operator definition registers Varlen only for `ascend910_93`
(910C); the `ascend910b` build list does not include it. This environment has
910B2 devices. Replacing that unrelated recurrent kernel or changing its chip
registration would require a separate correctness/porting effort. No such
substitution was made to obtain benchmark results.

The current registration/build support does not establish that a 910B port
is intrinsically impossible. Per user direction on 2026-10-28, no 910B port
or further task-pipeline E2E validation is requested in this task. This mode
is excluded from acceptance rather than recorded as a passed test.

Task-pipeline sampling transport has unit-test coverage, but
**real-weight task-pipeline E2E performance is not validated**. The failed
baseline launch and all four rank logs remain under
`local/npu_sampling/accepted_dflash_pipeline/`; the corresponding JSONL has
only a launch event and is excluded from performance summaries.

## Correctness and regression checks

The original measured build passed the following checks:

- `sampler_test`: 65 tests pass, including NPU categorical support/distribution,
  vocabulary tails and padding, FP32/BF16/FP16, fixed-seed isolation, mixed
  unseeded RNG, greedy preservation, native filter ties, raw selected logprobs,
  and dense-versus-sparse rejection/recovery.
- `slot_buffer_test`: 51 tests pass; packed/shared input: 18 tests pass;
  ExecutionConfig: seven tests pass. These cover propagation/reset of the two
  new host-derived sampling controls and serialization schema 10.
- Existing block-worker/recurrent task-pipeline tests: 23 pass. These do not replace
  a real-weight task-pipeline E2E run.
- The eight rounded-top-logprob verifier cases were rerun at 192 x 65536
  logits elements, beyond D's activation threshold; all pass. This rerun is a
  subset of the 65 sampler tests, not eight additional unique tests.
- An additional fixed-logits allocation/repeatability probe completed 600
  calls over batch sizes 1/49/128, FP32/BF16 and V=248320 without a token-ID
  mismatch. This tests the sampler independently of serving/model variation.

Test logs are retained under `/tmp/npu_sampling_final_extra_tests.log`,
`/tmp/npu_sampling_acceptance_pipeline.log`, `/tmp/npu_sampling_shared_memory_test.log`,
`/tmp/npu_sampling_config_check.log`,
`/tmp/npu_sampling_final_tie_regression.log` and
`/tmp/npu_sampling_repeatability_probe.log`; the final manifest records their
identities. Copies are also saved under `local/npu_sampling/validation/`.
Source formatting uses clang-format 20.1.6 and Python lint uses
the repository's Ruff configuration.

The original submission review passed 164 regression cases: the original 161 plus
three additional block-worker option cases. Its combined log is
`local/npu_sampling/validation/npu_sampling_review_regression.log`.
It also checks the benchmark controller against a live owned child process
under normal Python and `python -O`.
That historical probe exercises real process stopping, all six memory writes/readbacks,
PID preservation and invalid-mask rejection without using an NPU. Controller
writes and measurement checks use explicit runtime validation, so Python's
optimization mode cannot remove them.

### Automatic-dispatch follow-up validation

The revised production build passes 160 targeted tests: sampler 35, sampling
controls 24, filter masks 3, slot buffers 51, packed/shared inputs 18,
execution config 6 and recurrent/block-worker pipeline 23. Production runs only
the automatic verifier policy; the test-only build retains the normalization
and verification factorial combinations. The old configuration round-trip
case is removed with the six options. Independent Torch references check
raw selected logprobs and rounded top-token ties.

The production executable has no ablation symbol, and its help has none of the
six removed options. Four-device Qwen3.8-27B smoke coverage passes random B=8,
mixed B=32, selected logprobs B=128, top-5 B=8 and greedy B=8: all 184 requests
return the requested 16 completion tokens. This is a functional smoke test,
not a new end-to-end performance acceptance dataset.

Logs, command lines and production hashes are retained under
`local/npu_sampling/automatic_validation/`. The controller also passes real
PIE executable symbol lookup and memory-write/readback tests under normal
Python and `python -O`; production executables reject that controller.

The instrumented build passes all 65 sampler cases, including the D/G
factorial combinations. A new three-repeat, interleaved native cross-check
confirms the revised test control selects the measured paths:

| Workload | Baseline us | Optimized us | Speedup |
| --- | ---: | ---: | ---: |
| Ordinary random, BF16, B=32, V=248320, A+B+C | 4435.83 | 1151.78 | 3.85x |
| Ordinary mixed 25% random, BF16, B=32, V=248320, A+B+C | 44659.83 | 986.73 | 45.26x |
| Verification, FP32, B=32, S=7, K=16, D+E+G | 7265.51 | 2684.52 | 2.71x |

The new test controller also passes a real-weight DFlash2 smoke in one
resident four-rank group: masks 63 and 0, random B=1, 32 completion tokens.
All four PID/readback checks pass. This verifies switching and sparse proposal
transport; these two short windows are not added to the 400-workload performance
acceptance dataset. The build is restored to production afterward; the sampler
binary matches its validated production SHA, and the relinked service is
checked with the same functional smoke.

The verification allocator peak remains 636.57 MiB versus 424.43 MiB
(-33.3%). Its complete eight-way D/E/G factorial and raw samples are in
local/npu_sampling/automatic_validation/instrument_native.json. These are
new measurements of the test build; the original tables below keep their
original build identities.

### Distribution abstraction validation

Draft distributions now encapsulate deterministic, dense and sparse q behind
one value type. The verifier depends on normalization, acceptance correction
and residual recovery operations; representation validation stays with the
distribution. Preallocated logits are normalized at verification time, and
proposal token replacement retains distribution storage and input owners.
Unseeded verifier sampling uses the public categorical entry point, retaining
the NPU inverse-CDF dispatch and the existing Gumbel primitive.

The refactor passes 92 production regression cases (37 sampling, 29 CPU
controls/distribution, three masks and 23 pipeline cases) and 67 instrumented
sampling cases. Production has no ablation symbol. A four-device production
model smoke passes all 184 requests across random, mixed, selected-logprob,
top-5 and greedy modes.

Five-repeat native checks preserve the optimized B=32 results: ordinary
random 1150.92 us, ordinary mixed 981.14 us, and D+E+G verification 2680.74 us.
Against the published implementation retested in the same session, these
change by -0.07%, +0.44% and +0.32%, respectively; allocator peaks are unchanged.
Each executable interleaves its ablation masks in one process. A separate
published/refactored/refactored/published B=1 check collects 14 samples per
version and mask: D+E changes by -0.45%, and D+E+G by -2.29%. It does not
reproduce the initial small-batch slowdown seen in separate-process medians.

The serving check switches masks in one resident four-rank group per model,
using physical devices 4–7, three repeats and 64 output tokens. Ordinary uses
masks 0/7; DFlash2 uses 0/24/56/63. All 576 ordinary and 864 DFlash2 requests
complete successfully. The table shows medians for baseline versus all
applicable optimizations, without replacing the historical acceptance matrix.

| Workload | Baseline tokens/s | Optimized tokens/s | TPOT ms, baseline → optimized |
| --- | ---: | ---: | ---: |
| Ordinary random, B=32 | 836.51 | 932.04 | 35.15 → 31.35 |
| Ordinary mixed, B=32 | 410.22 | 946.55 | 75.09 → 30.87 |
| Ordinary greedy, B=32 | 975.41 | 975.70 | 29.94 → 29.91 |
| DFlash2 random, B=32 | 570.62 | 587.18 | 45.68 → 43.81 |
| DFlash2 mixed, B=32 | 208.32 | 213.74 | 141.40 → 138.38 |
| DFlash2 greedy, B=8 | 228.70 | 227.65 | 26.71 → 25.55 |

DFlash2 combinations remain sensitive to acceptance and scheduling variation:
D+E random throughput is 614.03 tokens/s, exceeding all-on in this short run.
Greedy is a control, not a claimed speedup: its D+E+G-only TPOT is 31.22 ms
versus baseline 26.71 ms, although throughput changes by -1.19%; all-on TPOT
is 25.55 ms. These variable control results are retained in the raw records.

Build logs, binary hashes, native repeats, switching readbacks and all serving
results are under local/npu_sampling/review_followup/.

## Native ablations

Representative ordinary BF16 mixed batches, V=248320, 25% random rows:

| Rows | Baseline us | A us | B us | A+B us | A+B+C us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3073.2 | 2792.6 | 1169.3 | 879.3 | 876.3 |
| 8 | 8487.0 | — | 1318.3 | — | 894.6 |
| 32 | 45170.5 | 43930.5 | 2150.8 | 1012.9 | 984.4 |
| 128 | 135666.4 | — | 6305.0 | 2345.0 | 2077.4 |

These rows come from the matched ordinary and small-batch native runs.
The final normalization-source retake uses five interleaved repetitions:

| BF16 selected logprobs, B=128, V=248320 | Median us | Extra allocator peak MiB |
| --- | ---: | ---: |
| Baseline | 16408.8 | 363.8 |
| A | 3808.4 | 404.5 |
| D | 16186.4 | 303.1 |
| A+D | 3783.9 | 343.9 |
| A+C | 3577.6 | 283.2 |
| A+C+D | 3538.6 | 76.6 |

A alone can increase temporary memory by introducing a full probability tensor.
C removes that cost. D's main additional benefit is memory; its standalone
latency benefit is small. Top-5 requests retain the legacy normalization and
therefore do not get D's selected-logprob memory reduction.

Final verification retake, FP32 target logits, B=32, S=7, V=248320, K=16,
without reported logprobs:

| Variant | Median us | Extra allocator peak MiB |
| --- | ---: | ---: |
| Baseline | 7271.0 | 636.6 |
| D | 6861.1 | 636.6 |
| E | 5123.3 | 424.4 |
| D+E | 4648.8 | 424.4 |
| G | 5276.3 | 636.6 |
| D+G | 4836.5 | 636.6 |
| E+G | 3188.4 | 424.4 |
| D+E+G | 2680.6 | 424.4 |

The full factorial includes every single addition, pair, and leave-one-out from
D+E+G. With selected logprobs enabled, baseline is 7914.6 us and D+E+G is
2709.7 us. The sparse proposal removes approximately 212.2 MiB of dense q
storage at B=32, S=7. The benchmark includes proposal materialization, using
one dense scatter, conservatively cheaper than the real producer's per-step
scatters.

Sparse proposals also have a small-batch tradeoff. At B=1, S=7, V=248320,
the native E-only case takes 1335.0 us versus 1257.3 us for the baseline
(6.2% slower), while temporary allocator peak falls from 34.01 to 27.38 MiB
(19.5% lower). D and G are inactive at this size. The long-output serving
check below does not show a sustained regression, but this native latency
cost remains part of the feature's acceptance evidence; E is not claimed
to improve latency at every shape.

## Serving ablations

### Ordinary sampling

The final run completed 168 measured workloads (three paired repetitions)
using the same worker PIDs **184473, 184474, 184475, 184476** throughout.
All requests completed with the requested 64 output tokens. There was no
compilation or trace export during these timings. Each cell below is a
median across repetitions. Percentage changes are the median of the three
within-repeat ratios, rather than a ratio of separately aggregated medians.

| Workload | Concurrency | Mask | Baseline TPOT ms | Variant TPOT ms | Baseline tok/s | Variant tok/s | Paired TPOT change | Paired throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 7 | 20.717 | 20.342 | 45.3 | 46.2 | -1.90% | +1.95% |
| random | 32 | 7 | 35.819 | 31.422 | 825.9 | 932.5 | -12.19% | +12.74% |
| random | 128 | 7 | 77.387 | 60.851 | 1553.6 | 1944.0 | -21.37% | +25.13% |
| mixed | 1 | 7 | 20.563 | 20.298 | 45.7 | 46.4 | -1.40% | +1.36% |
| mixed | 32 | 7 | 75.199 | 31.004 | 409.2 | 941.8 | -58.80% | +130.17% |
| mixed | 128 | 7 | 192.522 | 59.498 | 648.1 | 1981.4 | -69.13% | +205.07% |
| unseeded | 32 | 7 | 32.322 | 31.301 | 904.2 | 930.7 | -3.26% | +3.16% |
| logprobs | 128 | 13 | 77.866 | 61.202 | 1535.5 | 1921.1 | -21.40% | +25.11% |
| top5 | 128 | 13 | 78.856 | 62.763 | 1516.6 | 1873.2 | -20.43% | +23.52% |
| greedy | 32 | 7 | 30.082 | 30.162 | 962.5 | 955.4 | +0.21% | -0.74% |

Mixed concurrency 128 improves from 648.1 to 1981.4 tok/s (about 3.05x);
this large gain depends on skipping expensive greedy-row filtering. The
three paired TPOT ratios are 0.3084, 0.3087 and 0.3096. Random concurrency
128 improves throughput by about 25%. Greedy is a control: these features
are inactive, with +0.21% TPOT / -0.74% throughput in this run.

The concurrency-32 component comparison shows why joint measurements matter:

| Mask | Ordinary random TPOT ms | Ordinary mixed TPOT ms |
| --- | ---: | ---: |
| 0 | 35.819 | 75.199 |
| 1 | 31.379 | 73.711 |
| 2 | 35.822 | 32.542 |
| 3 | 31.513 | 31.124 |
| 5 | 31.451 | 73.645 |
| 7 | 31.422 | 31.004 |

A removes most seeded per-request overhead. B is inactive in an all-random
batch, but dominates the mixed-batch improvement. C contributes primarily
temporary-memory savings beyond A at these serving shapes. D alone on
selected logprobs has a paired TPOT ratio of 1.0013 and throughput ratio
of 0.9965; its additional acceptance benefit is the native memory reduction,
not a demonstrated standalone E2E latency gain. For top-5, D retains the
legacy normalization, as required for exact reported token IDs.

Full per-mask/per-repeat ratios, actual request usage and outputs remain in
`local/npu_sampling/same_process_ordinary_final.jsonl` and
`local/npu_sampling/e2e_summary.json`.

### DFlash2 ordinary execution path

The 64-token matrix completed 189 measured workloads, three paired repetitions
per variant, with worker PIDs **177095, 177096, 177097, 177098** unchanged.
It includes the full D/E/G factorial, A+B+C prefill controls (mask 7), and
all six options together (mask 63). Sparse producer and consumer are switched
as one E feature. Every request returned its requested output-token count.

| Workload | Concurrency | Baseline TPOT ms | Mask 63 TPOT ms | Baseline tok/s | Mask 63 tok/s | Paired TPOT change | Paired throughput change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 19.446 | 14.974 | 48.0 | 61.0 | -10.82% | +11.03% |
| random | 8 | 25.521 | 25.633 | 247.8 | 252.1 | +1.78% | +2.86% |
| random | 32 | 44.585 | 42.692 | 599.5 | 616.7 | -6.74% | +3.68% |
| mixed | 1 | 20.935 | 20.008 | 44.9 | 46.9 | -1.33% | +1.35% |
| mixed | 8 | 51.692 | 48.000 | 136.2 | 145.2 | -10.62% | +6.64% |
| mixed | 32 | 141.182 | 133.779 | 207.0 | 220.6 | -6.61% | +6.30% |

The concurrency-32 factorial gives the following serving view:

| Mask | Random TPOT ms | Random paired throughput change | Mixed TPOT ms | Mixed paired throughput change |
| --- | ---: | ---: | ---: | ---: |
| 0 | 44.585 | +0.00% | 141.182 | +0.00% |
| 8 | 44.857 | -2.71% | 136.539 | +4.18% |
| 16 | 44.114 | -0.58% | 138.332 | +5.71% |
| 24 | 42.505 | +3.06% | 135.328 | +4.49% |
| 32 | 45.945 | +0.54% | 134.790 | +4.94% |
| 40 | 43.746 | -2.23% | 140.363 | -0.82% |
| 48 | 44.426 | +0.70% | 137.918 | +5.51% |
| 56 | 44.301 | -3.21% | 137.464 | +1.04% |
| 7 | 44.384 | -2.61% | 140.623 | +2.76% |
| 63 | 42.692 | +3.68% | 133.779 | +6.30% |

The native factorial establishes reductions in verifier latency/memory with
fixed inputs. The short serving matrix supports the complete group at
concurrency 32 (roughly +3.7% random / +6.3% mixed throughput), but does not
establish a positive E2E latency benefit for every isolated feature. For
example, mask 56 has -3.2% random throughput in this short matrix despite a
lower median TPOT; it is not presented as an unconditional serving speedup.

DFlash2 request seeds are unsupported and acceptance lengths differ between
variants. At concurrency one, D and G are gated off, so masks 16 and 56
execute the same optimized path. Nevertheless, their short-request TPOT
medians are 18.979 and 21.726 ms: three short repetitions are too noisy to
infer their relative speed. The greedy short control also varies (mask 63
paired TPOT +9.1%, throughput -1.2%), with different committed-prefix lengths.
The long random and greedy controls below check these apparent regressions.

Raw data: `local/npu_sampling/same_process_dflash.jsonl`. The report does not
equate stochastic acceptance-length changes with kernel speedup.

### Long-output DFlash2 check

A separate resident four-rank run completed 28 workloads: concurrency one,
512 output tokens, seven paired repetitions of masks 0/16/56/63. Worker PIDs
**187506, 187507, 187508, 187509** remain unchanged. This extends the noisy
64-token single-request comparison; no CPU build/export overlaps the timing.

| Mask | TPOT ms | Tok/s | Paired TPOT change | Paired throughput change | Median committed tokens / draft | Paired TPOT ratio range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 19.267 | 51.44 | +0.00% | +0.00% | 2.769 | 1.000–1.000 |
| 16 | 18.093 | 54.74 | -4.70% | +4.88% | 2.960 | 0.899–1.037 |
| 56 | 17.665 | 56.05 | -8.30% | +8.96% | 3.000 | 0.726–0.982 |
| 63 | 18.070 | 54.81 | -6.34% | +6.67% | 2.909 | 0.879–1.021 |

No sustained small-batch regression is observed in this retake. However,
acceptance-length variation remains visible: masks 16 and 56 have the same
active implementation at this shape, and their different aggregate times
must not be attributed to D/G, which are gated off. The per-feature memory
and fixed-input latency claims come from the native factorial. Mask 63
improves paired throughput by 6.67% here; this does not establish a guaranteed
speedup for other prompts or acceptance distributions.

Raw data: `local/npu_sampling/same_process_dflash_long.jsonl`.

### Long-output greedy control

The last control run uses concurrency eight, 512 output tokens, five paired
repetitions, and unchanged worker PIDs 190176, 190177, 190178, 190179.

| Mask | TPOT ms | Tok/s | Paired TPOT change | Paired throughput change | Paired TPOT ratio range |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 22.233 | 332.52 | +0.00% | +0.00% | 1.000–1.000 |
| 56 | 21.704 | 337.40 | -0.69% | +3.22% | 0.905–1.118 |
| 63 | 22.221 | 333.23 | +3.16% | +0.04% | 0.920–1.071 |

The mask-63 throughput ratio is 1.0004, with essentially unchanged separate
median TPOT (22.233 versus 22.221 ms). The median paired TPOT ratio is 1.0316,
and individual paired ratios range from 0.920 to 1.071. This control does not
establish a systematic throughput regression or a greedy latency improvement;
acceptance/scheduling variability remains larger than the sampling effect.
It is included rather than dropping the short greedy control that looked
worse. Raw data: `local/npu_sampling/same_process_dflash_greedy_long.jsonl`.

### Acceptance scope

- A+B+C passes the tested ordinary sampling matrix, including mixed/random,
  seeded/unseeded and greedy controls. C mainly removes the temporary
  probability tensor beyond the batched A path.
- D is useful for selected-logprob temporary memory and jointly with other
  verifier features; a standalone serving-latency gain is not established.
  Top-logprob requests intentionally retain the original normalization.
- D+E+G has the strongest fixed-input verifier result (63.1% less native
  latency, 33.3% less temporary allocator peak at the representative shape).
  The all-options serving group has modest gains for the tested target/draft
  pair, with stochastic uncertainty documented above. Isolated/group results
  are not all positive in the short E2E matrix; no blanket per-feature E2E
  speedup is claimed.
- F (synthetic full-vocabulary top-k in top-p-only filtering) is rejected
  because it changes tie behavior. Full filtering fusion is excluded.
- All six optimizations apply automatically behind the existing input guards;
  production exposes no feature switches. Real-weight task-pipeline E2E
  is outside this user-approved 910B2 acceptance scope and does not block
  completion. No performance claim is made for that mode.

The accepted serving datasets contain 400 measured workloads in total:
168 ordinary, 189 short DFlash2, 28 long random DFlash2, and 15 long greedy
controls. Earlier restart-per-variant runs and the overlapping-build pilot
are excluded. The implementation is submitted in [PR #2603](https://github.com/xLLM-AI/xllm/pull/2603).

## Device trace evidence

The existing online Ascend profiling API is used after timing, followed by
explicit `msprof --export=on` for **every** PROF directory. CANN 9 rejects
aclprof APIs when `PROFILING_MODE=dynamic`; that attach-mode variable is
removed for this online-API capture method. The initial failed capture is
retained separately.

Six ordinary capture windows cover four ranks/devices each: 24 complete
exported timelines. All contain timestamped events; paths, byte counts,
SHA-256 hashes, device IDs, task types and operator totals are in
`local/npu_sampling/serving_repeatability/profile_ordinary/trace_index.json`.
No browser/Perfetto UI inspection was performed.

Rank 0 / physical device 4, 32-token capture workloads:

| Window | Baseline | Optimized | Observation |
| --- | ---: | ---: | --- |
| Random, concurrency 32: per-request uniform kernels | 1024 | 0 | Replaced by 33 fused categorical invocations. |
| Random: synchronous memcpy API calls | 101 | 35 | Copy directions are not present in these timeline API events. |
| Random: asynchronous memcpy API calls | 1371 | 347 | Includes generator-state traffic; these are whole-window counts. |
| Mixed, concurrency 32: ApplyTopKTopP kernel total ms | 1355.34 | 1.19 | Random-row compaction avoids the expensive mixed control path. |
| Selected logprobs, concurrency 128: full log-softmax calls | 34 | 4 | The compact path handles 30 sufficiently large batches; small batches retain the existing path. |

Counts are observed within their own capture windows and include prefill.
They explain the implementation; user-visible speedups are determined from
separate measurements with profiling disabled. Summed kernel durations are
not wall-clock latency.

DFlash2 adds three capture windows (masks 0/56/63), again on all four
ranks: **12 additional complete timelines**, for 36 exported timelines
overall. The DFlash2 trace index is
`local/npu_sampling/accepted_dflash/profile_dflash/trace_index.json`.
On device 4, baseline versus all options:

- The native ScatterElementsV2 kernel count falls from 98 to 14. Sparse q
  construction removes per-draft-step full-vocabulary scatters; residual
  support correction still scatters once per verifier step.
- The optimized window contains 13 compact BF16 row-LSE calls and 13 FP32
  fused recovery draws. BF16 fused draws cover bonus/prefill selection.
- Native LogSumExp ReduceMax totals fall from 4956.6 to 182.5 us; the new
  compact target LSE contributes 5513.4 us itself. Remaining native LSE
  operations normalize the small candidate distribution. These totals
  describe changed work, rather than a standalone per-request speedup.
- Uniform kernel counts are both 58, but their total duration drops from
  17904.3 to 256.1 us as vocabulary-sized noise is replaced by compact
  uniform controls where supported. Acceptance and draft-selector draws
  still remain; this is not a claim that all RNG launches disappear.

The different windows execute different stochastic verification histories.
Raw counts/durations are retained, and no whole-window count is normalized
into a wall-clock speedup. Profiling is disabled for the serving timing
tables. No browser/Perfetto UI inspection was performed for either set.

## Reproduction and artifacts

Build a dedicated instrumented service and native `npu_sampling_benchmark`
with `BUILD_NPU_SAMPLING_ABLATION=ON`. This build is for tests only; production
builds leave that CMake setting off and have no runtime ablation controls. AOT Python source changes must first regenerate the
affected kernels through `xllm/compiler/tilelang_launcher.py compile-kernels`;
a direct CMake rebuild alone does not regenerate them.

For example, after configuring the repository's NPU build:

```bash
python3 xllm/compiler/tilelang_launcher.py compile-kernels \
  --target ascend --device a2 \
  --output-root build/cmake.linux-aarch64-cpython-311/xllm/compiler/tilelang \
  --kernels seeded_sample sampling_logits --jobs 6
cmake -S . -B build/cmake.linux-aarch64-cpython-311 \
  -DBUILD_NPU_SAMPLING_ABLATION=ON
cmake --build build/cmake.linux-aarch64-cpython-311 \
  --target xllm npu_sampling_benchmark sampler_test -j16

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
build/lib.linux-aarch64-cpython-311/xllm/npu_sampling_benchmark \
  --benchmark_filter='sparse_verification/B:32/S:7/.*logprobs:0/' \
  --benchmark_repetitions=5 --benchmark_enable_random_interleaving=true \
  --benchmark_out=local/npu_sampling/verify_repro.json \
  --benchmark_out_format=json
```

Build directory names above describe this Python 3.11/aarch64 environment;
adapt them to the configured checkout. The benchmark's full Cartesian shape
grid is much larger than the selected acceptance subsets, so use a filter.

The serving tool and controller are checked in under `tools/`:

```bash
python3 tools/benchmark_npu_sampling.py ordinary \
  --target-model /data/models/Qwen3.8-27B --devices 4,5,6,7 \
  --artifacts local/npu_sampling/ordinary_repro

python3 tools/benchmark_npu_sampling.py dflash \
  --target-model /data/models/Qwen3.8-27B \
  --draft-model /home/lhp/models/z-lab/Qwen3.8-27B-DFlash2 \
  --devices 4,5,6,7 --profile \
  --artifacts local/npu_sampling/dflash_repro

python3 tools/benchmark_npu_sampling.py dflash \
  --target-model /data/models/Qwen3.8-27B \
  --draft-model /home/lhp/models/z-lab/Qwen3.8-27B-DFlash2 \
  --devices 4,5,6,7 --batch-sizes 1 --modes random \
  --tokens 512 --repeats 7 --masks 0 16 56 63 \
  --artifacts local/npu_sampling/dflash_long_repro
```

Run inside the matching CANN/PyTorch build environment. The controller must own
the service process tree and use an instrumented service plus matching
benchmark binary. The controller resolves only the test-specific mask symbol;
it refuses a production executable that lacks that symbol. Each run saves the actual command, executable
SHA, controller/source copies, PID readback, per-request usage, response hashes,
timing samples and speculative counters. `--task-pipeline` selects that mode;
`--profile-only` captures warmed windows without running the timing matrix.
The bit mask is only a test-build control. Reconfigure with
`-DBUILD_NPU_SAMPLING_ABLATION=OFF` and rebuild before deployment; normal
serving enables supported paths automatically.

Native raw files are under `local/npu_sampling/`:

- `acceptance_native_joint.json`: final normalization/verification factorial retake.
- `final_native_small_and_joint.json`: final small-batch filter gate and joint controls.
- `final_native_ordinary.json`, `final_native_filter49k.json`,
  `final_native_logprobs.json`, `final_native_sparse.json`: wider shape/mode coverage.
- `acceptance_manifest.json`: source/native identities and experiment notes.

Earlier restart-per-variant results, incomplete-model startup attempts, and the
brief CPU-build-overlap ordinary pilot repeat are exploratory. They are not
used as the final serving acceptance data.

The final `acceptance_manifest.json` preserves per-run executable/source SHA-256,
worker PIDs, complete-request counts, test logs and trace-index identities.
Native-stage manifests remain separate: older wide-shape measurements are not
relabeled as the final serving executable. The matching native benchmark hash
for the final joint retake is `5f358b7e23fea423cc887c975a622441913d34686d453ae28769279f30d36a26`.
Final serving executable SHA-256: `22d7d6eece90aeef313a282caa5070ee4f5e37fae40c02b47b13ca464504014e`.
