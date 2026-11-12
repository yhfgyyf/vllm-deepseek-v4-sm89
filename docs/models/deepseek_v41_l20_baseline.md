# DeepSeek V4.1 L20: current TP8 and historical Q

The selected service configuration as of 2026-11-12 is **TP8/PP1 + draft TP8**,
retaining Q's K1/K2/K3 and decode PCIe IPC. Both README commands retain the
service parameters, except for the user's requested memory-utilization change
from 0.98 to 0.96: 8192 batched tokens, 4 sequences, K5 adaptive, 1,048,576 context
limit, `FULL_AND_PIECEWISE` with capture48, `--gpu-memory-utilization 0.96`, and
`--no-enable-extensible-kv-cache`, without a fixed KV byte limit.

The recorded 0.98 launch allocated **5.107793 GiB KV per worker**; the latest
0.96 startup receipt records **4.160247 GiB** on each of the eight workers.
The change followed an OOM on a 17,630-token input in the Engram TP all-gather
prefill path. This publication reuses that startup receipt and does not restart
the service or qualify long-running stability. Eight-rank IPC
numerical, repeated-call and changed-input Graph checks passed; all eight
workers loaded the accepted K1/K2/K3 modules. The bounded novel C4 observation
at 0.98 was 288.979 aggregate output tokens/s for the complete batch and 402.005
tokens/s during a continuous four-request decode interval. It differs from
historical Q in cache hits, accepted drafts, output lengths and draft TP;
these are workload observations, not an isolated speedup or a global optimum.
One of four stories did not meet its requested 10,000-character length.

Q below remains the frozen, controlled historical experiment. Its performance
numbers and fixed 2 GiB allocation must not be relabeled as TP8 measurements.
No new wheel was published for either configuration. The new
[SM120 opt-in adapters](deepseek_v41_sm120_b12x.md) have separate qualification
limits and do not change L20 execution.

## Historical Q

The accepted 2026-11-11 baseline is **TP4/PP2 + K1 Attention + K2 mHC +
K3 Linear + decode FlashInfer PCIe IPC all-reduce** on eight L20 GPUs.
K1/K2/K3 tune existing SM89 kernel launches; they introduce no production
dependency. PCIe IPC uses the existing opt-in backend.

Build vLLM from this source and FlashInfer from the updated
[native V4.1 recipe](../../tools/recipes/patches/flashinfer-0.7.3-deepseek-v41-native.patch),
following the [build guide](deepseek_v41_sm89_sm120.md#build-the-current-source).
The published vLLM `vision12` and FlashInfer `vision3` wheels predate the
K1/K2/K3 changes. Validation used those wheels with the exact candidate Python
modules overlaid; no new complete wheel was built for Q.

## Configuration

The current TP8 command is in the [English README](../../README_EN.md#33-sm89-848-gb-gpus-l20--rtx-4090-48-gb)
and [Chinese README](../../README.md#33-sm89848gb-gpul20--rtx-4090-48gb).
The table below preserves Q's historical configuration.

| Setting | Accepted value |
| --- | --- |
| Target parallelism | TP4/PP2, GPU0-3 and GPU4-7 |
| DSpark | TP4 on GPU4-7, K=5, probabilistic draft, block rejection, adaptive verification |
| Scheduler | 8192 batched tokens, 4 sequences, prefill schedule interval 1 |
| Context and KV | 1,048,576 token limit; FP8 KV, block size 128, 2 GiB KV per worker |
| CUDA Graph | `FULL_AND_PIECEWISE`, `max_cudagraph_capture_size=48` |
| Communication | `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` |
| Other settings | EP, CED, Engram CPU offload, prefix caching, image limit 2 |

The library default for PCIe IPC remains off. On this configuration it handles
contiguous BF16 `[rows,5120]` messages up to 48 rows. The observed prefill
all-reduces and larger messages use the existing backend. Topology tuning and
workspace preparation run before graph capture; the two TP4 groups selected
the existing `pcieswitch-pairs` kernel synchronization path.

The optional TP2/PP4 layout is outside Q's PCIe IPC validation. Keep the flag
unset when using that layout. Changing scheduling, graph sizes, KV allocation,
or topology requires validation for the resulting configuration.

## Source identity

These are the exact candidate modules used for Q, based on vLLM commit
`04b886bbd1bf18ca94929b563ebad880466f0a7d`. K1 is distributed through the
FlashInfer recipe patch, applied to FlashInfer commit
`775207deb34055667df89ed618fb4b06fc84de74`.

| Component | SHA256 |
| --- | --- |
| K1: patched `flashinfer/mla/deepseek_v41.py` | `1dcf16378fe0e7d048585f5a629b44763001528f313b79b0a04f18fda688c0dd` |
| K2: `vllm/model_executor/kernels/mhc/deepseek_v41.py` | `6eabde538f41c5327ab87f57f685f17da8284efe1e424b5156e0de274ed9db9a` |
| K3: `vllm/model_executor/kernels/linear/mxfp8/ada.py` | `21763a4eb942ee81350fafbafb425b40c66309c4c86fd138167eb27ee016d416` |
| Native V4.1 FlashInfer recipe patch | `8b0fa40b7d72050cf2d9f7600d6a52a249f11eb3a2ffaa9c807591446d36b43b` |

K1 adjusts split/unified-tile selection and reserves a conservative workspace
bound. K2 uses eight projection splits for SM89 decode at hidden size 5120,
preserving FP32 weights and the existing Sinkhorn/collapse behavior. K3 tunes
tiles and split-K for six measured MXFP8 Linear shapes. Other shapes and
architectures retain their existing dispatch.

## Validation

The unchanged candidate hashes support reuse of the completed GPU validation:

| Check | Result |
| --- | --- |
| K1 numerical, dispatch and changed-input Graph replay | 69 exact replay cases; 108 pytest passed, 2 skipped |
| K1 CPU workspace/SM120 dispatch checks | 19,392 workspace bounds; 21,504 unchanged SM120 selections |
| K2 numerical and changed-input Graph replay | 32 exact replay cases; 20 pytest passed |
| K3 numerical and changed-input Graph replay | 147 exact candidate cases; 30 pytest passed |
| PCIe IPC, both TP4 groups | All 8 ranks passed numerical, repeated-call, changed-input Graph and workspace-lifecycle checks |
| PCIe IPC fallback | 64, 128 and 8192 rows used the existing backend |
| Full service | Chinese, images, tool round trip, 8 code cases, 32 concurrent code cases and 127,050-token retrieval passed |
| Full C4 workload | Three batches, each 40/40 requests and 40,960 output tokens; no prefix hits or preemption |

K1 passed its FP32 reference tolerance, with at most 0.015625 BF16 difference
from the old launch. K2's coefficient error against FP64 was at most
1.08179e-6; its carried collapse remained bitwise equal to the old path.
K3 matched the old BF16 output in all 147 cases. IPC and NCCL independently
passed a BF16 summation error bound; different reduction orders need not be
bitwise equal. These checks do not establish full-model output equivalence.

GPU regression commands, in the corresponding built environments:

```bash
# vLLM checkout, SM89
.venv/bin/python -m pytest -q tests/kernels/quantization/test_ada_mxfp8_linear.py
.venv/bin/python -m pytest -q tests/kernels/mhc/test_ada_mhc.py

# FlashInfer checkout after applying the recipe patch, SM89
/path/to/vllm/.venv/bin/python -m pytest -q tests/attention/test_deepseek_v41_mixed_sparse.py
```

The vLLM tests are selected by the existing L4 quantization CI step. The
FlashInfer tests live in the recipe patch and require a separately patched
FlashInfer checkout; the vLLM CI step does not run them.

## Service observations

The comparison uses the same frozen inputs and dependency stack. A is the
earlier TP4/PP2 baseline, P adds K1/K2/K3, and Q adds PCIe IPC to P. A was
measured earlier, not in an interleaved A/B/A sequence.

C1 requests use 8192/32768/131072 input tokens and 512 output tokens. Decode
counts actual tokens between first and last nonempty SSE events. Pairs are
selected only by acceptance-rate difference (at most 1 percentage point),
without reuse within a comparison or selection by speed.

| Input tokens | A/Q matched pairs | A to Q decode (tokens/s) | Mean paired speed change |
| --- | --- | --- | --- |
| 8192 | 8/9 | 113.00 to 127.38 | +12.74% |
| 32768 | 9/9 | 113.14 to 127.80 | +12.96% |
| 131072 | 7/9 | 113.40 to 127.85 | +12.77% |

Server prefill speed changed by +0.33%, +0.17% and -0.16%, respectively,
using all nine observations per input length. Unmatched decode samples were
retained. Relative to P, Q's matched C1 decode gains were 5.45%, 6.61% and 5.21%.

C4 uses the same 40 LongSWE prompts, totaling 1,335,265 input and 40,960 output
tokens per batch. Q's three results were 149.300, 150.297 and 147.272 aggregate
output tokens/s: **148.956 ± 1.541** (mean ± sample standard deviation), versus
A's **125.040 ± 2.332**, a 19.13% increase. Wall time includes prefill, queuing,
decode and drain, and excludes startup and warmup. Per-request acceptance and
interference are not fully matched, so this is not a pure decode speedup or a
LongSWE task-accuracy score.

An eight-worker Nsight trace confirmed NCCL all-reduces in observed prefill
and IPC all-reduces in observed decode. Trace event-loss warnings limit causal
attribution; all reported throughput measurements ran without profiling.

## Limits

SM120 validation used one RTX PRO 6000: 19 Attention, 3 mHC and 3 Dense cases,
plus 71 operator Graph checks. K1/K2 retained the original SM120 path and K3
did not select its SM89 implementation. The IPC module compiled and correctly
rejected one rank. **SM120 multi-GPU IPC and full-model serving remain
unverified.** Independent B12X component measurements are not part of Q.

The full-service quality checks are bounded regressions, not a complete model
evaluation. Task services were stopped after validation and original wheel
files restored; accepting Q does not deploy a running service or publish a
new wheel.
