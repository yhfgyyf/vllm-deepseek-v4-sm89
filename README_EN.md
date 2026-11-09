# DeepSeek-V4.1-Flash, DeepSeek-V4-Flash / Vision, and GLM-5.3-Flash on SM89 / SM120 — vLLM fork

<!-- markdownlint-disable MD060 -->

> 中文版见 [`README.md`](README.md)。
>
> This repository is based on
> [vllm-project/vllm](https://github.com/vllm-project/vllm). It runs
> DeepSeek-V4.1-Flash, DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, and
> GLM-5.3-Flash on SM89/Ada and SM120/RTX Blackwell.

The current source incorporates official main
`3213796cd32df625a0669e9423acb397b627b790`, checked on 2026-11-09, with Torch
2.14 / FlashInfer 0.7.3 and the SM89/SM120, CED, PP and DSpark adaptations.
The new `vision12` wheels are in the
[release](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/tag/v0.33.0-vision12-sm89-sm120-cu130);
installation commands are below.

The upgrade passed V4.1 TP4/PP2 full-model, image/OCR, Chinese, tool-call and
near-1M-context checks on 8x L20. Mean decode rates for 8K/32K/128K inputs and
512 outputs from the final wheel improved by 7.8%/13.0%/8.9% over the historical
stack; prefill was roughly unchanged. See the [validation record](docs/models/deepseek_v41_sm89_sm120.md#validation-scope).
The historical support matrix and `vision11` artifacts below still describe
the older vLLM 0.28 / Torch 2.13 / FlashInfer 0.6.18 stack.

## Historical release support matrix

| GPU architecture | Validated GPU | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash | DeepSeek-V4-Flash-Vision-Exp | GLM-5.3-Flash |
|---|---|---:|---:|---:|---:|
| SM89 / Ada | 8× RTX 4090 48 GB | Yes | Yes | Yes | Yes |
| SM120 / RTX Blackwell | 4× RTX PRO 6000 96 GB | Yes | Yes | Yes | Yes |

---

## Changelog

### 2026-11-09

- Merge official main, adapting model registration, the GPU runner, speculative
  decoding, cache management and Engram interfaces.
- Retain default TP4/PP2 with DSpark TP4 and at most four sequences, plus the
  optional TP2/PP4 target with draft TP4.
- Publish vLLM `0.33.0 / vision12` and paired FlashInfer `0.7.3 / vision3`
  wheels with Torch 2.14. Docker images retain the older stack.
- Fix native MoE loading and compressor ring-cache interfaces; validate full-model
  serving and near-1M context on 8x L20.
- Fix startup warmup and mixed-batch collective ordering for TP2/PP4 with draft TP4.

### 2026-09-16

- Published the public GHCR `vision11` image with paired FlashInfer `vision2`,
  adaptive verification, and CED image-input support; see Section 2.2.
- Updated the current prebuilt-wheel instructions to the `vision11` vLLM
  wheel with adaptive verification and CED image-input support; paired
  FlashInfer `vision2` is unchanged.
- Added an 8× RTX 4090 48 GB DeepSeek-V4.1-Flash launch configuration with
  an `auto` context limit, at most 4 sequences, memory utilization `0.98`,
  and a batch-token budget of `4096` (alternatively `2048`).

### 2026-09-12

- Added SM89 / SM120 adaptive verification support for DeepSeek-V4 / V4.1 in
  the current `main` source, including device-ragged metadata, padding, and
  FULL graph replay. SM120 operator and V4.1 full-model validation passed.
  At that time, the release's vLLM wheel was replaced by `vision10` with
  this adaptation. DSpark examples enable adaptive verification with explicit
  FULL graphs by default. `vision9` and earlier vLLM wheels do not contain it.
  Paired FlashInfer `vision2` is unchanged.
- Added native DeepSeek-V4.1-Flash model, Engram, DSpark, and experimental CED
  prefill support, published as the SM89+SM120 `vision9` wheels.
- Validated the full model server on 4× RTX PRO 6000 (SM120).
- For the tested long-text prefill workloads, the prefill proxy (input tokens /
  TTFT) roughly doubled relative to CED off; the gain remains workload-dependent.

### 2026-09-07

- Updated to the vLLM `0.28.1rc1` development series and published SM89+SM120 `vision8`
  wheels.
- Fixed the DeepSeek-V4 C128 sparse-attention crash caused by non-contiguous
  indices reported in [Issue #98](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/98)
  (PR #96).
- Fixed persistent Top-K candidate-buffer overflow and incorrect index
  selection (PR #97), and E8M0 scale compatibility in CUDA Triton block-FP8.

### 2026-09-02

- Fixed [Issue #90](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/90)
  and replaced the affected GitHub Release wheel with the `dev293` / `vision7`
  build.

### 2026-09-01

- Added DeepSeek-V4-Flash-Vision-Exp support for 4× RTX PRO 6000 (SM120) and
  8× RTX 4090 48 GB (SM89).
- Added SM89 deployment guidance and an 8-GPU DSpark launch command for
  DeepSeek-V4-Flash-Vision-Exp.
- Added an 8× RTX 4090 48 GB GLM-5.3-Flash launch command and changed the SM89
  DeepSeek-V4-Flash example to `--max-model-len auto`.

### 2026-08-31

- A community user successfully ran GLM-5.3-Flash on 8× RTX 4090 48 GB
  (SM89); see the [Issue #74 validation record](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993).

### 2026-08-30

- Updated the main branch to the vLLM `v0.28.1rc0-110` baseline while retaining
  the repository's validated DeepSeek-V4-Flash support on SM89.
- Added DeepSeek-V4-Flash and GLM-5.3-Flash support on RTX PRO 6000 (SM120).
- Published one SM89+SM120 vLLM wheel and the matching FlashInfer `0.6.18`
  wheel.
- Aligned the Python package version, Git source, and release artifacts.

Earlier SM89 builds and environments remain available in
[historical Releases](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases).

---

## 1. Current vision12 validated environment

| Item | Version / configuration |
|---|---|
| Operating system | Linux x86_64 |
| Python | 3.12 |
| CUDA toolkit | 13.2 build; CUDA 13.0 for local L20 JIT |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| Transformers | 5.22.2 |
| FlashInfer | `0.7.3+glm53.dsv41.vision3.sm89sm120.cu130.pt214` |
| vLLM | `0.33.0+glm53.dsv41.vision12.sm89sm120.cu130` |
| SM89 | 8x L20: full-model V4.1 TP4/PP2 and TP2/PP4 with draft TP4 |
| SM120 | RTX PRO 6000 Blackwell: operator and wheel validation |

The FlashInfer wheel is a Python/JIT source package. The first unseen model
shape is compiled once and then reused from the JIT cache.

---

## 2. Quick install (prebuilt wheels / container)

Install `vision12` vLLM and paired FlashInfer `vision3` in a new Python 3.12
environment. Both wheels come from the new
`v0.33.0-vision12-sm89-sm120-cu130` release:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/vllm-0.33.0%2Bglm53.dsv41.vision12.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=67228efddc16f0769e3578137a734e14cf2869b36610f37946419503d6d8c65e' \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/flashinfer_python-0.7.3%2Bglm53.dsv41.vision3.sm89sm120.cu130.pt214-py3-none-any.whl#sha256=b85753d53a0094796e865bffed11b2f55226d1991f776a3b9a76a2f0f77abecf' \
  'transformers==5.22.2' 'triton==3.8.0'
```

`uv` verifies both wheel hashes. The vLLM runtime code corresponds to main
commit `8e91ca5082c7c21dbdf1f03ae576fdca6c572072`, with SM89 / SM120 native
extensions and a release-profile Rust frontend. See the
[build and validation guide](docs/models/deepseek_v41_sm89_sm120.md) for details.

Optionally use `--index-url` for a PyPI mirror; the wheel still downloads
from the release URL above.

The older ACR `vision7` Docker image includes neither DeepSeek-V4.1-Flash nor
this adaptive adaptation. It cannot directly use the adaptive-enabled DSpark
commands below. Use the `vision12` wheels or current source;
the older ACR `vision7` configuration must keep adaptive disabled.

### 2.1 Install current main source (alternative to the wheel)

Follow the [current source build guide](docs/models/deepseek_v41_sm89_sm120.md)
to create a fresh environment and install the matching native FlashInfer 0.7.3
extension. Do not pin Torch 2.13 / Triton 3.7.1 or copy historical `vision11`
binaries into the upgraded source.

### 2.2 Historical vision11 Docker image (GHCR, no login required)

This image retains vLLM 0.28 / Torch 2.13 and does not include the framework upgrade.

[GHCR image](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pkgs/container/vllm-deepseek-v4-sm89)

Supported models: DeepSeek-V4.1-Flash, DeepSeek-V4-Flash,
DeepSeek-V4-Flash-Vision-Exp, and GLM-5.3-Flash.

Pull the image (no login required):

```bash
docker pull ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

Launch template: replace the local model paths and append the corresponding
model's launch arguments from the sections below. Pass environment variables
with `docker run -e` and use `--host 0.0.0.0` for the service address.

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130 \
  /models/model-directory \
  --host 0.0.0.0 --port 8000
```

---

## 3. DeepSeek-V4.1-Flash launch command

The following text-serving configuration was validated on 4× RTX PRO 6000
96 GB. It uses TP=4, expert parallelism, FP8 KV, Engram CPU offload, DSpark 5 (adaptive),
and experimental CED prefill. CPU offload also requires sufficient host memory.
V4.1 uses Model Runner V2; switching to the legacy V1 runner is unsupported:

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0
# Only historical vision11 artifacts need VLLM_USE_V2_MODEL_RUNNER=1.
unset TRITON_PTXAS_BLACKWELL_PATH

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len auto \
  --max-num-seqs 20 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --engram-config '{"cpu_offload":true}' \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --hf-overrides '{"ced_prefill":true}' \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'
```

Use `vllm serve` from the new environment above and replace the model path
with your checkpoint. Current source uses Triton 3.8's bundled compiler;
the Triton 3.7.1 `ptxas-blackwell` override belongs to the historical environment.
Initial JIT compilation and graph capture may take time. Measure throughput
only after the service is ready and warmed up.

CED is currently an experimental approximate text-prefill path. For the tested
long-text prefill workloads, the prefill proxy (input tokens / TTFT) roughly
doubled relative to CED off, but the gain depends on the prompt, length, and
runtime configuration. It is not guaranteed to be output-equivalent to the
non-CED path; validate quality for the target workload.

Current source retains CED single-image and multi-image input, DSpark, and
prefix caching, and adopts the shared upstream left/right visibility metadata
interface. V4.1 text and image tokens both use causal SWA; bidirectional image
visibility applies to the legacy V4 Vision model.
CED normally replays the last 128 query tokens, extending backward when the
boundary cuts through an image span, with tail capacity reserved for
`128 + vision_max_n_token`. Set a limit such as
`--limit-mm-per-prompt '{"image":2}'` as needed. This migration has operator
and component checks, plus one full-model two-image shape/OCR check. This is not
a broad image-quality evaluation. Non-image modalities,
prompt embeddings, and prompt logprobs are rejected before engine execution.

The image path has CPU regressions, small-weight SM120 GPU numerical tests,
and real-cache-manager image-hit and boundary-backoff tests. On 2026-09-16,
8× RTX 4090 48 GB completed single-concurrency 8K / 32K / 128K image and
tool-calling requests with CED and DSpark retained. Assess output quality
for the actual task; request completion does not mean every image answer
was correct.

### 3.1 DSpark enables adaptive verification by default

All DSpark launch examples below explicitly enable adaptive with FULL graphs.
This is the README example default, not a change to the generic
`SpeculativeConfig` default. Use the `vision12` wheels or current
`main` source and a DSpark checkpoint with a confidence head. Retain the model
path, TP/EP, FP8 KV, Engram, and CUDA toolchain settings.

| Parameter | Previous fixed DSpark 5 | Current adaptive default |
|---|---|---|
| `enable_adaptive_verification` | `false` | `true` |
| `--compilation-config` | Not explicitly set | `'{"cudagraph_mode":"FULL"}'` |
| `num_speculative_tokens` / draft / rejection | `5` / `probabilistic` / `block` | Unchanged |
| `--hf-overrides '{"ced_prefill":true}'` | Enables CED | May be retained; CED is not required for adaptive |

The V4.1 command ending is:

```bash
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'
```

Do not add `--enforce-eager` or switch graph mode to `PIECEWISE`. CED and
adaptive are independent: retain `ced_prefill=true` to combine them, or
remove that override for normal prefill. Real CED prefill/mixed steps still
follow their EAGER route, while pure decode can use FULL; requesting FULL
does not imply that all CED prefill runs inside a graph.

To restore fixed-draft verification, set `enable_adaptive_verification=false`;
FULL may remain enabled. Each model's existing draft count and sampling
method stay unchanged when enabling adaptive.

SM120 validation covers V4 / V4.1 operators, mixed prefill, padding, zero
draft budgets, FULL replay, and V4.1 full-model serving. The SM89 V4.1 image
and tool-calling runs are described above and in Section 8. These runtime
results do not establish full-model accuracy equivalence.

### 3.2 DeepSeek-V4.1-Flash throughput reference

Measured on 2026-09-12 using `vllm bench serve` on 4× RTX PRO 6000 Blackwell
Server Edition 96 GB (SM120), TP4/EP, FP8 KV, Engram CPU offload,
CED + DSpark5 adaptive, and FULL configured:

| Input / output tokens | Max concurrency | Requests | Aggregate output throughput | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|
| 8,192 / 512 | 10 | 200 | **296.61 tokens/s** | 1.464 s | 30.64 ms |

The service was warmed up; random inputs, temperature=0, ignore-eos, all
200 requests succeeded, and prefix-cache hits were zero. Throughput is total
output tokens / full-run duration, including prefill and decode, not
per-request or decode-only speed. This single run used the Vision9 /
FlashInfer Vision2 environment with this adaptive patch applied and is a
throughput reference for that environment. There is no same-workload baseline,
so it does not establish an adaptive speedup percentage or quality equivalence.

**Single-concurrency (C1)** results from the same patched environment follow.
Each input length had one warmup followed by four measured requests, all
with 512 output tokens. All 12 measured requests succeeded with zero
prefix-cache hits.

| Input / output tokens | Mean TTFT (ms) | Prefill proxy (tokens/s) | Decode (tokens/s) | DSpark acceptance rate | Mean accepted draft tokens | Mean acceptance length (including +1) |
|---|---:|---:|---:|---:|---:|---:|
| 8,192 / 512 | 913.30 | 8,969.81 | 267.11 | 84.11% | 4.206 | 5.206 |
| 32,768 / 512 | 3,279.21 | 9,992.67 | 271.67 | 85.89% | 4.295 | 5.295 |
| 131,072 / 512 | 15,529.54 | 8,440.21 | 80.00 | 4.64% | 0.232 | 1.232 |

Metric definitions:

- Prefill proxy is input tokens / TTFT in seconds. TTFT includes queueing,
  scheduling, prefill, and first-token work; it is not pure GPU-kernel or
  engine prefill throughput.
- Decode is `1000 / TPOT (ms)`. TTFT is averaged across the four measured
  requests; both throughput values are calculated per request and then
  averaged, not obtained by inverting the mean latency.
- Acceptance rate = total accepted draft tokens / total proposed draft
  tokens; mean accepted draft tokens = total accepted draft tokens / total
  draft rounds; vLLM mean acceptance length adds 1 for the extra target token.
  These three metrics use pooled counters from the four measured requests,
  not averages of ratios.

These are native vLLM counters: proposed drafts are counted before adaptive
trimming, and accepted tokens before stop/output-limit truncation. They are
not the acceptance rate of actual post-trim verification slots or the final
accepted-token count returned to clients. Random-input results are not a
quality evaluation.

### 3.3 SM89: 8×48 GB GPUs (L20 / RTX 4090 48 GB)

The following configuration is for **eight 48 GB SM89 GPUs**, excluding
standard 24 GB RTX 4090 cards. The default target uses TP4×PP2 (20/20 layers),
with DSpark TP4 and `max-num-seqs=4`; target TP2×PP4 (10/10/10/10 layers) is
optional. It retains EP, FP8 KV, Engram CPU offload, CED image input,
DSpark 5 adaptive, and FULL CUDA Graph. Engram CPU offload and checkpoint
loading both require sufficient host memory, so do not size host memory from
the GPU weight footprint alone.

Use the `vision12` wheels or updated `main` source for these PP/DSpark configurations.
Both layouts passed 40 LongSWE requests with 1,024 outputs each at concurrency
four on 8x L20. Completion times were 323.89 / 321.31 seconds, with output
throughput of 126.46 / 127.48 tokens/s: similar results in this single-batch
comparison. Check capacity and throughput on your actual GPUs and workload.
Configure the SM89 toolchain and launch in that environment:

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9
# Only historical vision11 artifacts need VLLM_USE_V2_MODEL_RUNNER=1.
unset TRITON_PTXAS_BLACKWELL_PATH
unset VLLM_PP_LAYER_PARTITION

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.98 \
  --enable-prefix-caching \
  --engram-config '{"cpu_offload":true}' \
  --load-format safetensors \
  --safetensors-load-strategy prefetch \
  --safetensors-prefetch-num-threads 2 \
  --limit-mm-per-prompt '{"image":2}' \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --hf-overrides '{"ced_prefill":true}' \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'
```

- For optional **target TP2×PP4 with DSpark TP4**, set
  `export VLLM_PP_LAYER_PARTITION=10,10,10,10`, change the parallel options to
  `--tensor-parallel-size 2 --pipeline-parallel-size 4`, and add
  `"draft_tensor_parallel_size":4` to the existing `--speculative-config`
  JSON. Keep the other parameters. The draft uses an independent TP4/EP4 group
  on the last four GPUs. Unset the layer-partition variable when returning to
  the default TP4×PP2 layout.
- To reduce peak prefill memory, replace `--max-num-batched-tokens 4096` with
  `--max-num-batched-tokens 2048` and leave the other parameters unchanged.
  This also affects workspace use, graph-capture memory, and the number of
  chunks for long inputs; do not assume that memory or throughput changes in
  the same proportion.
- Do not set an explicit KV-cache byte budget. With
  `--gpu-memory-utilization 0.98`, vLLM sizes the KV cache automatically after
  subtracting resident allocations, transient-peak headroom, and its CUDA
  Graph estimate. The remaining 2% is neither graph-only space nor a hard
  runtime memory limit.
- `--max-model-len auto` may resolve to a lower value when capacity is
  insufficient. If a full 1M-token context is required, confirm that the
  startup log resolves it to `1048576`; each request's input, including image
  tokens, plus output must remain within that limit. `--max-num-seqs 4` is a
  scheduling limit, not a guarantee that four simultaneous 1M-token requests
  will fit.
- Check OOMs, KV preemption/recomputation, latency, and output quality using
  actual long requests and the target mixed concurrency during deployment.

## 4. DeepSeek-V4-Flash launch commands

These DSpark commands enable adaptive verification with explicit FULL CUDA
Graphs by default.

### 4.1 SM89: 4× RTX 4090 48 GB

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend auto \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.986 \
  --cudagraph-capture-sizes 1 2 4 7 8 \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

The historical fixed-DSpark configuration was validated with 8K, 32K, and
128K inputs, 512 output tokens, and four concurrent 8K requests.

### 4.2 SM120: 4× RTX PRO 6000 96 GB

```bash
vllm serve /path/to/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend auto \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

---

## 5. DeepSeek-V4-Flash-Vision-Exp launch commands

These DSpark examples enable adaptive verification with explicit FULL CUDA
Graphs by default.

### 5.1 SM89: 8× RTX 4090 48 GB

```bash
vllm serve /path/to/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend auto \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.98 \
  --enable-prefix-caching \
  --interleave-mm-strings \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

4× RTX 4090 48 GB is not recommended for Vision-Exp with DSpark because the
draft model and KV cache do not have enough memory headroom. 8× RTX 4090
48 GB can enable DSpark with the command above; set
`--max-num-batched-tokens` to `4096`.

Historical SM89 runtime regression used 4× RTX 4090 48 GB without DSpark.

### 5.2 SM120: 4× RTX PRO 6000 96 GB

```bash
vllm serve /path/to/DeepSeek-V4-Flash-Vision-Exp \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend auto \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --interleave-mm-strings \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

## 6. GLM-5.3-Flash launch commands

### 6.1 SM89: 8× RTX 4090 48 GB

The following configuration is based on the community deployment validation
with the official FP8 weights in
[Issue #74](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993):

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_P2P_DISABLE=1
export VLLM_ENGINE_READY_TIMEOUT=3600

vllm serve /path/to/GLM-5.3-Flash \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 8 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --block-size 2304 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.96 \
  --enable-prefix-caching \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

### 6.2 SM120: 4× RTX PRO 6000 96 GB

```bash
vllm serve /path/to/GLM-5.3-Flash \
  --served-model-name zai-org/GLM-5.3-Flash \
  --tensor-parallel-size 4 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --block-size 2304 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --enable-prefix-caching \
  --port 8000
```

### 6.3 SM120 key parameters

| Option | Recommended value | Purpose |
|---|---:|---|
| `--tensor-parallel-size` | 4 | Four 96 GB RTX PRO 6000 GPUs |
| `--kv-cache-dtype` | `fp8` | Reduces long-context KV-cache memory |
| `--block-size` | `2304` | Model-wide block size for the GLM hybrid cache |
| `--max-model-len` | `auto` | Fits context capacity to the active memory profile |
| `--max-num-seqs` | 4 | Validated concurrent-sequence setting |
| `--max-num-batched-tokens` | 8192 | Chunked-prefill token budget |
| `--gpu-memory-utilization` | 0.97 | Reserves memory for the model, CUDA Graph, and KV cache |
| `--speculative-config` | MTP 5 | Enables five-token MTP speculative decoding |
| `--enable-prefix-caching` | enabled | Reuses repeated or shared prefixes |
| `glm45` / `glm47` | reasoning / tool parser | Parses reasoning output and tool calls |

---

## 7. GLM-5.3-Flash SM120 throughput baseline

These results are retained from the project's first complete SM120 benchmark.
The setup used 4× RTX PRO 6000, TP=4, FP8 KV, MTP=5, CUDA Graph,
`block-size=2304`, an 8192-token chunked-prefill budget, and prefix caching
disabled. Each input length was run five times with 512 output tokens; all 10
requests completed.

| Input → output | Mean / median prefill TPS | Mean / median decode TPS | Mean TTFT |
|---|---:|---:|---:|
| 8,192 → 512 | **9,919.34 / 9,904.75** | **158.47 / 175.25** | 825.88 ms |
| 32,768 → 512 | **9,841.51 / 9,843.08** | **199.00 / 208.74** | 3,329.62 ms |

Metric definitions:

- `Prefill TPS = input tokens / TTFT`
- `Decode TPS = 511 / (E2E - TTFT)`
- TTFT is an end-to-end approximation of time to first token. Decode throughput
  varies with MTP acceptance.

This table preserves the first SM120 baseline for comparison. It does not
guarantee identical results for every prompt, driver, or memory configuration.

### Long-context throughput

The first SM120 release also recorded 256K / 784K long-context runs with 512
output tokens and prefix-cache reuse:

| Input / output | 8192-token baseline | 4096-token steady state | Prefill change |
|---|---:|---:|---:|
| 256K / 512 | 27.865 s; 9,407.50 tok/s | 31.649 s; 8,282.96 tok/s | -11.95% |
| 784K / 512 | 101.473 s; 7,911.64 tok/s | 112.685 s; 7,124.45 tok/s | -9.95% |

Repeating the same prompt produced:

| Input | Prefix-cache hit rate | Repeat-request TTFT |
|---|---:|---:|
| 256K | 98.4375% | 1.308 s |
| 784K | 99.5855% | 3.117 s |

---

## 8. Correctness validation

- DeepSeek-V4.1-Flash, DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, and
  GLM-5.3-Flash passed server startup on SM120.
- DeepSeek-V4.1-Flash passed text serving, tool calling, DSpark, and experimental
  CED prefill validation on SM120. On 8× RTX 4090 48 GB (SM89), it completed
  single-concurrency 8K / 32K / 128K image and tool-calling requests with CED
  and DSpark retained. Assess image-answer quality separately for the task.
- DeepSeek-V4-Flash and GLM-5.3-Flash passed 8K/32K to 512-token tests on
  SM120.
- DeepSeek-V4-Flash passed 8K/32K/128K, four-concurrency, tool-calling, and UTF-8
  output tests on SM89.
- DeepSeek-V4-Flash-Vision-Exp passed server startup, single-image,
  multi-image, video, tool-calling, 8K/32K input, and UTF-8 output tests on
  4× RTX 4090 48 GB (SM89) without DSpark.
- A community user successfully served and ran GLM-5.3-Flash on 8× RTX 4090
  48 GB (SM89); see [Issue #74](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993).
- Both models passed multilingual output plus streaming and non-streaming tool
  calling checks.

---

## 9. License / provenance

The code is based on [vllm-project/vllm](https://github.com/vllm-project/vllm)
and remains under Apache-2.0. The FlashInfer wheel is based on
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
