# DeepSeek-V4-Flash on SM89 (Ada / RTX 4090) — vLLM fork

<!-- markdownlint-disable MD060 -->

> 中文版见 [`README.md`](README.md)。
> This repository is based on [vllm-project/vllm](https://github.com/vllm-project/vllm). It selectively backports current upstream `main` DeepSeek-V4 / DSpark fixes from the v0.27.1 era while preserving the SM89/Ada-specific FlashInfer sparse MLA path.

It extends vLLM's **DeepSeek-V4-Flash** inference from SM90/SM100/SM120 to **SM89 (Ada Lovelace: RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada)**. End-to-end validated on **4× RTX 4090 (48 GB)**: environment setup → operator tests → server startup → inference → performance / tool-calling — all passing.

## Changelog

### 2026-08-26

- Replaced the FlashInfer Python wheel in the current CUDA 13.2 release from `0.6.17+sm89.1` to `0.6.17+sm89.2`. The new wheel adds single-cache and dual-cache DSV4 sparse-MLA prefill dispatch and bounds protection for `num_heads=8`.
- Removed the old `sm89.1` wheel from the release. The existing vLLM wheel's historical metadata still pins `sm89.1`, so the quick-install command uses uv `--overrides` to select `sm89.2`. The source dependency is updated so future vLLM wheels pin `sm89.2` directly.

### 2026-08-22

- Selectively backported vLLM v0.27.1-era DeepSeek-V4 fixes and optimizations: tokenizer/parser fixes, DSpark target/draft backend alignment, DFlash hybrid causal metadata, mHC broadcast, the specialized DSV4 top-k path, sparse-index metadata optimizations, and sparse MLA / SWA boundary fixes. The SM89 FlashInfer path keeps its wide-eager CUDA Graph guard.
- Included the earlier SM89 paged-MQA-logits int32 addressing fix (PR #51) and Triton per-shape kernel-cache growth fix (PR #61). Reverted upstream scratch/stride experiments remain excluded, as does confidence-scheduled adaptive verification.
- Updated the recommended DSpark config to `method=dspark`, `num_speculative_tokens=7`, and `draft_sample_method=probabilistic`. Single-concurrency decode on 4× RTX 4090 is **366.95 / 327.38 tok/s** for `8K / 32K -> 1K`.
- Updated the release stack to CUDA toolkit **13.2**, `torch 2.13.0+cu130`, `triton 3.7.1`, `flashinfer-python 0.6.17+sm89.1`, and `flashinfer-cubin 0.6.17`. Source-vs-wheel 8K/32K A/B runs, tool calling, and UTF-8 output checks all passed.

### 2026-08-02

- Added support for the latest `DeepSeek-V4-Flash-0731` model and completed short-context, long-context, and GSM8K accuracy validation on 4× RTX 4090.
- Fixed FlashInfer SM89 sparse MLA decode accuracy by reconstructing the distributed query/KV UE8M0 scales for each MMA accumulator, handling boundary encodings, and supporting page block size 256.
- Install the matching vLLM and FlashInfer wheels from the [latest Release](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/latest) in the same dependency-resolution invocation.

### 2026-07-10

- Switched SM89 sparse MLA prefill/decode to the **FlashInfer 0.6.14 sparse MLA JIT fork**. The release includes the matching FlashInfer wheel, and runtime validation rejects an official package without the SM89 patch.
- Fixed Lightning Indexer scheduler metadata selection to use actual DeepGEMM hardware support instead of package presence. An installed DeepGEMM package no longer makes SM89 call an unsupported metadata API.
- The wheel build script now honors an explicit `VLLM_VERSION_OVERRIDE` and ignores historical CUDA/SM release tags during automatic version resolution, preventing a pre-build `setuptools_scm` parse failure.
- This release does not update `confidence_head` and does not include per-request adaptive ℓ; DSpark remains fixed at `ℓ=6`.
- On 4× RTX 4090, TP=4, single concurrency, all five requests per case passed. For `8K / 32K / 128K -> 1K`, Prefill TPS is **3515.72 / 4881.18 / 3812.00** and Decode TPS is **286.82 / 344.63 / 313.57**.

### 2026-07-06

- Added notes for the **SM80/A800 test adaptation**. The SM80 path is for experiments and self-testing only, not production-grade support.
- Completed DeepSeek-V4-Flash DSpark speculative decoding smoke and throughput tests on a 4× A800 server with `method=dspark`, `num_speculative_tokens=6`, `draft_sample_method=greedy`, the FlashInfer sampler, sparse MLA warmup, and `max-num-batched-tokens=16384`.
- Only decode-side results are reported: **229.8 tok/s/req** for 8k input -> 1k output, single concurrency; **274.2 tok/s/req** for 32k input -> 1k output, single concurrency. The matching no-DSpark `mbt16k` baselines are 57.6 and 58.1 tok/s/req.

### 2026-07-01

- Completed **DeepSeek-V4-Flash-DSpark** model adaptation with `method=dspark` speculative decoding. The current release wheel target is **CUDA 13.0 toolkit + torch 2.11.0+cu130**, and `vllm serve`, tool calling, and vLLM bench have been validated on CUDA 13.x / 4× RTX 4090.
- DSpark single-concurrency `8K / 32K / 128K` input and `1K` output cases all passed `10/10`; converted decode throughput is **355 / 336 / 219 tok/s**. Compared with the non-DSpark source-model baseline decode throughput of **~82 tok/s**, this is about **4.3× / 4.1× / 2.7×** faster.
- Recommended DSpark serving config: `gpu-memory-utilization=0.96`, `max-model-len=262144`, `max-num-batched-tokens=2048`, `max-num-seqs=4`, `block-size=256`, `kv-cache-dtype=fp8_ds_mla`.

---

## SM80/A800 Test Adaptation

The SM80/A800 path can now be used for DeepSeek-V4-Flash + DSpark speculative decoding self-tests, but it remains a test-only adaptation and is not a production support commitment. The validated A800 setup used:

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'
```

with the FlashInfer sampler, sparse MLA warmup, and `max-num-batched-tokens=16384`.

| input -> output | concurrency | DSpark decode | no-DSpark decode | decode speedup |
|---|---:|---:|---:|---:|
| 8,192 -> 1,024 | 1 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 -> 1,024 | 1 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

This table reports decode-side results only; SM80 long-context prefill needs separate evaluation.

---

## 1. Background: why this fork

DeepSeek-V4-Flash combines DeepSeek Sparse Attention (DSA / Lightning Indexer), FP4-expert MoE, and mHC. This fork ports FlashInfer's SM120 sparse MLA JIT kernels to **SM89** and keeps the Triton/torch auxiliary fallbacks required on Ada.

| Subsystem | Upstream (SM90/100) | SM89 (this fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA / FlashInfer sparse | **FlashInfer 0.6.17 sparse MLA JIT** |
| Lightning Indexer (FP8 MQA logits) | DeepGEMM | **Hardware-gated DeepGEMM / fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **SM89-compatible path** |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE (FP4 experts) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16** (FP4→FP16 dequant) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**Hardware fact:** Ada has FP8 tensor cores but **no FP4 tensor cores and no hardware microscaling MMA**, so the FP4 MoE must run through Marlin dequantization (slower than native FP4 MMA).

### SM89 changes

- The `flashinfer-python==0.6.17+sm89.2` sparse MLA JIT path admits exact capability `8.9`; other 8.x GPUs remain rejected. This version also supports DSV4 prefill with `num_heads=8`.
- `vllm/v1/attention/backends/mla/indexer.py` uses `is_deep_gemm_supported()` for scheduler metadata, preventing SM89 from calling the DeepGEMM metadata API.
- `vllm/models/deepseek_v4/compressor.py` and `vllm/utils/import_utils.py` select existing Triton/torch fallbacks on SM89 and avoid SM90+ CuTe-DSL instructions.
- MXFP4 MoE continues to select Marlin on SM89 instead of Blackwell-only DeepGEMM FP4.

### Upstream vLLM code updates (2026-08-22)

| Area | Upstream PRs | Update in this fork |
|---|---|---|
| Correctness | #51727 / #51296 | Fixed duplicated DeepSeek tokenizer vocab accounting and the reasoning parser's default thinking behavior. |
| DSpark | #52288 + #52809 semantics | Only DeepSeek-V4 drafts inherit the target attention backend; an explicit draft backend still wins. |
| DFlash | #47914 | Propagated hybrid causal metadata per KV group in both eager and FULL graph paths, avoiding mixed SWA/full-attention capture mismatches. |
| Decode | #48137 / #48660 / #47463 | Removed the repeated mHC decode copy, added the DSV4 top-k softplus/sqrt kernel, and moved dtype handling into the kernel. |
| Sparse index | #49486 / #50298 / #52084 / #51967 / #48957 | Skipped unnecessary short-context top-k work, reused the output buffer, updated worker/constexpr settings, and skipped empty C128 launches. |
| Sparse MLA | #51538 (selected fixes) | Backported SM89-relevant q-head, SWA-width, negative-index-length, top-k-boundary, and workspace-lane fixes without importing SM120-only implementations. |
| CUDA Graph | #51430 / #52401 / #52492 (SM89-specialized) | Kept the narrow-eager structure and correctness follow-ups, while retaining the wide-eager guard for SM89 FlashInfer sparse MLA to prevent indexer output-ordering failures. |

See [`deepseek-v4-sm89-upstream-pr-analysis.md`](deepseek-v4-sm89-upstream-pr-analysis.md) for the full applicability and revert analysis. This fork does not include confidence-scheduled adaptive verification (#47808/#52436), and it does not reintroduce upstream-reverted #50004/#49236.

---

## 2. Validated environment

| Item | Version |
|---|---|
| GPU | 4× RTX 4090 (48 GB) · compute capability **8.9** |
| Driver / CUDA toolkit | 595.x / **CUDA 13.2** (nvcc 13.2) |
| Python | 3.12 |
| torch / Triton | **2.13.0+cu130 / 3.7.1** |
| FlashInfer | **0.6.17+sm89.2 sparse MLA fork** + cubin 0.6.17 |
| vLLM | `0.23.1rc1.dev904+sm89.cu132`, CPython 3.12, SM89/Ada |

---

## 3. Quick install (prebuilt wheel)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download v0.23.1rc1.dev904-g998fd644b-cu132-sm89 \
  --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.17+sm89.2-*.whl' \
  --pattern 'flashinfer_cubin-0.6.17-*.whl' \
  --pattern 'vllm-*.cu132-cp312-cp312-linux_x86_64.whl' \
  --dir /tmp/vllm-sm89-release

printf '%s\n' \
  'flashinfer-python @ file:///tmp/vllm-sm89-release/flashinfer_python-0.6.17+sm89.2-py3-none-any.whl' \
  > /tmp/vllm-sm89-release/uv-overrides.txt

UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install \
  /tmp/vllm-sm89-release/flashinfer_cubin-0.6.17-*.whl \
  /tmp/vllm-sm89-release/vllm-*.cu132-cp312-cp312-linux_x86_64.whl \
  --overrides /tmp/vllm-sm89-release/uv-overrides.txt \
  --torch-backend=cu130
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

> The vLLM wheel in this release predates the H8 hotfix and still declares `flashinfer-python==0.6.17+sm89.1` in its package metadata. `--overrides` replaces that dependency during resolution with a direct reference to the local wheel, so the installed runtime wheel is `sm89.2`. As a result, `uv pip check` reports this one known metadata mismatch until the next vLLM wheel rebuild.

**Validated environment:**

- **Python 3.12**, Linux x86_64
- **4× RTX 4090 (SM89/Ada, 48 GB)**, 595.x driver, CUDA toolkit 13.2
- **torch 2.13.0+cu130**, **Triton 3.7.1**
- **FlashInfer 0.6.17+sm89.2 fork**; official 0.6.17 does not contain the sparse MLA SM89 JIT patch required by this release
- `flashinfer-cubin==0.6.17`; set `FLASHINFER_DISABLE_VERSION_CHECK=1` because the Python wheel carries the `+sm89.2` local-version suffix
- Wheel built with `TORCH_CUDA_ARCH_LIST=8.9+PTX`, targeting Ada/SM89

---

## 4. Build from source (clone this repo)

### 4.1 Python env + torch 2.13/cu130

```bash
uv venv --python 3.12
source .venv/bin/activate
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install torch==2.13.0 --torch-backend=cu130
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

Before running sparse MLA on SM89, install the FlashInfer 0.6.17 SM89 wheel from the same release as shown in section 3. If Aliyun is slow, use the Tencent Cloud or USTC mirror; the Tsinghua mirror is not recommended here.

### 4.2 Rust toolchain (vLLM builds the Rust frontend)

```bash
export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain 1.95 --profile minimal
source "$HOME/.cargo/env"
# Configure a crates mirror in ~/.cargo/config.toml:
#   [source.crates-io]
#   replace-with = "rsproxy-sparse"
#   [source.rsproxy-sparse]
#   registry = "sparse+https://rsproxy.cn/index/"
```

### 4.3 Clone this repo

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
```

### 4.4 Build / package a CUDA 13.2 wheel (Ada 8.9 only)

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.2
export VLLM_VERSION_OVERRIDE=0.23.1rc1.dev904+sm89.cu132
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=8 NVCC_THREADS=2

./build_wheel.sh
uv pip install --force-reinstall --no-deps dist-sm89/vllm-*.cu132-*.whl
```

> Ada does not support DeepGEMM kernels, but the package does not need to be manually removed; vLLM disables its scheduler metadata path using the hardware capability gate.
> To build an SM80/A100/A800 wheel, set `TORCH_CUDA_ARCH_LIST` to `8.0`.
> PyTorch still uses the official cu130 wheel; CUDA 13.2 is the local build toolkit. The release wheel is `vllm-0.23.1rc1.dev904+sm89.cu132-cp312-cp312-linux_x86_64.whl`.

---

## 5. Operator smoke test (no full model needed)

```python
import torch
from vllm.platforms import current_platform
print("cap:", current_platform.get_device_capability())          # (8, 9)
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm89
print("flashinfer sparse MLA SM89:", has_flashinfer_sparse_mla_sm89())  # True
from vllm.v1.attention.backends.mla.indexer import _uses_deep_gemm_scheduler_metadata
print("DeepGEMM scheduler metadata:", _uses_deep_gemm_scheduler_metadata())  # False
from vllm.utils.import_utils import has_cutedsl
print("has_cutedsl:", has_cutedsl())                             # False on SM89
```

---

## 6. Deployment (vllm serve)

### 6.1 Source model

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm serve /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 16 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --trust-remote-code --port 8000
```

### 6.2 DSpark speculative decoding model

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm serve /path/to/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-dspark \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.96 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --trust-remote-code --port 8000
```

Startup success markers: `Application startup complete.`, and the log shows `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`.

---

## 7. Test results (4× RTX 4090)

### 7.1 Inference correctness

```text
Q: Introduce the Great Wall in one sentence. (in Chinese)
A: A coherent, accurate one-sentence answer is returned, finish_reason=stop.
```

### 7.2 Max context (KV cache)

| max-model-len | max-num-seqs | GMU | GPU KV cache | per-request concurrency | startup |
|---|---|---|---|---|---|
| 262,144 (256K) | 16 | 0.97 | 972,374 tok | 3.71x | ✅ |
| 786,432 (768K) | 16 | 0.97 | 1,220,509 tok | 1.55x | ✅ |
| **1,048,576 (1M)** | 4 | 0.97 | **1,243,644 tok** | 1.19x | ✅ (model arch limit) |

Longest input that completed: **768K (786,000 tokens, prefill ~147 s)**. 1M starts and the kernels run correctly, but a full 1M single-prompt prefill is **impractically slow (>10 min)**. Day-to-day, **128K–256K** is recommended.

Input-length sweep (256K config, all succeeded): 64K (25 s) / 128K (37 s) / 200K (74 s) / 262K (71 s).

### 7.3 Non-DSpark decode performance (4× RTX 4090, single concurrency)

| input | decode |
|---|---:|
| 8,192 | **~82 tok/s** |
| 32,768 | **~82 tok/s** |

Decode is mainly bounded by Marlin MoE dequantization overhead (no FP4 tensor cores on Ada).

### 7.4 Tool call (`deepseek_v4` parser)

```text
Q: What's Beijing's weather today? Answer in Celsius. (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

### 7.5 DSpark speculative decoding (CUDA 13.2 / torch cu130, single concurrency)

Measured with vLLM's built-in `vllm bench serve`, random dataset with fixed lengths, `max-concurrency=1`, 5 requests per case, 1024 output tokens.

Stable config:

```bash
vllm serve /root/autodl-tmp/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-dspark \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.96 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --block-size 256 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_ds_mla \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

| Input -> output | Success | Prefill TPS | Decode TPS | Acceptance |
|---|---:|---:|---:|---:|
| 8,192 -> 1,024 | 5/5 | **4891.19** | **366.95** | 95.63% |
| 32,768 -> 1,024 | 5/5 | **4859.82** | **327.38** | 83.05% |

Conversion: `Prefill TPS = input_tokens / mean_TTFT`; `Decode TPS = 1000 / mean_TPOT(ms)`.
End-to-end output tok/s from `vllm bench` includes long-context prefill/TTFT and is not decode-only throughput. For example, 32K -> 512 at single concurrency reports 61.82 tok/s end to end, while TPOT 3.047 ms corresponds to **328.2 tok/s** decode-only.

### 7.6 Wheel-vs-source A/B smoke test

With identical DSpark settings, 512 output tokens, four requests, and concurrency four, wheel output throughput differed from source by **+2.80%** at 8K (216.64 vs 210.74 tok/s) and **-0.74%** at 32K (70.45 vs 70.98 tok/s). Both sides completed 4/4 requests, returned valid tool calls, produced clean mixed Chinese/English UTF-8 text, and logged no fatal errors.

## 8. License / provenance

Based on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) and its PR #41834. This fork keeps the same license. AI-assisted, human-validated.
