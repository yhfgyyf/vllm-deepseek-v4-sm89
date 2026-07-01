# DeepSeek-V4-Flash on SM89 (Ada / RTX 4090) — vLLM fork

> 中文版见 [`README.md`](README.md)。

> This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm). The branch already contains **PR #41834** (the SM120 portable Triton path) plus the **SM89/Ada enablement commits**.

It extends vLLM's **DeepSeek-V4-Flash** inference from SM90/SM100/SM120 to **SM89 (Ada Lovelace: RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada)**. End-to-end validated on **4× RTX 4090 (48 GB)**: environment setup → operator tests → server startup → inference → performance / tool-calling — all passing.

## Changelog

### 2026-07-01

- Completed **DeepSeek-V4-Flash-DSpark** model adaptation with `method=dspark` speculative decoding. The current release wheel target is **CUDA 13.0 toolkit + torch 2.11.0+cu130**, and `vllm serve`, tool calling, and vLLM bench have been validated on CUDA 13.x / 4× RTX 4090.
- DSpark single-concurrency `8K / 32K / 128K` input and `1K` output cases all passed `10/10`; converted decode throughput is **355 / 336 / 219 tok/s**. Compared with the non-DSpark source-model baseline decode throughput of **~82 tok/s**, this is about **4.3× / 4.1× / 2.7×** faster.
- Recommended DSpark serving config: `gpu-memory-utilization=0.96`, `max-model-len=262144`, `max-num-batched-tokens=2048`, `max-num-seqs=4`, `block-size=256`, `kv-cache-dtype=fp8_ds_mla`.

---

## 1. Background: why this fork

DeepSeek-V4-Flash combines DeepSeek Sparse Attention (DSA / Lightning Indexer) + FP4-expert MoE + mHC. Upstream defaults to **FlashMLA + DeepGEMM**, which are only built for Hopper / datacenter Blackwell. PR #41834 introduced a **portable Triton path** for **SM120 (consumer Blackwell)** to replace those kernels; this fork opens that path up further to **SM89**.

| Subsystem | Upstream (SM90/100) | SM89 (this fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA sparse | **Triton** (PR #41834 portable kernels) |
| Lightning Indexer (FP8 MQA logits) | DeepGEMM | **Triton / torch fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **Triton** (FP8 dot upcast to bf16) |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE (FP4 experts) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16** (FP4→FP16 dequant) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**Hardware fact:** Ada has FP8 tensor cores but **no FP4 tensor cores and no hardware microscaling MMA**, so the FP4 MoE must run through Marlin dequantization (slower than native FP4 MMA).

### SM89 changes (relative to PR #41834: 10 files, +294/-26)

- `vllm/v1/attention/backends/mla/sparse_mla_env.py` — central switch `is_ada_sm89()`; folds SM89 into the Triton sparse-MLA path.
- `vllm/utils/deep_gemm.py` / `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` — MQA-logits / HC-GEMM fallback dispatch extended to SM89.
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — Triton FP8 einsum extended to SM89 + bf16 upcast of the FP8 `tl.dot`.
- `vllm/model_executor/kernels/mhc/tilelang.py` — mHC TF32 path extended to SM89.
- `vllm/model_executor/layers/sparse_attn_indexer.py` / `v1/attention/backends/mla/indexer.py` — fix the init-time crash in `_sparse_indexer_requires_deep_gemm`; memory budget.
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability` made accurate.
- **`vllm/utils/import_utils.py` — `has_cutedsl()` returns False on SM89**.

> See [`SM89_DEEPSEEK_V4_NOTES.md`](SM89_DEEPSEEK_V4_NOTES.md) for details.

---

## 2. Validated environment

| Item | Version |
|---|---|
| GPU | 4× RTX 4090 (48 GB) · compute capability **8.9** |
| Driver / CUDA toolkit | 595.x / **CUDA 13.0** (nvcc 13.0) |
| Python | 3.12 |
| torch | **2.11.0+cu130** |
| vLLM | this fork release wheel = **0.23.1rc1.dev145+g<commit>.cu130**, built for SM89/Ada only |

---

## 3. Quick install (prebuilt wheel)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --torch-backend=cu130

gh release download \
  --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'vllm-0.23.*.cu130-cp312-cp312-linux_x86_64.whl' \
  --dir /tmp/vllm-sm89-cu130

uv pip install --force-reinstall --no-deps \
  /tmp/vllm-sm89-cu130/vllm-0.23.*.cu130-cp312-cp312-linux_x86_64.whl
```

**Validated environment:**
- **Python 3.12**, Linux x86_64
- **4× RTX 4090 (SM89/Ada, 48 GB)**, 595.x driver, CUDA toolkit 13.0
- **torch 2.11.0+cu130**
- Wheel built with `TORCH_CUDA_ARCH_LIST=8.9+PTX`, targeting Ada/SM89

---

## 4. Build from source (clone this repo)

### 4.1 Python env + torch cu130

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --torch-backend=cu130
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

### 4.2 Rust toolchain (vLLM 0.11 ships a Rust frontend)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95 --profile minimal
source "$HOME/.cargo/env"
```

### 4.3 Clone this repo

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
```

### 4.4 Build / package a CUDA 13.0 wheel (Ada 8.9 only)

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$HOME/.cargo/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0
export VLLM_VERSION_OVERRIDE=0.23.1rc1.dev145+g$(git rev-parse --short=9 HEAD).cu130
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=16 NVCC_THREADS=2

.venv/bin/python -m build --wheel --no-isolation
uv pip install --force-reinstall --no-deps dist/vllm-*.cu130-*.whl
```

> Do **not** install DeepGEMM (unsupported on Ada).
> Wheel names follow the release convention: `vllm-0.23.1rc1.dev145+g<commit>.cu130-cp312-cp312-linux_x86_64.whl`.

---

## 5. Operator smoke test (no full model needed)

```python
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_mla_env as e
print("cap:", current_platform.get_device_capability())          # (8, 9)
print("is_ada_sm89:", e.is_ada_sm89())                            # True
print("triton sparse mla:", e.is_triton_sparse_mla_enabled(torch.device("cuda:0")))  # True
from vllm.model_executor.layers.sparse_attn_indexer import _sparse_indexer_requires_deep_gemm as r
print("indexer needs deepgemm (fp8 cache):", r(False))           # False  <- key fix
from vllm.utils.import_utils import has_cutedsl
print("has_cutedsl:", has_cutedsl())                             # False on SM89
```

---

## 6. Deployment (vllm serve)

### 6.1 Source model

```bash
export VLLM_TRITON_MLA_SPARSE=1
vllm serve /path/to/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 16 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --trust-remote-code --port 8000
```

### 6.2 DSpark speculative decoding model

```bash
export VLLM_TRITON_MLA_SPARSE=1
vllm serve /path/to/DeepSeek-V4-Flash-DSpark \
  --served-model-name deepseek-v4-flash-dspark \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.96 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}' \
  --trust-remote-code --port 8000
```

Startup success markers: `Application startup complete.`, and the log shows `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`.

---

## 7. Test results (4× RTX 4090)

### 7.1 Inference correctness
```
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

### 7.3 Performance (single concurrency, 512 output tokens)
| input | TTFT | prefill | decode |
|---|---|---|---|
| 8,192 | 1.97 s | **~4,160 tok/s** | **~82 tok/s** |
| 32,768 | 7.81 s | **~4,195 tok/s** | **~82 tok/s** |

Decode ~82 tok/s is bounded by Marlin MoE dequantization overhead (no FP4 tensor cores on Ada).

### 7.4 Tool call (`deepseek_v4` parser)
```
Q: What's Beijing's weather today? Answer in Celsius. (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

### 7.5 DSpark speculative decoding (CUDA 13.x / torch cu130, single concurrency)

Measured with vLLM's built-in `vllm bench serve`, random dataset with fixed lengths, `max-concurrency=1`, 10 requests per case, 1024 output tokens.

Stable config:

```bash
vllm serve /root/autodl-tmp/DeepSeek-V4-Flash-DSpark \
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
  --speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'
```

| Input → output | Success | mean TTFT | mean TPOT | Prefill | Decode | output tok/s | total tok/s | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 → 1,024 | 10/10 | 2.428 s | 2.816 ms | **3,374 tok/s** | **355 tok/s** | 192.9 | 1,736.0 | 92.81% |
| 32,768 → 1,024 | 10/10 | 9.442 s | 2.974 ms | **3,470 tok/s** | **336 tok/s** | 82.0 | 2,706.6 | 89.12% |
| 131,072 → 1,024 | 10/10 | 42.829 s | 4.568 ms | **3,060 tok/s** | **219 tok/s** | 21.6 | 2,780.8 | 58.25% |

## 8. License / provenance

Based on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) and its PR #41834. This fork keeps the same license. AI-assisted, human-validated.
