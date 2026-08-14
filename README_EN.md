# DeepSeek-V4-Flash on SM80 / SM86 / SM89 — vLLM fork

> 中文版见 [`README.md`](README.md)。

> This repository is a fork of [vllm-project/vllm](https://github.com/vllm-project/vllm). The branch already contains **PR #41834** (the SM120 portable Triton path) plus the **SM80 / SM86 / SM89 enablement commits**.

It extends vLLM's **DeepSeek-V4-Flash** inference from SM90/SM100/SM120 to **SM80 / SM86 / SM89**, covering Ampere and Ada GPUs such as A100, RTX 3090, A10/A40, RTX 4090, L40/L40S/L4, and RTX 6000 Ada.

> ⚠️ Experimental fork. For self-testing DeepSeek-V4-Flash on SM80 / SM86 / SM89 GPUs only.
> The **SM80/A800 path is a test-only adaptation**, not a production support commitment.

### SM80/A800 status

The SM80 path has passed DeepSeek-V4-Flash DSpark speculative decoding smoke and throughput tests
on a 4× A800 server. The tested server used
`--speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'`,
the FlashInfer sampler, sparse MLA warmup, and `max-num-batched-tokens=16384`.

Decode-side results below use the same server's `mbt16k` no-DSpark run as baseline:

| input -> output | concurrency | DSpark decode | no-DSpark decode | decode speedup |
|---|---:|---:|---:|---:|
| 8,192 -> 1,024 | 1 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 -> 1,024 | 1 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

### Scope of the long-context patch

This branch fixes the FP8 MQA-logits query tile at `BLOCK_M=16` for long SM80
sequences, avoiding the severe register spill produced by `BLOCK_M=64` on A100.
It only addresses the long-context Lightning Indexer prefill hotspot. It does not
provide CPU/NUMA MoE offload, DSpark tuning, service orchestration, or an API
aggregation layer.

This document provides two separate reproduction routes:

- **Source route:** build this repository on an SM80 / SM86 / SM89 system with
  sufficient resources for the model.
- **Validated 2×A100 hybrid route:** use CPU MoE offload from `Lvllmds4-x v2.3.9`,
  then apply the equivalent M16 change and the correctness backports listed in
  Section 6.2.

Pulling this performance PR alone does **not** recreate the complete second route.
The documentation keeps the PR, public runtime, and correctness patches separate
to avoid attributing deployment changes to this PR.

---

## 1. Background: why this fork

DeepSeek-V4-Flash combines DeepSeek Sparse Attention (DSA / Lightning Indexer) + FP4-expert MoE + mHC. Upstream defaults to **FlashMLA + DeepGEMM**, which are only built for Hopper / datacenter Blackwell. PR #41834 introduced a **portable Triton path** for **SM120 (consumer Blackwell)** to replace those kernels; this fork opens that path up further to **SM80 / SM86 / SM89**.

| Subsystem | Upstream (SM90/100) | SM80 / SM86 / SM89 (this fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA sparse | **Triton** (PR #41834 portable kernels) |
| Lightning Indexer (FP8 MQA logits) | DeepGEMM | **Triton / torch fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **Triton** (FP8 dot upcast to bf16) |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE (FP4 experts) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16** (FP4→FP16 dequant) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**Hardware fact:** SM80 / SM86 do not have native FP8 tensor cores. SM89 has FP8 tensor cores, but no FP4 tensor cores and no hardware microscaling MMA. This fork uses Triton / torch fallbacks and Marlin WNA16 on these GPUs instead of DeepGEMM / FlashInfer FP4 kernels.

### SM80 / SM86 / SM89 changes

- `vllm/v1/attention/backends/mla/sparse_mla_env.py` — folds SM80 / SM86 / SM89 into the Triton sparse-MLA path.
- `vllm/utils/deep_gemm.py` / `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` — MQA-logits / HC-GEMM fallback dispatch extended to SM80 / SM86 / SM89.
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — Triton FP8 einsum extended to SM80 / SM86 / SM89 while preserving the bf16 B path.
- `vllm/model_executor/kernels/mhc/tilelang.py` — mHC TF32 path extended to SM80 / SM86 / SM89.
- `vllm/model_executor/layers/sparse_attn_indexer.py` / `v1/attention/backends/mla/indexer.py` — fix the init-time crash in `_sparse_indexer_requires_deep_gemm`; memory budget.
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability` made accurate.
- **`vllm/utils/import_utils.py` — `has_cutedsl()` returns False on SM80 / SM86 / SM89**.
- FP8 linear / MoE paths use Marlin W8A16 on SM80 / SM86 to avoid software-decode GEMMs on GPUs without native FP8 MMA.

> See [`SM89_DEEPSEEK_V4_NOTES.md`](SM89_DEEPSEEK_V4_NOTES.md) for details.

---

## 2. Supported hardware and validated environment

| Item | Version |
|---|---|
| Supported GPUs | **SM80** (A100), **SM86** (RTX 3090 / A10 / A40), **SM89** (RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada) |
| Driver / CUDA toolkit | 595.x / **CUDA 13.0** (wheel built with `/usr/local/cuda-13.0`, nvcc 13.0.48) |
| Python | 3.12 (historical validation used conda; uv is used below) |
| torch | **2.11.0+cu130** |
| vLLM | this fork = **0.23.1rc1.dev145** (DeepSeek-V4-Flash + SM80 / SM86 / SM89 changes), built from source |

---

## 3. Build from source (clone this repo)

### 3.1 Prerequisites

Install Git, uv, Rust 1.95, and a matching CUDA toolkit first. The repository-local
environment is created after clone so `.venv` cannot be placed in the wrong directory.

### 3.2 Rust toolchain (vLLM builds the Rust frontend)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.95 --profile minimal
source "$HOME/.cargo/env"
```

### 3.3 Clone this repo

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
git fetch https://github.com/fidonan/vllm-deepseek-v4-sm89.git \
  perf/sm80-long-context-mqa-block-m
git checkout -B perf-sm80-long-context-mqa-block-m FETCH_HEAD
git rev-parse HEAD
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

The public head branch above reproduces this change exactly before merge; after merge, use the
maintainer's final merged branch/tag and commit instead. Pin and record the printed
commit. The SM80 base branch does not yet contain
[PR #51](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/51), which is already
merged into `main`; synchronize that paged-MQA 64-bit addressing fix before a
long-context deployment. Its current single commit is
`3183db2f6c86803cc86d06223000d220e001baf8`; check the target revision with
`git merge-base --is-ancestor`, review it, and cherry-pick only when absent.

### 3.4 Build

```bash
uv pip install -U "setuptools>=77,<81" setuptools-rust numpy packaging wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
export MAX_JOBS=16 NVCC_THREADS=2
uv pip install -e . --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

> If you deploy on a single GPU architecture, you can narrow `TORCH_CUDA_ARCH_LIST` to the target architecture: `8.0` for A100, `8.6` for RTX 3090, or `8.9+PTX` for RTX 4090 / L40.
> Do **not** install DeepGEMM; this fork does not use that path on SM80 / SM86 / SM89.
> If after the build torchvision/torchaudio are the non-cu130 builds you will see `torchvision::nms does not exist`; fix with:
> `uv pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu130 torchvision torchaudio`

---

## 4. Optional: build a local wheel

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-} \
uv pip wheel . --no-build-isolation --no-deps -w dist/ \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

Install the locally built wheel:

```bash
uv pip install dist/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

The branch changes frequently. Prefer source install unless you have built and published a wheel that matches SM80 / SM86 / SM89.

---

## 5. Operator smoke test (no full model needed)

Run the repository tests after installation:

```bash
uv pip install -r requirements/test/cuda.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
.venv/bin/python test_sm80_ops.py
.venv/bin/python -m pytest -q tests/v1/attention/test_sm120_deepgemm_fallbacks.py
```

The final command checks long-sequence `BLOCK_M=16` selection. On SM80 it also
compares the complete M16 and M64 logits and requires element-wise equality.

```python
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_mla_env as e
print("cap:", current_platform.get_device_capability())          # (8, 9)
print("is_ampere_or_ada:", e.is_ampere_or_ada())                 # True on SM8x
print("triton sparse mla:", e.is_triton_sparse_mla_enabled(torch.device("cuda:0")))  # True
from vllm.model_executor.layers.sparse_attn_indexer import _sparse_indexer_requires_deep_gemm as r
print("indexer needs deepgemm (fp8 cache):", r(False))           # False  <- key fix
from vllm.utils.import_utils import has_cutedsl
print("has_cutedsl:", has_cutedsl())                             # False on SM89
```

---

## 6. Reproducible deployment

### 6.1 Source route

The following command uses the public DSpark model ID and records no machine path,
bind address, or credential. Set `TP_SIZE` to the number of GPUs selected by the
operator.

```bash
export MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-DSpark
export TP_SIZE="${TP_SIZE:?set TP_SIZE to the number of selected GPUs}"
export VLLM_TRITON_MLA_SPARSE=1
.venv/bin/vllm serve "$MODEL_ID" \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size "$TP_SIZE" \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 16 \
  --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --trust-remote-code
```

Startup success markers: `Application startup complete.`, and the log shows `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`.

### 6.2 Validated 2×A100 hybrid route

This route is for two 80-GiB A100 GPUs that cannot keep all MoE experts resident
in GPU memory. The measured runtime was
[Lvllmds4-x v2.3.9](https://github.com/guqiong96/Lvllmds4-x/releases/tag/lvllmds4-x-v2.3.9).
The asset is `lvllmds4_x-2.3.9-cp312-cp312-manylinux_2_34_x86_64.whl`, requiring
x86_64, Python 3.12, and glibc 2.34 or newer. Its SHA-256 is
`1357dd5e11d060cce973f84563d96bb18c1bd2bd7a4d05f642fe01629ba7ab62`.

```bash
export LVLLM_REPRO_DIR=.repro/lvllmds4x
mkdir -p "$LVLLM_REPRO_DIR"
uv venv --python 3.12 "$LVLLM_REPRO_DIR/.venv"
export LVLLM_WHEEL=lvllmds4_x-2.3.9-cp312-cp312-manylinux_2_34_x86_64.whl
export LVLLM_WHEEL_URL=https://github.com/guqiong96/Lvllmds4-x/releases/download/lvllmds4-x-v2.3.9/$LVLLM_WHEEL
curl -fL "$LVLLM_WHEEL_URL" -o "$LVLLM_REPRO_DIR/$LVLLM_WHEEL"
printf '%s  %s\n' \
  1357dd5e11d060cce973f84563d96bb18c1bd2bd7a4d05f642fe01629ba7ab62 \
  "$LVLLM_REPRO_DIR/$LVLLM_WHEEL" | sha256sum -c -
uv pip install --python "$LVLLM_REPRO_DIR/.venv/bin/python" \
  "$LVLLM_REPRO_DIR/$LVLLM_WHEEL"
```

Complete these four checks before startup:

1. Install and checksum the public wheel in an isolated environment; do not
   overwrite an existing vLLM environment.
   `.repro/` is repository-ignored and should be removed after the work; never
   commit the wheel or environment files.
2. Apply this branch's `_fp8_mqa_logits_block_m()` M16 change to that isolated
   runtime. Section 5 tests only validate the source route and cannot prove that
   the wheel was patched. From outside the repository, use the hybrid interpreter
   to inspect `vllm.__file__`, the target function source, and its SM80 long-sequence
   return value, then run an equivalent GPU-equality test. Stop if imports resolve
   to the source checkout.

   ```bash
   set -euo pipefail
   export LVLLM_REPRO_DIR="${LVLLM_REPRO_DIR:-.repro/lvllmds4x}"
   REPRO_ROOT="$(pwd -P)"
   CHECK_DIR="$(mktemp -d)"
   (
     cd "$CHECK_DIR"
     "$REPRO_ROOT/$LVLLM_REPRO_DIR/.venv/bin/python" - <<'PY'
   import inspect
   import vllm
   from vllm.models.deepseek_v4.nvidia.ops import sm12x_mqa

   print(vllm.__file__)
   print(inspect.getsource(sm12x_mqa._fp8_mqa_logits_block_m))
   assert sm12x_mqa._fp8_mqa_logits_block_m(4096, 16 * 1024 + 1) == 16
   PY
     "$REPRO_ROOT/$LVLLM_REPRO_DIR/.venv/bin/python" \
       "$REPRO_ROOT/benchmarks/kernels/benchmark_sm12x_mqa.py" \
       --contexts 98304 --repeats 1 --seed 0
   )
   rmdir "$CHECK_DIR"
   ```
3. Verify that the runtime contains the
   [paged-MQA 64-bit addressing fix](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/51)
   and the upstream vLLM mixed-precision/packed-KV zeroing fixes:
   [#47574](https://github.com/vllm-project/vllm/pull/47574),
   [#49623](https://github.com/vllm-project/vllm/pull/49623),
   [#49704](https://github.com/vllm-project/vllm/pull/49704),
   [#49903](https://github.com/vllm-project/vllm/pull/49903),
   [#50276](https://github.com/vllm-project/vllm/pull/50276), and
   [#52058](https://github.com/vllm-project/vllm/pull/52058).
   Long-running use should also include the scheduler per-step drain fix
   [#44490](https://github.com/vllm-project/vllm/pull/44490).
4. The CPU must be x86_64 with AVX2, the host must provide a NUMA runtime, and
   the isolated environment's libstdc++ must satisfy the wheel's GLIBCXX ABI.
   If the uv environment reports a GLIBCXX error, use a host toolchain, compatible
   container, or Conda environment that satisfies the ABI; do not replace system
   libraries in place.

Those correctness backports are independent from this performance PR. The measured
stack used local backports equivalent to these upstream fixes; the links here are
the recommended upstream set that converged after the benchmark, and that exact
set was not rerun as one complete end-to-end A/B. Because these cross-version
backports still require conflict resolution and retesting against the selected
revision, this section is a validated configuration checklist, not an unattended
installer. Without these fixes, an isolated kernel experiment is still useful,
but a long-context run must not be presented as a long-running stability result.

The following is the currently recommended reproduction profile; it is not the
exact per-row configuration for every historical sample in Section 7.5.
`GPU_DEVICES` is supplied by the operator after checking both the GPU-interconnect
topology and CPU/NUMA affinity. The thread and resident-layer values are not
universal defaults; use them only after checking GPU memory, host memory, and
bandwidth.

```bash
export MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-DSpark
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:?set the selected GPU IDs}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LVLLM_MOE_NUMA_ENABLED=1
export LK_THREADS=30
export OMP_NUM_THREADS=30
export LK_THREAD_BINDING=CPU_CORE
export LVLLM_GPU_PREFETCH_WINDOW=1
export LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=256
export LK_POWER_SAVING=0
export LVLLM_GPU_RESIDENT_MOE_LAYERS=0-28

.repro/lvllmds4x/.venv/bin/vllm serve "$MODEL_ID" \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --dtype bfloat16 \
  --max-num-seqs 2 \
  --enable-auto-tool-choice \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config \
    '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --disable-custom-all-reduce
```

The measured environment set `FLASHINFER_DISABLE_VERSION_CHECK=1`. This bypasses
a version guard and is not a general default. Set it for this pinned runtime only
after verifying the wheel release's FlashInfer version and binary ABI.

`LVLLM_*` and `LK_*` are specific to Lvllmds4-x and do not apply to standard
vLLM. Startup must finish weight loading, KV initialization, sparse-MLA warmup,
and Triton/CUDA-graph compilation. Do not begin the measured run until the service
is healthy, no worker has restarted, and both ranks have completed warmup.

### 6.3 Codex reproduction task

The following task can be pasted directly into Codex. It requires Codex to discover
devices and directories and prohibits writing discovered machine details back to
the repository:

```text
Read AGENTS.md and both README files completely before acting.
Reproduce the SM80 long-context MQA M16 result in an isolated environment.

1. Inspect GPU capability, topology, NUMA layout, free RAM, disk capacity, CUDA,
   Python, and the currently selected Git revision. Do not change running services.
2. Choose either the source-only route or the documented Lvllmds4-x hybrid route.
   Stop and report if the machine cannot hold the model and runtime state.
3. Pin every repository revision and verify every downloaded artifact checksum.
4. Verify the paged-MQA int64 addressing fix and every listed packed/mixed-KV
   zeroing fix before treating a long-context run as a stability test.
5. Apply only this branch's M16 selector change, then run the focused unit and
   GPU-equivalence tests. Do not add Router, protocol, API, or service-manager code.
6. Start the model in the foreground using endpoint and credentials supplied by
   the operator. Never invent or commit an address, port, username, absolute local
   path, credential, service name, prompt, or production log.
7. Warm each new Triton shape with a throwaway unique prefix. Benchmark another
   unique prefix at concurrency one, verify zero prefix-cache hits, and save the
   exact commit, parameters, JIT state, token counts, TTFT, and kernel timing.
8. Return a sanitized report and leave generated machine-specific files untracked.
```

---

## 7. Test results

### 7.1 Inference correctness (4× RTX 4090 source route)
```
Q: Introduce the Great Wall in one sentence. (in Chinese)
A: A coherent, accurate one-sentence answer is returned, finish_reason=stop.
```

### 7.2 Max context (4× RTX 4090 source route)
| max-model-len | max-num-seqs | GMU | GPU KV cache | per-request concurrency | startup |
|---|---|---|---|---|---|
| 262,144 (256K) | 16 | 0.97 | 972,374 tok | 3.71x | ✅ |
| 786,432 (768K) | 16 | 0.97 | 1,220,509 tok | 1.55x | ✅ |
| **1,048,576 (1M)** | 4 | 0.97 | **1,243,644 tok** | 1.19x | ✅ (model arch limit) |

Longest input that completed: **768K (786,000 tokens, prefill ~147 s)**. 1M starts and the kernels run correctly, but a full 1M single-prompt prefill is **impractically slow (>10 min)**. Day-to-day, **128K–256K** is recommended.

### 7.3 SM80/A800 DSpark decode performance (single concurrency, 1,024 output tokens)
| input | DSpark decode | no-DSpark decode | decode speedup |
|---|---:|---:|---:|
| 8,192 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

The SM80/A800 path is still a test-only adaptation. This table only reports decode-side results; long-context prefill needs separate evaluation.

### 7.4 SM80/A100 long-context MQA kernel A/B

This is the historical kernel A/B for the patch: same process and input, with each
`BLOCK_M` compiled separately; one untimed warmup after compilation, followed by
the median of three CUDA-event timings. `M` varies with `N` to keep the FP32 logits
workspace near 256 MiB. The fixed shape uses `num_heads=64`, `head_dim=128`; the
three `M` values are 2,730, 2,048, and 1,724. Inputs are FP8 E4M3 Q/K plus FP32
scales/weights. The public same-shape reproducer defaults to random seed 0; the
historical table used a fixed seed derived from each context length, so the script
validates the trend and numerical equality rather than promising identical
millisecond values. The lightweight equivalence test in Section 5 does not
reproduce these timing shapes.

The public reproducer is
[`benchmarks/kernels/benchmark_sm12x_mqa.py`](benchmarks/kernels/benchmark_sm12x_mqa.py).
It refuses non-SM80 devices and reports CUDA-event medians plus M16/M64 equality
for every shape. Spill counts came from the compiled Triton artifact metadata;
the values in this table are historical compiler-output values, not constants
across Triton/CUDA releases.

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm12x_mqa.py \
  --contexts 98304 131072 155648 \
  --repeats 3 \
  --seed 0 \
  --output-json mqa-kernel-results.json
```

| raw context | C4-compressed N | M16 | M64 | kernel speedup | M16 / M64 spill |
|---:|---:|---:|---:|---:|---:|
| 98,304 | 24,576 | **40.491 ms** | 643.797 ms | **15.90×** | 0 / 5,744 B/thread |
| 131,072 | 32,768 | **34.065 ms** | 1,003.002 ms | **29.44×** | 0 / 8,272 B/thread |
| 155,648 | 38,912 | **34.547 ms** | 644.409 ms | **18.65×** | 0 / 5,744 B/thread |

All three complete FP32-logits tensors, each about 256 MiB, satisfy
`torch.equal(M16, M64)` with no NaN/Inf mismatch. These values are single-kernel
latencies, **not** TTFT or end-to-end prefill tok/s.

### 7.5 SM80/A100 end-to-end observations

Every row below was actually run at concurrency one. Prefill throughput is
`prompt tokens / client TTFT`. The formal M16 samples use a fresh token sequence,
zero prefix-cache hits, and an already-warmed JIT shape.

| context | selector / state | client TTFT | observed prefill |
|---:|---|---:|---:|
| 65,536 | original selector already used M16; JIT-hot boundary control | 29.893 s | **2,192.4 tok/s** |
| 98,304 | original M64 selector; cold prefix | 237.989 s | 413.1 tok/s |
| 98,304 | M16; first shape/JIT | 105.971 s | 927.6 tok/s |
| 98,304 | M16; JIT-hot, cold prefix | 54.383 s | **1,807.6 tok/s** |
| 155,648 | original M64 selector; cold prefix | 730.664 s | 213.0 tok/s |
| 155,648 | M16; first shape/JIT | 138.660 s | 1,122.5 tok/s |
| 155,648 | M16; JIT-hot, cold prefix | 110.015 s | **1,414.8 tok/s** |

These end-to-end results are `n=1` runtime observations, not strict same-config
A/B results. The old baseline used `max-num-batched-tokens=16384` and resident
layers `0-27`; the M16 samples used `8192`, GMU `0.90`, resident layers `0-28`, and a separate
compile cache. Raw context 65,536 maps exactly to the C4 16,384 boundary, where the
old code already selected M16, so it is only a control. No patched 131,072-token
full-service cold-prefill run was performed; this document gives no end-to-end
speed or interpolated result for that length.

Do not reuse the measured prompt through the built-in warmup request. First send
an independent JIT-priming request of the same length with seed A, then start the
formal one-request run with seed B; disable request warmup and ready check in both
runs. The saved `input_lens` must match the target length, metrics must report a
zero cache-hit delta, and the measured request window must contain no new JIT log.
`input_lens[0] / ttfts[0]` is the client-observed TTFT rate used in this table; it
is not server `request_prefill_time`.

The example below uses the hybrid route; for the source route, point `BENCH_VLLM`
to `.venv/bin/vllm`.

```bash
export BENCH_VLLM=.repro/lvllmds4x/.venv/bin/vllm
"$BENCH_VLLM" bench serve \
  --backend openai \
  --base-url "${BENCH_BASE_URL:?set the operator-provided endpoint}" \
  --endpoint /v1/completions \
  --model deepseek-v4-flash \
  --tokenizer deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 \
  --dataset-name random \
  --random-input-len "${BENCH_TOKENS:?set the measured input length}" \
  --random-output-len 32 \
  --random-range-ratio 0 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --request-rate inf \
  --num-warmups 0 \
  --ready-check-timeout-sec 0 \
  --seed "${BENCH_SEED:?use a seed different from the priming request}" \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --result-dir "${BENCH_RESULT_DIR:?set an isolated result directory}" \
  --result-filename measured.json
```

### 7.6 Tool call (`deepseek_v4` parser)
```
Q: What's Beijing's weather today? Answer in Celsius. (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

---

## 8. Known limitations / risks

1. **MoE runs on Marlin**: correctness validated, but performance is below native FP4 MMA — the biggest remaining tuning opportunity.
2. **Performance is not tuned for the 4090**: the fused_moe / scaled_mm tuned configs only cover RTX PRO 6000 / GB10; the 4090 uses default heuristics (the log prints "Performance might be sub-optimal").
3. **Very long context**: 1M starts but prefill is impractically slow; single requests over 256K take several minutes.
4. The 4× RTX 4090 source route, 4× A800 DSpark decode, and 2× A100 hybrid route
   each have the tests explicitly labeled in this document. Other Ada GPUs
   (L40/L4, etc.) should work in principle but are untested.
5. **M16 only optimizes prefill:** paged decode does not call this non-paged MQA
   path. Decode throughput is still determined by long-context attention, MoE
   offload, and DSpark acceptance.
6. **Deployment dependency boundary:** this PR does not include the paged-MQA
   addressing or packed/mixed-KV zeroing backports. Do not claim the full stable
   environment was reproduced before checking the prerequisites in Section 6.2.
7. **Benchmark boundary:** the first new shape includes JIT cost, while an exact
   repeat uses the prefix cache. Neither replaces a formal cold-prefix sample with
   warmed JIT and zero cache hits.

---

## 9. License / provenance

Based on [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0) and its PR #41834. This fork keeps the same license. AI-assisted, human-validated.
