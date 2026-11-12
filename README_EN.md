# DeepSeek V4.1 / V4 / Vision and GLM-5.3 on SM89 / SM120

<!-- markdownlint-disable MD060 -->

[中文](README.md)

This [vLLM](https://github.com/vllm-project/vllm) fork serves
DeepSeek-V4.1-Flash, DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp and
GLM-5.3-Flash through an OpenAI-compatible API.

## 1. Requirements

| Component | Configuration |
| --- | --- |
| OS / Python | Linux x86_64 / Python 3.12 |
| SM89 examples | L20 or RTX 4090 48 GB; V4.1 uses eight GPUs |
| SM120 examples | Four RTX PRO 6000 Blackwell 96 GB GPUs |
| Current runtime | PyTorch 2.14.0+cu130, Triton 3.8.0, Transformers 5.22.2 |
| FlashInfer | This project's patched 0.7.3 extension; the upstream package is not a substitute |
| CUDA | 13.2 for source builds; local JIT in the serving examples uses 13.0 |

Use a complete model checkpoint. Engram CPU offload and model loading require
sufficient host RAM. Examples for 48 GB cards do not apply to standard 24 GB RTX 4090 GPUs.

## 2. Installation

### 2.1 Current source

For the L20 TP8 configuration or optional SM120 B12X operators below, install
current `main` and apply its matching FlashInfer recipe patch:

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
```

Then follow the [source build instructions](docs/models/deepseek_v41_sm89_sm120.md#build-the-current-source)
for vLLM and FlashInfer. Use a new Python 3.12 environment; do not reuse extensions
built against Torch 2.13.

### 2.2 Prebuilt wheels

The pinned `vision12` vLLM and `vision3` FlashInfer wheels below do not include
the current source's L20 tuning or optional SM120 B12X adapters. Use a source
installation when you need those features.

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/vllm-0.33.0%2Bglm53.dsv41.vision12.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=67228efddc16f0769e3578137a734e14cf2869b36610f37946419503d6d8c65e' \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/flashinfer_python-0.7.3%2Bglm53.dsv41.vision3.sm89sm120.cu130.pt214-py3-none-any.whl#sha256=b85753d53a0094796e865bffed11b2f55226d1991f776a3b9a76a2f0f77abecf' \
  'transformers==5.22.2' 'triton==3.8.0'
```

`uv` verifies both wheel SHA256 hashes.

### 2.3 Docker

The existing GHCR `vision11` image uses vLLM 0.28 / Torch 2.13 for older
environments. The current L20 TP8 example requires the source installation
above and should not be applied directly to this image.
The container template below requires a model path and model arguments for
that image version. Pass environment variables with `docker run -e`; V4.1 on
`vision11` also requires `-e VLLM_USE_V2_MODEL_RUNNER=1`.

```bash
docker pull ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130 \
  /models/model-directory \
  --host 0.0.0.0 --port 8000
```

## 3. DeepSeek-V4.1-Flash launch command

This template uses **4× RTX PRO 6000 96 GB, TP4/PP1, draft TP4 and DSpark5**.
**8×L20 defaults to DSpark7**, shown in Section 3.3. Replace the environment,
CUDA and model paths before running.

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0
unset TRITON_PTXAS_BLACKWELL_PATH

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len 1048576 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-extensible-kv-cache \
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
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true,"draft_tensor_parallel_size":4}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":48}'
```

### 3.1 Feature switches

- **DSpark**: use a checkpoint with a confidence head. Examples enable adaptive
  verification; set `enable_adaptive_verification=false` for fixed-draft verification.
- **CUDA Graph**: V4.1 uses `FULL_AND_PIECEWISE`; the other DSpark examples use
  `FULL`. Keep the corresponding graph mode when using adaptive and do not add
  `--enforce-eager`.
- **CED**: `ced_prefill=true` enables experimental approximate prefill. Remove
  that override for normal prefill. CED and adaptive are independent; assess
  output quality on your workload.
- **Images**: set the count limit with `--limit-mm-per-prompt '{"image":2}'`.
  V4.1 currently does not support non-image modalities, prompt embeddings or prompt logprobs.
- **SM120 B12X**: disabled by default. Follow the [installation and switch instructions](docs/models/deepseek_v41_sm120_b12x.md)
  for the pinned version and explicit opt-in. GPU integration qualification of
  the new adapters is pending.

Initial JIT compilation and graph capture may take time; wait for server
readiness before sending requests. `max-model-len` limits request length and
`max-num-seqs` limits scheduling; they do not guarantee that every concurrent
request can use the maximum context at once.

### 3.2 L20 performance reference

8×L20, TP8/PP1, draft TP8, DSpark7, memory utilization 0.96, with CED and adaptive:

| Input → output tokens | Median prefill | Median decode | DSpark acceptance |
| --- | ---: | ---: | ---: |
| 17,630 → 2,641 | **7,849.510 tokens/s** | **395.850 tokens/s** | **87.135177%** |

Synthetic copying measured on 2026-11-12: three single-request runs,
temperature=0, reasoning_effort=none and zero prefix hits. Prefill = input tokens /
server prefill seconds; decode = `(output tokens - 1) / server decode seconds`;
acceptance = accepted/proposed. These are single-request phase rates; actual
performance depends on the input and acceptance rate.

### 3.3 SM89: 8×48 GB GPUs (L20 / RTX 4090 48 GB)

The current-source L20 configuration uses **TP8/PP1, draft TP8, DSpark7 and
0.96 automatic KV**, with PCIe IPC, EP, CED and Engram CPU offload. It schedules
up to 8192 tokens and four sequences, with graph capture capped at 48 rows.

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
unset TRITON_PTXAS_BLACKWELL_PATH
unset VLLM_PP_LAYER_PARTITION

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len 1048576 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --prefill-schedule-interval 1 \
  --gpu-memory-utilization 0.96 \
  --no-enable-extensible-kv-cache \
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
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true,"draft_tensor_parallel_size":8}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","max_cudagraph_capture_size":48}'
```

Keep `--no-enable-extensible-kv-cache` for conventional automatic KV sizing.
Target TP8 requires draft TP8. IPC uses the original communication backend for
unsupported messages. If memory is insufficient, lower utilization or the batch
budget and check capacity against your workload.

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

The following example uses the official FP8 weights://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993):

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

## 7. Call the API

After the server is ready, use the OpenAI-compatible API. The `model` must
match `--served-model-name`; this example uses the V4.1 server above:

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

## 8. Documentation

- [Source build and dependencies](docs/models/deepseek_v41_sm89_sm120.md)
- [L20 configuration and performance](docs/models/deepseek_v41_l20_baseline.md)
- [Optional SM120 operators](docs/models/deepseek_v41_sm120_b12x.md)
- [Release and measurement history](docs/models/sm89_sm120_history.md)

## 9. License / provenance

The code is based on [vllm-project/vllm](https://github.com/vllm-project/vllm)
and remains under Apache-2.0. The FlashInfer wheel is based on
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
