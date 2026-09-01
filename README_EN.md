# DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, and GLM-5.3-Flash on SM89 / SM120 — vLLM fork

<!-- markdownlint-disable MD060 -->

> 中文版见 [`README.md`](README.md)。
>
> This repository is based on
> [vllm-project/vllm](https://github.com/vllm-project/vllm). It runs
> DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, and GLM-5.3-Flash on
> SM89/Ada and SM120/RTX Blackwell.

The current source is based on vLLM `v0.28.1rc0-289` and is paired with
FlashInfer `0.6.18`. Validated configurations include
**4×/8× RTX 4090 48 GB** and **4× RTX PRO 6000 Blackwell 96 GB** systems.

## Support matrix

| GPU architecture | Validated GPU | DeepSeek-V4-Flash | DeepSeek-V4-Flash-Vision-Exp | GLM-5.3-Flash |
|---|---|---:|---:|---:|
| SM89 / Ada | 8× RTX 4090 48 GB | Yes | Yes | Yes |
| SM120 / RTX Blackwell | 4× RTX PRO 6000 96 GB | Yes | Yes | Yes |

---

## Changelog

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

## 1. Validated environment

| Item | Version / configuration |
|---|---|
| Operating system | Linux x86_64 |
| Python | 3.12 |
| CUDA toolkit | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| FlashInfer | `0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213` |
| vLLM | `0.28.1rc0.dev289` SM89+SM120 vision build |
| SM89 | 4×/8× RTX 4090 48 GB |
| SM120 | 4× RTX PRO 6000 Blackwell 96 GB |

The FlashInfer wheel is a Python/JIT source package. The first unseen model
shape is compiled once and then reused from the JIT cache.

---

## 2. Quick install (prebuilt wheels)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download v0.28.1rc0-vision-sm89-sm120-cu130 \
  --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213-*.whl' \
  --pattern 'vllm-*glm53.dsv4.vision*.sm89sm120.cu130-*.whl' \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-sm89-sm120-vision-release

cd /tmp/vllm-sm89-sm120-vision-release
sha256sum -c SHA256SUMS

UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install ./vllm-*glm53.dsv4.vision*.sm89sm120.cu130-*.whl \
  --torch-backend=cu130
```

The vLLM wheel installs the paired FlashInfer wheel from a pinned URL in the
same release. `SHA256SUMS` verifies both wheels downloaded above.

If the Aliyun mirror is slow, replace it with the Tencent Cloud or USTC PyPI
mirror.

---

## 3. DeepSeek-V4-Flash launch commands

### 3.1 SM89: 4× RTX 4090 48 GB

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
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --port 8000
```

This preserves the established SM89 deployment profile. It has been validated
with 8K, 32K, and 128K inputs, 512 output tokens, and four concurrent 8K
requests.

### 3.2 SM120: 4× RTX PRO 6000 96 GB

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
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
  --port 8000
```

---

## 4. DeepSeek-V4-Flash-Vision-Exp launch commands

### 4.1 SM89: 8× RTX 4090 48 GB

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
  '{"method":"dspark","num_speculative_tokens":3}' \
  --port 8000
```

4× RTX 4090 48 GB is not recommended for Vision-Exp with DSpark because the
draft model and KV cache do not have enough memory headroom. 8× RTX 4090
48 GB can enable DSpark with the command above; set
`--max-num-batched-tokens` to `4096`.

The 8-GPU command above is the recommended deployment configuration. This
round of SM89 runtime regression used 4× RTX 4090 48 GB without DSpark.

### 4.2 SM120: 4× RTX PRO 6000 96 GB

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
  '{"method":"dspark","num_speculative_tokens":3}' \
  --port 8000
```

## 5. GLM-5.3-Flash launch commands

### 5.1 SM89: 8× RTX 4090 48 GB

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

### 5.2 SM120: 4× RTX PRO 6000 96 GB

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

### 5.3 SM120 key parameters

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

## 6. GLM-5.3-Flash SM120 throughput baseline

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

## 7. Correctness validation

- DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, and GLM-5.3-Flash passed
  server startup on SM120.
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

## 8. License / provenance

The code is based on [vllm-project/vllm](https://github.com/vllm-project/vllm)
and remains under Apache-2.0. The FlashInfer wheel is based on
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
