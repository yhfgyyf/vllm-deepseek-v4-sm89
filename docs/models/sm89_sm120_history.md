# SM89 / SM120 release and measurement history

<!-- markdownlint-disable MD060 -->

These sections were moved from the READMEs on 2026-11-12. Dates, software
versions and measurement definitions belong to the original runs; they do
not qualify the current source or its launch defaults. See the
[usage guide](../../README_EN.md) for installation and serving, and the
[L20 configuration](deepseek_v41_l20_baseline.md) for the later TP8/K7 record.

## Historical release support matrix

| GPU architecture | Validated GPU | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash | DeepSeek-V4-Flash-Vision-Exp | GLM-5.3-Flash |
| --- | --- | ---: | ---: | ---: | ---: |
| SM89 / Ada | 8× RTX 4090 48 GB | Yes | Yes | Yes | Yes |
| SM120 / RTX Blackwell | 4× RTX PRO 6000 96 GB | Yes | Yes | Yes | Yes |

---

## Changelog

### 2026-11-12

- Update the L20 example to TP8/PP1, draft TP8, DSpark7, 0.96 automatic KV and capture48, retaining all Q optimizations.
- Add the matched-input L20 K7 prefill/decode and acceptance results, preserving the historical K5 comparison and synthetic single-request scope.
- Add opt-in SM120 B12X small-batch Attention/mHC/Linear adapters with native fallbacks; GPU integration qualification is pending.
- This is a source update, not a new wheel or Docker release.

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
  adaptive verification, and CED image-input support.
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

### 2026-09-03

- Updated the SM89+SM120 Docker image to fix the problem reported in
  [Issue #95](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/95).

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

## DeepSeek-V4.1-Flash throughput reference

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
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
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

## GLM-5.3-Flash SM120 throughput baseline

These results are retained from the project's first complete SM120 benchmark.
The setup used 4× RTX PRO 6000, TP=4, FP8 KV, MTP=5, CUDA Graph,
`block-size=2304`, an 8192-token chunked-prefill budget, and prefix caching
disabled. Each input length was run five times with 512 output tokens; all 10
requests completed.

| Input → output | Mean / median prefill TPS | Mean / median decode TPS | Mean TTFT |
| --- | ---: | ---: | ---: |
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
| --- | ---: | ---: | ---: |
| 256K / 512 | 27.865 s; 9,407.50 tok/s | 31.649 s; 8,282.96 tok/s | -11.95% |
| 784K / 512 | 101.473 s; 7,911.64 tok/s | 112.685 s; 7,124.45 tok/s | -9.95% |

Repeating the same prompt produced:

| Input | Prefix-cache hit rate | Repeat-request TTFT |
| --- | ---: | ---: |
| 256K | 98.4375% | 1.308 s |
| 784K | 99.5855% | 3.117 s |

---

## Historical correctness validation

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

## Legacy vision7 container (Chinese)

> **注意：** 以下历史 `vision7` 镜像不包含 DeepSeek-V4.1-Flash 或本次 adaptive
> 适配，不能直接使用下文默认开启 adaptive 的 DSpark 命令。请使用上面的 GHCR
> `vision11` 镜像、wheel 或源码安装流程；参见[安装指南](../../README.md#2-安装)。旧镜像的历史配置须关闭 adaptive。

镜像地址：

```text
crpi-6uvuk5v2ux77q4n9.cn-shanghai.personal.cr.aliyuncs.com/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc0-vision7-sm89-sm120-cu130
```

| 项目 | 值 |
| --- | --- |
| 平台 | Linux x86_64 / `linux/amd64` |
| vLLM | `0.28.1rc0.dev293+gcb7a435391.glm53.dsv4.vision7.sm89sm120.cu130` |
| FlashInfer | `0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213` |
| PyTorch / CUDA | `2.13.0+cu130` / CUDA 13.0 JIT toolchain |
| 镜像大小 | 9.27 GB 未压缩；约 4.33 GB Registry 传输量 |
| Digest | `sha256:1a120648d54ebb90aad91bef2a620e5779c627426ad9f0a34367303b4cbe659a` |

直接拉取镜像：

```bash
docker pull \
  crpi-6uvuk5v2ux77q4n9.cn-shanghai.personal.cr.aliyuncs.com/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc0-vision7-sm89-sm120-cu130
```

镜像入口是 `vllm serve`。使用后文启动参数时，将命令开头的
`vllm serve /path/to/model` 替换为：

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  crpi-6uvuk5v2ux77q4n9.cn-shanghai.personal.cr.aliyuncs.com/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc0-vision7-sm89-sm120-cu130 \
  /models/model-directory
```

其余模型参数按旧镜像的历史配置设置。镜像已完成 SM120 GPU 运行验证和 SM89
目标编译验证，不代表它支持本次 adaptive 适配。

---
