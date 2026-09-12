# DeepSeek-V4.1-Flash、DeepSeek-V4-Flash / Vision 与 GLM-5.3-Flash on SM89 / SM120 — vLLM fork

<!-- markdownlint-disable MD060 -->

> English version: [`README_EN.md`](README_EN.md)
>
> 本仓库基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)，用于在
> SM89/Ada 与 SM120/RTX Blackwell 上运行 DeepSeek-V4.1-Flash、
> DeepSeek-V4-Flash、DeepSeek-V4-Flash-Vision-Exp 和 GLM-5.3-Flash。

当前代码为 vLLM `0.28.1rc1.dev517` 开发版，配套 FlashInfer `0.6.18`。已验证配置包括
**4×/8× RTX 4090 48GB** 和 **4× RTX PRO 6000 Blackwell 96GB**。

## 支持矩阵

| GPU 架构 | 已验证 GPU | DeepSeek-V4.1-Flash | DeepSeek-V4-Flash | DeepSeek-V4-Flash-Vision-Exp | GLM-5.3-Flash |
|---|---|---:|---:|---:|---:|
| SM89 / Ada | 8× RTX 4090 48GB | 待验证 | 是 | 是 | 是 |
| SM120 / RTX Blackwell | 4× RTX PRO 6000 96GB | 是 | 是 | 是 | 是 |

---

## Changelog

### 2026-09-12

- 增加 DeepSeek-V4.1-Flash 原生模型、Engram、DSpark 和实验性 CED prefill
  支持，发布 SM89+SM120 `vision9` wheel。
- 在 4× RTX PRO 6000（SM120）上完成完整模型服务验证；SM89 本轮仅完成
  离线编译验证，未在 SM89 硬件上执行内核数值测试或完整模型服务验证。
- 已测长文本 prefill 中，按输入 token 数 / TTFT 估算的 prefill 代理指标相对
  关闭 CED 时约翻倍；实际收益取决于 prompt 和运行配置。

### 2026-09-07

- 更新至 vLLM `0.28.1rc1` 开发版，发布 SM89+SM120 `vision8` wheel。
- 修复 [Issue #98](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/98)
  中 DeepSeek-V4 C128 稀疏注意力索引不连续导致的崩溃（PR #96）。
- 修复 persistent Top-K 候选缓冲区溢出导致的索引选择错误（PR #97），
  并修复 CUDA Triton block-FP8 路径的 E8M0 scale 兼容性。

### 2026-09-03

- 更新 SM89+SM120 Docker 镜像，修复
  [Issue #95](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/95) 反馈的问题。

### 2026-09-02

- 修复 [Issue #90](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/90)，
  GitHub Release 中有问题的 vLLM wheel 已替换为 `dev293` / `vision7` 构建。
- 重新发布包含 DeepSeek-V4-Flash-Vision-Exp 支持的 SM89+SM120 Docker
  镜像，并完成 SM120 GPU、模型模块和视频解码验证。

### 2026-09-01

- 增加 DeepSeek-V4-Flash-Vision-Exp 支持；4× RTX PRO 6000（SM120）和
  8× RTX 4090 48GB（SM89）均已支持。
- 增加 DeepSeek-V4-Flash-Vision-Exp 的 SM89 部署说明和 8 卡 DSpark
  启动命令。
- 增加 8× RTX 4090 48GB 的 GLM-5.3-Flash 启动命令，并将 DeepSeek-V4-Flash
  的 SM89 示例改为 `--max-model-len auto`。

### 2026-08-31

- 社区用户已在 8× RTX 4090 48GB（SM89）上成功运行 GLM-5.3-Flash，参见
  [Issue #74 的验证记录](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993)。
- 发布统一的 SM89+SM120 Docker 镜像到阿里云上海 ACR。

### 2026-08-30

- 将主分支更新到 vLLM `v0.28.1rc0-110` 基线，并保留本仓库已经验证的
  DeepSeek-V4-Flash SM89 支持。
- 增加 RTX PRO 6000（SM120）上的 DeepSeek-V4-Flash 和 GLM-5.3-Flash 支持。
- 发布统一的 SM89+SM120 vLLM wheel，以及配套的 FlashInfer `0.6.18` wheel。
- Python 包版本、Git 源码和 release 制品使用同一组版本标记。

早期 SM89 版本和对应环境仍保留在
[历史 Releases](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases) 中。

---

## 1. 已验证环境

| 项目 | 版本 / 配置 |
|---|---|
| 操作系统 | Linux x86_64 |
| Python | 3.12 |
| CUDA toolkit | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1（`ptxas-blackwell` CUDA 13.1） |
| Transformers | 5.16.1 |
| FlashInfer | `0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213` |
| vLLM | `0.28.1rc1.dev517+glm53.dsv41.vision9.sm89sm120.cu130` |
| SM89 | 4×/8× RTX 4090 48GB |
| SM120 | 4× RTX PRO 6000 Blackwell 96GB |

FlashInfer wheel 是 Python/JIT 源码包。首次遇到新的模型 shape 时会进行一次 JIT
编译，后续启动会复用缓存。

---

## 2. 快速安装

### 2.1 预编译 wheel

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

gh release download v0.28.1rc1-vision9-sm89-sm120-cu130 \
  --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.18+glm53.dsv41.vision2.sm89sm120.cu130.pt213-*.whl' \
  --pattern 'vllm-*glm53.dsv41.vision9.sm89sm120.cu130-*.whl' \
  --pattern SHA256SUMS \
  --dir /tmp/vllm-sm89-sm120-vision9-release

cd /tmp/vllm-sm89-sm120-vision9-release
sha256sum -c SHA256SUMS

UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
uv pip install ./vllm-*glm53.dsv41.vision9.sm89sm120.cu130-*.whl \
  --torch-backend=cu130
uv pip install 'transformers==5.16.1' 'triton==3.7.1'
```

vLLM wheel 会通过锁定的依赖 URL 安装同一 Release 中配套的 FlashInfer wheel。
下载到本地的两个 wheel 均由 `SHA256SUMS` 校验。

如果阿里云镜像速度较慢，可以替换为腾讯云或中科大 PyPI 镜像。

### 2.2 从源码完整构建

以下流程会重新编译 vLLM 的 C++ / CUDA 扩展，并同时生成 SM89 与 SM120
目标代码。`requirements/cuda.txt` 会安装本次 Release 配套的 FlashInfer wheel。
构建前需准备 CUDA 13.0 toolkit、C++ 编译器和支持 Rust 2024 edition 的
Rust/Cargo。

```bash
git clone --branch main \
  https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89

uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
uv pip install -r requirements/cuda.txt --torch-backend=cu130
uv pip install 'transformers==5.16.1' 'triton==3.7.1'

export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST='8.9;12.0'
export MAX_JOBS=4

VLLM_VERSION_OVERRIDE='0.28.1rc1.dev517+glm53.dsv41.vision9.sm89sm120.cu130' \
  uv build --wheel --no-build-isolation
uv pip install --no-build-isolation \
  dist/vllm-0.28.1rc1.dev517+glm53.dsv41.vision9.sm89sm120.cu130-*.whl
```

### 2.3 Docker 镜像（阿里云上海 ACR）

> **注意：** 以下现有 Docker 镜像不包含 DeepSeek-V4.1-Flash 支持；V4.1
> 请使用本次 Release wheel 或上述源码构建流程。

镜像地址：

```text
crpi-6uvuk5v2ux77q4n9.cn-shanghai.personal.cr.aliyuncs.com/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc0-vision7-sm89-sm120-cu130
```

| 项目 | 值 |
|---|---|
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

其余模型参数保持不变。镜像已完成 SM120 GPU 运行验证和 SM89 目标编译验证。

---

## 3. DeepSeek-V4.1-Flash 启动命令（SM120）

以下为 4× RTX PRO 6000 96GB 上验证过的文本服务配置。它使用 TP=4、EP、
FP8 KV、Engram CPU offload、DSpark 5 和实验性 CED prefill；CPU offload
还需预留充足的主机内存。V4.1 使用 Model Runner V2，不支持切换到旧 V1 runner：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_CUDA_ARCH_LIST=12.0
export VLLM_USE_V2_MODEL_RUNNER=1
export TRITON_PTXAS_BLACKWELL_PATH="$(
  "$VIRTUAL_ENV/bin/python" -c \
    'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas-blackwell")'
)"
"$TRITON_PTXAS_BLACKWELL_PATH" --version

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
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

`ptxas-blackwell --version` 必须显示 CUDA 13.1 工具链。

CED 当前是实验性的近似文本 prefill 路径。在已测长文本 prefill 中，按输入 token
数 / TTFT 估算的 prefill 代理指标相对关闭 CED 时约翻倍，但收益会随 prompt、
长度和运行配置变化。它不保证与非 CED 路径输出等价；部署前应按实际任务验证
质量。CED 不支持多模态输入、prompt embeddings 或 prompt logprobs，因此启用
CED 时不要发送图片或其他多模态内容。

SM120 已完成完整模型服务验证；SM89 本轮仅完成离线编译验证，未在 SM89
硬件上执行内核数值测试或完整模型服务验证。

## 4. DeepSeek-V4-Flash 启动命令

### 4.1 SM89：4× RTX 4090 48GB

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

该配置保持原有 SM89 部署口径，已验证 8K、32K、128K 输入，512 输出，以及
4 并发 8K 输入。

### 4.2 SM120：4× RTX PRO 6000 96GB

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

## 5. DeepSeek-V4-Flash-Vision-Exp 启动命令

4× RTX 4090 48GB 可以运行 DeepSeek-V4-Flash-Vision-Exp，但不建议启用
DSpark：目标模型和 draft model 会占用大部分显存，剩余空间不足以提供实用的
draft KV cache。8× RTX 4090 48GB 可以启用 DSpark，建议将
`--max-num-batched-tokens` 设置为 `4096`。

### 5.1 SM120：4× RTX PRO 6000 96GB

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

### 5.2 SM89：8× RTX 4090 48GB（DSpark）

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

上述 8 卡命令是推荐部署配置；本轮 SM89 实机回归使用 4× RTX 4090 48GB，
且未启用 DSpark。

## 6. GLM-5.3-Flash 启动命令

### 6.1 SM89：8× RTX 4090 48GB

以下参数参考社区用户在
[Issue #74](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993)
中使用官方 FP8 权重完成的部署验证：

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

### 6.2 SM120：4× RTX PRO 6000 96GB

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

### 6.3 SM120 关键参数

| 参数 | 推荐值 | 说明 |
|---|---:|---|
| `--tensor-parallel-size` | 4 | 4 张 96GB RTX PRO 6000 |
| `--kv-cache-dtype` | `fp8` | 降低长上下文 KV cache 显存占用 |
| `--block-size` | `2304` | GLM 混合 cache 使用的模型级 block size |
| `--max-model-len` | `auto` | 根据当前显存和启动配置自动计算容量 |
| `--max-num-seqs` | 4 | 已验证的并发上限配置 |
| `--max-num-batched-tokens` | 8192 | chunked prefill token budget |
| `--gpu-memory-utilization` | 0.97 | 为模型、CUDA Graph 和 KV cache 分配显存 |
| `--speculative-config` | MTP 5 | 启用五 token MTP 推测解码 |
| `--enable-prefix-caching` | 开启 | 复用重复或共享前缀 |
| `glm45` / `glm47` | reasoning / tool parser | 推理输出和工具调用解析 |

---

## 7. GLM-5.3-Flash SM120 吞吐基线

以下数据来自本项目第一版 SM120 完整基准，使用 4× RTX PRO 6000、TP=4、
FP8 KV、MTP=5、CUDA Graph、`block-size=2304`、chunked prefill 8192，关闭
prefix cache。每个输入长度运行 5 次，输出均为 512 tokens，10/10 请求成功。

| 输入 → 输出 | Prefill TPS 均值 / 中位数 | Decode TPS 均值 / 中位数 | 平均 TTFT |
|---|---:|---:|---:|
| 8,192 → 512 | **9,919.34 / 9,904.75** | **158.47 / 175.25** | 825.88 ms |
| 32,768 → 512 | **9,841.51 / 9,843.08** | **199.00 / 208.74** | 3,329.62 ms |

计算口径：

- `Prefill TPS = input tokens / TTFT`
- `Decode TPS = 511 / (E2E - TTFT)`
- TTFT 是端到端首 token 延迟近似值；decode 会随 MTP 接受率波动。

这组数字用于保留第一版 SM120 的可比基线，不代表本次 wheel 在所有输入内容、
驱动版本或显存配置下都能得到相同结果。

### 长上下文吞吐

首版 SM120 release 还记录了 256K / 784K 长上下文、512 输出、prefix cache
相关数据：

| 输入 / 输出 | 8192 chunk 基线 | 4096 chunk 稳态 | Prefill 变化 |
|---|---:|---:|---:|
| 256K / 512 | 27.865s；9,407.50 tok/s | 31.649s；8,282.96 tok/s | -11.95% |
| 784K / 512 | 101.473s；7,911.64 tok/s | 112.685s；7,124.45 tok/s | -9.95% |

重复相同 prompt 时：

| 输入 | Prefix 命中率 | 重复请求 TTFT |
|---|---:|---:|
| 256K | 98.4375% | 1.308s |
| 784K | 99.5855% | 3.117s |

---

## 8. 正确性验证

- DeepSeek-V4.1-Flash、DeepSeek-V4-Flash、DeepSeek-V4-Flash-Vision-Exp
  与 GLM-5.3-Flash
  在 SM120 上均通过服务启动。
- DeepSeek-V4.1-Flash 在 SM120 上通过文本服务、工具调用、DSpark 和实验性
  CED prefill 验证；SM89 本轮仅完成离线编译验证，未在 SM89 硬件上执行
  内核数值测试或完整模型服务验证。
- DeepSeek-V4-Flash 与 GLM-5.3-Flash 在 SM120 上通过 8K/32K 输入和
  512 输出测试。
- DeepSeek-V4-Flash 在 SM89 上通过 8K/32K/128K、4 并发、工具调用和 UTF-8
  输出测试。
- DeepSeek-V4-Flash-Vision-Exp 在 4× RTX 4090 48GB（SM89）上通过服务启动、
  单图、多图、视频、工具调用、8K/32K 输入和 UTF-8 输出测试；该 4 卡配置未启用
  DSpark。
- GLM-5.3-Flash 已由社区用户在 8× RTX 4090 48GB（SM89）上成功启动并完成
  推理，参见 [Issue #74](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/issues/74#issuecomment-5474430993)。
- 两个模型均通过中英文、多语言字符、流式与非流式工具调用检查。

---

## 9. License / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)，沿用
Apache-2.0 协议。FlashInfer wheel 基于
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)。
