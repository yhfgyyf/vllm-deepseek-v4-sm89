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

- 在当前 `main` 源码中增加 DeepSeek-V4 / V4.1 的 SM89 / SM120 adaptive
  verification 适配，覆盖 device-ragged metadata、padding 和 FULL graph 回放。
  SM120 已完成算子及 V4.1 整模型验证；SM89 adaptive 实卡验证仍待完成。
  最新 Release 中的 vLLM wheel 已替换为包含该适配的 `vision10`，DSpark 示例默认开启
  adaptive verification，并显式使用 FULL graph；`vision9` 及更早的 vLLM
  wheel 不包含该适配。配套 FlashInfer `vision2` 保持不变。
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
| vLLM | `0.28.1rc1.dev517+glm53.dsv41.vision10.sm89sm120.cu130` |
| SM89 | 4×/8× RTX 4090 48GB |
| SM120 | 4× RTX PRO 6000 Blackwell 96GB |

FlashInfer wheel 是 Python/JIT 源码包。首次遇到新的模型 shape 时会进行一次 JIT
编译，后续启动会复用缓存。

---

## 2. 快速安装

### 2.1 预编译 wheel

> **Adaptive verification 需要本次 `vision10` vLLM wheel 或本仓库当前
> `main` 源码。** `vision9` 及更早的 vLLM whl 包不包含该适配，不能直接使用
> 下文默认开启 adaptive 的 DSpark 命令。配套 FlashInfer `vision2` 无需更换。

最新 Release 沿用 `v0.28.1rc1-vision9-sm89-sm120-cu130` tag，但其中的
vLLM 资产已更新为 `vision10`；配套 FlashInfer `vision2` 和依赖下载地址不变。
在 Python 3.12 虚拟环境中，直接通过下面的确切 wheel URL 安装：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130/vllm-0.28.1rc1.dev517%2Bglm53.dsv41.vision10.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=459a9638502de1cce8a1d1043bbe89e290afaa034a82008da92285bd1c9b72e5' \
  'transformers==5.16.1' 'triton==3.7.1'
```

`uv` 会校验 URL 中的 SHA256，并自动安装配套 FlashInfer `vision2`，
无需单独下载或安装 FlashInfer。已激活兼容环境时，只需执行 `uv pip install`。
该 wheel 基于源码提交 `1cf1104417873d65e4ad353ea904f602d16204f2` 构建，
复用已审计且未改动的 Vision9 原生二进制；Release tag 对应的源码归档未移动。

PyPI 依赖可按需通过 `--index-url` 使用镜像源；wheel 仍从上述 Release URL 下载。

### 2.2 从当前 main 源码安装（wheel 的替代方案）

以下流程以 editable 方式直接安装本仓库当前 `main` 源码，并编译 SM89 与
SM120 的 C++ / CUDA 扩展，不生成用于分发的新 wheel。
`requirements/cuda.txt` 会安装现有 Release 配套的 FlashInfer wheel。
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

uv pip install --no-build-isolation -e . --torch-backend=cu130

.venv/bin/python -I -c 'import vllm; print(vllm.__file__)'
```

最后的路径应指向当前源码目录下的 `vllm/__init__.py`，不能仍指向旧 wheel 的
`site-packages/vllm`。后续启动使用此 `.venv` 中的 `vllm` 命令；更新已有 checkout
时先同步本仓库 `main`，再执行上述安装。

### 2.3 Docker 镜像（阿里云上海 ACR）

> **注意：** 以下现有 Docker 镜像不包含 DeepSeek-V4.1-Flash 或本次 adaptive
> 适配，不能直接使用下文默认开启 adaptive 的 DSpark 命令。请使用 `vision10`
> wheel 或上述源码安装流程；旧镜像的历史配置须关闭 adaptive。

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

其余模型参数按旧镜像的历史配置设置。镜像已完成 SM120 GPU 运行验证和 SM89
目标编译验证，不代表它支持本次 adaptive 适配。

---

## 3. DeepSeek-V4.1-Flash 启动命令（SM120）

以下为 4× RTX PRO 6000 96GB 上验证过的文本服务配置。它使用 TP=4、EP、
FP8 KV、Engram CPU offload、DSpark 5（adaptive）和实验性 CED prefill；CPU offload
还需预留充足的主机内存。V4.1 使用 Model Runner V2，不支持切换到旧 V1 runner：

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
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
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'
```

使用上述新环境中的 `vllm serve`，并将模型路径替换为自己的 checkpoint。
`ptxas-blackwell --version` 必须显示 CUDA 13.1 工具链（已验证 13.1.80）；
这不改变 CUDA toolkit / PyTorch 的 cu130 配套。首次 JIT 和 graph capture
可能耗时，应在服务就绪及预热完成后测吞吐。

CED 当前是实验性的近似文本 prefill 路径。在已测长文本 prefill 中，按输入 token
数 / TTFT 估算的 prefill 代理指标相对关闭 CED 时约翻倍，但收益会随 prompt、
长度和运行配置变化。它不保证与非 CED 路径输出等价；部署前应按实际任务验证
质量。CED 不支持多模态输入、prompt embeddings 或 prompt logprobs。包含
issue #111 修复的源码会在正常 API 请求进入推理引擎前拒绝这些组合，返回客户端
错误，而不是让 worker 退出；已发布的 vision10 wheel 尚不包含这项前端保护。
需要图片或其他多模态输入时，请移除 `--hf-overrides '{"ced_prefill":true}'` 或改成
`'{"ced_prefill":false}'`；无需因此关闭 DSpark/adaptive。

SM120 已完成完整模型服务验证；SM89 本轮仅完成离线编译验证，未在 SM89
硬件上执行内核数值测试或完整模型服务验证。

### 3.1 DSpark 默认开启 adaptive verification

本文所有 DSpark 启动示例均显式开启 adaptive，并配置 FULL；这是 README
示例的默认设置，不改变 `SpeculativeConfig` 的通用默认值。先安装 `vision10`
wheel 或当前 `main` 源码，并使用带 confidence head 的 DSpark checkpoint。
保留模型路径、TP/EP、FP8 KV、Engram、Model Runner V2 和相应 CUDA 工具链设置。

| 参数 | 旧固定 DSpark 5 | 当前默认 adaptive |
|---|---|---|
| `enable_adaptive_verification` | `false` | `true` |
| `--compilation-config` | 未显式指定 | `'{"cudagraph_mode":"FULL"}'` |
| `num_speculative_tokens` / draft / rejection | `5` / `probabilistic` / `block` | 不变 |
| `--hf-overrides '{"ced_prefill":true}'` | 开启 CED | 可保留；CED 不是 adaptive 的必要条件 |

V4.1 的命令结尾为：

```bash
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'
```

不要增加 `--enforce-eager` 或将 graph 模式改为 `PIECEWISE`。CED 与 adaptive
是独立选项：保留 `ced_prefill=true` 可组合使用；移除该 override 即可采用普通
prefill。CED 的真实 prefill/mixed 步骤仍遵循其 EAGER 路由，纯 decode 可使用
FULL；显式 FULL 不表示所有 CED prefill 都在 graph 中执行。

若要恢复固定草稿验证，将 `enable_adaptive_verification` 改为 `false`；可保留
FULL 配置。各模型原有的草稿数量和采样方式不因开启 adaptive 而改变。

本次在 SM120 上验证了 V4 / V4.1 算子、mixed prefill、padding、零草稿预算和
FULL 回放，并完成 V4.1 整模型 serving；不将这些结果扩展为 SM89 实卡验证或
完整模型精度无损保证。

### 3.2 DeepSeek-V4.1-Flash 吞吐参考

2026-09-12，4× RTX PRO 6000 Blackwell Server Edition 96GB（SM120），
TP4/EP、FP8 KV、Engram CPU offload、CED + DSpark5 adaptive、FULL 配置下，
使用 `vllm bench serve` 测得：

| 输入 / 输出 tokens | 最大并发 | 请求数 | 聚合输出吞吐 | 平均 TTFT | 平均 TPOT |
|---|---:|---:|---:|---:|---:|
| 8,192 / 512 | 10 | 200 | **296.61 tokens/s** | 1.464 s | 30.64 ms |

服务已预热，随机输入、temperature=0、ignore-eos，200/200 请求成功，
前缀缓存命中为 0。吞吐为完整测试周期的总输出 tokens / 总耗时，包含
prefill 和 decode，不是单请求或纯 decode 速度。此为一次测量，来自已应用
本次 adaptive 补丁的 Vision9 / FlashInfer Vision2 环境，非新 Vision10 wheel
的重新压测；无同负载基线，不据此声称 adaptive 加速比例或质量无损。

同一已打补丁环境中的**单并发（C1）**结果如下。每种输入长度先预热 1 次，
再测 4 次正式请求；每次输出均为 512 tokens，12 次正式请求全部成功，
前缀缓存命中均为 0。

| 输入 / 输出 tokens | 平均 TTFT (ms) | Prefill 代理 (tokens/s) | Decode (tokens/s) | DSpark 接受率 | 平均接受草稿 tokens | 平均接受长度（含 +1） |
|---|---:|---:|---:|---:|---:|---:|
| 8,192 / 512 | 913.30 | 8,969.81 | 267.11 | 84.11% | 4.206 | 5.206 |
| 32,768 / 512 | 3,279.21 | 9,992.67 | 271.67 | 85.89% | 4.295 | 5.295 |
| 131,072 / 512 | 15,529.54 | 8,440.21 | 80.00 | 4.64% | 0.232 | 1.232 |

统计口径：

- Prefill 代理为输入 tokens / TTFT（秒）。TTFT 包含排队、调度、prefill
  和首 token 工作，因此不是纯 GPU 算子或引擎 prefill 吞吐。
- Decode 为 `1000 / TPOT（ms）`。TTFT 取 4 次正式请求的算术平均；
  两项吞吐均先逐次计算再取平均，不使用平均延迟的倒数。
- 接受率 = 接受草稿 tokens 总数 / 提出草稿 tokens 总数；平均接受草稿数
  = 接受草稿 tokens 总数 / 草稿轮次总数；vLLM 平均接受长度再加 1
  （额外 target token）。
  这三项使用 4 次正式请求的累计计数，不直接平均每次请求的比率。

上述为 vLLM 原生计数：proposed 在 adaptive 裁剪前记录，accepted 在
stop / 输出上限截断前记录；不等于裁剪后实际验证槽位的接受率或最终返回
客户端的 accepted-token 数。随机输入的这些结果不代表质量评估。

## 4. DeepSeek-V4-Flash 启动命令

以下命令已默认开启 adaptive。V4 本轮验证限于 SM120 算子和 FULL 回放，
未重新执行 V4 整模型 serving；SM89 adaptive 实卡验证仍待完成。

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
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

原固定 DSpark 配置曾验证 8K、32K、128K 输入、512 输出及 4 并发 8K 输入；
这些历史结果不代表新增 adaptive 参数后的 SM89 实卡验证。

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
  '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

---

## 5. DeepSeek-V4-Flash-Vision-Exp 启动命令

以下 DSpark 示例也默认开启 adaptive；本轮未进行 Vision-Exp 多模态 adaptive
整模型回归，部署前需验证实际输入，历史多模态结果不作为本次验证证据。

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
  '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
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
  '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --port 8000
```

历史 SM89 实机回归使用 4× RTX 4090 48GB，且未启用 DSpark；上述 8 卡
adaptive 命令不属于已完成的 SM89 实机验证范围。

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
