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
| SM89 / Ada | 8× RTX 4090 48GB | 是 | 是 | 是 | 是 |
| SM120 / RTX Blackwell | 4× RTX PRO 6000 96GB | 是 | 是 | 是 | 是 |

---

## Changelog

### 2026-09-16

- 发布公开的 GHCR `vision11` Docker 镜像，支持免登录拉取，包含完整 CUDA JIT
  开发依赖和 CED 图片支持。
- 当前安装入口更新为包含 adaptive verification 和 CED 图片支持的 `vision11`
  vLLM wheel，配套 FlashInfer `vision2` 保持不变。
- 增加 8× RTX 4090 48GB 的 DeepSeek-V4.1-Flash 启动示例：上下文长度 `auto`、
  最多 4 条序列、显存利用率 `0.98`、batch token 预算 `4096`（可改为 `2048`）。

### 2026-09-12

- 在当前 `main` 源码中增加 DeepSeek-V4 / V4.1 的 SM89 / SM120 adaptive
  verification 适配，覆盖 device-ragged metadata、padding 和 FULL graph 回放。
  SM120 已完成算子及 V4.1 整模型验证。
  当时 Release 中的 vLLM wheel 更新为包含该适配的 `vision10`，DSpark 示例默认开启
  adaptive verification，并显式使用 FULL graph；`vision9` 及更早的 vLLM
  wheel 不包含该适配。配套 FlashInfer `vision2` 保持不变。
- 增加 DeepSeek-V4.1-Flash 原生模型、Engram、DSpark 和实验性 CED prefill
  支持，发布 SM89+SM120 `vision9` wheel。
- 在 4× RTX PRO 6000（SM120）上完成完整模型服务验证。
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
| vLLM | `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130` |
| SM89 | 4×/8× RTX 4090 48GB |
| SM120 | 4× RTX PRO 6000 Blackwell 96GB |

FlashInfer wheel 是 Python/JIT 源码包。首次遇到新的模型 shape 时会进行一次 JIT
编译，后续启动会复用缓存。

---

## 2. 快速安装

### 2.1 预编译 wheel

> **下文命令使用 `vision11` vLLM wheel、第 2.3 节的 GHCR 镜像，或本仓库当前
> `main` 源码**，包含 adaptive verification 和 CED 图片支持。
> 配套 FlashInfer `vision2` 保持不变。

最新 Release 沿用 `v0.28.1rc1-vision9-sm89-sm120-cu130` tag，
当前使用其中的 `vision11` vLLM 资产；配套 FlashInfer `vision2` 和依赖下载地址不变。
在 Python 3.12 虚拟环境中，直接通过下面的确切 wheel URL 安装：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.28.1rc1-vision9-sm89-sm120-cu130/vllm-0.28.1rc1.dev517%2Bglm53.dsv41.vision11.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=e1c8313e6a8b58ec3feecaffb37fc3fda8e61ba4ecff624853b24216b7eb97ed' \
  'transformers==5.16.1' 'triton==3.7.1'
```

`uv` 会校验 URL 中的 SHA256，并自动安装配套 FlashInfer `vision2`，
无需单独下载或安装 FlashInfer。已激活兼容环境时，只需执行 `uv pip install`。
该 wheel 的 Python 源码对应提交 `86d35745c8797c9d2524877fb235084a6a5f5a7d`，
复用已审计的 SM89 / SM120 原生二进制；Release tag 对应的源码归档未移动。

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

### 2.3 Docker 镜像（GHCR，公开免登录）

[GHCR 镜像页面](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pkgs/container/vllm-deepseek-v4-sm89)
提供当前 `vision11` 镜像，无需 `docker login` 即可拉取：

```bash
docker pull \
  ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

镜像平台为 `linux/amd64`，包含 `vision11` vLLM、配套 FlashInfer `vision2`、
PyTorch `2.13.0+cu130` 和 CUDA 13.0 JIT 工具链，支持当前 adaptive verification
和 CED 图片代码路径。入口为 `vllm serve`。

固定此版本时可将上述 `:tag` 替换为 `@sha256:...`，完整 manifest digest 为：

```text
sha256:af58a59d32d65fbed1785f265bc9b9969a1010b8e1c6a588087adf796021911e
```

[构建与校验记录](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/actions/runs/35093489937)：
已核对 11,177 个 vLLM/FlashInfer 文件，并通过 CUDA 头文件与编译链接、
图片/视频解码、SM89 冷缓存 JIT 检查；远端镜像及所有 15 个层已验证匿名访问。
这次云端构建未进行真机 GPU 或整模型推理复测。

### 2.4 历史 Docker 镜像（阿里云上海 ACR，vision7）

> **注意：** 以下历史 `vision7` 镜像不包含 DeepSeek-V4.1-Flash 或本次 adaptive
> 适配，不能直接使用下文默认开启 adaptive 的 DSpark 命令。请使用上面的 GHCR
> `vision11` 镜像、wheel 或源码安装流程；旧镜像的历史配置须关闭 adaptive。

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

## 3. DeepSeek-V4.1-Flash 启动命令

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

CED 当前是实验性的近似 prefill 路径。在已测长文本 prefill 中，按输入 token
数 / TTFT 估算的 prefill 代理指标相对关闭 CED 时约翻倍，但收益会随 prompt、
长度和运行配置变化。它不保证与非 CED 路径输出等价；部署前应按实际任务验证
质量，也不能把文本测试的收益推广到图片请求。

本仓库源码在 PR #112 的前端保护基础上增加了 CED 图片输入路径，按请求中的
全部图片范围处理单图和多图；图片与最后 128 个 query token 相交时，回放边界
向前扩展以保留完整图片，不会自动关闭 CED。可保留 DSpark 和 prefix caching。
按需设置 `--limit-mm-per-prompt '{"image":2}'`；每张图片必须能放入一个 prefill
chunk，扩展回放也必须满足 batch token 预算。视觉模型每卡预留
`128 + vision_max_n_token` 行回放状态，增加 TP 不会按比例减少这部分显存。
非图片模态、prompt embeddings 和 prompt logprobs 仍在进入引擎前被拒绝。

图片路径已有 CPU 回归、SM120 单卡小型权重 GPU 数值测试，以及真实缓存管理器
的图片命中和边界回退测试。2026-09-16 在 8× RTX 4090 48GB 上，保留 CED 和
DSpark，完成了单并发 8K / 32K / 128K 的图片与工具调用请求。输出质量仍需按
实际任务评估，请求完成不等同于所有图片答案正确。

`vision11` wheel 和 GHCR 镜像已包含 CED 图片支持，使用图片时可保留
`--hf-overrides '{"ced_prefill":true}'` 和 DSpark/adaptive。

### 3.1 DSpark 默认开启 adaptive verification

本文所有 DSpark 启动示例均显式开启 adaptive，并配置 FULL；这是 README
示例的默认设置，不改变 `SpeculativeConfig` 的通用默认值。使用 `vision11`
wheel、GHCR 镜像或当前 `main` 源码，并使用带 confidence head 的 DSpark checkpoint。
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

SM120 已覆盖 V4 / V4.1 算子、mixed prefill、padding、零草稿预算和 FULL 回放，
并完成 V4.1 整模型 serving；SM89 的 V4.1 图片与工具调用记录见本节和第 8 节。
这些运行结果不构成完整模型精度无损保证。

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
本次 adaptive 补丁的 Vision9 / FlashInfer Vision2 环境，仅作该环境的吞吐参考；
无同负载基线，不据此声称 adaptive 加速比例或质量无损。

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

### 3.3 SM89：8× RTX 4090 48GB

以下是 **8 张 48GB 版 RTX 4090** 的启动配置，不适用于普通 24GB 版。
保留 TP8/EP、FP8 KV、Engram CPU offload、CED 图片输入、DSpark 5 adaptive
和 FULL CUDA Graph。Engram CPU offload 与 checkpoint 加载还需要充足的主机
内存，不能只按 GPU 权重大小估算整机内存需求。

先按第 2.1 节安装 **vision11 wheel**，或按第 2.2 节安装包含 CED 图片支持的
当前 `main` 源码，也可使用第 2.3 节的 GHCR `vision11` 镜像；历史 ACR `vision7`
镜像不能用于本示例的 CED 图片路径。在该环境中启动：

```bash
source /path/to/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=8.9
export TORCH_CUDA_ARCH_LIST=8.9
export VLLM_USE_V2_MODEL_RUNNER=1
unset TRITON_PTXAS_BLACKWELL_PATH

vllm serve /path/to/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
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

- 若需要降低 prefill 峰值显存，将 `--max-num-batched-tokens 4096` 替换为
  `--max-num-batched-tokens 2048`，其余参数不变。此调整也会影响工作区、图捕获
  阶段显存和长输入分块数量，不能假设显存或吞吐按同一比例变化。
- **不设置显式 KV 字节预算**，由 `gpu-memory-utilization=0.98` 自动分配 KV。
  当前实现会先扣除常驻占用、临时峰值余量和 CUDA Graph 估算；剩余 2% 不是
  Graph 专用区，也不是运行时显存硬上限。
- `max-model-len=auto` 可能因容量不足而下调。若目标是 1M 上下文，确认启动日志
  最终长度为 `1048576`，且实际输入（含图片 tokens）与输出之和不超过该长度。
  `max-num-seqs=4` 是调度上限，不保证能同时容纳 4 条各 1M 的请求。
- 部署时按实际长请求和目标混合并发检查 OOM、KV 抢占重算、时延及输出质量。

## 4. DeepSeek-V4-Flash 启动命令

以下 DSpark 命令默认开启 adaptive verification，并显式使用 FULL CUDA Graph。

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

历史固定 DSpark 配置曾验证 8K、32K、128K 输入、512 输出及 4 并发 8K 输入。

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

以下 DSpark 示例默认开启 adaptive verification，并显式使用 FULL CUDA Graph。

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

历史 SM89 实机回归使用 4× RTX 4090 48GB，且未启用 DSpark。

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
  CED prefill 验证；在 8× RTX 4090 48GB（SM89）上完成保留 CED + DSpark 的
  单并发 8K / 32K / 128K 图片与工具调用请求。图片答案质量按实际任务单独评估。
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
