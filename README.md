# DeepSeek-V4-Flash on SM89 (Ada / RTX 4090) — vLLM fork

<!-- markdownlint-disable MD060 -->

> English version: [`README_EN.md`](README_EN.md)
> 本仓库是 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork，跟进当前上游 `main`，并增加 SM89/Ada 的 FlashInfer sparse MLA 适配。

把 vLLM 的 **DeepSeek-V4-Flash** 推理从 SM90/SM100/SM120 扩展到 **SM89(Ada Lovelace：RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada)**。已在 **4× RTX 4090 (48GB)** 上完整验证:环境搭建 → 算子测试 → 启动 → 推理 → 性能/工具调用 全部通过。

## Changelog

### 2026-08-02

- 适配最新 `DeepSeek-V4-Flash-0731` 模型，并在 4× RTX 4090 上完成短上下文、长上下文和 GSM8K 准确性测试。
- 修复 FlashInfer SM89 sparse MLA decode 精度问题：为每个 MMA accumulator 正确恢复分布式 query/KV UE8M0 scale，处理边界编码并支持 page block size 256。
- 安装时请使用[最新 Release](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/latest)中配套的 vLLM 和 FlashInfer wheel，并在同一次依赖解析中安装。

### 2026-07-10

- SM89 sparse MLA 的 prefill/decode 路径切换到 **FlashInfer 0.6.14 sparse MLA JIT fork**；release 同时提供匹配的 FlashInfer wheel，运行时会拒绝未包含 SM89 补丁的官方包。
- 修复 Lightning Indexer 仅按“是否安装 DeepGEMM”生成 scheduler metadata 的问题；现在按当前 GPU 的实际 DeepGEMM 支持能力判断，SM89 无需卸载 DeepGEMM 环境即可避开不支持的 metadata 路径。
- wheel 构建脚本会优先使用显式 `VLLM_VERSION_OVERRIDE`，自动推导版本时忽略历史 CUDA/SM Release tag，避免 `setuptools_scm` 在编译前解析失败。
- 本次不更新 `confidence_head`，也不包含 per-request adaptive ℓ；DSpark 继续使用固定 `ℓ=6`。
- 4× RTX 4090、TP=4、单并发、每组 5 次均 `5/5` 成功。`8K / 32K / 128K -> 1K` 的 Prefill TPS 为 **3515.72 / 4881.18 / 3812.00**，Decode TPS 为 **286.82 / 344.63 / 313.57**。

### 2026-07-06

- 增加 **SM80/A800 测试性适配**说明。SM80 路径仅用于自测和实验，不代表生产级支持。
- 在 4× A800 上完成 DeepSeek-V4-Flash DSpark 推测解码冒烟与吞吐测试，测试参数为 `method=dspark`、`num_speculative_tokens=6`、`draft_sample_method=greedy`，并开启 FlashInfer sampler、sparse MLA warmup、`max-num-batched-tokens=16384`。
- 只记录 decode 侧结果：8k 输入、1k 输出、单并发为 **229.8 tok/s/req**；32k 输入、1k 输出、单并发为 **274.2 tok/s/req**。对应无 DSpark `mbt16k` 基线分别为 57.6 和 58.1 tok/s/req。

### 2026-07-01

- 完成 **DeepSeek-V4-Flash-DSpark** 模型适配，支持 `method=dspark` 推测解码；当前 release wheel 打包目标切换为 **CUDA 13.0 工具链 + torch 2.11.0+cu130**，并已在 CUDA 13.x / 4× RTX 4090 上验证 `vllm serve`、tool call 和 vLLM bench。
- DSpark 单并发 `8K / 32K / 128K` 输入、`1K` 输出均 `10/10` 成功;decode 折算为 **355 / 336 / 219 tok/s**。相比非 DSpark 源模型基线 decode **~82 tok/s**，分别提升约 **4.3× / 4.1× / 2.7×**。
- 推荐 DSpark 服务配置:`gpu-memory-utilization=0.96`、`max-model-len=262144`、`max-num-batched-tokens=2048`、`max-num-seqs=4`、`block-size=256`、`kv-cache-dtype=fp8_ds_mla`。

---

## SM80/A800 测试性适配

SM80/A800 路径已经可以用于 DeepSeek-V4-Flash + DSpark 推测解码自测，但仍是测试性适配，不是生产支持承诺。当前已验证的 A800 配置使用：

```bash
--speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'
```

并开启 FlashInfer sampler、sparse MLA warmup、`max-num-batched-tokens=16384`。

| 输入 -> 输出 | 并发 | DSpark decode | 无 DSpark decode | decode 提升 |
|---|---:|---:|---:|---:|
| 8,192 -> 1,024 | 1 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 -> 1,024 | 1 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

这里仅列 decode 结果；SM80 长上下文 prefill 仍需单独评估。

---

## 1. 背景:为什么需要这个 fork

DeepSeek-V4-Flash 使用 DeepSeek 稀疏注意力(DSA / Lightning Indexer)+ FP4 专家 MoE + mHC。本 fork 将 FlashInfer 的 SM120 sparse MLA JIT 内核移植到 **SM89**，并保留 Ada 所需的 Triton/torch 辅助算子 fallback。

| 子系统 | 上游(SM90/100) | SM89(本 fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA / FlashInfer sparse | **FlashInfer 0.6.14 sparse MLA JIT** |
| Lightning Indexer(FP8 MQA logits) | DeepGEMM | **按硬件能力门控的 DeepGEMM / fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **SM89 兼容路径** |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE(FP4 专家) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16**(FP4→FP16 反量化) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**硬件事实**:Ada 有 FP8 张量核，但**没有 FP4 张量核、没有硬件 microscaling MMA**，所以 FP4 MoE 只能走 Marlin 反量化(比原生 FP4 MMA 慢)。

### SM89 相关改动

- `flashinfer-python==0.6.14` 的 sparse MLA JIT 路径开放到精确 capability `8.9`，其它 8.x GPU 仍拒绝。
- `vllm/v1/attention/backends/mla/indexer.py` 按 `is_deep_gemm_supported()` 生成 scheduler metadata，避免 SM89 误走 DeepGEMM metadata API。
- `vllm/models/deepseek_v4/compressor.py` 和 `vllm/utils/import_utils.py` 在 SM89 上选择现有 Triton/torch fallback，避开 SM90+ CuTe-DSL 指令。
- MXFP4 MoE 在 SM89 上继续选择 Marlin，不会误选 Blackwell-only DeepGEMM FP4。

---

## 2. 已验证环境

| 项 | 版本 |
|---|---|
| GPU | 4× RTX 4090 (48GB) · compute capability **8.9** |
| 驱动 / CUDA toolkit | 595.x / **CUDA 13.0**(nvcc 13.0) |
| Python | 3.12 |
| torch | **2.11.0+cu130** |
| FlashInfer | **0.6.14 SM89 sparse MLA fork** |
| vLLM | 本 fork 的 CUDA 13.0 / CPython 3.12 wheel，只为 SM89/Ada 编译 |

---

## 3. 快速安装(预编译 wheel，免编译)

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install torch==2.11.0 flashinfer-cubin==0.6.13 --torch-backend=cu130
gh release download --repo yhfgyyf/vllm-deepseek-v4-sm89 \
  --pattern 'flashinfer_python-0.6.14*sm89*.whl' \
  --pattern 'vllm-*.cu130-cp312-cp312-linux_x86_64.whl' \
  --dir /tmp/vllm-sm89-release
uv pip install /tmp/vllm-sm89-release/flashinfer_python-0.6.14*sm89*.whl
uv pip install /tmp/vllm-sm89-release/vllm-*.cu130-cp312-cp312-linux_x86_64.whl \
  --torch-backend=cu130
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

**已验证过的环境**:

- **Python 3.12** · Linux x86_64
- **4× RTX 4090 (SM89/Ada, 48GB)** · 驱动 595.x · CUDA toolkit 13.0
- **torch 2.11.0+cu130**
- **FlashInfer 0.6.14 SM89 fork**；官方 0.6.14 不包含本 release 所需的 SM89 sparse MLA JIT 补丁
- `flashinfer-cubin==0.6.13`；运行前设置 `FLASHINFER_DISABLE_VERSION_CHECK=1`，SM89 sparse MLA 仍由 0.6.14 fork 源码 JIT
- wheel 使用 `TORCH_CUDA_ARCH_LIST=8.9+PTX` 编译，面向 Ada/SM89

---

## 4. 源码安装(clone 本仓库编译)

### 4.1 Python 环境 + torch cu130

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --torch-backend=cu130 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

运行 SM89 sparse MLA 前，还需按第 3 节安装同一 release 中的 FlashInfer 0.6.14 SM89 wheel。

### 4.2 Rust 工具链(vLLM 构建需要 Rust frontend)

```bash
export RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y --default-toolchain 1.95 --profile minimal
source "$HOME/.cargo/env"
# ~/.cargo/config.toml 配 crates 镜像:
#   [source.crates-io]
#   replace-with = "rsproxy-sparse"
#   [source.rsproxy-sparse]
#   registry = "sparse+https://rsproxy.cn/index/"
```

### 4.3 clone 本仓库

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
```

### 4.4 编译 / 打包 CUDA 13.0 wheel(只为 Ada 8.9 编译)

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

> Ada 不支持 DeepGEMM kernel，但无需手工卸载 DeepGEMM 包；vLLM 会按硬件能力关闭其 scheduler metadata 路径。
> 如果要为 SM80/A100/A800 构建 wheel，把 `TORCH_CUDA_ARCH_LIST` 改成 `8.0`。
> wheel 文件名遵循 release 命名:`vllm-0.23.1rc1.dev145+g<commit>.cu130-cp312-cp312-linux_x86_64.whl`。

---

## 5. 算子级自检(无需起完整模型)

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

## 6. 部署(vllm serve)

### 6.1 源模型

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

### 6.2 DSpark 推测解码模型

```bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
vllm serve /path/to/DeepSeek-V4-Flash-DSpark \
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
  --speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}' \
  --trust-remote-code --port 8000
```

启动成功标志:`Application startup complete.`，日志里能看到 `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`。

---

## 7. 测试结果(4× RTX 4090)

### 7.1 推理正确性

```text
Q: 用一句话介绍长城。
A: 长城是中国古代为抵御北方游牧民族入侵而修筑的、横跨多个朝代、绵延数千公里的
   军事防御工程，也是世界文化遗产中象征中华民族坚韧精神的伟大奇迹。   (finish_reason=stop)
```

### 7.2 最大上下文(KV cache)

| max-model-len | max-num-seqs | GMU | GPU KV cache | 单请求并发 | 启动 |
|---|---|---|---|---|---|
| 262,144 (256K) | 16 | 0.97 | 972,374 tok | 3.71x | ✅ |
| 786,432 (768K) | 16 | 0.97 | 1,220,509 tok | 1.55x | ✅ |
| **1,048,576 (1M)** | 4 | 0.97 | **1,243,644 tok** | 1.19x | ✅(模型架构上限) |

实测能跑完的最长输入:**768K(786,000 token，prefill ~147s)**。1M 可启动、kernel 数值正确，但**满 1M 单次 prefill 极慢(>10 min)，不实用**。日常推荐 **128K~256K**。

输入长度 sweep(256K 配置，均成功):64K(25s)/128K(37s)/200K(74s)/262K(71s)。

### 7.3 非 DSpark decode 性能(4× RTX 4090，单并发)

| 输入 | Decode |
|---|---:|
| 8,192 | **~82 tok/s** |
| 32,768 | **~82 tok/s** |

Decode 主要受 Marlin MoE 反量化开销影响(Ada 无 FP4 张量核)。

### 7.4 Tool call(`deepseek_v4` parser)

```text
Q: 北京今天天气怎么样？请用摄氏度回答。  (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

### 7.5 DSpark 推测解码(CUDA 13.x / torch cu130, 单并发)

vLLM 自带 `vllm bench serve`，random dataset 固定长度，`max-concurrency=1`，每组 5 次，输出 1024 token。

稳定配置:

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

| 输入 → 输出 | 成功 | Prefill TPS | Decode TPS |
|---|---:|---:|---:|
| 8,192 → 1,024 | 5/5 | **3515.72** | **286.82** |
| 32,768 → 1,024 | 5/5 | **4881.18** | **344.63** |
| 131,072 → 1,024 | 5/5 | **3812.00** | **313.57** |

折算口径：`Prefill TPS = input_tokens / mean_TTFT`；`Decode TPS = 1000 / mean_TPOT(ms)`。

## 8. 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)(Apache-2.0)及其 PR #41834。本 fork 沿用同协议。AI 辅助完成，人工验证。
