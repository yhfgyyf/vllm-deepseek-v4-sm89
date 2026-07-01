# DeepSeek-V4-Flash on SM89 (Ada / RTX 4090) — vLLM fork

> English version: [`README_EN.md`](README_EN.md)

> 本仓库是 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork，分支已包含 **PR #41834**(SM120 可移植 Triton 路径)+ **SM89/Ada 适配 commit**。

把 vLLM 的 **DeepSeek-V4-Flash** 推理从 SM90/SM100/SM120 扩展到 **SM89(Ada Lovelace：RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada)**。已在 **4× RTX 4090 (48GB)** 上完整验证:环境搭建 → 算子测试 → 启动 → 推理 → 性能/工具调用 全部通过。

## Changelog

### 2026-07-01

- 完成 **DeepSeek-V4-Flash-DSpark** 模型适配，支持 `method=dspark` 推测解码；当前 release wheel 打包目标切换为 **CUDA 13.0 工具链 + torch 2.11.0+cu130**，并已在 CUDA 13.x / 4× RTX 4090 上验证 `vllm serve`、tool call 和 vLLM bench。
- DSpark 单并发 `8K / 32K / 128K` 输入、`1K` 输出均 `10/10` 成功;decode 折算为 **355 / 336 / 219 tok/s**。相比非 DSpark 源模型基线 decode **~82 tok/s**，分别提升约 **4.3× / 4.1× / 2.7×**。
- 推荐 DSpark 服务配置:`gpu-memory-utilization=0.96`、`max-model-len=262144`、`max-num-batched-tokens=2048`、`max-num-seqs=4`、`block-size=256`、`kv-cache-dtype=fp8_ds_mla`。在 4× RTX 4090、`fp8_ds_mla`、`block-size=256` 的实测配置下，GPU KV cache 总容量约 **972,374 tokens**；这是所有活跃请求共享的 KV cache token 池，不是单请求上下文长度。

---

## 1. 背景:为什么需要这个 fork

DeepSeek-V4-Flash 用了 DeepSeek 稀疏注意力(DSA / Lightning Indexer)+ FP4 专家 MoE + mHC。上游默认走 **FlashMLA + DeepGEMM**(只编译 Hopper / 数据中心 Blackwell)。PR #41834 为 **SM120(消费级 Blackwell)** 引入了一套**可移植 Triton 路径**替代这些内核;本 fork 在其之上把这套路径进一步放开到 **SM89**。

| 子系统 | 上游(SM90/100) | SM89(本 fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA sparse | **Triton**(PR #41834 可移植内核) |
| Lightning Indexer(FP8 MQA logits) | DeepGEMM | **Triton / torch fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **Triton**(FP8 dot 加 bf16 upcast) |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE(FP4 专家) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16**(FP4→FP16 反量化) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**硬件事实**:Ada 有 FP8 张量核，但**没有 FP4 张量核、没有硬件 microscaling MMA**，所以 FP4 MoE 只能走 Marlin 反量化(比原生 FP4 MMA 慢)。

### SM89 相关改动(相对 PR #41834，10 文件 +294/-26)

- `vllm/v1/attention/backends/mla/sparse_mla_env.py` — 中央开关 `is_ada_sm89()`，把 SM89 并入 Triton 稀疏 MLA 路径。
- `vllm/utils/deep_gemm.py` / `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` — MQA logits / HC GEMM fallback dispatch 扩到 SM89。
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — Triton FP8 einsum 扩到 SM89 + FP8 `tl.dot` 的 bf16 upcast。
- `vllm/model_executor/kernels/mhc/tilelang.py` — mHC TF32 路径扩到 SM89。
- `vllm/model_executor/layers/sparse_attn_indexer.py` / `v1/attention/backends/mla/indexer.py` — 修复构造期会崩的 `_sparse_indexer_requires_deep_gemm`、内存预算。
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability` 修准确。
- **`vllm/utils/import_utils.py` — `has_cutedsl()` 在 SM89 返回 False**。

> 详见 [`SM89_DEEPSEEK_V4_NOTES.md`](SM89_DEEPSEEK_V4_NOTES.md)。

---

## 2. 已验证环境

| 项 | 版本 |
|---|---|
| GPU | 4× RTX 4090 (48GB) · compute capability **8.9** |
| 驱动 / CUDA toolkit | 595.x / **CUDA 13.0**(nvcc 13.0) |
| Python | 3.12 |
| torch | **2.11.0+cu130** |
| vLLM | 本 fork release wheel = **0.23.1rc1.dev145+g<commit>.cu130**，只为 SM89/Ada 编译 |

---

## 3. 快速安装(预编译 wheel，免编译)

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

**已验证过的环境**:
- **Python 3.12** · Linux x86_64
- **4× RTX 4090 (SM89/Ada, 48GB)** · 驱动 595.x · CUDA toolkit 13.0
- **torch 2.11.0+cu130**
- wheel 使用 `TORCH_CUDA_ARCH_LIST=8.9+PTX` 编译，面向 Ada/SM89

---

## 4. 源码安装(clone 本仓库编译)

### 4.1 Python 环境 + torch cu130

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.11.0 --torch-backend=cu130
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
```

### 4.2 Rust 工具链(vLLM 0.11 有 Rust frontend)

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

> DeepGEMM **不要**装(Ada 不支持)。
> wheel 文件名遵循 release 命名:`vllm-0.23.1rc1.dev145+g<commit>.cu130-cp312-cp312-linux_x86_64.whl`。

---

## 5. 算子级自检(无需起完整模型)

```python
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_mla_env as e
print("cap:", current_platform.get_device_capability())          # (8, 9)
print("is_ada_sm89:", e.is_ada_sm89())                            # True
print("triton sparse mla:", e.is_triton_sparse_mla_enabled(torch.device("cuda:0")))  # True
from vllm.model_executor.layers.sparse_attn_indexer import _sparse_indexer_requires_deep_gemm as r
print("indexer needs deepgemm (fp8 cache):", r(False))           # False ← 关键
from vllm.utils.import_utils import has_cutedsl
print("has_cutedsl:", has_cutedsl())                             # False on SM89
```

---

## 6. 部署(vllm serve)

### 6.1 源模型

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

### 6.2 DSpark 推测解码模型

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

启动成功标志:`Application startup complete.`，日志里能看到 `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`。

---

## 7. 测试结果(4× RTX 4090)

### 7.1 推理正确性
```
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

### 7.3 性能(单并发，输出 512 token)
| 输入 | TTFT | Prefill | Decode |
|---|---|---|---|
| 8,192 | 1.97s | **~4,160 tok/s** | **~82 tok/s** |
| 32,768 | 7.81s | **~4,195 tok/s** | **~82 tok/s** |

Decode ~82 tok/s 受 Marlin MoE 反量化开销影响(Ada 无 FP4 张量核)。

### 7.4 Tool call(`deepseek_v4` parser)
```
Q: 北京今天天气怎么样？请用摄氏度回答。  (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

### 7.5 DSpark 推测解码(CUDA 13.x / torch cu130, 单并发)

vLLM 自带 `vllm bench serve`，random dataset 固定长度，`max-concurrency=1`，每组 10 次，输出 1024 token。

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

| 输入 → 输出 | 成功 | mean TTFT | mean TPOT | Prefill | Decode | output tok/s | total tok/s | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,192 → 1,024 | 10/10 | 2.428s | 2.816ms | **3,374 tok/s** | **355 tok/s** | 192.9 | 1,736.0 | 92.81% |
| 32,768 → 1,024 | 10/10 | 9.442s | 2.974ms | **3,470 tok/s** | **336 tok/s** | 82.0 | 2,706.6 | 89.12% |
| 131,072 → 1,024 | 10/10 | 42.829s | 4.568ms | **3,060 tok/s** | **219 tok/s** | 21.6 | 2,780.8 | 58.25% |

## 8. 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)(Apache-2.0)及其 PR #41834。本 fork 沿用同协议。AI 辅助完成，人工验证。
