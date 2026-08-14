# DeepSeek-V4-Flash on SM80 / SM86 / SM89 — vLLM fork

> English version: [`README_EN.md`](README_EN.md)

> 本仓库是 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 fork，分支已包含 **PR #41834**(SM120 可移植 Triton 路径)+ **SM80 / SM86 / SM89 适配 commit**。

把 vLLM 的 **DeepSeek-V4-Flash** 推理从 SM90/SM100/SM120 扩展到 **SM80 / SM86 / SM89**，覆盖 Ampere 和 Ada GPU，例如 A100、RTX 3090、A10/A40、RTX 4090、L40/L40S/L4、RTX 6000 Ada。

> ⚠️ 实验性 fork。仅供在 SM80 / SM86 / SM89 GPU 上自测 DeepSeek-V4-Flash。
> 其中 **SM80/A800 适配仍是测试性适配**，不是生产支持承诺。

### SM80/A800 当前状态

SM80 路径已在 4× A800 上完成 DeepSeek-V4-Flash DSpark 推测解码冒烟与吞吐测试。测试配置使用
`--speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"greedy"}'`、
FlashInfer sampler、sparse MLA warmup、`max-num-batched-tokens=16384`。

Decode 侧结果如下，基线为同一服务器上无 DSpark 的 `mbt16k` 结果：

| 输入 -> 输出 | 并发 | DSpark decode | 无 DSpark decode | decode 提升 |
|---|---:|---:|---:|---:|
| 8,192 -> 1,024 | 1 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 -> 1,024 | 1 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

### 本次长上下文补丁的边界

本分支把 SM80 长序列 FP8 MQA logits 的 query tile 固定为 `BLOCK_M=16`，避免
`BLOCK_M=64` 在 A100 上产生严重的寄存器溢出。它只修复 Lightning Indexer 的
长上下文 prefill 热点，不包含 CPU/NUMA MoE offload、DSpark 参数、服务编排或 API
聚合层。

下面分别给出两条复现路线：

- **源码路线**：构建本仓库，适合已有足够模型运行资源的 SM80 / SM86 / SM89 环境。
- **已验证的 2×A100 混合路线**：使用 `Lvllmds4-x v2.3.9` 的 CPU MoE offload，
  再应用本分支等价的 M16 改动和第 6.2 节列出的正确性回移。

仅拉取这个性能 PR **不能单独复刻**第二条路线的完整运行环境。文档会明确区分本
PR、公开运行时和正确性补丁，避免把部署改造误写成本 PR 的能力。

---

## 1. 背景:为什么需要这个 fork

DeepSeek-V4-Flash 用了 DeepSeek 稀疏注意力(DSA / Lightning Indexer)+ FP4 专家 MoE + mHC。上游默认走 **FlashMLA + DeepGEMM**(只编译 Hopper / 数据中心 Blackwell)。PR #41834 为 **SM120(消费级 Blackwell)** 引入了一套**可移植 Triton 路径**替代这些内核;本 fork 在其之上把这套路径进一步放开到 **SM80 / SM86 / SM89**。

| 子系统 | 上游(SM90/100) | SM80 / SM86 / SM89(本 fork) |
|---|---|---|
| Sparse MLA attention | FlashMLA sparse | **Triton**(PR #41834 可移植内核) |
| Lightning Indexer(FP8 MQA logits) | DeepGEMM | **Triton / torch fallback** |
| o_proj FP8 einsum | DeepGEMM `fp8_einsum` | **Triton**(FP8 dot 加 bf16 upcast) |
| mHC pre/post GEMM | DeepGEMM / TileLang | **TileLang TF32** |
| MoE(FP4 专家) | DeepGEMM / FlashInfer-CUTLASS FP4 | **Marlin WNA16**(FP4→FP16 反量化) |
| Indexer Q rope+quant / KV dequant | **CuTe-DSL** | **Triton/torch fallback** |

**硬件事实**:SM80 / SM86 没有原生 FP8 张量核;SM89 有 FP8 张量核，但没有 FP4 张量核、没有硬件 microscaling MMA。因此本 fork 在这些 GPU 上使用 Triton / torch fallback 和 Marlin WNA16 路径替代 DeepGEMM / FlashInfer FP4 内核。

### SM80 / SM86 / SM89 相关改动

- `vllm/v1/attention/backends/mla/sparse_mla_env.py` — 把 SM80 / SM86 / SM89 并入 Triton 稀疏 MLA 路径。
- `vllm/utils/deep_gemm.py` / `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` — MQA logits / HC GEMM fallback dispatch 扩到 SM80 / SM86 / SM89。
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — Triton FP8 einsum 扩到 SM80 / SM86 / SM89，并保留 bf16 B 路径。
- `vllm/model_executor/kernels/mhc/tilelang.py` — mHC TF32 路径扩到 SM80 / SM86 / SM89。
- `vllm/model_executor/layers/sparse_attn_indexer.py` / `v1/attention/backends/mla/indexer.py` — 修复构造期会崩的 `_sparse_indexer_requires_deep_gemm`、内存预算。
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability` 修准确。
- **`vllm/utils/import_utils.py` — `has_cutedsl()` 在 SM80 / SM86 / SM89 返回 False**。
- FP8 linear / MoE 路径在 SM80 / SM86 上使用 Marlin W8A16，避免无原生 FP8 MMA 时的软件解码 GEMM。

> 详见 [`SM89_DEEPSEEK_V4_NOTES.md`](SM89_DEEPSEEK_V4_NOTES.md)。

---

## 2. 适配硬件与已验证环境

| 项 | 版本 |
|---|---|
| 适配 GPU | **SM80**(A100), **SM86**(RTX 3090 / A10 / A40), **SM89**(RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada) |
| 驱动 / CUDA toolkit | 595.x / **CUDA 13.0**(wheel 构建使用 `/usr/local/cuda-13.0`, nvcc 13.0.48) |
| Python | 3.12(历史验证使用 conda；下文用 uv 创建隔离环境) |
| torch | **2.11.0+cu130** |
| vLLM | 本 fork = **0.23.1rc1.dev145**(DeepSeek-V4-Flash + SM80 / SM86 / SM89 改动)，源码编译 |

---

## 3. 源码安装(clone 本仓库编译)

### 3.1 前置工具

确保主机已经安装 Git、uv、Rust 1.95 和匹配的 CUDA toolkit。仓库内隔离环境在 clone
之后创建，避免把 `.venv` 建到错误目录。

### 3.2 Rust 工具链(vLLM 构建需要 Rust frontend)

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

### 3.3 clone 本仓库

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

上面的公开 head 分支用于在合并前精确复现本改动；合并后应改为维护者最终合入的 branch/tag
和 commit。请固定并保存输出的 commit。当前 SM80 基线尚未包含已经合入 `main` 的
[PR #51](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/51)；长上下文部署前还要
同步这项 paged-MQA 64 位寻址修复。当前单提交为
`3183db2f6c86803cc86d06223000d220e001baf8`；先用 `git merge-base --is-ancestor`
检查目标 revision，未包含时再审查并 cherry-pick，不要重复应用。

### 3.4 编译

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

> 如果只在单一 GPU 架构上部署，可以把 `TORCH_CUDA_ARCH_LIST` 缩小到目标架构，例如 A100 用 `8.0`，RTX 3090 用 `8.6`，RTX 4090 / L40 用 `8.9+PTX`。
> DeepGEMM **不要**装(SM80 / SM86 / SM89 不走该路径)。
> 编译完 torchvision/torchaudio 若是非 cu130 版会报 `torchvision::nms does not exist`，修:
> `uv pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu130 torchvision torchaudio`

---

## 4. 可选:构建本地 wheel

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-} \
uv pip wheel . --no-build-isolation --no-deps -w dist/ \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

安装自己构建的 wheel:

```bash
uv pip install dist/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

当前分支更新频繁，建议优先从源码安装;只有在你自己构建并发布了匹配 SM80 / SM86 / SM89 的 wheel 后，再使用预编译 wheel 安装。

---

## 5. 算子级自检(无需起完整模型)

安装后先运行仓库自带的测试：

```bash
uv pip install -r requirements/test/cuda.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
.venv/bin/python test_sm80_ops.py
.venv/bin/python -m pytest -q tests/v1/attention/test_sm120_deepgemm_fallbacks.py
```

最后一条命令包含长序列 `BLOCK_M=16` 选择测试；在 SM80 上还会比较 M16 与 M64 的
完整 logits，并要求逐元素一致。

```python
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_mla_env as e
print("cap:", current_platform.get_device_capability())          # (8, 9)
print("is_ampere_or_ada:", e.is_ampere_or_ada())                 # True on SM8x
print("triton sparse mla:", e.is_triton_sparse_mla_enabled(torch.device("cuda:0")))  # True
from vllm.model_executor.layers.sparse_attn_indexer import _sparse_indexer_requires_deep_gemm as r
print("indexer needs deepgemm (fp8 cache):", r(False))           # False ← 关键
from vllm.utils.import_utils import has_cutedsl
print("has_cutedsl:", has_cutedsl())                             # False on SM89
```

---

## 6. 可复现部署

### 6.1 本仓库源码路线

以下命令使用公开的 DSpark 模型 ID，不写入机器路径、监听地址或凭据。`TP_SIZE` 应由
部署者按实际可用 GPU 数设置。

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

启动成功标志:`Application startup complete.`，日志里能看到 `Using 'MARLIN' Mxfp4 MoE backend` / `Using FP8 indexer cache`。

### 6.2 已验证的 2×A100 混合路线

这条路线用于两张 80 GiB A100 无法把全部 MoE 专家常驻显存的场景。实测使用的运行时为
[Lvllmds4-x v2.3.9](https://github.com/guqiong96/Lvllmds4-x/releases/tag/lvllmds4-x-v2.3.9)，
资产名为 `lvllmds4_x-2.3.9-cp312-cp312-manylinux_2_34_x86_64.whl`，要求 x86_64、
Python 3.12 和 glibc 2.34 或更新版本；SHA-256 为
`1357dd5e11d060cce973f84563d96bb18c1bd2bd7a4d05f642fe01629ba7ab62`。

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

在启动前必须完成四项检查：

1. 在隔离环境安装并校验上面的公开 wheel；不要覆盖已有 vLLM 环境。
   `.repro/` 已被仓库忽略，完成后应清理；不要提交 wheel 或环境文件。
2. 将本分支 `_fp8_mqa_logits_block_m()` 的 M16 改动应用到该隔离运行时。第 5 节测试
   只验证源码路线，不能用来证明 wheel 已被修改；应在仓库目录之外用 hybrid interpreter
   导入 `vllm`，检查 `vllm.__file__`、目标函数源码和 SM80 长序列返回值，再运行同等的
   GPU 等价测试。若导入路径落到源码 checkout，立即停止。

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
3. 确认运行时含有 [paged-MQA 64 位寻址修复](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/pull/51)，
   以及上游 vLLM 的 mixed-precision/packed KV zeroing 修复：
   [#47574](https://github.com/vllm-project/vllm/pull/47574)、
   [#49623](https://github.com/vllm-project/vllm/pull/49623)、
   [#49704](https://github.com/vllm-project/vllm/pull/49704)、
   [#49903](https://github.com/vllm-project/vllm/pull/49903)、
   [#50276](https://github.com/vllm-project/vllm/pull/50276) 和
   [#52058](https://github.com/vllm-project/vllm/pull/52058)。
   长期运行还应包含 scheduler 每步 drain 修复
   [#44490](https://github.com/vllm-project/vllm/pull/44490)。
4. CPU 必须为 x86_64 且支持 AVX2；宿主需提供 NUMA runtime，隔离环境的 libstdc++
   也必须满足 wheel 的 GLIBCXX ABI。若 uv 环境 import 报 GLIBCXX 错误，应换用满足
   ABI 的宿主工具链、兼容容器或 Conda 环境，不要就地替换系统库。

这些正确性回移与本性能 PR 是相互独立的。实测栈使用了与这些上游修复等价的本地回移；
文档列出的是基准完成后上游收敛的推荐修复集合，并未把该集合重新作为整体跑完同一轮
端到端 A/B。由于这些跨版本回移仍需按目标 revision 解决冲突和复测，本节是已验证配置
清单，不宣称是一条无需审查的自动安装脚本。缺少这些修复时，可以做隔离 kernel 实验，
但不要把超长上下文结果当作长期运行稳定性结论。

下面是当前推荐复现的运行参数，不等同于第 7.5 节每个历史样本的逐项配置。
`GPU_DEVICES` 由部署者传入，必须同时检查 GPU interconnect 拓扑和 CPU/NUMA affinity
后选择同一高速互联组；线程数和常驻层不是通用最优值，应在显存、主存和带宽检查通过
后再使用。

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

实测环境设置过 `FLASHINFER_DISABLE_VERSION_CHECK=1`。这会绕过版本保护，不应作为通用
默认值；只有在核对 wheel release 的 FlashInfer 版本和二进制 ABI 后，才能为兼容该固定
运行时显式设置。

`LVLLM_*` 和 `LK_*` 是 Lvllmds4-x 专用开关，不适用于普通 vLLM。本配置启动阶段需要
完成权重加载、KV 初始化、sparse MLA warmup 和 Triton/CUDA graph 编译；只有服务健康、
无 worker 重启且两个 rank 都完成 warmup 后才能开始正式测试。

### 6.3 交给 Codex 的复现任务

可以把下面这段直接交给 Codex。它要求 Codex 自己发现设备和目录，并禁止把发现到的
机器信息写回仓库：

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

## 7. 测试结果

### 7.1 推理正确性(4× RTX 4090 源码路线)
```
Q: 用一句话介绍长城。
A: 长城是中国古代为抵御北方游牧民族入侵而修筑的、横跨多个朝代、绵延数千公里的
   军事防御工程，也是世界文化遗产中象征中华民族坚韧精神的伟大奇迹。   (finish_reason=stop)
```

### 7.2 最大上下文(4× RTX 4090 源码路线)
| max-model-len | max-num-seqs | GMU | GPU KV cache | 单请求并发 | 启动 |
|---|---|---|---|---|---|
| 262,144 (256K) | 16 | 0.97 | 972,374 tok | 3.71x | ✅ |
| 786,432 (768K) | 16 | 0.97 | 1,220,509 tok | 1.55x | ✅ |
| **1,048,576 (1M)** | 4 | 0.97 | **1,243,644 tok** | 1.19x | ✅(模型架构上限) |

实测能跑完的最长输入:**768K(786,000 token，prefill ~147s)**。1M 可启动、kernel 数值正确，但**满 1M 单次 prefill 极慢(>10 min)，不实用**。日常推荐 **128K~256K**。

### 7.3 SM80/A800 DSpark decode 性能(单并发，输出 1,024 token)
| 输入 | DSpark decode | 无 DSpark decode | decode 提升 |
|---|---:|---:|---:|
| 8,192 | **229.8 tok/s/req** | 57.6 tok/s/req | **3.99×** |
| 32,768 | **274.2 tok/s/req** | 58.1 tok/s/req | **4.72×** |

SM80/A800 仍是测试性适配。这里仅列 decode 结果，长上下文 prefill 仍需单独评估。

### 7.4 SM80/A100 长上下文 MQA kernel A/B

这是本补丁的历史 kernel A/B 记录：同一进程、同一输入，每种 `BLOCK_M` 独立编译；编译后
先执行一次不计时 warmup，再用 CUDA event 计时 3 次取中位数。`M` 随 `N` 调整，使
FP32 logits workspace 约为 256 MiB。固定 `num_heads=64`、`head_dim=128`；三行的
`M` 分别为 2,730、2,048、1,724，输入为 FP8 E4M3 Q/K、FP32 scale/weights。公开
同 shape 复测脚本默认随机 seed 为 0；历史表使用按上下文派生的固定 seed，因此它用来
复核性能趋势和数值等价，不保证逐毫秒复刻历史表。第 5 节的轻量等价测试不复现这些
计时 shape。

公开复测脚本为
[`benchmarks/kernels/benchmark_sm12x_mqa.py`](benchmarks/kernels/benchmark_sm12x_mqa.py)；
它会拒绝非 SM80 设备，并输出各 shape 的 CUDA-event 中位数和 M16/M64 等价性结果。
寄存器 spill 数来自当次 Triton 编译产物元数据，表中的值属于历史编译环境，不能视为
跨 Triton/CUDA 版本恒定值。

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm12x_mqa.py \
  --contexts 98304 131072 155648 \
  --repeats 3 \
  --seed 0 \
  --output-json mqa-kernel-results.json
```

| 原始上下文 | C4 压缩后 N | M16 | M64 | kernel 加速 | M16 / M64 spill |
|---:|---:|---:|---:|---:|---:|
| 98,304 | 24,576 | **40.491 ms** | 643.797 ms | **15.90×** | 0 / 5,744 B/thread |
| 131,072 | 32,768 | **34.065 ms** | 1,003.002 ms | **29.44×** | 0 / 8,272 B/thread |
| 155,648 | 38,912 | **34.547 ms** | 644.409 ms | **18.65×** | 0 / 5,744 B/thread |

三组约 256 MiB 的完整 FP32 logits 都满足 `torch.equal(M16, M64)`，没有 NaN/Inf 差异。
这里的毫秒数是单个 MQA kernel latency，**不是** TTFT 或端到端 prefill tok/s。

### 7.5 SM80/A100 端到端观测

以下是已经实际执行过的单并发样本；prefill 速度按 `prompt tokens / client TTFT` 计算。
M16 正式样本使用全新 token 序列、prefix-cache 命中为 0，且该 shape 的 JIT 已预热。

| 上下文 | 选择器 / 状态 | client TTFT | 观测 prefill |
|---:|---|---:|---:|
| 65,536 | 原选择器已使用 M16；JIT-hot 边界对照 | 29.893 s | **2,192.4 tok/s** |
| 98,304 | 原选择器 M64；cold prefix | 237.989 s | 413.1 tok/s |
| 98,304 | M16；首次 shape/JIT | 105.971 s | 927.6 tok/s |
| 98,304 | M16；JIT-hot、cold prefix | 54.383 s | **1,807.6 tok/s** |
| 155,648 | 原选择器 M64；cold prefix | 730.664 s | 213.0 tok/s |
| 155,648 | M16；首次 shape/JIT | 138.660 s | 1,122.5 tok/s |
| 155,648 | M16；JIT-hot、cold prefix | 110.015 s | **1,414.8 tok/s** |

这些端到端数据是 `n=1` 的运行观测，不是严格的同配置 A/B：旧基线使用
`max-num-batched-tokens=16384`、常驻层 `0-27`；M16 样本使用 `8192`、GMU `0.90`、
常驻层 `0-28`，并使用独立编译缓存。65,536 正好映射到 C4 的 16,384 边界，旧代码本来就选
M16，因此只能作为控制样本。没有执行过补丁后的 131,072 全服务 cold-prefill，本文不
提供该长度的端到端速度或插值结果。

复测时不要使用内置 warmup 请求复用正式 prompt。应先以 seed A 独立发送一次相同长度
的 JIT priming 请求，再以 seed B 启动正式单请求测试；两次都关闭 request warmup 和
ready check。正式结果必须同时满足：保存的 `input_lens` 等于目标长度、metrics 显示
cache-hit 增量为 0、请求窗口没有新 JIT 日志。`input_lens[0] / ttfts[0]` 才是本表使用的
client-observed TTFT rate，而不是服务端 `request_prefill_time`。

下面以混合路线为例；源码路线把 `BENCH_VLLM` 指向 `.venv/bin/vllm`。

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

### 7.6 Tool call(`deepseek_v4` parser)
```
Q: 北京今天天气怎么样？请用摄氏度回答。  (tools=[get_weather])
→ finish_reason: tool_calls
→ get_weather  arguments: {"city": "北京", "unit": "celsius"}   ✅
```

---

## 8. 已知限制 / 风险

1. **MoE 走 Marlin**:正确性已验证，但性能比原生 FP4 MMA 低。性能调优空间最大的一块。
2. **性能未针对 4090 调优**:fused_moe / scaled_mm 的 tuned config 只覆盖 RTX PRO 6000 / GB10，4090 用默认 heuristic(日志会有 "Performance might be sub-optimal" 提示)。
3. **超长上下文**:1M 可启动但 prefill 慢到不实用;>256K 单请求约数分钟。
4. 4× RTX 4090 源码路线、4× A800 DSpark decode 和 2× A100 混合路线分别有本文标注
   的测试；其它 Ada 卡(L40/L4 等)原理相同但未实测。
5. **M16 只优化 prefill**：paged decode 不调用这个非 paged MQA 路径；decode 吞吐仍由
   长上下文 attention、MoE offload 和 DSpark 接受率共同决定。
6. **部署依赖边界**：本 PR 不包含 paged-MQA 寻址和 packed/mixed-KV zeroing 回移；
   未核对第 6.2 节正确性前置项时，不应声称已经复现完整稳定环境。
7. **基准边界**：首次新 shape 包含 JIT 成本，exact-repeat 会命中 prefix cache。两者都
   不能替代“JIT 已热 + 新 prefix + cache hit 为 0”的正式 cold-prefill 样本。

---

## 9. 许可 / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)(Apache-2.0)及其 PR #41834。本 fork 沿用同协议。AI 辅助完成，人工验证。
