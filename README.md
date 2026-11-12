# DeepSeek V4.1 / V4 / Vision 与 GLM-5.3 on SM89 / SM120

<!-- markdownlint-disable MD060 -->

[English](README_EN.md)

基于 [vLLM](https://github.com/vllm-project/vllm) 的推理服务，支持
DeepSeek-V4.1-Flash、DeepSeek-V4-Flash、DeepSeek-V4-Flash-Vision-Exp 和
GLM-5.3-Flash，可通过 OpenAI 兼容 API 调用。

## 1. 环境要求

| 项目 | 配置 |
| --- | --- |
| 系统 / Python | Linux x86_64 / Python 3.12 |
| SM89 示例 | L20、RTX 4090 48GB；V4.1 使用 8 张卡 |
| SM120 示例 | RTX PRO 6000 Blackwell 96GB，4 张卡 |
| 当前运行栈 | PyTorch 2.14.0+cu130、Triton 3.8.0、Transformers 5.22.2 |
| FlashInfer | 本项目配套的 0.7.3 扩展，不能直接用上游包替代 |
| CUDA | 源码构建使用 13.2；下方服务示例的本地 JIT 使用 13.0 |

模型路径须指向完整 checkpoint。Engram CPU offload 和模型加载需要充足的
主机内存；48GB GPU 示例不适用于普通 24GB 版 RTX 4090。

## 2. 安装

### 2.1 当前源码

使用下方 L20 TP8 配置或 SM120 可选 B12X 算子时，请安装当前 `main`，
并应用配套的 FlashInfer recipe patch：

```bash
git clone https://github.com/yhfgyyf/vllm-deepseek-v4-sm89.git
cd vllm-deepseek-v4-sm89
```

然后按[源码构建说明](docs/models/deepseek_v41_sm89_sm120.md#build-the-current-source)
安装 vLLM 和 FlashInfer。使用新的 Python 3.12 环境，不复用 Torch 2.13 的扩展。

### 2.2 预编译 wheel

`vision12` vLLM 与 `vision3` FlashInfer 使用下方固定版本；它们不包含当前源码的
L20 算子调优及可选 SM120 B12X 适配。需要这些功能时使用源码安装。

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install --torch-backend=cu130 \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/vllm-0.33.0%2Bglm53.dsv41.vision12.sm89sm120.cu130-cp312-cp312-linux_x86_64.whl#sha256=67228efddc16f0769e3578137a734e14cf2869b36610f37946419503d6d8c65e' \
  'https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/v0.33.0-vision12-sm89-sm120-cu130/flashinfer_python-0.7.3%2Bglm53.dsv41.vision3.sm89sm120.cu130.pt214-py3-none-any.whl#sha256=b85753d53a0094796e865bffed11b2f55226d1991f776a3b9a76a2f0f77abecf' \
  'transformers==5.22.2' 'triton==3.8.0'
```

`uv` 会校验两个 wheel 的 SHA256。

### 2.3 Docker

现有 GHCR `vision11` 镜像使用 vLLM 0.28 / Torch 2.13，适合保留旧环境；
当前源码的 L20 TP8 示例需要上面的源码安装，不能直接套用到此镜像。
下方是容器模板，需替换模型路径并追加所用镜像版本的模型参数。
环境变量通过 `docker run -e` 传入；`vision11` 的 V4.1 还需
`-e VLLM_USE_V2_MODEL_RUNNER=1`。

```bash
docker pull \
  ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130
```

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /path/to/models:/models:ro \
  ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130 \
  /models/model-directory \
  --host 0.0.0.0 --port 8000
```

## 3. DeepSeek-V4.1-Flash 启动命令

下方是 **4× RTX PRO 6000 96GB、TP4/PP1、draft TP4、DSpark5** 模板。
**8×L20 默认使用 DSpark7**，命令见 3.3。替换虚拟环境、CUDA 和模型路径后运行。

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

### 3.1 功能开关

- **DSpark**：checkpoint 需要 confidence head。示例开启 adaptive verification；
  要使用固定草稿验证，将 `enable_adaptive_verification` 改为 `false`。
- **CUDA Graph**：V4.1 使用 `FULL_AND_PIECEWISE`，其余 DSpark 示例使用 `FULL`。
  启用 adaptive 时保留对应配置，不添加 `--enforce-eager`。
- **CED**：`ced_prefill=true` 开启实验性近似 prefill；移除此 override 可使用普通
  prefill。CED 与 adaptive 可独立设置，输出质量需按实际任务确认。
- **图片**：通过 `--limit-mm-per-prompt '{"image":2}'` 设置数量上限。
  V4.1 当前不支持非图片模态、prompt embeddings 或 prompt logprobs。
- **SM120 B12X**：默认关闭。按[安装与开关说明](docs/models/deepseek_v41_sm120_b12x.md)
  使用固定版本并显式启用；新适配的 GPU 集成验证尚未完成。

首次启动的 JIT 和 Graph capture 可能较慢，等服务就绪后再调用。
`max-model-len` 是请求长度上限，`max-num-seqs` 是调度上限，两者不保证所有
并发请求都能同时使用最大上下文。

### 3.2 L20 性能参考

8×L20、TP8/PP1、draft TP8、DSpark7、显存利用率 0.96，启用 CED 和 adaptive：

| 输入 → 输出 tokens | Prefill 中位数 | Decode 中位数 | DSpark 命中率 |
| --- | ---: | ---: | ---: |
| 17,630 → 2,641 | **7,849.510 tokens/s** | **395.850 tokens/s** | **87.135177%** |

2026-11-12 的合成复制负载，3 次单请求，temperature=0、reasoning_effort=none、
前缀命中为零。Prefill = 输入 tokens / 服务端 prefill 秒数；decode =
`(输出 tokens - 1) / 服务端 decode 秒数`；命中率 = accepted/proposed。
这是单请求阶段速度，实际任务表现会随输入和接受率变化。

### 3.3 SM89：8×48GB GPU（L20 / RTX 4090 48GB）

当前源码的 L20 配置为 **TP8/PP1、draft TP8、DSpark7、0.96 自动 KV**，
启用 PCIe IPC、EP、CED 和 Engram CPU offload。调度预算为 8192 tokens，
最多 4 条序列，Graph capture 上限为 48 行。

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

保留 `--no-enable-extensible-kv-cache`，让服务使用常规自动 KV 分配。
target TP8 的 draft 也必须是 TP8。IPC 对不支持的消息使用原通信后端。
显存不足时降低利用率或调度预算，再根据实际负载确认容量。

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

## 6. GLM-5.3-Flash 启动命令

### 6.1 SM89：8× RTX 4090 48GB

以下示例使用官方 FP8 权重：

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

## 7. 调用 API

服务就绪后，通过 OpenAI 兼容接口调用。`model` 与启动时的
`--served-model-name` 保持一致；下面使用 V4.1 示例：

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

## 8. 更多文档

- [源码构建与依赖](docs/models/deepseek_v41_sm89_sm120.md)
- [L20 配置与性能说明](docs/models/deepseek_v41_l20_baseline.md)
- [SM120 可选算子设置](docs/models/deepseek_v41_sm120_b12x.md)
- [历史发布与测量记录](docs/models/sm89_sm120_history.md)

## 微信用户交流群

<details>
<summary>点击展开微信群二维码</summary>

<p align="center">
  <img src="docs/assets/wechat-user-group.png" alt="Strata、vLLM、SGLang AI 推理讨论群二维码" width="420">
</p>

</details>

---

## 9. License / 来源

代码基于 [vllm-project/vllm](https://github.com/vllm-project/vllm)，沿用
Apache-2.0 协议。FlashInfer wheel 基于
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)。
