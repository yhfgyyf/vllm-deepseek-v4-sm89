# DeepSeek V4.1 upstream migration ledger

Pinned upstream: `3213796cd32df625a0669e9423acb397b627b790` (2026-11-09).

This ledger reconciles the earlier 24 correctness and 21 performance findings against the active SM89/SM120 source route. A source entry is not an end-to-end validation or speedup claim. See [build and validation scope](deepseek_v41_sm89_sm120.md).

| ID | Audited change | Active source | Integration decision |
| --- | --- | --- | --- |
| B01 | DSML tool_calls 起始标记兼容 | [`deepseek_v41.py`](../../vllm/parser/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B02 | 尊重生成提示与末条 assistant 续写参数 | [`deepseek_v41.py`](../../vllm/tokenizers/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B03 | 先物化消息 content 迭代器再 deepcopy | [`deepseek_v41.py`](../../vllm/tokenizers/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B04 | tool_reference 结果与 Anthropic 工具消息分组 | [`deepseek_v41.py`](../../vllm/tokenizers/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B05 | 更新 reasoning effort 映射 | [`deepseek_v41_encoding.py`](../../vllm/tokenizers/deepseek_v41_encoding.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B06 | 接受 Responses API 文本类型 | [`deepseek_v41.py`](../../vllm/tokenizers/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B07 | 对齐 1017 版消息编码 | [`deepseek_v41_encoding.py`](../../vllm/tokenizers/deepseek_v41_encoding.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B08 | 图片数据与实际渲染顺序对齐 | [`deepseek_v41.py`](../../vllm/tokenizers/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B09 | 允许 DSML 参数省略 string 属性 | [`deepseek_v41.py`](../../vllm/parser/deepseek_v41.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B10 | 增量流出字符串工具参数 | [`deepseek_v4.py`](../../vllm/parser/deepseek_v4.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B11 | DSpark 图 padding 行不参与 MoE fused top-k | [`router.py`](../../vllm/models/deepseek_v4_1/nvidia/router.py) | 原生 router 传播 padding mask；保留稳定 softplus 和 128×3/384×6 选择。 |
| B12 | 压缩器 ring slot 避开 null block | [`compressor.py`](../../vllm/models/deepseek_v4_1/compressor.py) | 原生压缩器同步 null-block 防护。 |
| B13 | 补齐启动时 prefill 显存预算 | [`model.py`](../../vllm/models/deepseek_v4_1/nvidia/model.py) | 启动 profile 路径包含 Engram 临时张量。 |
| B14 | 按最坏长宽比估算 vision token 数 | [`mm_preprocess.py`](../../vllm/models/deepseek_v4_1/common/mm_preprocess.py) | 原生 processor 迁移最坏长宽比预算。 |
| B15 | PP 分段入口注入 Engram | [`model.py`](../../vllm/models/deepseek_v4_1/nvidia/model.py) | PP 入口的三维 residual 分支执行 Engram 注入。 |
| B16 | 为不支持 EPLB 的 DSpark draft 清理 EPLB 状态 | [`utils.py`](../../vllm/v1/worker/gpu/spec_decode/dspark/utils.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| B17 | 避免重复 dummy 初始化和随机填充巨型 CPU 表 | [`engram.py`](../../vllm/models/deepseek_v41/common/engram.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B18 | Engram offload 尊重 meta 初始化 | [`engram.py`](../../vllm/models/deepseek_v41/common/engram.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B19 | 加载 EPLB 冗余专家槽位 | [`model.py`](../../vllm/models/deepseek_v4_1/nvidia/model.py) | 原生专家映射传递冗余槽位数；EPLB 整模型未测。 |
| B20 | attention sink 使用统一 weight_loader | [`attention.py`](../../vllm/models/deepseek_v4_1/attention.py) | 原生 sink 使用当前统一参数 loader。 |
| B21 | DSpark 不再丢弃有效的末尾 prefix block | [`speculative.py`](../../vllm/config/speculative.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| B22 | DSpark 草稿宽度与 MTP stage 数校验分离 | [`config.py`](../../vllm/v1/worker/gpu/spec_decode/dspark/config.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| B23 | 限制 DSpark profile 查询批次 | [`model_runner.py`](../../vllm/v1/worker/gpu/model_runner.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| B24 | InstantTensor 将 offloaded Engram 权重先加载到 CPU | [`default_loader.py`](../../vllm/model_executor/model_loader/default_loader.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P01 | DSpark 先合并 HC 再 SP all-gather | [`dspark.py`](../../vllm/models/deepseek_v4_1/nvidia/dspark.py) | 先 collapse，再 SP all-gather。 |
| P02 | DSpark context 只执行 KV 插入 | [`dspark.py`](../../vllm/models/deepseek_v4_1/nvidia/dspark.py) | 以零 query heads 调用原生 KV 插入内核，保留 528-byte ABI。 |
| P03 | 合并各 DSpark 层的 context WKV 投影 | [`dspark.py`](../../vllm/models/deepseek_v4_1/nvidia/dspark.py) | 合并 context WKV 投影，再逐层 norm/insert。 |
| P04 | HC 展开使用专用向量化复制 | [`mhc.py`](../../vllm/models/deepseek_v41/nvidia/ops/mhc.py) | 原生模型直接导入上游 hc_expand；eager/graph 位级回归已覆盖。 文件与固定上游逐字节一致。 |
| P05 | ring slot kernel 避免按运行时参数反复 JIT | [`compressor.py`](../../vllm/models/deepseek_v4_1/compressor.py) | 原生 ring-slot kernel 同步动态参数处理。 |
| P06 | 流式加载 VL 权重 | [`vl_model.py`](../../vllm/models/deepseek_v4_1/nvidia/vl_model.py) | 保留 iterable 流式映射和原生模型子 loader。 |
| P07 | AV draft budget 使用专用 Triton kernel | [`adaptive_verification.py`](../../vllm/v1/worker/gpu/spec_decode/adaptive_verification.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| P08 | greedy draft 分词表 argmax | [`speculator.py`](../../vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| P09 | Markov bias 与 Gumbel sampling 融合 | [`speculator.py`](../../vllm/v1/worker/gpu/spec_decode/dspark/speculator.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| P10 | offloaded Engram 异步预取 | [`engram.py`](../../vllm/models/deepseek_v41/common/engram.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P11 | Engram host table 大页打包 | [`engram.py`](../../vllm/models/deepseek_v41/common/engram.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P12 | Engram WKV 沿 TP 分片 | [`engram.py`](../../vllm/models/deepseek_v41/common/engram.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P13 | 融合 mHC post 与下一次 delayed pre | [`deepseek_v41.py`](../../vllm/model_executor/kernels/mhc/deepseek_v41.py) | 局部权重比较后，仅 1–32 tokens 使用上游融合；更大批次保留 TF32x3；不跨 Engram 注入融合。 |
| P14 | ViT encoder CUDA graph 支持及后续修正 | [`vl_model.py`](../../vllm/models/deepseek_v4_1/nvidia/vl_model.py) | 接入上游 vision graph mixin 和当前接口；整模型图像推理未测。 |
| P15 | 多张图片打包执行 ViT | [`vision.py`](../../vllm/models/deepseek_v4/common/vision.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P16 | 融合 token 到 request 的元数据映射 | [`backend.py`](../../vllm/v1/attention/backend.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 文件与固定上游逐字节一致。 |
| P17 | 补齐 DFlash/DSpark 输入准备 JIT warmup | [`speculator.py`](../../vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| P18 | DSpark 不占用 target 的额外 draft token 预算 | [`config.py`](../../vllm/v1/worker/gpu/spec_decode/dflash/config.py) | 新版 speculator config 管理独立 query pass 的预算，DSpark 继承并处理 anchor。 文件与固定上游逐字节一致。 |
| P19 | query RMSNorm 与量化融合补齐 PDL launch 参数 | [`query_quant.py`](../../vllm/models/deepseek_v4_1/common/ops/query_quant.py) | 原生 query quant 同步 PDL launch 参数；能力检查控制启用。 |
| P20 | 减少 sparse attention 元数据临时操作 | [`sparse_swa.py`](../../vllm/v1/attention/backends/mla/sparse_swa.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |
| P21 | FULL draft graph 内更新 metadata，跳过 eager 重建 | [`speculator.py`](../../vllm/v1/worker/gpu/spec_decode/dflash/speculator.py) | 沿用当前上游共享实现；源代码接入不等于整模型功能或性能已验证。 |

Native cache layouts, E8M0 boundary conversion, stable target/draft routing and MXFP8 quantization order remain native. mHC fusion is selected by tested token shape. Other GPU architectures keep upstream dispatch. Full-model serving, image quality, EPLB reload and throughput are not established by this ledger.
