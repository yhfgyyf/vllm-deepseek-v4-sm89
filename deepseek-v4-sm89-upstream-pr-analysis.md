# vLLM 上游 PR 对 yhfgyyf/vllm-deepseek-v4-sm89 的适用性分析

<!-- markdownlint-disable MD028 MD060 -->

- **分析日期**：2026-08-20；实现状态复核于 2026-08-22
- **分析对象**：[yhfgyyf/vllm-deepseek-v4-sm89](https://github.com/yhfgyyf/vllm-deepseek-v4-sm89)（DeepSeek-V4-Flash 在 SM89/Ada 上的 vLLM 适配 fork，专用 FlashInfer 0.6.17+sm89.1 sparse MLA JIT）
- **上游范围**：vllm-project/vllm `v0.24.0`（2026-06-27）→ [`main@7ca49fbe4bab`](https://github.com/vllm-project/vllm/commit/7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6)（2026-08-22）
- **本次增量**：原报告 HEAD `0fb168e6ee` → `4666a8ba9ed5`，共 312 个主线提交
- **验证方法**：对比官方 `origin/main` 完整提交历史、PR 页面/合并状态与 fork 当前代码；仅把已合并 PR 计入“官方现行代码”，仍 open 的 PR 单列

---

## 0. 2026-08-22 实现状态

本报告第 1～8 节保留了移植前的分析语境；其中“fork 缺失”表示 2026-08-20 快照时的状态。当前 release 工作树已经完成以下选择性回移植，最终状态以本节为准。

| 状态 | PR / 更新 | 当前实现 |
|---|---|---|
| ✅ 已移植 | #51727 / #51296 | DeepSeek tokenizer vocab size 与 reasoning parser 默认 thinking 修复。 |
| ✅ 已移植 | #52288 + #52809 语义 | 仅 DeepSeek-V4 draft 在未显式配置时继承 target attention backend。 |
| ✅ 已移植 | #47914 | DFlash eager/FULL graph 均按 KV group 传递 hybrid causal metadata。 |
| ✅ 已移植 | #48137 / #48660 / #47463 | mHC decode copy 消除、DSV4 top-k softplus/sqrt kernel 和 kernel 内 dtype 处理。 |
| ✅ 已移植 | #49486 / #50298 / #52084 / #51967 / #48957 | 短上下文 top-k 快捷路径、输出 buffer 复用、workers/constexpr 和空 C128 launch 优化。 |
| ⚙️ SM89 特化 | #51430 / #52401 / #52492 | 保留结构和 correctness follow-up；SM89 FlashInfer sparse MLA 继续使用宽 eager CUDA Graph guard。 |
| ⚙️ 选择性抽取 | #51538 | 仅采用与 SM89 相关的 q-head、SWA width、负 index length、top-k 边界和 workspace lane 修复。 |
| ⛔ 未移植 | #47808 / #52436 | 未启用 confidence-scheduled adaptive verification；当前推荐固定 k=7 probabilistic。 |
| ⛔ 保持回滚 | #50004 / #49236 | 不重新引入上游已经回滚的 adaptive stride 和 model-wide eager scratch pool。 |

此外，fork 已包含自身 PR #51 的 SM89 paged MQA logits int32 地址溢出修复和 PR #61 的 Triton per-shape kernel cache 增长修复。依赖同步到 torch 2.13.0/cu130、Triton 3.7.1、FlashInfer 0.6.17、CUTLASS DSL 4.6.2 和 TileLang 0.1.12；SM89 release 使用 `flashinfer-python==0.6.17+sm89.1`。

---

## 1. fork 基线判断

fork 代码中可验证的同步点约为**上游 2026-07-06/07 的 main**，叠加：

- SM89 补丁（FlashInfer JIT、Marlin FP4 MoE、Triton/torch fallback）
- 自研 DSpark 支持（2026-07-01，对应上游 #46995 同期版本）
- 0731 checkpoint 适配（2026-08-02，含 UE8M0 scale、边界编码、block 256 等自研修复）

**结论**：fork 的可验证上游共同祖先仍是 2026-07-07 的 `93e2ab7111`；原报告“多数后续修复/优化未进入 fork”的判断仍成立。此次新增主线中，最直接的 SM89 修复是 #52288；#50004 与 #49236 已被上游回滚，不能再按旧建议移植。

| 证据 | 说明 |
|---|---|
| fork 有 `dspark.py:273` draft_id_to_target_id | → 已含 #47429 |
| fork 有 `attention.py:623` kv_quant_mode | → 已含 #47716 |
| fork 有 `backend.py:471/520` token_to_req_indices 缓存 | → 已含 #47474 |
| fork 的 `model.py:1078` 仍有 `unsqueeze(-2).repeat` | → 缺 #48137 |
| fork 的 `attention.py:432` 仍是宽 eager region | → 缺 #51430 |
| fork 的 `cache_utils.py:618` 仍有 `torch.full` | → 缺 #50298 |
| fork 的 tokenizer 仍有 `__len__` 覆盖 | → 缺 #51727 |

---

## 2. 已在 fork 中（无需重复移植）

| PR | 修复/优化内容 | 验证依据 |
|---|---|---|
| [#47429](https://github.com/vllm-project/vllm/pull/47429) | DSpark 补上 `draft_id_to_target_id = None` | fork `dspark.py:273`，与上游逐字一致 |
| [#47716](https://github.com/vllm-project/vllm/pull/47716) | fp8_ds_mla KV cache reshape 修复 | fork `attention.py:623` 已有 `kv_quant_mode=get_kv_quant_mode(...)` |
| [#47474](https://github.com/vllm-project/vllm/pull/47474) | 缓存 `token_to_req_indices`（kernel 5~6×） | fork `vllm/v1/attention/backend.py:471,520` 已有实现 |

---

## 3. 缺失且值得移植的 Bug 修复

| 优先级 | PR | 修复内容 | 说明 |
|---|---|---|---|
| **高** | [#51727](https://github.com/vllm-project/vllm/pull/51727) | tokenizer vocab size 多算（`added_vocab` 重复计数）导致 **guided decoding 崩溃** | fork 的 `vllm/tokenizers/deepseek_v4.py` 中旧 `__len__` 覆盖确认还在；上游仅删除 5 行，改动极小 |
| **中高** | [#51296](https://github.com/vllm-project/vllm/pull/51296) | parser thinking 默认值与 tokenizer 不一致 | fork 的 `vllm/parser/deepseek_v4.py:117` 仍为 `thinking: bool = False` → 默认 CONTENT；上游修复后空 kwargs 应为 REASONING，影响 reasoning 解析正确性 |
| **中高** | [#52288](https://github.com/vllm-project/vllm/pull/52288) | DSpark 未显式指定 backend 时继承 DeepSeek-V4 target backend，避免 target/draft KV 布局分叉 | fork `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:26` 仍直接使用可能为 `None` 的 `speculative_config.attention_backend`；移植时应同时采用 open follow-up #52809 的“仅 DSV4 继承”限定 |
| 条件 | [#52401](https://github.com/vllm-project/vllm/pull/52401) + [#52492](https://github.com/vllm-project/vllm/pull/52492) | 修复 #51430 后 MRV1 输出损坏，以及 breakable graph 把短上下文 indexer 快捷路径固化进图的问题 | fork 仍是宽 eager region，当前不触发；若移植 #51430，二者必须作为 correctness bundle 一起评估 |
| 条件 | [#52436](https://github.com/vllm-project/vllm/pull/52436) | adaptive DSpark draft budget 为 0 时修正 structured-output grammar bitmask 映射 | fork 尚无 #47808 adaptive verification；只在引入该功能时需要 |
| 低 | [#49415](https://github.com/vllm-project/vllm/pull/49415) | DSpark 草稿共享专家 padding（仅 TP>8 触发） | fork 用 TP4，不触发；移植成本低 |
| 低 | [#47493](https://github.com/vllm-project/vllm/pull/47493) | TP16 输出乱码（FlashInfer 索引重映射 + 512B 对齐） | fork 用 TP4；对齐部分 fork 已是 `[512]`；若日后跑 A800 TP8/TP16 需复核 |

---

## 4. 缺失且值得移植的吞吐优化（按收益排序）

| 优先级 | PR | 优化内容 | 上游量化数字 | fork 现状 |
|---|---|---|---|---|
| **高** | [#51430](https://github.com/vllm-project/vllm/pull/51430) | 收窄 eager CUDA graph 区域（Q 投影 / fused norm+RoPE / KV 插入等移出 eager） | **Mean TTFT −53.7%**（41.45→19.20 ms，V4-Flash 4×GB200 短 prefill） | fork `attention.py:432` 仍是宽 eager region；必须连同 #52401/#52492 移植，不能再搭配已回滚的 #49236 |
| **高** | [#48137](https://github.com/vllm-project/vllm/pull/48137) | 去掉 decode 每步 `hidden_states.unsqueeze(-2).repeat(1, hc_mult, 1)` 冗余拷贝 | E2E **TPOT −1.8%** | fork `model.py:1078` 确认仍是旧代码，改动约 10 行 |
| 中高 | [#48660](https://github.com/vllm-project/vllm/pull/48660) | dsv4 专用 routing kernel（`dsv4_topk.py` + `topk_softplus_sqrt` CUDA kernel） | E2E **TPOT −2.94%** | fork 仍走旧 `fused_topk_bias` 路径；需编译新 .cu kernel |
| 中高 | [#49486](https://github.com/vllm-project/vllm/pull/49486) | 候选数 ≤ topk 时跳过 topk 与 router 计算 | Decode 场景 E2E **TTFT −3.4%** | fork 无此优化；若进入 captured/breakable graph，必须带 #52492 |
| 中 | [#50298](https://github.com/vllm-project/vllm/pull/50298) | 传 out tensor 消除 `torch.full` kernel 调用 | kernel **1.88×** | fork `common/ops/cache_utils.py:618` 确认仍是 `torch.full` |
| 中 | [#52084](https://github.com/vllm-project/vllm/pull/52084) | sparse top-k metadata kernel workers 128→256 | kernel **+15.2%~36.9%**；配对 serving 吞吐 **+1.56%**、TPOT **−2.79%** | fork `cache_utils.py:628` 仍为 128 workers；Triton 改动小，需在 SM89 复测占用率 |
| 中低 | [#51967](https://github.com/vllm-project/vllm/pull/51967) | 把 global top-k kernel 的 stride/topk/block size 标为 `tl.constexpr` | kernel **+15.1%**；serving 吞吐 **+0.50%**、TPOT **−0.98%** | fork `cache_utils.py:518` 的参数仍是运行时值，改动小 |
| 中 | [#48957](https://github.com/vllm-project/vllm/pull/48957) | 跳过空 c128 kernel launch | kernel **~2×** | fork 无 |
| 中 | [#47463](https://github.com/vllm-project/vllm/pull/47463) | `fused_topk_bias` dtype 转换移进 kernel | kernel **1.5~2×** | fork `fused_topk_bias_router.py` 确认仍是 kernel 外 `.to()` |
| 低 | [#50312](https://github.com/vllm-project/vllm/pull/50312) | PP 末 rank 无 MTP 时不分配/拷贝 buffer | **省 448 MiB 显存** | fork 只按 last rank 分配，未按 MTP 开关；fork 跑 PP=1 且 DSpark 需要 MTP，收益有限 |

> 注：上游数字来自各自 PR 描述中的评测（GB200、MI355X、A10 等环境各异），不可跨 PR 直接相加；kernel 级数字为微基准。SM89/4090 上 #51430 的实际收益需实测（短 prefill 的 graph 收益机理一致，但幅度不同）。

> **上游回滚门禁**：#50004 已由 #51318 回滚；#49236 已由 #52836 回滚。两项都不再属于官方现行实现，详见 6.2。

---

## 5. 不适用 / 不建议移植（含原因）

### 平台限定（ROCm / XPU）

- ROCm：[#48519](https://github.com/vllm-project/vllm/pull/48519)（sparse prefill kernel 1.83~2.13× / TTFT −10%）、[#47718](https://github.com/vllm-project/vllm/pull/47718)（两阶段 compressor，TTFT −4~6%）、[#48788](https://github.com/vllm-project/vllm/pull/48788)（gfx950 reducer 占位优化）、[#46275](https://github.com/vllm-project/vllm/pull/46275)（gfx942 split sparse decode）、[#46122](https://github.com/vllm-project/vllm/pull/46122)（AITER MoE）、[#49714](https://github.com/vllm-project/vllm/pull/49714)、[#46730](https://github.com/vllm-project/vllm/pull/46730)、[#51145](https://github.com/vllm-project/vllm/pull/51145)、[#47017](https://github.com/vllm-project/vllm/pull/47017)、[#51821](https://github.com/vllm-project/vllm/pull/51821)、[#52212](https://github.com/vllm-project/vllm/pull/52212)、[#52566](https://github.com/vllm-project/vllm/pull/52566)、[#52737](https://github.com/vllm-project/vllm/pull/52737)
- XPU：[#50434](https://github.com/vllm-project/vllm/pull/50434)、[#48476](https://github.com/vllm-project/vllm/pull/48476)、[#45991](https://github.com/vllm-project/vllm/pull/45991)、[#47677](https://github.com/vllm-project/vllm/pull/47677)

### 硬件/特性不匹配

| PR | 原因 |
|---|---|
| [#43008](https://github.com/vllm-project/vllm/pull/43008) cluster-cooperative topK v2 | 要求 SM90+（TMA/DSMEM），SM89 不可用 |
| [#47229](https://github.com/vllm-project/vllm/pull/47229) 更好的 MXFP8 量化 kernel | CUTeDSL 内核，fork 无 cutedsl |
| [#48167](https://github.com/vllm-project/vllm/pull/48167) FlashInfer 非因果草稿注意力 | Blackwell（SM100/120）路径修复；fork 的 SM89 JIT 路径已验证可用 |
| [#51768](https://github.com/vllm-project/vllm/pull/51768) MRV1 piecewise CUDA graphs 防护 | 官方最新已由 #52401 改为“MRV1 保留宽 eager、MRV2 使用窄 eager”；fork 当前本来就是宽 eager，无需单独移植 #51768 |
| [#51395](https://github.com/vllm-project/vllm/pull/51395) 禁用 FlashInfer sparse MLA dense prefill | 明确限定 SM120；fork 是 SM89 自定义 JIT |
| [#52217](https://github.com/vllm-project/vllm/pull/52217) sparse MLA mask 128-bit 向量加载 | 针对 FA4/CuTeDSL，评测为 B200；fork 无对应 `sparse_mla_mask.py` 路径 |
| [#51538](https://github.com/vllm-project/vllm/pull/51538) DSV4 sparse MLA plain/MTP/DSpark 七项修复 | 整包针对 SM120 FlashInfer sparse MLA，不能整 PR 移植；仅在 fork 出现相同 SWA width、负 indexer length 或 top-k 越界症状时逐项抽取 |
| [#51092](https://github.com/vllm-project/vllm/pull/51092) EAGLE3 崩溃 | fork 用 DSpark，不用 EAGLE3 |
| [#49634](https://github.com/vllm-project/vllm/pull/49634) Quark MXFP4 崩溃 | fork 用 Marlin FP4 路径 |
| [#48256](https://github.com/vllm-project/vllm/pull/48256) 均匀 page-size 误路由防护 | 影响被误路由进 DeepseekV4 packing 的**其他** MLA+SWA 模型；fork 只服务 DSV4 |
| [#48642](https://github.com/vllm-project/vllm/pull/48642) fp8_ds_mla dense prefill | fork 的 prefill 走 FlashInfer sparse JIT，非 FlashMLA dense 路径 |
| [#50693](https://github.com/vllm-project/vllm/pull/50693) DSpark warmup 崩溃 | fork 的 nvidia 路径无该 `topk_indices_buffer` assert（assert 只在 fork 的 amd/rocm.py） |
| [#51602](https://github.com/vllm-project/vllm/pull/51602) parallel_drafting_token_id init | fork 的 DSpark 无 parallel drafting 功能 |
| [#46986](https://github.com/vllm-project/vllm/pull/46986) / [#46973](https://github.com/vllm-project/vllm/pull/46973) deepseek_v2 共享 backbone | fork 无 `vllm/models/deepseek_v2/` 目录，V4 模型独立实现 |
| [#46789](https://github.com/vllm-project/vllm/pull/46789) / [#51434](https://github.com/vllm-project/vllm/pull/51434) Sequence Parallelism | 需要 DP/EP 配置；fork 是单机 TP4；#51434 针对 V3.2 |

### 上游基础设施 / 非正常 serving 路径

| PR | 不建议直接移植的原因 |
|---|---|
| [#51704](https://github.com/vllm-project/vllm/pull/51704) KV-cache layout `customize_spec` | 是 18 文件的布局标准化系列中间件，不是独立 DSV4 修复；fork 与当前上游接口代际不同 |
| [#52550](https://github.com/vllm-project/vllm/pull/52550) 统一 `indexer_kv_dtype` | 配置/API 迁移，无独立性能收益；fork 仍用 `use_fp4_indexer_cache`，仅在同步整套新 indexer 基础设施时需要 |
| [#52626](https://github.com/vllm-project/vllm/pull/52626) / [#51368](https://github.com/vllm-project/vllm/pull/51368) mHC broadcast buffer | 分别修复 RL/refit 后 CUDA graph 读旧权重，以及 `--load-format dummy`；正常静态权重 serving 不触发，fork 也没有同名 broadcast buffer 实现 |
| [#52842](https://github.com/vllm-project/vllm/pull/52842) / [#52044](https://github.com/vllm-project/vllm/pull/52044) | 仅 CI fixture / MoE benchmark 识别，不改变线上推理路径 |

---

## 6. 上游状态变化与条件项

### 6.1 #48047（q-head padding 移除）——取决于你们的 FlashInfer JIT

- 上游 [#48047](https://github.com/vllm-project/vllm/pull/48047) 在 FlashInfer ≥0.6.14 下移除了 sparse-MLA 的 q-head padding（kernel 已支持 h_q ∈ {8,16,32,64,128}）。
- fork 的 `flashinfer_sparse.py:240` 仍把 h_q 硬 pad 到 64/128（"FP8 decode kernel only supports h_q = 64 or 128"）。
- 你们 TP4 下每 rank 32 head → pad 到 64，**decode 侧有约一半 padding 浪费**。
- **前提**：需先确认你们的 SM89 JIT fork 的 decode kernel 是否支持 h_q=32；支持则直接可用。

### 6.2 两项旧优化已被官方回滚

| 旧 PR | 当前官方状态 | 对 fork 的结论 |
|---|---|---|
| [#50004](https://github.com/vllm-project/vllm/pull/50004) adaptive C128A width | [#51318](https://github.com/vllm-project/vllm/pull/51318) 完整回滚：运行时 packed row stride 与 capture-time stride 不同，长上下文会读错行；修正版 [#52823](https://github.com/vllm-project/vllm/pull/52823) 截止快照仍 open | **不要移植 #50004**；fork 本来就没有 `active_topk_width`，当前是安全的固定 stride |
| [#49236](https://github.com/vllm-project/vllm/pull/49236) eager scratch reuse | [#52836](https://github.com/vllm-project/vllm/pull/52836) 完整回滚：model-wide scratch 跨 layer/stream 复用时缺少 allocator 的 event lifetime tracking，可能覆盖仍在消费的数据 | **不要移植 #49236**；#51430 在 MRV2 上的主要收益不依赖该 scratch pool |

### 6.3 DSpark 最新合并链（fork 的 DSpark 是 7 月初早期版本）

| PR | 官方现状 | SM89 fork 结论 |
|---|---|---|
| [#52288](https://github.com/vllm-project/vllm/pull/52288) target backend fallback | 已合并；后续 [#52809](https://github.com/vllm-project/vllm/pull/52809) 截止快照仍 open，拟把继承范围限定为 DSV4 | fork 明确缺失；建议直接采用“仅 `model_type == deepseek_v4` 时继承”的最终语义 |
| [#47808](https://github.com/vllm-project/vllm/pull/47808) confidence-scheduled verification | 已合并；高并发动态缩减验证预算；SM100 才有 varlen FULL graph 快路，非 SM100 回退 PIECEWISE | fork 完全缺失且改动跨约 40 个文件；SM89 只能视为独立大功能实验，不能当作小型性能 backport |
| [#52436](https://github.com/vllm-project/vllm/pull/52436) budget=0 grammar bitmask | 已合并，是 #47808 的 correctness follow-up | 仅在引入 #47808 时一并移植 |
| [#51538](https://github.com/vllm-project/vllm/pull/51538) plain/MTP/DSpark sparse MLA bundle | 已合并，验证平台为 8×RTX PRO 6000 Blackwell（SM120） | 不整包移植；只按症状审计 SWA width、负 indexer length、top-k clamp 等通用片段 |
| [#49969](https://github.com/vllm-project/vllm/pull/49969) / [#49731](https://github.com/vllm-project/vllm/pull/49731) | PR 描述、模型接线与评测都限定 Qwen3 DSpark；分别是 top-k Markov 与 TP replicated Markov head（#49731 另改通用 embedding/logits helper） | 原报告把 merge commit SHA 当成 PR 标签且误判为 DSV4 通用优化；当前 DSV4 fork 不适用 |
| [#50911](https://github.com/vllm-project/vllm/pull/50911) | TokenSpeed MLA non-causal DSpark，文件/评测针对 Kimi-K3+B200 | 不走 SM89 FlashInfer JIT，当前 fork 不适用 |

DSpark 仍是 fork decode 吞吐的核心（原报告测得 355/336/219 tok/s vs 82 tok/s 基线），但最新可直接 backport 的是 #52288，而不是原报告列出的三个 Qwen3/Kimi 专用 PR。

---

## 7. 建议移植顺序

1. **#51727**（guided decoding 崩溃，删几行）
2. **#51296**（parser 默认值，几行）
3. **#52288 + #52809 的 DSV4 限定语义**（target/draft backend 一致性，小改动）
4. **#48137**（TPOT −1.8%，约 10 行）
5. **#51430 + #52401 + #52492**（窄 eager region 及两个 correctness follow-up）
6. **#49486 / #48660**（decode/TTFT 优化）
7. 小型 Triton/kernel 项：**#52084、#51967、#50298、#48957、#47463**

条件实验：只有确认 SM89 PIECEWISE graph 路径与高并发收益后，才评估 **#47808 + #52436**。

**注意事项**：

- **禁止按旧顺序移植 #50004 或 #49236**；二者都已被官方主线回滚。
- #51430 必须带 #52401/#52492；fork 的 FlashInfer JIT 路径是特有实现，移植后必须跑完整精度验证（GSM8K、长上下文、DSpark）再上线。
- 每个 PR 移植前建议用 `git show <commit>` 提取完整 diff，评估与 fork 改动的冲突面。

---

## 附录：已进入官方 `main` 的 PR 清单与 fork 结论

| PR / 关系链 | 官方 `main` 状态 | fork 结论 |
|---|---|---|
| #47429 / #47474 / #47716 | 已合并、现行 | ✅ 已在 fork |
| #51727 / #51296 / #52288 | 已合并、现行 bugfix | ⚠️ 缺失；依次为高 / 中高 / 中高优先级 |
| #51430 → #52401 / #52492 | 已合并；后两项是 correctness follow-up | ⚠️ 缺失；必须按完整链移植 |
| #48137 / #48660 / #49486 | 已合并、现行性能优化 | ⚠️ 缺失；高至中高优先级 |
| #50298 / #52084 / #51967 / #48957 / #47463 | 已合并、现行小型 kernel/Triton 优化 | ⚠️ 缺失；逐项 SM89 微基准后采用 |
| #50312 / #49415 / #47493 | 已合并、现行 | ⚠️ 缺失，但当前 TP4/PP1 场景优先级低 |
| #48047 | 已合并、现行 | ⚠️ 缺失；取决于 SM89 JIT 是否支持 `h_q=32` |
| #47808 → #52436 | 已合并；后者是 adaptive verification correctness follow-up | ⚠️ 缺失；SM89 条件实验，不是小 backport |
| #51538 | 已合并、现行 SM120 sparse-MLA bundle | ⚠️ 仅按相同症状抽取通用片段，不整包移植 |
| #50004 → #51318 | #50004 已被完整回滚 | ⛔ 不移植旧实现；fork 当前固定 stride 是安全状态 |
| #49236 → #52836 | #49236 已被完整回滚 | ⛔ 不移植 model-wide eager scratch pool |
| #51704 / #52550 | 已合并、现行基础设施/API 迁移 | ❌ 不独立移植 |
| #52626 / #51368 | 已合并、现行 mHC refit/dummy-load 修复 | 条件项；正常静态 serving 不触发 |
| #51395 / #52217 | 已合并、现行 SM120 / FA4 优化 | ❌ SM89 不适用 |
| #51821 / #52212 / #52566 / #52737 | 已合并、现行 ROCm 项 | ❌ NVIDIA SM89 不适用 |
| #52842 / #52044 | 已合并、仅 CI / benchmark | ❌ 无线上代码可移植 |
| #49969 / #49731 / #50911 / #52197 / #52188 | 已合并、Qwen3 DSpark / Kimi-K3 TokenSpeed/DCP 专用 | ❌ 当前 DSV4 SM89 fork 不适用 |
| #43008 / #47229 / #48167 / #51768 / #51092 / #49634 / #48256 / #48642 / #50693 / #51602 / #46986 / #46973 / #46789 / #51434 | 已合并 | ❌ 硬件、runner 或特性不匹配，原因见第 5 节 |
| #48519 / #47718 / #48788 / #46275 / #46122 / #49714 / #46730 / #51145 / #47017 / #50434 / #48476 / #45991 / #47677 | 已合并 | ❌ ROCm/XPU 限定 |

---

## 8. 截止快照仍 open 的高相关 PR（不属于官方现行代码）

| PR | open 内容 | 对 SM89 fork 的处理 |
|---|---|---|
| [#52165](https://github.com/vllm-project/vllm/pull/52165) | 从 DeepSeek-V4 checkpoint config 识别 DSpark drafter，避免把 0731/0813 的 DSpark 权重误路由成 MTP | 平台无关且直接相关；在 PR 合并前可参考其 config 判定，但不能记为 upstream 已修复 |
| [#52809](https://github.com/vllm-project/vllm/pull/52809) | 把 #52288 的 backend 继承限定到 DeepSeek-V4，避免破坏 Kimi 等其他 DSpark 架构 | #52288 backport 应采用该限定语义，但不能把 #52809 写成“已进 main” |
| [#52823](https://github.com/vllm-project/vllm/pull/52823) | 用固定 capacity stride 重新实现被 #51318 回滚的 adaptive C128A width | 等合并与上游精度结论稳定后再评估，不能复用 #50004 旧实现 |
| [#52696](https://github.com/vllm-project/vllm/pull/52696) | sparse indexer 使用 FP16 logits；CUDA 微基准与 DSV4/GLM 长上下文验证 | 改动跨 DeepGEMM/CUDA/top-k，需单独做 SM89 性能与精度验证 |
| [#53040](https://github.com/vllm-project/vllm/pull/53040) | 把 shared experts 融入 SM100 MegaMoE | 明确 SM100；fork 的 SM89 Marlin 路径不适用 |
| [#53074](https://github.com/vllm-project/vllm/pull/53074) | `extract_hidden_states` 时隔离 hidden-state cache 与 DSV4 MLA groups | 仅使用 hidden-state extraction/connector 时相关；普通 serving 不触发 |
| [#51802](https://github.com/vllm-project/vllm/pull/51802) | 补 NVIDIA TileLang mHC warmup | fork 有 NVIDIA mHC/TileLang 路径；需先在 SM89 验证 TileLang 支持与实际 warmup 缺口 |
| [#51202](https://github.com/vllm-project/vllm/pull/51202) | 跳过空的 FlashInfer prefill/decode slice | 实现类明确是 `DeepseekV4FlashInferSM120Attention`，SM89 JIT 不直接适用 |
| [#52986](https://github.com/vllm-project/vllm/pull/52986) / [#53022](https://github.com/vllm-project/vllm/pull/53022) | DSV4 LoRA / static expert maps | 分别只在 live-adapter LoRA、EPLB/static placement 场景相关；不属于当前 TP4 常规 serving 必选项 |
| [#52865](https://github.com/vllm-project/vllm/pull/52865) / [#51262](https://github.com/vllm-project/vllm/pull/51262) / [#53071](https://github.com/vllm-project/vllm/pull/53071) / [#52254](https://github.com/vllm-project/vllm/pull/52254) / [#52255](https://github.com/vllm-project/vllm/pull/52255) / [#51229](https://github.com/vllm-project/vllm/pull/51229) / [#50684](https://github.com/vllm-project/vllm/pull/50684) | parser/tokenizer/renderer 的 streaming tool 参数、trailing system、unknown role、thinking tags/effort、空 think block、tools 保留/顺序修复 | 与 API 行为相关、与 kernel 无关；按 fork 实际前端需求单独跟踪 |
| [#52795](https://github.com/vllm-project/vllm/pull/52795) | 在 DSV4+SM90 启用 adaptive verification | SM90 专用，不能视为 SM89 支持证据 |

> OPEN 状态快照截至 2026-08-20；后续状态变化需重新核对 GitHub，不能自动视为已进入官方 `main`。

*初版基于 fork 浅克隆完成；本次更新使用官方 `origin/main` 完整历史、逐 PR GitHub 状态与 fork 代码锚点交叉验证。*
