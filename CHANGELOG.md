# Changelog

## 2026-08-26 - FlashInfer SM89.2 H8 prefill hotfix

### 中文

- Release 中的 FlashInfer Python wheel 更新为 `0.6.17+sm89.2`，增加 DSV4 sparse MLA prefill `num_heads=8` 的单缓存、双缓存分发和边界保护。
- 现有 vLLM wheel 的依赖元数据从 `flashinfer-python==0.6.17+sm89.1` 更新为 `==0.6.17+sm89.2`；除 `METADATA` 和据此重建的 `RECORD` 外，其余 4471 个文件逐字节一致。
- Python 3.12 / manylinux x86_64 的 uv 无覆盖参数 dry-run 成功解析 196 个包。本次按要求未进行服务器或 GPU 运行测试。

### English

- Updated the Release FlashInfer Python wheel to `0.6.17+sm89.2`, adding single-cache and dual-cache dispatch plus boundary guards for DSV4 sparse MLA prefill with `num_heads=8`.
- Updated the existing vLLM wheel metadata from `flashinfer-python==0.6.17+sm89.1` to `==0.6.17+sm89.2`. All 4,471 files other than `METADATA` and the regenerated `RECORD` remain byte-identical.
- A no-override uv dry-run for Python 3.12 / manylinux x86_64 resolved 196 packages. Server and GPU runtime tests were intentionally not run for this update.

## 2026-08-22 - vLLM upstream sync, DSpark k=7, and CUDA 13.2 wheels

### 中文

- 选择性回移植 vLLM v0.27.1 时期的 DeepSeek-V4 更新：#51727/#51296 tokenizer/parser 修复、#52288 加 #52809 语义的 DSpark backend 继承、#47914 DFlash hybrid causal metadata、#48137 mHC copy 消除、#48660/#47463 DSV4 top-k，以及 #49486/#50298/#52084/#51967/#48957 sparse-index 优化。
- 从 #51538 抽取 SM89 适用的 sparse MLA/SWA correctness 修复；保留 #51430/#52401/#52492 的结构和 follow-up，但 SM89 FlashInfer sparse MLA 继续使用宽 eager CUDA Graph guard。未引入 #47808/#52436 adaptive verification，也未重新引入已回滚的 #50004/#49236。
- 合入 SM89 paged MQA logits int32 地址溢出修复（PR #51）和 Triton per-shape kernel cache 增长修复（PR #61）。
- DSpark 推荐参数更新为 `method=dspark`、`num_speculative_tokens=7`、`draft_sample_method=probabilistic`。4× RTX 4090 上 `8K / 32K -> 1K` 单并发 decode 为 366.95 / 327.38 tok/s。
- CUDA 依赖同步到当前上游版本：`torch 2.13.0+cu130`、`triton 3.7.1`、CUTLASS DSL 4.6.2、QuACK 0.6.4、TileLang 0.1.12、Tokenspeed MLA 0.1.8；SM89 release 使用 CUDA toolkit 13.2、`flashinfer-python 0.6.17+sm89.1` 和 `flashinfer-cubin 0.6.17`。
- 新 wheel 环境通过 `uv pip check`、原生扩展导入、SM89 sparse MLA、8K/32K 源码 A/B、工具调用和 UTF-8 输出测试。

### English

- Selectively backported vLLM v0.27.1-era DeepSeek-V4 updates: #51727/#51296 tokenizer/parser fixes, #52288 with #52809 semantics for DSpark backend inheritance, #47914 DFlash hybrid causal metadata, #48137 mHC copy removal, #48660/#47463 DSV4 top-k, and #49486/#50298/#52084/#51967/#48957 sparse-index optimizations.
- Extracted the SM89-relevant sparse MLA/SWA correctness fixes from #51538. The #51430/#52401/#52492 structure and follow-ups are present, but SM89 FlashInfer sparse MLA retains its wide-eager CUDA Graph guard. #47808/#52436 adaptive verification and upstream-reverted #50004/#49236 remain excluded.
- Included the SM89 paged-MQA-logits int32 addressing fix (PR #51) and Triton per-shape kernel-cache growth fix (PR #61).
- Updated the recommended DSpark config to `method=dspark`, `num_speculative_tokens=7`, and `draft_sample_method=probabilistic`. Single-concurrency decode on 4× RTX 4090 is 366.95 / 327.38 tok/s for `8K / 32K -> 1K`.
- Synced CUDA dependencies to current upstream versions: `torch 2.13.0+cu130`, `triton 3.7.1`, CUTLASS DSL 4.6.2, QuACK 0.6.4, TileLang 0.1.12, and Tokenspeed MLA 0.1.8. The SM89 release uses CUDA toolkit 13.2, `flashinfer-python 0.6.17+sm89.1`, and `flashinfer-cubin 0.6.17`.
- The fresh wheel environment passed `uv pip check`, native-extension imports, SM89 sparse MLA detection, source-vs-wheel 8K/32K A/B runs, tool calling, and UTF-8 output checks.

## 2026-07-10 - FlashInfer 0.6.14 sparse MLA on SM89

### 中文

- 将 SM89 sparse MLA prefill/decode 切换到 FlashInfer 0.6.14 SM89 JIT fork，并为 release 增加匹配的 FlashInfer wheel。
- Lightning Indexer scheduler metadata 改为按 `is_deep_gemm_supported()` 判断；安装了 DeepGEMM 包但硬件不支持的 SM89 不再调用其 metadata API。
- wheel 构建脚本优先使用显式 `VLLM_VERSION_OVERRIDE`，自动推导时忽略历史 CUDA/SM Release tag，避免 `setuptools_scm` 在编译前解析失败。
- 本次不更新 `confidence_head`，不包含 per-request adaptive ℓ，DSpark 固定使用 `ℓ=6`。
- 4× RTX 4090、TP=4、单并发、每组 5 次全部成功。`8K / 32K / 128K -> 1K` 的 Prefill TPS 为 3515.72 / 4881.18 / 3812.00，Decode TPS 为 286.82 / 344.63 / 313.57。

### English

- Switched SM89 sparse MLA prefill/decode to the FlashInfer 0.6.14 SM89 JIT fork and added the matching FlashInfer wheel to the release.
- Gated Lightning Indexer scheduler metadata with `is_deep_gemm_supported()` so SM89 does not call the metadata API merely because the DeepGEMM package is installed.
- Made the wheel build script honor an explicit `VLLM_VERSION_OVERRIDE` and ignore historical CUDA/SM release tags during automatic version resolution, preventing a pre-build `setuptools_scm` parse failure.
- Left `confidence_head` unchanged and excluded per-request adaptive ℓ; DSpark remains fixed at `ℓ=6`.
- On 4× RTX 4090, TP=4, single concurrency, all five requests per case passed. For `8K / 32K / 128K -> 1K`, Prefill TPS is 3515.72 / 4881.18 / 3812.00 and Decode TPS is 286.82 / 344.63 / 313.57.

## 2026-07-06 - SM80/A800 DSpark test adaptation

### 中文

- 增加 DeepSeek-V4-Flash 的 SM80/A800 测试性适配说明。SM80 路径仅用于自测和实验，不代表生产级支持。
- 增加 DSpark 推测解码说明，测试参数为 `method=dspark`、`num_speculative_tokens=6`、`draft_sample_method=greedy`。
- 构建环境切换到 CUDA 13.0 / PyTorch cu130，wheel 构建显式使用 `/usr/local/cuda-13.0`。
- 记录 4× A800 上的 decode 结果：8k 输入、1k 输出、单并发为 229.8 tok/s/req；32k 输入、1k 输出、单并发为 274.2 tok/s/req。对应无 DSpark `mbt16k` 基线分别为 57.6 和 58.1 tok/s/req。

### English

- Added notes for the SM80/A800 DeepSeek-V4-Flash test adaptation. The SM80 path is for experiments and self-testing only, not production-grade support.
- Documented DSpark speculative decoding with `method=dspark`, `num_speculative_tokens=6`, and `draft_sample_method=greedy`.
- Moved the documented build environment to CUDA 13.0 / PyTorch cu130, with wheel builds explicitly using `/usr/local/cuda-13.0`.
- Recorded decode-side 4× A800 results: 229.8 tok/s/req for 8k input -> 1k output, single concurrency; 274.2 tok/s/req for 32k input -> 1k output, single concurrency. The matching no-DSpark `mbt16k` baselines are 57.6 and 58.1 tok/s/req.
