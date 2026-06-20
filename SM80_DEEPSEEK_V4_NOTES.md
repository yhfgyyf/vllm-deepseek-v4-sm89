# DeepSeek-V4-Flash on SM80 (Ampere) — experimental branch

Branch: `sm80-deepseek-v4-flash`
Base: `sm89-deepseek-v4-flash` (which extends PR #41834's portable Triton
DeepSeek-V4 path from SM12x to SM89/Ada).

This branch extends the portable Triton DeepSeek-V4 path to **SM80 / SM86
Ampere** (A100 / A800 / A30, A10 / A40 / RTX 3090 / A6000).

> Status: **UNTESTED on hardware.** Code complete; awaiting operator
> validation + `vllm serve` on a remote A100. Build for Ampere only
> (`TORCH_CUDA_ARCH_LIST="8.0+PTX"` for A100, `8.6+PTX` for A10/3090).

---

## The one thing that makes SM80 different from SM89

SM89 (Ada) has **FP8 tensor cores**; SM80/SM86 (Ampere) **does not**. The
central gate is:

```
vllm/platforms/cuda.py::supports_fp8()  ->  has_device_capability(89)  ->  False on SM80
```

The portable Triton path was already written to **dequantize FP8 -> TF32/bf16
before every matmul**, and Ampere has TF32 + bf16 tensor cores. So the attention
/ indexer / einsum / MHC kernels run on SM80 unchanged once the capability gates
are widened. The only genuinely FP8-tensor-core-dependent kernels are the two
that use a native FP8 `tl.dot`; both are now given a bf16-upcast path.

What is still NOT covered (hardware wall, same as SM89): the MoE FP4 expert GEMM.
No SM 8.x has FP4 tensor cores, so the experts fall back to **Marlin WNA16**
(FP4->FP16 dequant, SM80+), exactly as on SM89.

---

## What changed vs the SM89 branch

### A. Capability gates widened `(8,9)` -> `is_device_capability_family(80)`

`family(80)` matches any SM 8.x (8.0 / 8.6 / 8.9), so SM89 behaviour is a strict
subset and is preserved. 8 sites:

- `vllm/v1/attention/backends/mla/sparse_mla_env.py` — `is_ada_sm89()` renamed to
  `is_ampere_or_ada()` (+ `_is_ampere_or_ada_device`); folds SM 8.x into
  `is_triton_sparse_mla_enabled[_for_platform]()`, `matmul_decode`, C128A topk.
- `vllm/utils/deep_gemm.py` — `_use_sm12x_mqa_fallback()` (5 MQA / HC GEMM
  dispatchers).
- `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` —
  `_use_sm12x_fallback()` (2 fused-topk guards).
- `vllm/model_executor/kernels/mhc/tilelang.py` — MHC TF32 prenorm GEMM gate.
- `vllm/model_executor/layers/sparse_attn_indexer.py` —
  `_sparse_indexer_requires_deep_gemm()` (avoids init-time crash with FP8-Q cache).
- `vllm/v1/attention/backends/mla/indexer.py` — shared 256 MB transient-logits
  budget.
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability()` now
  accepts `major == 8`.
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — `_use_..._triton_fp8_einsum`
  accepts `major in (8, 12)`. The existing `upcast_fp8 = not family(120)` already
  fires on SM80.
- `vllm/utils/import_utils.py` — `has_cutedsl()` returns False on all SM 8.x
  (CuTe-DSL targets SM90+ and emits PTX ptxas rejects on Ampere/Ada).

### B. The dense/attention FP8 linear path (the real SM80 work)

DeepSeek-V4-Flash quantizes **all dense + attention linears to FP8 block (128)**
(`quant_config.py`: *"DeepSeek V4 checkpoints always use FP8 block quantization
for linear/attention layers"*). On SM80 the block-FP8 kernel priority list
(`kernels/linear/__init__.py::_POSSIBLE_FP8_BLOCK_KERNELS`) resolves to the
**Triton** backend (DeepGEMM/FlashInfer/Cutlass-block need SM90+; Marlin
explicitly rejects block-FP8), whose kernel `_w8a8_triton_block_scaled_mm` does a
**native FP8 `tl.dot`** — which Ampere cannot lower.

Fix (same pattern as the o_proj einsum): add an `UPCAST_FP8` constexpr to
`_w8a8_triton_block_scaled_mm` and upcast both operands to bf16 before the dot.
The launcher `w8a8_triton_block_scaled_mm` sets
`upcast_fp8 = is_cuda and not supports_fp8() and A/B are e4m3`, so SM89/SM90/
SM100/SM12x keep the native FP8 dot unchanged and only Ampere upcasts. Block
scales are applied after the dot, so the upcast is numerically lossless.
File: `vllm/model_executor/layers/quantization/utils/fp8_utils.py`.

This single kernel covers every dense/attention linear (q_proj, kv_a/kv_b,
o_proj dense GEMM, shared expert, the compressor's fused_wqa_wkv) at once.

### Verified safe without changes
- Sparse-MLA attention (`sparse_mla_kernels.py`): `tl.dot(q, kv)` operands are
  the **dequantized** (bf16) KV from `dequantize_and_gather_k_cache`, not FP8.
- MQA logits (`sm12x_mqa.py`): FP8 loaded then `.to(tl.float32)`, `tl.dot(...,
  input_precision="tf32")`. TF32 works on Ampere.
- Activation/scale quantization: elementwise (CUDA cores), arch-independent.
- Per-tensor/per-channel FP8 (`torch._scaled_mm`/cutlass): **not used** by this
  model — all linears are block-FP8 — so the SM89-only `_scaled_mm` is never hit.

---

## Build on the SM80 server

```bash
# uv per AGENTS.md (never system pip/python)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate

# A100 = 8.0; A10/A40/3090/A6000 = 8.6. Full source build (C/C++ changes on
# this branch; cannot use VLLM_USE_PRECOMPILED). CMake skips the SM90/SM100
# FlashMLA sources and still builds Marlin WNA16 (SM80+). DeepGEMM is NOT
# installed — the Triton fallbacks replace it.
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
uv pip install -e . --no-build-isolation --torch-backend=auto
```

`vllm._flashmla_C` being unavailable after the build is expected.

---

## Operator validation (run first, no full model needed)

```bash
.venv/bin/python test_sm80_ops.py
```

The script checks (1) the SM80 gates report enabled, (2) `has_cutedsl()` is
False, (3) the block-FP8 Triton GEMM with `UPCAST_FP8` matches a bf16 dequant
reference within tolerance — this is the B-1 fix and the highest-risk path.

---

## Serve

```bash
VLLM_TRITON_MLA_SPARSE=1 \
.venv/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --trust-remote-code
# add --kernel-config moe_backend=marlin if MoE backend auto-selection errors
```

Same launch-flag rules as SM89: do NOT pass `moe_backend=deep_gemm_mega_moe`,
`--attention-backend FLASHINFER_MLA_SPARSE_DSV4`, or `use_fp4_indexer_cache`.

---

## Known risks to watch during testing

1. **Block-FP8 GEMM `UPCAST_FP8`** — the dense/attention workhorse. Validate
   `test_sm80_ops.py` passes before trusting outputs.
2. **o_proj FP8 einsum** — also upcasts; if o_proj is wrong, confirm the
   `UPCAST_FP8` branch fired.
3. **Marlin MoE FP4 experts** — same correctness/perf risk as SM89.
4. **Perf is untuned** — no Ampere fused_moe / scaled_mm tuned configs; default
   heuristics. The bf16-upcast block GEMM is slower than native FP8 MMA.
5. **bf16 throughput on A100** is good, but the FP8->bf16 upcast doubles operand
   read bandwidth in the block GEMM; expect lower tok/s than SM89.
