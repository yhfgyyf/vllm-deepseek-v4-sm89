# DeepSeek-V4-Flash on SM89 (Ada Lovelace) — experimental branch

Branch: `sm89-deepseek-v4-flash`
Base: PR #41834 head (`[New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash`),
itself based on upstream vLLM `main`.

This branch extends PR #41834's **portable Triton DeepSeek-V4 path** (built for
SM12x / consumer Blackwell) to also run on **SM89 / Ada Lovelace**
(RTX 4090, L40 / L40S, L4, RTX 6000 Ada, A6000 Ada).

> Status: **validated on 4× RTX 4090 (SM89), 2026-06-16.** vLLM 0.11.1 built
> from source (torch 2.11.0+cu128, CUDA 12.8), `vllm serve DeepSeek-V4-Flash`
> with TP=4, `--kv-cache-dtype fp8_ds_mla`, `--enforce-eager`. Startup reached
> "Application startup complete"; chat completions return correct, fluent
> output. MoE auto-selected the **MARLIN** backend; indexer used the **FP8**
> cache; FlashMLA/DeepGEMM correctly reported unsupported and were bypassed.

---

## Why this can work on SM89

The PR replaced every SM90/SM100-only kernel (FlashMLA sparse, DeepGEMM
`fp8_einsum` / `mqa_logits` / MXFP4 grouped GEMM, DeepGEMM HC GEMM) with
portable Triton / torch fallbacks. Those Triton kernels use **TF32/FP32**
`tl.dot` (Ampere+), so Ada runs them. The single FP8-operand `tl.dot`
(o_proj einsum) is given a bf16-upcast path for Ada.

## What is NOT covered (hardware wall, not a gate)

Ada has **FP8 tensor cores but no FP4 tensor cores and no hardware
microscaling MMA**. So the MoE FP4 expert GEMM (FlashInfer CUTLASS
MXFP4×MXFP8 / TRTLLM MXFP4 / DeepGEMM MXFP4) **cannot** run natively on SM89.
The model's standard fused-MoE path auto-falls-back to the **Marlin WNA16**
backend (FP4→FP16 dequant, SM80+), which works on Ada but is slower than
native FP4 MMA. The DeepGEMM "MegaMoE" path stays SM100-only and is not used
unless explicitly requested.

---

## Required / recommended launch flags on SM89

| Setting | Why |
| --- | --- |
| Do **NOT** pass `--kernel-config moe_backend=deep_gemm_mega_moe` | MegaMoE requires SM100; default `auto` → Marlin on Ada. |
| Do **NOT** pass `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` | That path has no Triton fallback; default backend does. |
| Do **NOT** enable `use_fp4_indexer_cache` | FP4 indexer cache is SM100-only; FP8-Q indexer is the default and is covered. |
| KV cache: `--kv-cache-dtype fp8_ds_mla` (the DSA fp8 layout) | Matches the indexer/sparse-MLA fp8 path. |
| (optional) `VLLM_TRITON_MLA_SPARSE=1` | Force the Triton path. It is auto-enabled on SM89, so usually unnecessary; `=0` force-disables (will then crash — only for A/B). |
| (optional) `--kernel-config moe_backend=marlin` | Pin Marlin explicitly if auto-selection ever errors. |

If the Marlin MoE rounds intermediate/hidden up, that is expected
(`mxfp4_round_up_hidden_size_and_intermediate_size`).

---

## Build on the SM89 server

```bash
# uv per AGENTS.md (never system pip/python)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install   # optional, for linting

# Build for Ada ONLY. This makes CMake skip the SM90/SM100 FlashMLA CUDA
# sources (they would fail to compile for 8.9) while still building the
# Marlin MoE WNA16 kernels (SM80+). DeepGEMM is a separate package and is
# intentionally NOT installed — the Triton fallbacks replace it.
#
# NOTE: cannot use VLLM_USE_PRECOMPILED — this branch carries C/C++ changes
# and is not a release tag, so a full source build is required.
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
uv pip install -e . --no-build-isolation --torch-backend=auto
```

It is fine (expected) for `vllm._flashmla_C` to be unavailable after the
build — `is_flashmla_*_supported()` returns False on Ada and all call sites
are routed to the Triton path instead.

---

## Smoke test (no full model needed)

```bash
# 1) The portable-path gates report SM89 as enabled:
.venv/bin/python - <<'PY'
import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import sparse_mla_env as e
cap = current_platform.get_device_capability()
print("device capability:", cap)
print("is_ada_sm89:", e.is_ada_sm89())
print("triton sparse mla (platform):", e.is_triton_sparse_mla_enabled_for_platform())
print("triton sparse mla (device):", e.is_triton_sparse_mla_enabled(torch.device("cuda:0")))
print("matmul decode:", e.triton_sparse_mla_matmul_decode_enabled())
PY

# 2) Indexer no longer demands DeepGEMM for the FP8-Q path on Ada:
.venv/bin/python - <<'PY'
from vllm.model_executor.layers.sparse_attn_indexer import (
    _sparse_indexer_requires_deep_gemm,
)
print("requires_deep_gemm(fp8 cache):", _sparse_indexer_requires_deep_gemm(False))  # expect False
print("requires_deep_gemm(fp4 cache):", _sparse_indexer_requires_deep_gemm(True))   # expect True
PY
```

## Serve

Use the official DeepSeek-V4-Flash command but drop EP/MegaMoE and keep TP
small for a single Ada box, e.g.:

```bash
VLLM_TRITON_MLA_SPARSE=1 \
.venv/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code
# add --kernel-config moe_backend=marlin if MoE backend auto-selection errors
```

---

## What changed (8 files)

Central switch — `vllm/v1/attention/backends/mla/sparse_mla_env.py`:
adds `is_ada_sm89()` / `_is_sm89_device()` and folds SM89 into
`is_triton_sparse_mla_enabled[_for_platform]()`,
`triton_sparse_mla_matmul_decode_enabled()`, and the C128A topk tuning.

Dispatch sites extended from `family(120)` to `family(120) or SM89`:
- `vllm/utils/deep_gemm.py` — `_use_sm12x_mqa_fallback()` for the 5
  `fp8_fp4_*mqa_*` + `tf32_hc_prenorm_gemm` dispatchers.
- `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` —
  `_use_sm12x_fallback()` for the 2 fused-topk guards.
- `vllm/model_executor/kernels/mhc/tilelang.py` — MHC TF32 prenorm GEMM gate.
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` — Triton FP8 einsum
  dispatch + **bf16 upcast** of the FP8 `tl.dot` for Ada (constexpr `UPCAST_FP8`).
- `vllm/model_executor/layers/sparse_attn_indexer.py` —
  `_sparse_indexer_requires_deep_gemm()` (prevents an init-time crash on Ada
  with the default FP8-Q cache).
- `vllm/v1/attention/backends/mla/indexer.py` — SM89 shares SM12x's smaller
  transient-logits memory budget.
- `vllm/models/deepseek_v4/sparse_mla.py` — `supports_compute_capability()`
  made accurate (adds 12 and exact 8.9; defensive — this method does not gate
  the model-provided backend).
- `vllm/utils/import_utils.py` — `has_cutedsl()` returns False on SM89. The
  DeepSeek-V4 indexer Q-rope-quant and KV-dequant-gather paths pick a CuTe-DSL
  (`nvidia-cutlass-dsl`) kernel whenever the package is importable (it ships as
  a FlashInfer dependency). Those CuTe-DSL kernels target SM90+ and emit
  `mul.bf16x2` / `cvt.bf16.f16` PTX that ptxas rejects on Ada, crashing engine
  init during warmup. Disabling cutedsl on SM89 forces the Triton/torch
  fallbacks. **This was found during hardware bring-up — the static audit
  missed it because the dispatch keys on package availability, not capability.**

Deliberately **unchanged** (correct as-is for SM89):
- MoE FP4 backends (`cutlass_moe`, `flashinfer_cutlass_moe`,
  `flashinfer_b12x_moe`) — Blackwell FP4 MMA; SM89 falls through to Marlin.
- `scaled_mm/cutlass.py` — SM89 uses the standard CUTLASS FP8 path already.
- `marlin_utils.py` / `fp8_utils.py` — SM89 already takes the right branch.
- `indexer.py::_uses_deep_gemm_scheduler_metadata` — `has_deep_gemm()` is
  False on SM89, so it is already disabled.
- SM120-only perf heuristics (short-row topk decode, `low_m_limit`).

---

## Known risks to watch during testing

1. **FP8 `tl.dot` on Ada** — mitigated by the bf16 upcast in `fp8_einsum.py`.
   If o_proj is wrong, check the `UPCAST_FP8` path fired.
2. **Marlin MoE correctness/perf** for the FP4 experts on this routing — the
   highest-risk area. Verify GSM8K / a few prompts before trusting outputs.
3. **TileLang MHC kernels** compiling for sm_89 — if MHC fails, the TF32
   path may need the same upcast treatment or a torch fallback.
4. **Perf is untuned** — the fused_moe / scaled_mm tuned configs only cover
   RTX PRO 6000 / GB10; SM89 uses default heuristics.
5. **MTP next_n flattening** — with MTP the indexer uses `use_flattening`;
   exercise both MTP on/off.
