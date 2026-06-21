#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 (Ampere) bring-up smoke test for the DeepSeek-V4-Flash portable path.

Run on the A100 box BEFORE serving the full model:

    .venv/bin/python test_sm80_ops.py

Checks:
  1. The portable-Triton gates report SM 8.x as enabled.
  2. has_cutedsl() is False on SM 8.x (forces Triton/torch indexer fallbacks).
  3. The block-FP8 Triton GEMM (DECODE_E4M3 path) matches a bf16 dequant
     reference -- the dense/attention linear workhorse and the highest-risk
     SM80 change (B-1). Ampere cannot represent fp8e4nv, so the kernel decodes
     e4m3 from uint8 in-register.
  4. The o_proj FP8 einsum (same DECODE_E4M3 path) matches its reference.
"""

import torch

from vllm.platforms import current_platform


def check_gates() -> None:
    import vllm.v1.attention.backends.mla.sparse_mla_env as e
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _sparse_indexer_requires_deep_gemm,
    )
    from vllm.utils.import_utils import has_cutedsl

    cap = current_platform.get_device_capability()
    print(f"device capability: {cap}")
    print(f"supports_fp8 (Ada+ only): {current_platform.supports_fp8()}")
    print(f"is_ampere_or_ada: {e.is_ampere_or_ada()}")
    print(
        f"triton sparse mla (platform): {e.is_triton_sparse_mla_enabled_for_platform()}"
    )
    print(
        "triton sparse mla (device): "
        f"{e.is_triton_sparse_mla_enabled(torch.device('cuda:0'))}"
    )
    print(f"matmul decode: {e.triton_sparse_mla_matmul_decode_enabled()}")
    print(f"has_cutedsl (expect False on SM 8.x): {has_cutedsl()}")
    print(
        "indexer requires_deep_gemm(fp8 cache) "
        f"(expect False): {_sparse_indexer_requires_deep_gemm(False)}"
    )
    print(
        "indexer requires_deep_gemm(fp4 cache) "
        f"(expect True):  {_sparse_indexer_requires_deep_gemm(True)}"
    )

    assert e.is_ampere_or_ada(), "is_ampere_or_ada() should be True on SM 8.x"
    assert e.is_triton_sparse_mla_enabled_for_platform()
    assert not has_cutedsl(), "CuTe-DSL must be disabled on SM 8.x"
    assert _sparse_indexer_requires_deep_gemm(False) is False
    print("  [OK] gates\n")


def check_block_fp8_gemm() -> None:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm,
    )

    torch.manual_seed(0)
    M, N, K = 128, 256, 256
    block_n = block_k = 128
    dev = "cuda"

    # Random fp8 operands (e4m3) + positive per-(token-group)/per-block scales.
    a_f = (torch.randn(M, K, device=dev) * 0.25).clamp(-3, 3)
    b_f = (torch.randn(N, K, device=dev) * 0.25).clamp(-3, 3)
    A = a_f.to(torch.float8_e4m3fn)
    B = b_f.to(torch.float8_e4m3fn)
    As = torch.rand(M, K // block_k, device=dev, dtype=torch.float32) * 0.5 + 0.5
    Bs = (
        torch.rand(N // block_n, K // block_k, device=dev, dtype=torch.float32) * 0.5
        + 0.5
    )

    out = w8a8_triton_block_scaled_mm(
        A, B, As, Bs, [block_n, block_k], output_dtype=torch.bfloat16
    )

    # Reference: dequantize to fp32 (per-block scales broadcast), then A @ B^T.
    a_deq = A.to(torch.float32) * As.repeat_interleave(block_k, dim=1)
    b_deq = B.to(torch.float32) * Bs.repeat_interleave(
        block_n, dim=0
    ).repeat_interleave(block_k, dim=1)
    ref = (a_deq @ b_deq.t()).to(torch.bfloat16)

    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    max_abs = diff.max().item()
    max_rel = rel.max().item()
    print(f"block-FP8 GEMM  max_abs={max_abs:.4f}  max_rel={max_rel:.4f}")
    # bf16 output + fp8 operand quantization -> a few % is expected/healthy.
    assert max_rel < 0.06, f"block-FP8 GEMM rel error too high: {max_rel}"
    print("  [OK] block-FP8 DECODE_E4M3 GEMM\n")


def check_o_proj_einsum() -> None:
    from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
        deepseek_v4_sm12x_fp8_einsum,
    )

    torch.manual_seed(0)
    T, G, H, R = 16, 2, 128, 128  # tokens, groups, hidden, out_rank
    dev = "cuda"
    a_f = (torch.randn(T, G, H, device=dev) * 0.25).clamp(-3, 3)
    b_f = (torch.randn(G, R, H, device=dev) * 0.25).clamp(-3, 3)
    a = a_f.to(torch.float8_e4m3fn)
    b = b_f.to(torch.float8_e4m3fn)
    a_scale = torch.rand(T, G, H // 128, device=dev, dtype=torch.float32) * 0.5 + 0.5
    b_scale = (
        torch.rand(G, R // 128, H // 128, device=dev, dtype=torch.float32) * 0.5 + 0.5
    )
    out = torch.empty(T, G, R, device=dev, dtype=torch.float32)
    deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, out)

    # Reference: bhr,hdr->bhd with block scales (here 1 hidden/out block each).
    a_deq = a.to(torch.float32) * a_scale[:, :, 0].unsqueeze(-1)
    b_deq = b.to(torch.float32) * b_scale[:, 0, 0].view(G, 1, 1)
    ref = torch.einsum("tgh,grh->tgr", a_deq, b_deq)

    diff = (out - ref).abs()
    rel = diff / (ref.abs() + 1e-3)
    max_abs = diff.max().item()
    max_rel = rel.max().item()
    print(f"o_proj einsum   max_abs={max_abs:.4f}  max_rel={max_rel:.4f}")
    assert max_rel < 0.06, f"o_proj einsum rel error too high: {max_rel}"
    print("  [OK] o_proj FP8 einsum DECODE_E4M3\n")


def main() -> None:
    if not current_platform.is_cuda():
        raise SystemExit("CUDA device required")
    check_gates()
    check_block_fp8_gemm()
    check_o_proj_einsum()
    print("ALL SM80 OP CHECKS PASSED")


if __name__ == "__main__":
    main()
