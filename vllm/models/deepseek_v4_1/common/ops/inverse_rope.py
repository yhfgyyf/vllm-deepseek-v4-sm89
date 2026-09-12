# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused inverse RoPE and grouped BF16 layout for V4.1's official wo_a."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _inverse_rope_group_kernel(
    x,
    positions,
    cos_sin,
    out,
    TOKENS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    X_TOKEN_STRIDE: tl.constexpr,
    X_HEAD_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, 512)
    base = x + token * X_TOKEN_STRIDE + head * X_HEAD_STRIDE
    values = tl.load(base + d).to(tl.float32)
    mate = tl.load(base + (d ^ 1)).to(tl.float32)
    pair = (d - 448) // 2
    position = tl.load(positions + token)
    cs = cos_sin + position.to(tl.int64) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), d >= 448, other=1).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), d >= 448, other=0).to(tl.float32)
    rotated = values * c + tl.where((d & 1) == 0, mate * s, -mate * s)
    group = head // HEADS_PER_GROUP
    within = head % HEADS_PER_GROUP
    offset = ((group.to(tl.int64) * TOKENS + token) * HEADS_PER_GROUP + within) * 512
    tl.store(out + offset + d, rotated)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 32,
    tma_aligned_scales: bool = False,
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inverse-rotated BF16 [T,G,D], backed by a grouped contiguous buffer."""
    if quantize:
        raise ValueError("V4.1 wo_a uses the official BF16 grouped-GEMM contract.")
    if not o.is_cuda or o.dtype != torch.bfloat16 or o.ndim != 3:
        raise ValueError("Expected CUDA BF16 attention output [tokens, heads, 512].")
    tokens, heads, dim = o.shape
    if (nope_dim, rope_dim, dim) != (448, 64, 512):
        raise ValueError("V4.1 inverse RoPE requires 448 NoPE and 64 RoPE dimensions.")
    if n_groups <= 0 or heads_per_group <= 0 or heads != n_groups * heads_per_group:
        raise ValueError("Local heads must divide evenly into the wo_a groups.")
    if o.stride(-1) != 1 or positions.numel() < tokens:
        raise ValueError("Invalid attention output or position layout.")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("Expected rotary cache [positions, 64].")
    if positions.device != o.device or cos_sin_cache.device != o.device:
        raise ValueError("Output, positions and rotary cache must share a CUDA device.")
    if (
        positions.ndim != 1
        or not positions.is_contiguous()
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected contiguous integer positions.")
    if cos_sin_cache.stride(1) != 1:
        raise ValueError("Rotary rows must be contiguous.")
    grouped = torch.empty(
        (n_groups, tokens, heads_per_group * 512),
        dtype=o.dtype,
        device=o.device,
    )
    if tokens:
        _inverse_rope_group_kernel[(tokens, heads)](
            o,
            positions,
            cos_sin_cache,
            grouped,
            tokens,
            heads_per_group,
            o.stride(0),
            o.stride(1),
            cos_sin_cache.stride(0),
            enable_fp_fusion=False,
            num_warps=4,
        )
    return grouped.transpose(0, 1), torch.empty(0, device=o.device, dtype=torch.float32)
