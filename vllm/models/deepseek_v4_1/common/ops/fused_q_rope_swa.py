# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused BF16 query RoPE and V4.1 FP8/group32 SWA-cache insertion."""

import torch

from vllm.triton_utils import tl, triton

from .quant_utils import SWA_ROW_BYTES, _rotate_gptj_512, _store_swa_row


@triton.jit(do_not_specialize=["num_insert"])
def _fused_q_rope_swa_kernel(
    q,
    kv,
    positions,
    cos_sin,
    cache,
    slots,
    out,
    num_insert,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_ROW_STRIDE: tl.constexpr,
    CACHE_PAGE_SIZE: tl.constexpr,
    CACHE_NUM_PAGES: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    dims = tl.arange(0, 512)
    if head < NUM_HEADS:
        x = tl.load(q + token * Q_TOKEN_STRIDE + head * Q_HEAD_STRIDE + dims)
        position = tl.load(positions + token)
        rotated = _rotate_gptj_512(x.to(tl.float32), position, cos_sin, COS_STRIDE)
        tl.store(out + (token * NUM_HEADS + head) * 512 + dims, rotated)
    elif token < num_insert:
        slot = tl.load(slots + token)
        if (slot >= 0) & (slot < CACHE_NUM_PAGES * CACHE_PAGE_SIZE):
            x = tl.load(kv + token * KV_STRIDE + dims).to(tl.float32)
            position = tl.load(positions + token)
            rotated = _rotate_gptj_512(x, position, cos_sin, COS_STRIDE)
            dst = (
                cache
                + (slot // CACHE_PAGE_SIZE).to(tl.int64) * CACHE_PAGE_STRIDE
                + (slot % CACHE_PAGE_SIZE) * CACHE_ROW_STRIDE
            )
            _store_swa_row(rotated, dst)


def fused_q_rope_swa_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    swa_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE without Q normalization, and store only valid SWA slots."""
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 query [tokens, heads, 512].")
    tokens, heads, _ = q.shape
    if not q.is_cuda or heads not in (8, 16, 32, 64):
        raise ValueError("V4.1 query requires CUDA and 8/16/32/64 local heads.")
    if kv.shape != (tokens, 512) or kv.dtype != q.dtype:
        raise ValueError("Expected BF16 KV [tokens, 512].")
    if q.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("Q/KV last dimension must be contiguous.")
    if positions.ndim != 1 or positions.numel() < tokens:
        raise ValueError("Positions must cover all query rows, including padding.")
    if slot_mapping.ndim != 1 or slot_mapping.numel() > tokens:
        raise ValueError("SWA slots cannot exceed the query token count.")
    if swa_cache.ndim != 3 or swa_cache.shape[-1] != SWA_ROW_BYTES:
        raise ValueError("V4.1 SWA requires 528-byte interleaved cache rows.")
    if swa_cache.dtype != torch.uint8 or swa_cache.stride(-1) != 1:
        raise ValueError("V4.1 SWA cache must use contiguous uint8 rows.")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != 64:
        raise ValueError("Expected rotary cache [positions, 64].")
    if (
        cos_sin_cache.stride(1) != 1
        or not positions.is_contiguous()
        or not slot_mapping.is_contiguous()
    ):
        raise ValueError("Rotary rows, positions and slots must be contiguous.")
    if positions.dtype not in (torch.int32, torch.int64) or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Positions and slots must use integer tensors.")
    if swa_cache.stride(1) < SWA_ROW_BYTES or swa_cache.stride(0) < swa_cache.shape[
        1
    ] * swa_cache.stride(1):
        raise ValueError("SWA cache pages and rows must not overlap.")
    for tensor in (kv, positions, cos_sin_cache, swa_cache, slot_mapping):
        if tensor.device != q.device:
            raise ValueError("All query/cache tensors must use the same CUDA device.")
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if tokens:
        _fused_q_rope_swa_kernel[(tokens, heads + 1)](
            q,
            kv,
            positions,
            cos_sin_cache,
            swa_cache,
            slot_mapping,
            out,
            slot_mapping.numel(),
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            cos_sin_cache.stride(0),
            swa_cache.stride(0),
            swa_cache.stride(1),
            swa_cache.shape[1],
            swa_cache.shape[0],
            heads,
            enable_fp_fusion=False,
            num_warps=4,
        )
    return out
