# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer K production for DeepSeek V4.1 kv-source layers.

In v4.1 the index key is derived from the *main* compressor's latent:
``k = k_norm(wk(latent))`` (reference model.py Indexer.forward), then RoPE'd
at the group's first-token position and MXFP4-quantized into the paged
indexer K cache. The ``wk(latent)`` GEMM runs separately; this kernel fuses the
remaining k_norm → RoPE → quant → paged store, one program per token.

Unlike the legacy (v4.0) indexer path there is no per-token pooling from a
compressor state cache: the latent already stands for a whole group, so only
group-boundary tokens ``(position + 1) % compress_ratio == 0`` produce a key.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .quant_utils import MXFP4_BLOCK_SIZE, _ceil_log2_scale, _fp32x2_to_fp4x2


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
) -> None:
    """k_norm → RoPE → quant → paged store for indexer keys.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int64 token positions.
        cos_sin_cache: [max_pos, rope_head_dim] GPT-J layout (cos half, then
            sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        k_cache: uint8 paged indexer cache [num_blocks, block_size, row_bytes].
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        use_fp4_cache: Must be True: MXFP4 with UE8M0 scales per 32 elements.
    """
    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    if not use_fp4_cache:
        raise ValueError("V4.1 native indexer requires MXFP4 keys")
    if not k_pre.is_cuda or k_cache.dtype != torch.uint8:
        raise ValueError("Expected CUDA BF16 keys and uint8 cache")
    if k_cache.ndim != 3 or k_cache.shape[-1] != 68 or k_cache.stride(-1) != 1:
        raise ValueError(
            "Indexer cache requires 64 value bytes and 4 scale bytes per slot"
        )
    if k_cache.stride(0) < k_cache.shape[1] * 68:
        raise ValueError(
            "Indexer pages must contain non-overlapping segregated payloads"
        )
    if rms_norm_weight.shape != (128,) or not rms_norm_weight.is_contiguous():
        raise ValueError("Expected contiguous 128-element index K normalization weight")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError(
            "Expected GPT-J rotary cache with 64 contiguous elements per row"
        )
    if any(
        t.device != k_pre.device
        for t in (positions, cos_sin_cache, rms_norm_weight, k_cache, kv_slot_mapping)
    ):
        raise ValueError("All indexer tensors must use the same CUDA device")
    if positions.dtype not in (
        torch.int32,
        torch.int64,
    ) or kv_slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("Positions and slots must be integer tensors")
    if not positions.is_contiguous() or not kv_slot_mapping.is_contiguous():
        raise ValueError("Positions and slots must be contiguous")
    if num_tokens == 0:
        return

    head_dim = k_pre.shape[1]
    token_stride = head_dim // 2
    scale_dim = head_dim // MXFP4_BLOCK_SIZE
    _indexer_k_norm_rope_quant_store_kernel[(num_tokens,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        NUM_PAGES=k_cache.shape[0],
        NATIVE_FP4=current_platform.is_device_capability_family(120),
        num_warps=1,
        enable_fp_fusion=False,
    )


@triton.jit
def _indexer_k_norm_rope_quant_store_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    NATIVE_FP4: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if (kv_slot_idx < 0) | (kv_slot_idx >= NUM_PAGES * kv_cache_block_size):
        return
    position = tl.load(positions_ptr + token_idx)
    # Only the last token of a group publishes that group's index key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, HEAD_SIZE)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block).to(tl.float32)
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)
    k = k.to(tl.float32)

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))  # each [NUM_PAIRS]
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ── Paged cache pointers (segregated: values first, then scales) ──
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    # MXFP4: each 32-element block = 16 consecutive even/odd pairs, so
    # tiling the halves into (N_BLOCKS, 16) lands one block per row.
    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
    HALF_BLOCK: tl.constexpr = 16
    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))

    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )
    amax = tl.maximum(amax, 6.0 * (2**-126))
    # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored (exp + 127).
    log2_ratio = _ceil_log2_scale(amax * (1.0 / 6.0))
    inv_scale = tl.exp2(-log2_ratio.to(tl.float32))
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

    inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(
        even_2d * inv_scale_col, odd_2d * inv_scale_col, NATIVE_FP4
    )  # (N_BLOCKS, HALF_BLOCK) uint8
    packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)
