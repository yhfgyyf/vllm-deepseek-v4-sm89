# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native V4.1 index-query RoPE and MXFP4 quantization."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .quant_utils import MXFP4_BLOCK_SIZE, _quantize_mxfp4_pair


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_mxfp4_kernel(
    pos_ptr,
    # Index Q RoPE input (fp/bf16)
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # MXFP4 Q outputs
    index_q_mxfp4_ptr,  # uint8, (T, H, HEAD_DIM // 2)
    index_q_mxfp4_stride0,
    index_q_mxfp4_stride1,
    index_q_scale_ptr,  # uint8 ue8m0, (T, H, HEAD_DIM // BLOCK)
    index_q_scale_stride0,
    index_q_scale_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
    # Weights (NO per-token q_scale fold for MXFP4; per-block scales stay
    # with the Q values in the output scale tensor).
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    NATIVE_FP4: tl.constexpr,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    NUM_NOPE_BLOCKS: tl.constexpr = INDEX_Q_NOPE_DIM // MXFP4_BLOCK
    NUM_ROPE_BLOCKS: tl.constexpr = INDEX_Q_ROT_DIM // MXFP4_BLOCK
    HALF_BLOCK: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)
    tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
    tl.static_assert(MXFP4_BLOCK % 2 == 0)

    tok_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)

    q_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    out_base = (
        index_q_mxfp4_ptr
        + tok_idx * index_q_mxfp4_stride0
        + head_idx * index_q_mxfp4_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + tok_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )

    half_off = tl.arange(0, HALF_BLOCK)

    # ---- NoPE blocks: direct load, pair as (even-index, odd-index) values ----
    for b in tl.static_range(NUM_NOPE_BLOCKS):
        base = b * MXFP4_BLOCK
        x_lo = tl.load(q_base + base + half_off * 2).to(tl.float32)
        x_hi = tl.load(q_base + base + half_off * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi, NATIVE_FP4)
        tl.store(out_base + base // 2 + half_off, packed)
        tl.store(scale_base + b, ue8m0)

    # ---- RoPE blocks: apply GPT-J interleaved RoPE to the block's 16 pairs,
    # then quantize. Each block covers HALF_BLOCK (=16) cos/sin pairs. ----
    rot_q_base = q_base + INDEX_Q_NOPE_DIM
    for b in tl.static_range(NUM_ROPE_BLOCKS):
        pair_off = b * HALF_BLOCK + half_off  # indices in [0, HALF_ROT_DIM)
        cos_b = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + pair_off
        ).to(tl.float32)
        sin_b = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + pair_off
            + INDEX_Q_HALF_ROT_DIM
        ).to(tl.float32)
        x_even = tl.load(rot_q_base + pair_off * 2).to(tl.float32)
        x_odd = tl.load(rot_q_base + pair_off * 2 + 1).to(tl.float32)
        r_even = x_even * cos_b - x_odd * sin_b
        r_odd = x_odd * cos_b + x_even * sin_b
        # bf16 roundtrip for parity with the FP8 kernel / reference numerics.
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd, NATIVE_FP4)
        rope_byte_off = (INDEX_Q_NOPE_DIM + b * MXFP4_BLOCK) // 2
        tl.store(out_base + rope_byte_off + half_off, packed)
        tl.store(scale_base + NUM_NOPE_BLOCKS + b, ue8m0)

    # MXFP4 weight-fold contract:
    #   index_weights_out = index_weights * softmax_scale * head_scale
    # NOTE: q_scale is NOT folded here (contrast with the FP8 kernel above).
    # MXFP4 Q emits a separate ue8m0 scale tensor of shape
    # (T, H, HEAD_DIM // MXFP4_BLOCK) alongside the packed values, so each
    # per-block scale is applied by the downstream MXFP4 logits kernel when
    # dequantizing Q — there is no per-token scalar to fold into `weights`.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return packed Q, row-major UE8M0 scales and scale-independent weights."""
    if not use_fp4:
        raise ValueError("V4.1 native indexer requires MXFP4 Q and K.")
    if not index_q.is_cuda or index_q.dtype != torch.bfloat16:
        raise ValueError("V4.1 indexer Q must be CUDA BF16.")
    if index_q.ndim != 3 or index_q.shape[2] != 128:
        raise ValueError("V4.1 indexer Q must have shape [tokens, heads, 128].")
    tokens, heads = positions.numel(), index_q.shape[1]
    if positions.ndim != 1 or tokens > index_q.shape[0]:
        raise ValueError("Invalid indexer position shape.")
    if index_weights.shape != (tokens, heads) or index_weights.stride(-1) != 1:
        raise ValueError("Indexer weights must have shape [tokens, heads].")
    if index_q.stride(-1) != 1 or index_q_cos_sin_cache.shape[1] != 64:
        raise ValueError("Invalid indexer Q or rotary cache layout.")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
    ):
        raise ValueError("Expected contiguous integer positions.")
    if index_q_cos_sin_cache.stride(1) != 1:
        raise ValueError("Rotary rows must be contiguous.")
    if any(
        t.device != index_q.device
        for t in (positions, index_q_cos_sin_cache, index_weights)
    ):
        raise ValueError("All index query tensors must share a CUDA device.")
    packed = torch.empty((tokens, heads, 64), dtype=torch.uint8, device=index_q.device)
    scales = torch.empty((tokens, heads, 4), dtype=torch.uint8, device=index_q.device)
    weights = torch.empty((tokens, heads), dtype=torch.float32, device=index_q.device)
    if tokens:
        _fused_indexer_q_rope_mxfp4_kernel[(tokens, heads)](
            positions,
            index_q,
            index_q.stride(0),
            index_q.stride(1),
            index_q_cos_sin_cache,
            index_q_cos_sin_cache.stride(0),
            32,
            packed,
            packed.stride(0),
            packed.stride(1),
            scales,
            scales.stride(0),
            scales.stride(1),
            128,
            MXFP4_BLOCK_SIZE,
            index_weights,
            index_weights.stride(0),
            index_weights_softmax_scale,
            index_weights_head_scale,
            weights,
            weights.stride(0),
            NATIVE_FP4=current_platform.is_device_capability_family(120),
            enable_fp_fusion=False,
            num_warps=1,
        )
    return (packed, scales), weights
