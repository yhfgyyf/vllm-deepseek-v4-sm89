# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 state saving/compression and independently schedulable cache insertion."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .quant_utils import GLOBAL_ROW_BYTES, _rotate_gptj_512, _store_global_row

if current_platform.is_rocm():
    from vllm.platforms.rocm import _ON_GFX950
else:
    _ON_GFX950 = False

# The ring and raw rows a ratio-2 request program pools are not adjacent, so
# reading them as one tile means joining two pointers. ROCm's Triton fails to
# legalize `tt.join` on pointers, so there the two rows are loaded separately.
_JOIN_ROW_PTRS = not current_platform.is_rocm()


def fused_save_compress_norm(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """Pool each closed group into a normalized BF16 latent; save FP32 states.

    The latent feeds the main-cache insert and the indexer K path, which the
    attention layer schedules on separate streams.

    Ratio 2 keeps one ring block per request holding the open group's rows:
    position ``p`` lives in row ``p % capacity`` and ``slot_mapping`` encodes
    ``block * capacity + p % capacity``. The grid has one program per request
    followed by one per pair of packed tokens. A request program handles the
    group that the chunk's first token closes with its predecessor's ring row,
    then stores the chunk's last ``capacity`` rows to the ring; because the
    same program does both, ring reads and writes never race. A pair program
    handles the group that ends inside its pair, reading both rows from the
    raw input. Ratio 1 has no ring and one program per token; ``slot_mapping``
    then only marks valid tokens.

    Args:
        kv_score: FP32 [tokens, 512] for CR1, [tokens, 1024] for CR2.
        positions: Absolute positions of the packed request tokens.
        state_cache: Ring FP32 [blocks, capacity, 1024] KV/score states (CR2).
        slot_mapping: Ring slots (CR2) or main-cache slots (CR1).
        query_start_loc: [num_reqs + 1] token offsets of each request's chunk.
        token_to_req_indices: Request indices for the packed token rows.
        rms_norm_weight: BF16 [512] normalization weight.
        rms_norm_eps: RMSNorm epsilon.
        compress_ratio: Group size, either 1 or 2.
        latent_out: BF16 [tokens, 512], written only at valid group boundaries.
    """
    assert compress_ratio in (1, 2)
    assert kv_score.dtype == torch.float32
    assert kv_score.shape[1] == 512 * compress_ratio and kv_score.stride(1) == 1
    assert latent_out.shape == (kv_score.shape[0], 512)
    assert latent_out.is_contiguous() and latent_out.dtype == torch.bfloat16
    assert positions.is_contiguous() and slot_mapping.is_contiguous()
    # Rows stay 64-byte aligned, as the kernel's tl.multiple_of hint promises.
    assert kv_score.stride(0) % 16 == 0
    if compress_ratio == 2:
        assert state_cache is not None and query_start_loc is not None
        assert token_to_req_indices is not None
        assert query_start_loc.is_contiguous()
        assert token_to_req_indices.is_contiguous()
        assert state_cache.dtype == torch.float32
        assert state_cache.shape[2] == 1024 and state_cache.stride(2) == 1
        assert state_cache.stride(1) % 16 == 0
        state_stride, state_row_stride, state_block = (
            state_cache.stride(0),
            state_cache.stride(1),
            state_cache.shape[1],
        )
        num_reqs = query_start_loc.numel() - 1
    else:
        state_cache = query_start_loc = token_to_req_indices = None
        state_stride = state_row_stride = state_block = 1
        num_reqs = 0
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(kv_score.shape[0], positions.numel())
    if num_tokens == 0:
        return
    grid = num_reqs + triton.cdiv(num_tokens, compress_ratio)
    _fused_save_compress_norm_kernel[(grid,)](
        kv_score,
        positions,
        state_cache,
        slot_mapping,
        query_start_loc,
        token_to_req_indices,
        rms_norm_weight,
        latent_out,
        num_tokens,
        num_reqs,
        RAW_STRIDE=kv_score.stride(0),
        STATE_STRIDE=state_stride,
        STATE_ROW_STRIDE=state_row_stride,
        STATE_BLOCK=state_block,
        COMPRESS_RATIO=compress_ratio,
        EPS=rms_norm_eps,
        JOIN_ROW_PTRS=_JOIN_ROW_PTRS,
        num_warps=4,
        **({"launch_pdl": False} if current_platform.is_cuda() else {}),
    )


@triton.jit(do_not_specialize=["num_tokens", "num_reqs"])
def _fused_save_compress_norm_kernel(
    raw,
    positions,
    state,
    state_slots,
    query_start_loc,
    req_ids,
    norm_weight,
    latent,
    num_tokens,
    num_reqs,
    RAW_STRIDE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_ROW_STRIDE: tl.constexpr,
    STATE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    EPS: tl.constexpr,
    JOIN_ROW_PTRS: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, 512)

    if COMPRESS_RATIO == 2:  # noqa: SIM102 (constexpr branch, then runtime)
        if pid < num_reqs:
            # Request program: the group closed by the chunk's first token,
            # then the ring tail store.
            start = tl.load(query_start_loc + pid)
            end = tl.load(query_start_loc + pid + 1)
            if start >= end:
                return
            state_slot = tl.load(state_slots + start)
            if state_slot < 0:
                return
            position = tl.load(positions + start)
            if (position + 1) % 2 == 0:
                ring = state + (state_slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE
                current = raw + start.to(tl.int64) * RAW_STRIDE
                if JOIN_ROW_PTRS:
                    # Joining pointers hides row alignment from Triton; restate it.
                    rows = tl.multiple_of(tl.join(prev, current)[:, None], (16, 16))
                    kv = tl.load(rows + d[None, :])
                    score = tl.load(rows + 512 + d[None, :])
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    pooled = tl.sum(kv * tl.softmax(score, 0), 0)
                else:
                    kv_prev = tl.load(prev + d)
                    score_prev = tl.load(prev + 512 + d)
                    kv_current = tl.load(current + d)
                    score_current = tl.load(current + 512 + d)
                    # Every lane has read the ring before the tail store below
                    # writes it.
                    tl.debug_barrier()
                    peak = tl.maximum(score_prev, score_current)
                    weight_prev = tl.exp(score_prev - peak)
                    weight_current = tl.exp(score_current - peak)
                    pooled = (kv_prev * weight_prev + kv_current * weight_current) / (
                        weight_prev + weight_current
                    )
                _store_latent(pooled, start, norm_weight, latent, EPS)
            num_rows = tl.minimum(end - start, STATE_BLOCK)
            for k in tl.range(0, num_rows):
                token = end - num_rows + k
                slot = tl.load(state_slots + token)
                if slot >= 0:
                    row = (
                        state
                        + (slot // STATE_BLOCK).to(tl.int64) * STATE_STRIDE
                        + (slot % STATE_BLOCK) * STATE_ROW_STRIDE
                    )
                    src = raw + token.to(tl.int64) * RAW_STRIDE
                    tl.store(row + d, tl.load(src + d))
                    tl.store(row + 512 + d, tl.load(src + 512 + d))
            return

    # Group program: one token (ratio 1) or the token of the pair that ends
    # a group (ratio 2). A chunk's first token belongs to its request program.
    t = (pid - num_reqs) * COMPRESS_RATIO
    if COMPRESS_RATIO == 2:
        t += (tl.load(positions + t) + 1) % 2
    state_slot = tl.load(state_slots + t, t < num_tokens, other=-1)
    if state_slot < 0:
        return
    if COMPRESS_RATIO == 1:
        pooled = tl.load(raw + t.to(tl.int64) * RAW_STRIDE + d)
    else:
        if t == tl.load(query_start_loc + tl.load(req_ids + t)):
            return
        rows = raw + (t.to(tl.int64) - 1 + tl.arange(0, 2)[:, None]) * RAW_STRIDE
        kv = tl.load(rows + d[None, :])
        score = tl.load(rows + 512 + d[None, :])
        pooled = tl.sum(kv * tl.softmax(score, 0), 0)
    _store_latent(pooled, t, norm_weight, latent, EPS)


@triton.jit
def _store_latent(pooled, t, norm_weight, latent, EPS: tl.constexpr):
    d = tl.arange(0, 512)
    weight = tl.load(norm_weight + d).to(tl.float32)
    variance = tl.sum(pooled * pooled, 0) / 512
    normed = pooled * tl.rsqrt(variance + EPS) * weight
    tl.store(latent + t.to(tl.int64) * 512 + d, normed.to(tl.bfloat16))


@triton.jit
def _rope_fp4_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    slots,
    COS_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    NATIVE_FP4: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + token)
    if (slot < 0) | (slot >= NUM_PAGES * PAGE_SIZE):
        return
    position = tl.load(positions + token)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    x = tl.load(latent + token * 512 + tl.arange(0, 512)).to(tl.float32)
    position = position // COMPRESS_RATIO * COMPRESS_RATIO
    rotated = _rotate_gptj_512(x, position, cos_sin, COS_STRIDE)
    dst = cache + (slot // PAGE_SIZE).to(tl.int64) * PAGE_STRIDE
    dst += (slot % PAGE_SIZE) * ROW_STRIDE
    _store_global_row(rotated, dst, NATIVE_FP4)


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Fuse post-RoPE FP4/group16 conversion and paged global-KV insertion."""
    if compress_ratio not in (1, 2) or fp8_scale is not None:
        raise ValueError("V4.1 global KV requires C1/C2 NVFP4 without global scale.")
    if not latent.is_cuda or latent.dtype != torch.bfloat16:
        raise ValueError("V4.1 compressed latent must be CUDA BF16.")
    if latent.ndim != 2 or latent.shape[1] != 512 or not latent.is_contiguous():
        raise ValueError("Expected contiguous latent [tokens, 512].")
    if kv_cache.ndim != 3 or kv_cache.shape[-1] != GLOBAL_ROW_BYTES:
        raise ValueError("V4.1 global KV requires 288-byte interleaved rows.")
    if kv_cache.dtype != torch.uint8 or kv_cache.stride(-1) != 1:
        raise ValueError("V4.1 global KV cache must use contiguous uint8 rows.")
    if kv_cache.stride(1) < GLOBAL_ROW_BYTES or kv_cache.stride(0) < kv_cache.shape[
        1
    ] * kv_cache.stride(1):
        raise ValueError("Global KV pages and rows must not overlap.")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.stride(1) != 1
    ):
        raise ValueError("Expected GPT-J rotary cache with contiguous 64-element rows.")
    if positions.dtype not in (torch.int32, torch.int64) or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Positions and slots must use integer tensors.")
    tokens = slot_mapping.numel()
    if tokens > min(latent.shape[0], positions.numel()):
        raise ValueError("Cache slots exceed latent/position rows.")
    if not slot_mapping.is_contiguous() or not positions.is_contiguous():
        raise ValueError("Cache slots and positions must be contiguous.")
    for tensor in (positions, cos_sin_cache, kv_cache, slot_mapping):
        if tensor.device != latent.device:
            raise ValueError("All compressor tensors must use the same CUDA device.")
    if tokens:
        _rope_fp4_insert_kernel[(tokens,)](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            cos_sin_cache.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.shape[1],
            kv_cache.shape[0],
            compress_ratio,
            current_platform.is_device_capability_family(120),
            enable_fp_fusion=False,
            num_warps=4,
        )
