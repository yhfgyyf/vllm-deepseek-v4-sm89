# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped W8A16 `wo_a` for DeepSeek V4.1 (enabled on SM120; portable Triton).

Keeps the checkpoint's FP8 `wo_a` with UE8M0 K32 scales and dequantizes each
tile to BF16 in registers. Dequantization is exact, so the BF16 MMA with FP32
accumulation matches the official BF16-weight GEMM up to accumulation order.
Split-K partials are reduced in a fixed order. Large token counts dequantize
the weight into scratch and run the production BF16 grouped GEMM.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import current_workspace_manager

_SCALE_GROUP = 32
_MAX_DIRECT_TOKENS = 128


@triton.jit(do_not_specialize=["T"])
def _woa_w8a16_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    part_ptr,
    T,
    R,
    stride_xg,
    stride_xt,
    stride_wg,
    stride_wr,
    stride_sg,
    stride_sr,
    stride_out,
    stride_part_k,
    stride_part_t,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    tiles_n = tl.cdiv(R, BN)
    group = pid_n // tiles_n
    rows = pid_m * BM + tl.arange(0, BM)
    cols = (pid_n % tiles_n) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    offs_s = tl.arange(0, BK // 32)
    row_mask = rows < T
    col_mask = cols < R
    x_base = x_ptr + group.to(tl.int64) * stride_xg + rows[:, None] * stride_xt
    w_base = w_ptr + group.to(tl.int64) * stride_wg + cols[:, None] * stride_wr
    s_base = s_ptr + group.to(tl.int64) * stride_sg + cols[:, None] * stride_sr
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_begin = pid_k * K_PER_SPLIT
    for step in range(0, K_PER_SPLIT, BK):
        k = k_begin + step
        x = tl.load(x_base + (k + offs_k)[None, :], mask=row_mask[:, None], other=0.0)
        w = tl.load(w_base + (k + offs_k)[None, :], mask=col_mask[:, None], other=0.0)
        s = tl.load(
            s_base + (k // 32 + offs_s)[None, :], mask=col_mask[:, None], other=127
        )
        scale = tl.exp2(s.to(tl.float32) - 127.0)
        w = tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * scale[:, :, None]
        w = tl.reshape(w, (BN, BK)).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(w), acc)
    out_cols = group * R + cols
    mask = row_mask[:, None] & col_mask[None, :]
    if SPLIT_K == 1:
        tl.store(
            out_ptr + rows[:, None] * stride_out + out_cols[None, :],
            acc.to(tl.bfloat16),
            mask=mask,
        )
    else:
        tl.store(
            part_ptr
            + pid_k * stride_part_k
            + rows[:, None] * stride_part_t
            + out_cols[None, :],
            acc,
            mask=mask,
        )


@triton.jit
def _woa_split_reduce_kernel(
    part_ptr,
    out_ptr,
    N,
    stride_part_k,
    stride_part_t,
    stride_out,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            part_ptr + split * stride_part_k + row * stride_part_t + cols,
            mask=mask,
            other=0.0,
        )
    tl.store(out_ptr + row * stride_out + cols, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + offs // 32, mask=mask, other=127).to(tl.float32)
    tl.store(out_ptr + offs, (w * tl.exp2(s - 127.0)).to(tl.bfloat16), mask=mask)


def _config(tokens: int, num_groups: int, depth: int) -> tuple[int, int, int, int]:
    """(BM, BN, BK, split-K), from SM120 sweeps at R=1024, D=4096."""
    if tokens <= 16:
        block_m, block_n, block_k = 16, 64, 128 if num_groups == 1 else 64
    elif tokens <= 32:
        block_m, block_n, block_k = 32, 64, 128
    elif tokens <= 64:
        block_m, block_n, block_k = 64, 128, 64
    else:
        block_m, block_n, block_k = 64, 128, 128
    split = 8 if tokens <= 64 else 4
    while split > 1 and depth % (split * block_k):
        split //= 2
    return block_m, block_n, block_k, split


def woa_w8a16_workspace_bytes(num_groups: int, rank: int, depth: int) -> int:
    """Largest scratch any call can request, for reservation at init."""
    partial = 8 * _MAX_DIRECT_TOKENS * num_groups * rank * 4
    return max(partial, num_groups * rank * depth * 2)


def woa_w8a16(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Compute `out[t, g*R + r] = sum_d x[g, t, d] * W[g, r, d]`.

    Args:
        x: BF16 `[G, T, D]` grouped attention output.
        weight: float8_e4m3fn `[G, R, D]`.
        scale: uint8 UE8M0 `[G, R, D // 32]`.

    Returns:
        BF16 `[T, G * R]`, the `wo_b` input layout.

    """
    num_groups, tokens, depth = x.shape
    rank = weight.shape[1]
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or weight.shape != (num_groups, rank, depth)
        or scale.shape != (num_groups, rank, depth // _SCALE_GROUP)
        or depth % 128
    ):
        raise ValueError("Unexpected V4.1 W8A16 wo_a operands.")
    if x.stride(2) != 1 or not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("V4.1 W8A16 wo_a requires unit inner strides.")
    if tokens > _MAX_DIRECT_TOKENS:
        (dequantized,) = current_workspace_manager().get_simultaneous(
            ((num_groups, rank, depth), torch.bfloat16)
        )
        numel = weight.numel()
        block = 4096
        _dequant_kernel[(triton.cdiv(numel, block),)](
            weight, scale, dequantized, numel, BLOCK=block
        )
        return torch.einsum("bhd,hrd->bhr", x.transpose(0, 1), dequantized).flatten(1)
    out = torch.empty((tokens, num_groups * rank), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    block_m, block_n, block_k, split = _config(tokens, num_groups, depth)
    part = out
    if split > 1:
        (part,) = current_workspace_manager().get_simultaneous(
            ((split, tokens, num_groups * rank), torch.float32)
        )
    grid = (
        triton.cdiv(tokens, block_m),
        num_groups * triton.cdiv(rank, block_n),
        split,
    )
    _woa_w8a16_kernel[grid](
        x,
        weight,
        scale,
        out,
        part,
        tokens,
        rank,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        part.stride(0) if split > 1 else 0,
        part.stride(1) if split > 1 else 0,
        BM=block_m,
        BN=block_n,
        BK=block_k,
        K_PER_SPLIT=depth // split,
        SPLIT_K=split,
        num_warps=4,
        num_stages=3,
    )
    if split > 1:
        block = 1024
        _woa_split_reduce_kernel[(tokens, triton.cdiv(num_groups * rank, block))](
            part,
            out,
            num_groups * rank,
            part.stride(0),
            part.stride(1),
            out.stride(0),
            SPLIT_K=split,
            BLOCK=block,
            num_warps=4,
        )
    return out
