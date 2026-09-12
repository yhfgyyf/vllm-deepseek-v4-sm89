# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native delayed mHC with shape-specific projection for Ada and RTX Blackwell."""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _prenorm_project_kernel(
    x,
    weight,
    mixed,
    squared,
    TOKENS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    cols = tl.arange(0, 32)
    offsets = tl.arange(0, BK)
    acc = tl.zeros((BM, 32), tl.float32)
    sumsq = tl.zeros((BM,), tl.float32)
    for i in range(K // SPLITS // BK):
        ks = split * (K // SPLITS) + i * BK + offsets
        a = tl.load(x + rows[:, None] * K + ks[None, :], rows[:, None] < TOKENS, 0)
        b = tl.load(weight + cols[None, :] * K + ks[:, None], cols[None, :] < N, 0)
        a = a.to(tl.float32)
        # FN is FP32: TF32x3 retains its precision without a BF16 weight copy.
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
        sumsq += tl.sum(a * a, 1)
    base = split.to(tl.int64) * TOKENS
    tl.store(
        mixed + (base + rows[:, None]) * N + cols[None, :],
        acc,
        (rows[:, None] < TOKENS) & (cols[None, :] < N),
    )
    tl.store(squared + base + rows, sumsq, rows < TOKENS)


def _native_hc_prenorm_gemm(x, fn, mixes, sqrsum):
    tokens, width = x.shape
    splits = mixes.shape[0]
    if width % (64 * splits):
        raise ValueError("V4.1 mHC width must be divisible by 64 * split count.")
    _prenorm_project_kernel[(triton.cdiv(tokens, 16), splits)](
        x,
        fn,
        mixes,
        sqrsum,
        tokens,
        width,
        fn.shape[0],
        splits,
        16,
        64,
        num_warps=4,
        num_stages=2,
    )


def mhc_pre_delayed_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate next coefficients while collapsing with the previous pre-mix."""
    from .deepseek_v41_tilelang import (
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )

    if not residual.is_cuda or residual.dtype != torch.bfloat16:
        raise ValueError("V4.1 mHC requires CUDA BF16 residual streams.")
    if residual.ndim != 3 or not residual.is_contiguous():
        raise ValueError("Expected contiguous residual [tokens, 4, hidden].")
    tokens, streams, hidden = residual.shape
    if streams != 4:
        raise ValueError("V4.1 native mHC requires four residual streams.")
    if x is None:
        x = residual.view(tokens, streams * hidden)
    if x.ndim != 2 or x.shape[0] != tokens or not x.is_contiguous():
        raise ValueError("Invalid mHC projection input.")
    if x.dtype != torch.bfloat16:
        raise ValueError("mHC projection input must be BF16.")
    width = x.shape[1]
    mix_size = streams * (streams + 2)
    if fn.shape != (mix_size, width) or fn.dtype != torch.float32:
        raise ValueError("mHC projection weight must be FP32 [24, input_width].")
    if not fn.is_contiguous() or hc_scale.shape != (3,) or hc_base.shape != (mix_size,):
        raise ValueError("Invalid mHC coefficient layout.")
    if (
        hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
        or not hc_scale.is_contiguous()
        or not hc_base.is_contiguous()
    ):
        raise ValueError("mHC scale/base must be contiguous FP32.")
    if any(
        t is not None and t.device != residual.device
        for t in (x, fn, hc_scale, hc_base, pre_mix, norm_weight)
    ):
        raise ValueError("All mHC tensors must share a CUDA device.")
    if pre_mix is not None and (
        pre_mix.shape != (tokens, streams)
        or pre_mix.dtype != torch.float32
        or not pre_mix.is_contiguous()
    ):
        raise ValueError("Carried mHC coefficients must be FP32 [tokens, 4].")
    if norm_weight is not None and (
        norm_weight.shape != (hidden,)
        or norm_weight.dtype != torch.bfloat16
        or not norm_weight.is_contiguous()
    ):
        raise ValueError("mHC input normalization weight must be BF16 [hidden].")
    next_pre = torch.empty(
        (tokens, streams), device=residual.device, dtype=torch.float32
    )
    post = torch.empty_like(next_pre)
    comb = torch.empty(
        (tokens, streams * streams), device=residual.device, dtype=torch.float32
    )
    layer_input = torch.empty(
        (tokens, hidden), device=residual.device, dtype=residual.dtype
    )
    outputs = (
        post.unsqueeze(-1),
        comb.view(tokens, streams, streams),
        layer_input,
        next_pre,
    )
    if not tokens:
        return outputs
    use_gemv = tokens <= 32
    splits = 1 if use_gemv else 16 if tokens <= 128 else 4 if tokens <= 1024 else 1
    mixes = torch.empty(
        (splits, tokens, mix_size), device=residual.device, dtype=torch.float32
    )
    sqrsum = torch.empty((splits, tokens), device=residual.device, dtype=torch.float32)
    if use_gemv:
        from .tilelang import _tilelang_hc_prenorm_gemm

        _tilelang_hc_prenorm_gemm(x, fn, mixes, sqrsum, width // streams, streams)
    else:
        _native_hc_prenorm_gemm(x, fn, mixes, sqrsum)
    fields = dict(
        hidden_size=hidden,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=splits,
        hc_mult=streams,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=width,
    )
    tensors = (mixes, sqrsum, hc_scale, hc_base, residual, post, comb, layer_input)
    carried = pre_mix if pre_mix is not None else post
    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(*tensors, carried, next_pre, **fields)
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            *tensors,
            norm_weight,
            carried,
            next_pre,
            norm_eps=norm_eps,
            **fields,
        )
    return outputs


@triton.jit
def _collapse_kernel(x, coeff, out, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accum = tl.zeros((BLOCK,), tl.float32)
    for stream in tl.static_range(4):
        value = tl.load(x + (token * 4 + stream) * HIDDEN + col, col < HIDDEN, 0)
        weight = tl.load(coeff + token * 4 + stream)
        accum += value.to(tl.float32) * weight
    tl.store(out + token * HIDDEN + col, accum, col < HIDDEN)


def hc_collapse_triton(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("V4.1 collapse requires CUDA [tokens, 4, hidden].")
    if x.dtype != torch.bfloat16 or pre_mix.dtype != torch.float32:
        raise ValueError("Expected BF16 residual and FP32 pre-mix.")
    if (
        not x.is_contiguous()
        or not pre_mix.is_contiguous()
        or pre_mix.shape != x.shape[:2]
    ):
        raise ValueError("Invalid collapse layout.")
    if pre_mix.device != x.device:
        raise ValueError("Collapse inputs must share a CUDA device.")
    out = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    if x.shape[0]:
        _collapse_kernel[(x.shape[0], triton.cdiv(x.shape[2], 1024))](
            x,
            pre_mix,
            out,
            x.shape[2],
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def _pre_fake(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    x=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    tokens, streams, hidden = residual.shape
    return (
        torch.empty((tokens, streams, 1), device=residual.device, dtype=torch.float32),
        torch.empty(
            (tokens, streams, streams), device=residual.device, dtype=torch.float32
        ),
        torch.empty((tokens, hidden), device=residual.device, dtype=residual.dtype),
        torch.empty((tokens, streams), device=residual.device, dtype=torch.float32),
    )


def _collapse_fake(x, pre_mix):
    return torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="mhc_pre_delayed_tilelang",
    op_func=mhc_pre_delayed_tilelang,
    mutates_args=[],
    fake_impl=_pre_fake,
)
direct_register_custom_op(
    op_name="hc_collapse_triton",
    op_func=hc_collapse_triton,
    mutates_args=[],
    fake_impl=_collapse_fake,
)
