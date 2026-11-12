# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM120 adapters for the pinned V4.1 B12X component experiments."""

import functools
import importlib.metadata
from dataclasses import replace
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
_PLANS: dict[tuple, Any] = {}


def enabled() -> bool:
    return (
        envs.VLLM_DEEPSEEK_V41_B12X
        and current_platform.is_cuda()
        and current_platform.is_device_capability(120)
    )


@functools.cache
def _modules():
    try:
        if importlib.metadata.version("b12x") != "1.3.0":
            raise ImportError("the measured adapter uses B12X 1.3.0")
        from b12x import preparation
        from b12x.attention import compressed_sparse_mla
        from b12x.gemm import mxfp8_linear
        from b12x.norm import mhc
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_DEEPSEEK_V41_B12X requires the tested B12X source "
            "e9ce547767ff9ee6509faf294fa1b4e2380dfbf5 (1.3.0); see "
            "docs/models/deepseek_v41_sm120_b12x.md."
        ) from exc
    return preparation, compressed_sparse_mla, mxfp8_linear, mhc


def _prepared(key: tuple, build):
    plan = _PLANS.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            logger.warning_once(
                "V4.1 B12X %s shape was not warmed before capture; using native.",
                key[0],
            )
            return None
        with torch.accelerator.device_index(key[1].index):
            preparation = _modules()[0]
            plan = build()
            preparation.prepare_default(
                plan.request(
                    name=f"vllm-v41-{key[0]}",
                    prepare_call=lambda state: preparation.PreparedCall(
                        run=lambda: None
                    ),
                )
            )
        _PLANS[key] = plan
    return plan


def try_mhc(residual, fn, scale, base, pre_mix, norm_weight, rms_eps, hc_eps, norm_eps):
    """Called after native validation, only for the carried, full-width input."""
    preparation, _, _, mhc = _modules()
    tokens, _, hidden = residual.shape
    key = ("mhc", residual.device, hidden, rms_eps, hc_eps, norm_eps)

    def build():
        return mhc.plan(
            mhc.Caps(device=residual.device, max_tokens=64, hidden_size=hidden),
            invocation=preparation.FrozenMapping(
                dict(
                    operation="pre",
                    output_mode="provided",
                    expanded_residual=True,
                    lagged_mix=True,
                    has_norm_weight=True,
                    norm_weight_dtype="bfloat16",
                    has_fn_bf16=False,
                    rms_eps=rms_eps,
                    hc_eps=hc_eps,
                    sinkhorn_iters=20,
                    norm_eps=norm_eps,
                )
            ),
        )

    plan = _prepared(key, build)
    if plan is None:
        return None
    (spec,) = plan.scratch_specs()
    next_pre = torch.empty_like(pre_mix)
    binding = mhc.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=residual.device),
        tokens=tokens,
        out=torch.empty_like(residual),
        y=torch.empty((tokens, hidden), dtype=residual.dtype, device=residual.device),
        post=torch.empty_like(pre_mix),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=residual.device),
        pre_out=next_pre,
    )
    _, post, comb, y = mhc.run_pre(
        residual,
        fn,
        scale,
        base,
        binding=binding,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=20,
        norm_eps=norm_eps,
    )
    return post[..., None], comb, y, next_pre


def _mma_scale_view(scale: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    """View FlashInfer F8_128x4 storage without copying or requantizing it."""
    m_tiles, k_tiles = (rows + 127) // 128, (width + 127) // 128
    return (
        scale.view(torch.float8_e8m0fnu)
        .view(m_tiles, k_tiles, 32, 4, 4)
        .permute(2, 3, 0, 4, 1)
        .unsqueeze(-1)
    )


def try_mxfp8(input_mxfp8, weight, input_scale, weight_scale, out_dtype):
    rows, width = input_mxfp8.shape
    n = weight.shape[0]
    if not (
        enabled()
        and 0 < rows <= 32
        and (n, width) in ((1152, 5120), (1792, 5120))
        and out_dtype == torch.bfloat16
    ):
        return None
    _, _, linear, _ = _modules()
    key = ("linear", input_mxfp8.device, rows, n, width)
    query = linear.FixedBlockscaledQuery(
        recipe="mxfp8",
        call_kind="packed",
        max_rows=rows,
        in_features=width,
        padded_in_features=width,
        out_features=n,
        input_dtype="float8_e4m3fn",
        output_dtype="bfloat16",
        expected_m=rows,
        source_scale_form="swizzled",
    )
    plan = _prepared(key, lambda: linear.plan(query))
    if plan is None:
        return None
    # Same prequantized entry as mxfp8_linear.mm((values, scales), packed_weight).
    # Borrow native weight/scale storage instead of keeping a second weight copy.
    return torch.ops.b12x.blockscaled_packed_mxfp8_prequantized(
        input_mxfp8,
        input_scale,
        weight,
        _mma_scale_view(weight_scale, n, width),
        plan.handle,
        out_dtype,
        None,
    )


def try_attention(
    q,
    swa_cache,
    global_cache,
    swa_slots,
    global_slots,
    scale,
    sinks,
    positions,
    cos_sin,
    grouped_output,
    token_offset,
) -> bool:
    if not (
        enabled()
        and 0 < q.shape[0] <= 32
        and q.shape[1] in (16, 32)
        and q.is_contiguous()
        and global_cache is not None
        and global_slots is not None
        and swa_slots.shape[1] == 128
        and global_slots.shape[1] > 0
        and swa_cache.shape[1] == 32
        and global_cache.shape[1] in (64, 128)
        and swa_cache.stride(1) == 528
        and swa_cache.stride(2) == 1
        and global_cache.stride(1) == 288
        and global_cache.stride(2) == 1
    ):
        return False
    _, mla, _, _ = _modules()
    swa = swa_cache.as_strided((swa_cache.shape[0], 32 * 528), (swa_cache.stride(0), 1))
    page = global_cache.shape[1]
    indexed = global_cache.as_strided(
        (global_cache.shape[0], page * 288), (global_cache.stride(0), 1)
    )
    rows, heads, _ = q.shape
    key = (
        "attention",
        q.device,
        heads,
        tuple(swa.shape),
        tuple(swa.stride()),
        tuple(indexed.shape),
        tuple(indexed.stride()),
        page,
        global_slots.shape[1],
        sinks is not None,
    )

    def build():
        # The measured decode plans used capacity 64, including DSpark M6.
        query = torch.empty((64, heads, 512), dtype=q.dtype, device=q.device)
        caps = mla.Caps(
            device=q.device,
            num_q_heads=heads,
            max_q_rows=64,
            max_width=128 + global_slots.shape[1],
            swa_width=128,
            indexed_width=global_slots.shape[1],
            swa_page_size=32,
            indexed_page_size=page,
            max_page_table_width=0,
            cache_format="deepseek_v41",
            mode="decode",
            use_cuda_graph=True,
        )
        invocation = mla.invocation_from_tensors(
            q=query,
            swa_k_cache=swa,
            indexed_k_cache=indexed,
            attn_sink=sinks,
            out=torch.empty_like(query),
        )
        plan = mla.plan(caps, invocation=invocation)
        config = plan.contract.configure(plan.query, device=None).default
        return mla.plan(
            caps,
            invocation=invocation,
            override=replace(config, v41_compute_mode="bf16", v41_heads_per_block=16),
        )

    plan = _prepared(key, build)
    if plan is None:
        return False
    (spec,) = plan.scratch_specs()
    binding = mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=q.device),
        q=q,
        swa_indices=swa_slots,
        swa_lengths=torch.full((rows,), 128, dtype=torch.int32, device=q.device),
        indexed_indices=global_slots,
        indexed_lengths=torch.full(
            (rows,), global_slots.shape[1], dtype=torch.int32, device=q.device
        ),
    )
    # vLLM has already translated global slots to physical rows. No page remap.
    core_output = torch.empty_like(q)
    mla.run(
        binding=binding,
        swa_k_cache=swa,
        swa_page_size=32,
        indexed_k_cache=indexed,
        indexed_page_size=page,
        sm_scale=scale,
        attn_sink=sinks,
        out=core_output,
        cache_format="deepseek_v41",
    )
    from vllm.models.deepseek_v4_1.common.ops.inverse_rope import (
        _inverse_rope_group_kernel,
    )

    _inverse_rope_group_kernel[(rows, heads)](
        core_output,
        positions,
        cos_sin,
        grouped_output[:, token_offset:],
        grouped_output.shape[1],
        heads // grouped_output.shape[0],
        core_output.stride(0),
        core_output.stride(1),
        cos_sin.stride(0),
        enable_fp_fusion=False,
        num_warps=4,
    )
    return True
