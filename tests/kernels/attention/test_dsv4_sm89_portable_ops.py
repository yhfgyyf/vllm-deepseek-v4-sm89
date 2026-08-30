# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness checks for portable DeepSeek V4 SM89 kernels."""

import importlib.util
from pathlib import Path

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import HAS_TRITON


def _load_leaf_module(name: str, relative_path: str):
    path = Path(__file__).parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fp8_einsum_module = _load_leaf_module(
    "dsv4_sm89_fp8_einsum_ops",
    "vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py",
)
sm12x_mqa_module = _load_leaf_module(
    "dsv4_sm89_sm12x_mqa_ops",
    "vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and an active Triton backend",
)


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-3.0, 3.0).to(torch.float8_e4m3fn)


def test_sm89_fp8_einsum_triton_matches_reference() -> None:
    torch.manual_seed(0)
    a = _to_fp8(torch.randn((3, 2, 128), device="cuda"))
    b = _to_fp8(torch.randn((2, 128, 128), device="cuda"))
    a_scale = torch.rand((3, 2, 1), device="cuda") * 0.1 + 0.01
    b_scale = torch.rand((2, 1, 1), device="cuda") * 0.1 + 0.01
    out = torch.empty((3, 2, 128), device="cuda", dtype=torch.bfloat16)

    fp8_einsum_module._deepseek_v4_sm89_fp8_einsum(a, a_scale, b, b_scale, out)

    ref = torch.einsum(
        "tgh,grh->tgr",
        a.float() * a_scale.float(),
        b.float() * b_scale.float(),
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_sm89_fp8_einsum_accepts_checkpoint_scale_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        fp8_einsum_module.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    torch.manual_seed(4)
    num_tokens, num_groups, hidden_size, out_rank = 3, 2, 128, 128
    a = _to_fp8(torch.randn((num_tokens, num_groups, hidden_size), device="cuda"))
    b = _to_fp8(torch.randn((num_groups * out_rank, hidden_size), device="cuda"))
    a_scale = torch.rand((num_tokens, num_groups, 1), device="cuda") * 0.1 + 0.01
    b_scale = torch.tensor([[123], [125]], dtype=torch.uint8, device="cuda").view(
        torch.float8_e8m0fnu
    )
    out = torch.empty(
        (num_tokens, num_groups, out_rank),
        device="cuda",
        dtype=torch.bfloat16,
    )

    fp8_einsum_module.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        (1, 128, 128),
    )

    b = b.view(num_groups, out_rank, hidden_size)
    b_scale = fp8_einsum_module._upcast_e8m0_to_fp32(b_scale).view(num_groups, 1, 1)
    ref = torch.einsum(
        "tgh,grh->tgr",
        a.float() * a_scale.float(),
        b.float() * b_scale.float(),
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_sm89_mqa_logits_triton_matches_reference() -> None:
    torch.manual_seed(1)
    num_q, num_heads, head_dim, seq_len = 5, 4, 64, 9
    q = _to_fp8(torch.randn((num_q, num_heads, head_dim), device="cuda"))
    k = _to_fp8(torch.randn((seq_len, head_dim), device="cuda"))
    k_scale = torch.rand((seq_len,), device="cuda") * 0.1 + 0.01
    weights = torch.rand((num_q, num_heads), device="cuda")
    starts = torch.tensor([0, 1, 2, 0, 3], dtype=torch.int32, device="cuda")
    ends = torch.tensor([9, 8, 9, 4, 7], dtype=torch.int32, device="cuda")

    actual = sm12x_mqa_module.fp8_mqa_logits_triton(
        q,
        (k, k_scale),
        weights,
        starts,
        ends,
        native_fp8=False,
    )

    scores = torch.einsum("qhd,sd->qhs", q.float(), k.float())
    scores = torch.relu(scores * k_scale.float()[None, None, :])
    ref = (scores * weights.float()[:, :, None]).sum(dim=1)
    offsets = torch.arange(seq_len, device="cuda")
    valid = (offsets[None, :] >= starts[:, None]) & (offsets[None, :] < ends[:, None])
    ref = ref.masked_fill(~valid, float("-inf"))
    torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)


def test_sm89_paged_mqa_logits_triton_matches_reference() -> None:
    torch.manual_seed(2)
    batch, next_n, num_heads, head_dim = 2, 2, 4, 64
    block_size, num_blocks, max_model_len = 4, 5, 8
    q = _to_fp8(torch.randn((batch, next_n, num_heads, head_dim), device="cuda"))
    kv_cache = torch.empty(
        (num_blocks, block_size, 1, head_dim + 4), dtype=torch.uint8, device="cuda"
    )
    kv_values, kv_scales = sm12x_mqa_module._view_packed_fp8_paged_mqa_kv_cache(
        kv_cache, head_dim
    )
    values = _to_fp8(torch.randn_like(kv_values.float()))
    scales = torch.rand_like(kv_scales.float()) * 0.1 + 0.01
    kv_values.copy_(values)
    kv_scales.copy_(scales)
    weights = torch.rand((batch * next_n, num_heads), device="cuda")
    context_lens = torch.tensor([[8, 5], [7, 4]], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device="cuda")

    actual = sm12x_mqa_module.fp8_paged_mqa_logits_triton(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    ref = torch.full((batch * next_n, max_model_len), float("-inf"), device="cuda")
    for b in range(batch):
        for n in range(next_n):
            row = b * next_n + n
            for token in range(int(context_lens[b, n])):
                block = int(block_tables[b, token // block_size])
                pos = token % block_size
                k = values[block, pos, 0].float()
                scale = scales[block, pos, 0, 0].float()
                score = torch.einsum("hd,d->h", q[b, n].float(), k)
                ref[row, token] = (torch.relu(score * scale) * weights[row]).sum()
    torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)


def test_sm89_tf32_hc_prenorm_gemm_triton_matches_reference() -> None:
    torch.manual_seed(3)
    m, k, n, num_split = 5, 128, 16, 2
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    fn = torch.randn((n, k), device="cuda", dtype=torch.float32)
    out = torch.empty((num_split, m, n), device="cuda", dtype=torch.float32)
    sqrsum = torch.empty((num_split, m), device="cuda", dtype=torch.float32)

    sm12x_mqa_module.tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)

    torch.testing.assert_close(
        out.sum(dim=0),
        x.float() @ fn.T,
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        sqrsum.sum(dim=0),
        x.float().square().sum(dim=1),
        rtol=5e-2,
        atol=5e-2,
    )
