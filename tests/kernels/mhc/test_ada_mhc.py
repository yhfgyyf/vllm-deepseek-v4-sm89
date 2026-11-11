# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.mhc.deepseek_v41 import mhc_pre_delayed_tilelang
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.is_device_capability(89)):
    pytest.skip("Ada mHC requires SM89", allow_module_level=True)


@pytest.mark.parametrize("rows", [1, 6, 8, 32, 33])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("fused_norm", [False, True])
@torch.inference_mode()
def test_ada_mhc_split_projection_and_mutable_graph(rows, carried, fused_norm):
    torch.manual_seed(0)
    residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 20480, device="cuda") * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0], device="cuda")
    base = torch.randn(24, device="cuda")
    pre_mix = torch.rand(rows, 4, device="cuda") if carried else None
    norm = (
        torch.empty(5120, device="cuda", dtype=torch.bfloat16).uniform_(0.5, 1.5)
        if fused_norm
        else None
    )
    args = (residual, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20)
    expected = mhc_pre_delayed_torch(*args, pre_mix=pre_mix)
    if fused_norm:
        y = expected[2].double()
        y = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-20)) * norm
        expected = (*expected[:2], y.bfloat16(), expected[3])
    actual = mhc_pre_delayed_tilelang(
        *args, pre_mix=pre_mix, norm_weight=norm, norm_eps=1e-20
    )
    for index, (observed, reference) in enumerate(zip(actual, expected, strict=True)):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(
            observed,
            reference,
            atol=1e-4 if index == 2 else 1e-5,
            rtol=0.008 if index == 2 else 1e-4,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mhc_pre_delayed_tilelang(
            *args, pre_mix=pre_mix, norm_weight=norm, norm_eps=1e-20
        )
    residual.neg_()
    if pre_mix is not None:
        pre_mix.mul_(0.75)
    graph.replay()
    eager = mhc_pre_delayed_tilelang(
        *args, pre_mix=pre_mix, norm_weight=norm, norm_eps=1e-20
    )
    for observed, reference in zip(captured, eager, strict=True):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)
