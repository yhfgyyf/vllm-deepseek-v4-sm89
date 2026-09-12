# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the V4.1 routed SwiGLU MXFP8 fusion."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import (
    _deepseek_v41_swiglu_router_weight_mxfp8_impl,
    deepseek_v41_swiglu_router_weight,
    deepseek_v41_swiglu_router_weight_mxfp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_quantize_dequantize,
)


def _reference(
    gate_up: torch.Tensor,
    router_weights: torch.Tensor,
    clamp_limit: float,
) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    activated = F.silu(gate.clamp(max=clamp_limit))
    activated = activated * up.clamp(min=-clamp_limit, max=clamp_limit)
    activated = activated * router_weights[:, None]
    rounded = activated.to(torch.bfloat16).float()
    blocked = rounded.view(rounded.shape[0], -1, 32)
    amax = blocked.abs().amax(dim=-1).clamp_min(1e-4)
    scale_exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(scale_exp).unsqueeze(-1)
    quantized = (blocked / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).view_as(rounded).to(torch.bfloat16)


def _inputs(rows: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(rows * 1009 + width)
    gate_up = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16).mul_(4)
    router_weights = torch.rand(rows, device="cuda", dtype=torch.float32)
    gate_up[0].zero_()
    router_weights[0] = 1e-5
    if rows > 1:
        gate_up[1] = torch.linspace(
            -20, 20, 2 * width, device="cuda", dtype=torch.bfloat16
        )
        router_weights[1] = 0.23
    return gate_up, router_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("rows", "width"), [(1, 32), (7, 288), (33, 576), (65, 1152)])
def test_v41_activation_fusion_matches_two_op_and_independent_reference(
    rows: int, width: int
) -> None:
    gate_up, router_weights = _inputs(rows, width)
    fused = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
    baseline = torch.empty_like(fused)

    deepseek_v41_swiglu_router_weight(
        baseline, gate_up, router_weights, clamp_limit=10.0
    )
    baseline = mxfp8_quantize_dequantize(baseline)
    deepseek_v41_swiglu_router_weight_mxfp8(
        fused, gate_up, router_weights, clamp_limit=10.0
    )

    assert torch.equal(fused, baseline)
    torch.testing.assert_close(
        fused.float(),
        _reference(gate_up, router_weights, 10.0).float(),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v41_activation_fusion_replays_with_fresh_inputs() -> None:
    rows, width = 6, 288
    gate_up, router_weights = _inputs(rows, width)
    output = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
    deepseek_v41_swiglu_router_weight_mxfp8(
        output, gate_up, router_weights, clamp_limit=10.0
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_swiglu_router_weight_mxfp8(
            output, gate_up, router_weights, clamp_limit=10.0
        )

    for seed in (11, 29):
        torch.manual_seed(seed)
        gate_up.normal_()
        router_weights.uniform_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(),
            _reference(gate_up, router_weights, 10.0).float(),
            rtol=2e-2,
            atol=2e-3,
        )


def test_v41_activation_fusion_requires_bf16_group32() -> None:
    with pytest.raises(AssertionError):
        _deepseek_v41_swiglu_router_weight_mxfp8_impl(
            torch.empty(1, 32),
            torch.empty(1, 64),
            torch.empty(1),
            10.0,
        )
    with pytest.raises(AssertionError):
        _deepseek_v41_swiglu_router_weight_mxfp8_impl(
            torch.empty(1, 33, dtype=torch.bfloat16),
            torch.empty(1, 66, dtype=torch.bfloat16),
            torch.empty(1),
            10.0,
        )
