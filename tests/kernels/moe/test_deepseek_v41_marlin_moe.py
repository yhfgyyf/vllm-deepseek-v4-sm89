# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end reference check for the V4.1 packed expert path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    deepseek_v41_swiglu_router_weight,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_quantize_dequantize,
)
from vllm.scalar_type import scalar_types


def _dequantize_mxfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=values.device,
        dtype=torch.float32,
    )
    codes = torch.stack((values & 0xF, (values >> 4) & 0xF), dim=-1).flatten(-2)
    decoded = lookup[codes.long()]
    block_scales = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    return (decoded * block_scales).to(torch.bfloat16)


def _quantize_and_pack_padded_experts(
    experts: int, hidden_size: int, logical_intermediate: int, packed_intermediate: int
) -> tuple[torch.Tensor, ...]:
    w1_source = torch.zeros(
        experts,
        2 * packed_intermediate,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2_source = torch.zeros(
        experts,
        hidden_size,
        packed_intermediate,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w1_source[:, :logical_intermediate].normal_(std=hidden_size**-0.5)
    w1_source[
        :,
        packed_intermediate : packed_intermediate + logical_intermediate,
    ].normal_(std=hidden_size**-0.5)
    w2_source[:, :, :logical_intermediate].normal_(std=logical_intermediate**-0.5)

    w1_checkpoint, w1_checkpoint_scale = mxfp4_quantize(w1_source)
    w2_checkpoint, w2_checkpoint_scale = mxfp4_quantize(w2_source)
    w1_ref = _dequantize_mxfp4(w1_checkpoint, w1_checkpoint_scale)
    w2_ref = _dequantize_mxfp4(w2_checkpoint, w2_checkpoint_scale)
    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    w1, w2, w1_scale, w2_scale, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        layer,
        w1_checkpoint,
        w2_checkpoint,
        w1_checkpoint_scale,
        w2_checkpoint_scale,
        None,
        None,
    )
    return w1_ref, w2_ref, w1, w2, w1_scale, w2_scale


def _reference(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    hidden = mxfp8_quantize_dequantize(hidden)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(w1.size(0)):
        positions = (topk_ids == expert).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_ids, route_ids = positions.unbind(dim=1)
        gate_up = hidden[token_ids] @ w1[expert].T
        gate, up = gate_up.float().chunk(2, dim=-1)
        gate = torch.clamp(gate, max=10.0)
        up = torch.clamp(up, min=-10.0, max=10.0)
        activated = mxfp8_quantize_dequantize(
            (F.silu(gate) * up * topk_weights[token_ids, route_ids].unsqueeze(1))
            .to(hidden.dtype)
            .contiguous()
        )
        expert_output = activated @ w2[expert].T
        output.index_add_(0, token_ids, expert_output.float())
    return output.to(hidden.dtype)


@torch.inference_mode()
def test_deepseek_v41_marlin_moe_matches_full_quantized_reference():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 9), (12, 0)):
        pytest.skip("requires SM89 or SM120")

    torch.manual_seed(7)
    tokens, hidden_size, logical_intermediate, intermediate = 8, 256, 96, 128
    experts, topk = 4, 2
    hidden = (
        torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        / hidden_size**0.5
    )
    w1_ref, w2_ref, w1, w2, w1_scale, w2_scale = _quantize_and_pack_padded_experts(
        experts, hidden_size, logical_intermediate, intermediate
    )
    assert not torch.count_nonzero(w1_ref[:, logical_intermediate:intermediate])
    assert not torch.count_nonzero(w1_ref[:, intermediate + logical_intermediate :])
    assert not torch.count_nonzero(w2_ref[:, :, logical_intermediate:])
    topk_ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
        .view(tokens, topk)
        .remainder(experts)
    )
    topk_weights = torch.softmax(
        torch.randn(tokens, topk, device="cuda", dtype=torch.float32), dim=-1
    )

    expected = _reference(hidden, w1_ref, w2_ref, topk_weights, topk_ids)
    actual = fused_marlin_moe(
        hidden,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        quant_type_id=scalar_types.float4_e2m1f.id,
        global_num_experts=experts,
        activation=MoEActivation.SILU,
        activation_config=ApplyMoEActivationConfig(clamp_limit=10.0),
        mxfp8_activation=True,
        router_weight_before_fc2_quant=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0.1, atol=0.02)
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference_rms = expected.float().square().mean().sqrt().clamp_min(1e-12)
    assert error / reference_rms < 0.12


@torch.inference_mode()
def test_deepseek_v41_fused_activation_matches_official_fp32_order():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability not in ((8, 9), (12, 0)):
        pytest.skip("requires SM89 or SM120")

    gate_up = torch.linspace(
        -12, 12, 3 * 128, device="cuda", dtype=torch.bfloat16
    ).view(3, 128)
    gate_up[1].zero_()
    gate_up[2].fill_(1e-5)
    router_weights = torch.tensor(
        [0.23, 0.37, 1e-5], device="cuda", dtype=torch.float32
    )
    output = torch.empty(3, 64, device="cuda", dtype=torch.bfloat16)

    gate, up = gate_up.float().chunk(2, dim=-1)
    expected = (
        F.silu(torch.clamp(gate, max=10.0))
        * torch.clamp(up, min=-10.0, max=10.0)
        * router_weights[:, None]
    ).to(torch.bfloat16)
    deepseek_v41_swiglu_router_weight(output, gate_up, router_weights, 10.0)

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)
    expected_qdq = mxfp8_quantize_dequantize(expected)
    actual_qdq = mxfp8_quantize_dequantize(output)
    torch.testing.assert_close(actual_qdq, expected_qdq, rtol=2e-2, atol=2e-3)


def test_deepseek_v41_router_weight_is_before_fc2_quantization():
    """Pin the V4.1 order independently of the CUDA Marlin implementation."""
    torch.manual_seed(3)
    gate = torch.randn(5, 64, dtype=torch.float32).clamp(-12, 12)
    up = torch.randn(5, 64, dtype=torch.float32).clamp(-12, 12)
    weights = torch.tensor([0.1, 0.25, 0.5, 0.75, 1.0])[:, None]
    activated = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)

    def e8m0_g32(x: torch.Tensor) -> torch.Tensor:
        groups = x.float().reshape(x.shape[0], -1, 32)
        amax = groups.abs().amax(-1).clamp_min(1e-4)
        exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
        scale = torch.exp2(exponent).unsqueeze(-1)
        out = (groups.clamp(-448 * scale, 448 * scale) / scale).to(
            torch.float8_e4m3fn
        ).to(torch.float32) * scale
        return out.reshape_as(x)

    before = e8m0_g32((activated * weights).to(torch.bfloat16))
    after = e8m0_g32(activated.to(torch.bfloat16)) * weights
    assert not torch.equal(before, after)
