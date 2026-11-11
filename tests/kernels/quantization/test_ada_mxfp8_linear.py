# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.ada import AdaMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform

if not (current_platform.is_cuda() and current_platform.is_device_capability(89)):
    pytest.skip("Ada MXFP8 requires SM89", allow_module_level=True)


@pytest.mark.parametrize(
    ("input_size", "output_size"),
    [
        (1280, 4096),
        (1280, 8192),
        (2048, 5120),
        (6144, 6400),
        (5120, 1152),
        (5120, 1792),
    ],
)
@pytest.mark.parametrize("rows", [1, 8, 9, 32, 33])
@torch.inference_mode()
def test_ada_mxfp8_small_rows_numerics_and_mutable_graph(
    input_size: int, output_size: int, rows: int
) -> None:
    torch.manual_seed(0)
    weight = (
        torch.randn(output_size, input_size, dtype=torch.bfloat16, device="cuda")
        * input_size**-0.5
    )
    wq, ws = _mxfp8_e4m3_quantize_torch(weight, min_amax=1e-4)
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(wq, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ws, requires_grad=False)
    kernel = AdaMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(
            weight_shape=(output_size, input_size), model_profile="deepseek_v41"
        )
    )
    kernel.process_weights_after_loading(layer)
    x = torch.randn(1, rows, input_size, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(output_size, dtype=torch.bfloat16, device="cuda") * 0.01
    xq, xs, dtype, shape = kernel._quantize_input(x, input_size)
    quantized = QuantizedActivation(xq, xs, dtype, shape, kMxfp8Dynamic)

    def dequant(data, scale):
        return data.double() * scale.view(
            torch.float8_e8m0fnu
        ).double().repeat_interleave(32, dim=-1)

    reference = (dequant(xq, xs) @ dequant(wq, ws).T + bias.double()).reshape(
        1, rows, output_size
    )
    actual = kernel.apply_weights(layer, x, bias)
    assert actual.shape == reference.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), reference, rtol=0.008, atol=2e-4)
    torch.testing.assert_close(
        actual, kernel.apply_weights(layer, quantized, bias), rtol=0, atol=0
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernel.apply_weights(layer, x, bias)
    x.neg_()
    bias.neg_()
    graph.replay()
    torch.testing.assert_close(
        captured, kernel.apply_weights(layer, x, bias), rtol=0, atol=0
    )
