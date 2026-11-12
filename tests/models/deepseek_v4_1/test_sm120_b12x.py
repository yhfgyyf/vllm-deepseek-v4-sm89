# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 adapter numerics and replay; run GPU cases in the documented B12X pin."""

import importlib.metadata

import pytest
import torch

from vllm.model_executor.kernels import deepseek_v41_b12x
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("rows,width", [(6, 5120), (1152, 5120), (1792, 5120)])
def test_mxfp8_mma_view_preserves_scale_bytes_and_storage(rows, width):
    """Each logical scale must address the same byte in both kernel ABIs."""
    torch.manual_seed(0)
    scales = torch.randint(0, 255, (rows, width // 32), dtype=torch.uint8)
    storage = swizzle_mxfp8_scale(scales, M=rows, K=width)
    mma = deepseek_v41_b12x._mma_scale_view(storage, rows, width)
    r = torch.arange(rows)[:, None]
    c = torch.arange(width // 32)[None, :]
    observed = mma.view(torch.uint8)[r % 32, r // 32 % 4, r // 128, c % 4, c // 4, 0]
    torch.testing.assert_close(observed, scales, rtol=0, atol=0)
    assert mma.data_ptr() == storage.data_ptr()
    storage.fill_(127)
    assert (mma.view(torch.uint8) == 127).all()


def _require_sm120_b12x():
    if not (current_platform.is_cuda() and current_platform.is_device_capability(120)):
        pytest.skip("requires an idle SM120 GPU")
    assert importlib.metadata.version("b12x") == "1.3.0", (
        "Install the documented, measured B12X revision before this GPU suite."
    )


@pytest.mark.parametrize("rows", [1, 6, 8, 24, 32, 33, 128])
@torch.inference_mode()
def test_sm120_mhc_carried_norm_and_mutable_graph(rows, monkeypatch):
    _require_sm120_b12x()
    from vllm.model_executor.kernels.mhc.deepseek_v41 import (
        mhc_pre_delayed_tilelang,
    )

    torch.manual_seed(0)
    residual = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 20480, device="cuda") * 0.02
    scale = torch.tensor([0.5, 0.25, 1.0], device="cuda")
    base = torch.randn(24, device="cuda")
    incoming = torch.rand(rows, 4, device="cuda")
    norm = torch.ones(5120, dtype=torch.bfloat16, device="cuda")

    def run():
        return mhc_pre_delayed_tilelang(
            residual,
            fn,
            scale,
            base,
            1e-20,
            1e-6,
            1e-6,
            2.0,
            20,
            pre_mix=incoming,
            norm_weight=norm,
            norm_eps=1e-6,
        )

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "0")
    expected = run()
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "1")
    actual = run()
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(
            a,
            b,
            rtol=0 if rows > 32 else 0.008 if index == 2 else 2e-5,
            atol=0 if rows > 32 else 5e-5 if index == 2 else 2e-6,
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    residual.neg_()
    incoming.mul_(0.75)
    graph.replay()
    for a, b in zip(captured, run(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 6, 8, 24, 32, 33, 128])
@pytest.mark.parametrize("n", [1152, 1792])
@torch.inference_mode()
def test_sm120_linear_keeps_activation_quantization_and_mutable_graph(
    rows, n, monkeypatch
):
    _require_sm120_b12x()
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    torch.manual_seed(0)
    width = 5120
    w = torch.randn(n, width, dtype=torch.bfloat16, device="cuda") * width**-0.5
    wq, ws = _mxfp8_e4m3_quantize_torch(w, min_amax=1e-4)
    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(wq, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(ws, requires_grad=False)
    config = Mxfp8LinearLayerConfig(
        weight_shape=(n, width), model_profile="deepseek_v41"
    )
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "0")
    native = FlashInferCutlassMxfp8LinearKernel(config)
    native.process_weights_after_loading(layer)
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "1")
    candidate = FlashInferCutlassMxfp8LinearKernel(config)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    x[0].mul_(1e-8)  # The V4.1 minimum-amax boundary must remain in force.
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.01
    expected = native.apply_weights(layer, x, bias)
    actual = candidate.apply_weights(layer, x, bias)
    xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True, min_amax=1e-4)
    quantized = QuantizedActivation(xq, xs, x.dtype, x.shape, kMxfp8Dynamic)
    torch.testing.assert_close(
        actual, candidate.apply_weights(layer, quantized, bias), rtol=0, atol=0
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=0 if rows > 32 else 0.008, atol=0 if rows > 32 else 2e-4
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate.apply_weights(layer, x, bias)
    x.neg_()
    bias.neg_()
    graph.replay()
    torch.testing.assert_close(
        captured, candidate.apply_weights(layer, x, bias), rtol=0, atol=0
    )


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("rows", [6, 32, 33])
@pytest.mark.parametrize("global_count", [0, 128])
@torch.inference_mode()
def test_sm120_attention_physical_slots_inverse_rope_offset_and_graph(
    heads, rows, global_count, monkeypatch
):
    _require_sm120_b12x()
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
        _native_mixed_attention,
    )

    torch.manual_seed(42)
    swa = torch.empty((4, 32, 528), dtype=torch.uint8, device="cuda")
    swa.view(128, 528)[:, :512] = (
        torch.randint(-4, 5, (128, 512), device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    swa.view(128, 528)[:, 512:] = 127
    cache = torch.empty((2, 128, 288), dtype=torch.uint8, device="cuda")
    cache[:, :, :256] = torch.randint(
        0, 256, (2, 128, 256), dtype=torch.uint8, device="cuda"
    )
    cache[:, :, 256:] = 56
    q = (torch.randn(rows, heads, 512, device="cuda") * 0.2).bfloat16()
    sw = torch.arange(128, dtype=torch.int32, device="cuda").repeat(rows, 1)
    glob = sw + 128 if global_count else None
    sw[:, 97:] = -1
    if glob is not None:
        glob[:, 91:] = -1
    sinks = torch.linspace(-1, 1, heads, device="cuda")
    angle = torch.randn(rows, 32, device="cuda")
    rotary = torch.cat((angle.cos(), angle.sin()), -1)
    positions = torch.arange(rows, device="cuda")

    def run():
        grouped = torch.full(
            (heads // 8, rows + 2, 4096), 17, device="cuda", dtype=torch.bfloat16
        )
        _native_mixed_attention(
            q,
            swa,
            cache if global_count else None,
            sw,
            glob,
            512**-0.5,
            sinks=sinks,
            positions=positions,
            cos_sin=rotary,
            grouped_out=grouped,
            token_offset=1,
        )
        return grouped

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "0")
    expected = run()
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_B12X", "1")
    actual = run()
    fallback = rows > 32 or not global_count
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, expected, rtol=0 if fallback else 0.03, atol=0 if fallback else 0.03
    )
    assert (actual[:, 0] == 17).all() and (actual[:, -1] == 17).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    q.neg_()
    if glob is not None:
        glob[:, :7] = -1
    graph.replay()
    torch.testing.assert_close(captured, run(), rtol=0, atol=0)
