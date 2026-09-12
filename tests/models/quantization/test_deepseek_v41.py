# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for DeepSeek-V4.1 native quantization."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mxfp8 import flashinfer as mxfp8_flashinfer
from vllm.model_executor.kernels.linear.mxfp8.ada import AdaMxfp8LinearKernel
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutlassMxfp8LinearKernel,
    _deepseek_v41_engram_mxfp8_tactic,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinExpertsBase,
)
from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.modelopt import (
    KMxfp8Static,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.mxfp4 import (
    DeepseekV41Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_quantize_dequantize_impl,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.models.deepseek_v4.quant_config import (
    DeepseekV4FP8Config as LegacyDeepseekV4FP8Config,
)
from vllm.models.deepseek_v4_1 import quant_config as v41_quant_config
from vllm.models.deepseek_v4_1.quant_config import (
    DeepseekV4FP8Config,
    DeepseekV41WoALinearMethod,
)


def _config_dict() -> dict:
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }


def test_official_mxfp8_quantization_uses_k32_and_minimum_amax():
    x = torch.zeros((3, 64), dtype=torch.bfloat16)
    x[1] = torch.linspace(-1, 1, 64).to(torch.bfloat16)
    x[2].fill_(1e-5)

    quantized, scale = _mxfp8_e4m3_quantize_torch(x, min_amax=1e-4)

    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.uint8
    assert scale.shape == (3, 2)
    expected_min_scale = int(torch.ceil(torch.log2(torch.tensor(1e-4 / 448)))) + 127
    assert scale[0].tolist() == [expected_min_scale, expected_min_scale]
    assert scale[2].tolist() == [expected_min_scale, expected_min_scale]


def test_packed_moe_activation_boundary_matches_quantize_dequantize_reference():
    x = torch.linspace(-2, 2, 3 * 64, dtype=torch.float32).reshape(3, 64)
    x = x.to(torch.bfloat16)
    quantized, scale = _mxfp8_e4m3_quantize_torch(x, min_amax=1e-4)
    expected = dequant_mxfp8_to_bf16(quantized, scale)

    actual = _mxfp8_quantize_dequantize_impl(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v41_router_weight_precedes_bf16_cast_and_second_k32_quantization():
    activation = torch.linspace(-1, 1, 32, dtype=torch.bfloat16).unsqueeze(0)
    router_weight = torch.tensor(0.23, dtype=torch.float32)

    official = _mxfp8_quantize_dequantize_impl(
        (activation.float() * router_weight).to(torch.bfloat16)
    )
    output_weighted = (
        _mxfp8_quantize_dequantize_impl(activation).float() * router_weight
    )

    assert (official.float() - output_weighted).abs().max() > 0.005


def test_v41_rejects_global_marlin_fp8_input_override(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(
        marlin_utils,
        "get_marlin_input_dtype",
        lambda: torch.float8_e4m3fn,
    )
    method = DeepseekV41Mxfp4MoEMethod.__new__(DeepseekV41Mxfp4MoEMethod)

    with pytest.raises(ValueError, match="VLLM_MARLIN_INPUT_DTYPE must be unset"):
        method.process_weights_after_loading(SimpleNamespace())


def test_v41_moe_quant_config_enables_official_router_weight_order():
    method = DeepseekV41Mxfp4MoEMethod.__new__(DeepseekV41Mxfp4MoEMethod)
    method.mxfp4_backend = Mxfp4MoeBackend.MARLIN
    layer = SimpleNamespace(
        w13_weight_scale=torch.empty(0),
        w2_weight_scale=torch.empty(0),
        swiglu_limit=10.0,
    )

    config = method.get_fused_moe_quant_config(layer)

    assert config is not None
    assert config.router_weight_before_fc2_quant
    assert config.quant_dtype == "mxfp8"
    assert config.weight_quant_dtype == "mxfp4"


def test_marlin_accepts_v41_moe_pair_without_narrowing_legacy_support():
    assert MarlinExpertsBase._supports_quant_scheme(kMxfp4Static, kMxfp8Dynamic)
    assert MarlinExpertsBase._supports_quant_scheme(kMxfp4Static, kMxfp4Static)


def test_marlin_defers_activation_quantization_only_for_v41_router_boundary():
    v41 = SimpleNamespace(router_weight_before_fc2_quant=True)
    legacy = SimpleNamespace(router_weight_before_fc2_quant=False)
    get_property = MarlinExpertsBase.expects_unquantized_inputs.fget

    assert get_property(v41)
    assert not get_property(legacy)


@pytest.mark.parametrize(
    ("capability", "expected_backend"),
    [
        (89, Mxfp4MoeBackend.MARLIN),
        (120, Mxfp4MoeBackend.MARLIN),
    ],
)
def test_v41_moe_selector_requests_native_mxfp8_activation(
    monkeypatch, capability, expected_backend
):
    requested = []

    class ExactKernel:
        @staticmethod
        def is_supported_config(*args):
            requested.append((args[2], args[3]))
            return True, None

    monkeypatch.setattr(
        mxfp4_oracle.current_platform,
        "is_device_capability_family",
        lambda family: capability == family,
    )
    monkeypatch.setattr(
        mxfp4_oracle.current_platform,
        "is_device_capability",
        lambda exact: capability == exact,
    )
    monkeypatch.setattr(
        mxfp4_oracle,
        "backend_to_kernel_cls",
        lambda backend: [ExactKernel],
    )
    config = SimpleNamespace(
        moe_backend="auto",
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )

    backend, kernel = mxfp4_oracle.select_deepseek_v41_mxfp4_moe_backend(config)

    assert backend == expected_backend
    assert kernel is ExactKernel
    assert requested == [(kMxfp4Static, kMxfp8Dynamic)]


def test_v41_explicit_b12x_rejects_inexact_activation_floor():
    config = SimpleNamespace(
        moe_backend="b12x",
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )

    with pytest.raises(NotImplementedError, match="minimum amax of 1e-4"):
        mxfp4_oracle.select_deepseek_v41_mxfp4_moe_backend(config)


@pytest.mark.parametrize(
    ("tp", "logical_intermediate", "packed_intermediate"),
    [(2, 1152, 1152), (4, 576, 576), (8, 288, 320)],
)
def test_v41_marlin_moe_tp_shapes_are_padded_without_changing_hidden(
    tp, logical_intermediate, packed_intermediate
):
    method = DeepseekV41Mxfp4MoEMethod.__new__(DeepseekV41Mxfp4MoEMethod)
    parallel_config = SimpleNamespace(
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_nixl_ep_kernels=False,
    )

    hidden, intermediate = method.maybe_roundup_sizes(
        hidden_size=5120,
        intermediate_size_per_partition=2304 // tp,
        act_dtype=torch.bfloat16,
        moe_parallel_config=parallel_config,
    )

    assert hidden == 5120
    assert logical_intermediate == 2304 // tp
    assert intermediate == packed_intermediate


def test_v41_64_alignment_does_not_change_legacy_marlin_alignment(monkeypatch):
    monkeypatch.setattr(mxfp4_oracle.current_platform, "is_xpu", lambda: False)

    hidden, intermediate = (
        mxfp4_oracle.mxfp4_round_up_hidden_size_and_intermediate_size(
            Mxfp4MoeBackend.MARLIN,
            hidden_size=5120,
            intermediate_size=576,
        )
    )

    assert hidden == 5120
    assert intermediate == 640


def test_v41_dense_selector_excludes_legacy_w8a16_fallback(monkeypatch):
    platform = linear_kernels.current_platform._enum
    monkeypatch.setitem(
        linear_kernels._POSSIBLE_MXFP8_KERNELS,
        platform,
        [MarlinMxfp8LinearKernel],
    )

    with pytest.raises(ValueError, match="Failed to find a kernel"):
        linear_kernels.init_mxfp8_linear_kernel(model_profile="deepseek_v41")


def test_checkpoint_scale_loader_expands_32_output_rows():
    captured = {}

    def base_loader(param, loaded_weight):
        captured["loaded_weight"] = loaded_weight

    loader = KMxfp8Static.get_scale_weight_loader(
        base_loader, SimpleNamespace(scale_block_size=(32, 32))
    )
    encoded = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8)

    loader(None, encoded)

    expanded = captured["loaded_weight"]
    assert expanded.shape == (64, 2)
    torch.testing.assert_close(expanded[:32], encoded[:1].expand(32, -1))
    torch.testing.assert_close(expanded[32:], encoded[1:].expand(32, -1))


def test_ada_prequantized_contract_is_row_major_k32():
    data = torch.zeros((2, 64), dtype=torch.float8_e4m3fn)
    scale = torch.full((2, 2), 127, dtype=torch.uint8)
    activation = QuantizedActivation(
        data=data,
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=torch.Size([2, 64]),
        quant_key=kMxfp8Dynamic,
    )

    actual_data, actual_scale, dtype, shape = AdaMxfp8LinearKernel._quantize_input(
        activation, 64
    )

    assert actual_data.data_ptr() == data.data_ptr()
    assert actual_scale.data_ptr() == scale.data_ptr()
    assert dtype == torch.bfloat16
    assert shape == torch.Size([2, 64])


@pytest.mark.parametrize(
    ("tokens", "output_size", "expected_split"),
    [(1, 288, 32), (16, 576, 16), (128, 576, 4), (128, 16384, 1)],
)
def test_ada_split_k_fills_small_output_grids(tokens, output_size, expected_split):
    assert AdaMxfp8LinearKernel._split_k(tokens, output_size) == expected_split


@pytest.mark.parametrize(
    ("tokens", "expected_tactic"),
    [(1, 1), (4, 1), (8, 1), (16, 1), (32, 1), (128, 2), (1024, 4)],
)
def test_v41_engram_mxfp8_uses_profiled_sm120_tactics(tokens, expected_tactic):
    assert _deepseek_v41_engram_mxfp8_tactic(tokens, n=25600, k=6144) == expected_tactic


def test_v41_engram_mxfp8_tactics_do_not_capture_other_linears():
    assert _deepseek_v41_engram_mxfp8_tactic(1, n=1792, k=5120) is None
    assert _deepseek_v41_engram_mxfp8_tactic(33, n=25600, k=6144) is None


def test_v41_custom_op_falls_back_for_unprofiled_shape(monkeypatch):
    x = torch.zeros((33, 128), dtype=torch.float8_e4m3fn)
    weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
    scale = torch.full((128,), 127, dtype=torch.uint8)
    called = {}

    def public_mm(*args, **kwargs):
        called.update(kwargs)
        return torch.zeros((33, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(mxfp8_flashinfer.vllm_flashinfer, "mm_mxfp8", public_mm)
    output = mxfp8_flashinfer._deepseek_v41_sm120_mxfp8_impl(
        x, weight, scale, scale, torch.bfloat16
    )

    assert output.shape == (33, 128)
    assert called == {"out_dtype": torch.bfloat16, "backend": "cutlass"}


def test_v41_engram_mxfp8_tactics_are_flashinfer_version_pinned(monkeypatch):
    check = mxfp8_flashinfer._has_deepseek_v41_sm120_mxfp8_tactics
    check.cache_clear()
    monkeypatch.setattr(
        mxfp8_flashinfer.importlib.metadata,
        "version",
        lambda _: mxfp8_flashinfer._DEEPSEEK_V41_MXFP8_TACTIC_FLASHINFER_VERSION,
    )
    assert check()
    check.cache_clear()
    monkeypatch.setattr(
        mxfp8_flashinfer.importlib.metadata,
        "version",
        lambda _: "unsupported",
    )
    assert not check()
    check.cache_clear()


def test_v41_engram_mxfp8_tactics_fail_closed_without_package_metadata(monkeypatch):
    check = mxfp8_flashinfer._has_deepseek_v41_sm120_mxfp8_tactics
    check.cache_clear()

    def missing(_distribution):
        raise mxfp8_flashinfer.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(mxfp8_flashinfer.importlib.metadata, "version", missing)
    assert not check()
    check.cache_clear()


def test_flashinfer_mxfp8_default_profile_keeps_public_cutlass_path(monkeypatch):
    x = torch.zeros((1, 128), dtype=torch.bfloat16)
    xq = torch.zeros((1, 128), dtype=torch.float8_e4m3fn)
    x_scale = torch.full((128,), 127, dtype=torch.uint8)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((128,), 127, dtype=torch.uint8),
        requires_grad=False,
    )
    called = {}

    monkeypatch.setattr(
        mxfp8_flashinfer,
        "mxfp8_e4m3_quantize",
        lambda *args, **kwargs: (xq, x_scale),
    )

    def public_mm(*args, **kwargs):
        called.update(kwargs)
        return torch.zeros((1, 128), dtype=torch.bfloat16)

    monkeypatch.setattr(mxfp8_flashinfer.vllm_flashinfer, "mm_mxfp8", public_mm)
    kernel = FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearLayerConfig())

    output = kernel.apply_weights(layer, x)

    assert output.shape == (1, 128)
    assert called == {"out_dtype": torch.bfloat16, "backend": "cutlass"}


def test_v41_engram_mxfp8_dynamic_compile_keeps_one_graph(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires an SM120 GPU")
    if not mxfp8_flashinfer._has_deepseek_v41_sm120_mxfp8_tactics():
        pytest.skip("requires the profiled FlashInfer tactic ABI")

    n, k = mxfp8_flashinfer._DEEPSEEK_V41_ENGRAM_MXFP8_SHAPE
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.full((n, k), 0.25, dtype=torch.float8_e4m3fn, device="cuda"),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.full((n, k // 32), 127, dtype=torch.uint8, device="cuda"),
        requires_grad=False,
    )
    kernel = FlashInferCutlassMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(model_profile="deepseek_v41")
    )
    kernel.process_weights_after_loading(layer)
    graphs = []
    workspace_keys = []
    inductor_backend = torch._dynamo.lookup_backend("inductor")
    from flashinfer import utils as flashinfer_utils

    get_cache_buf = flashinfer_utils._get_cache_buf

    def get_cache_buf_spy(name, *args, **kwargs):
        workspace_keys.append(name)
        return get_cache_buf(name, *args, **kwargs)

    monkeypatch.setattr(flashinfer_utils, "_get_cache_buf", get_cache_buf_spy)

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return inductor_backend(graph, _example_inputs)

    def apply(x):
        return kernel.apply_weights(layer, x)

    compiled = torch.compile(apply, backend=backend, fullgraph=True, dynamic=True)
    for tokens in (4, 128):
        x = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
        expected = apply(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert len(graphs) == 1
    assert "mm_mxfp8_workspace" in workspace_keys
    assert "deepseek_v41_sm120_mxfp8_workspace" not in workspace_keys


def test_v41_override_does_not_capture_legacy_model_types():
    v41_hf = SimpleNamespace(model_type="deepseek_v41")
    v4_hf = SimpleNamespace(model_type="deepseek_v4")

    assert (
        DeepseekV4FP8Config.override_quantization_method(_config_dict(), None, v41_hf)
        == "deepseek_v41_fp8"
    )
    assert (
        DeepseekV4FP8Config.override_quantization_method(_config_dict(), None, v4_hf)
        is None
    )
    assert (
        LegacyDeepseekV4FP8Config.override_quantization_method(
            _config_dict(), None, v4_hf
        )
        == "deepseek_v4_fp8"
    )


def test_v41_linear_dispatch_keeps_only_official_wo_a_unquantized():
    config = DeepseekV4FP8Config.from_config(_config_dict())
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)

    method = config.get_quant_method(layer, "model.layers.0.self_attn.q_a_proj")
    wo_a_method = config.get_quant_method(layer, "wo_a")

    assert isinstance(method, ModelOptLinearMethod)
    assert layer._mxfp8_model_profile == "deepseek_v41"
    assert isinstance(wo_a_method, DeepseekV41WoALinearMethod)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_v41_wo_a_loader_slices_weight_and_block_scales_on_output_axis(monkeypatch, tp):
    method = DeepseekV41WoALinearMethod.__new__(DeepseekV41WoALinearMethod)
    layer = torch.nn.Module()
    rank = tp - 1
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: tp
    )

    def load_column(param, loaded_weight):
        param.load_column_parallel_weight(loaded_weight)

    method.create_weights(
        layer,
        input_size_per_partition=4096,
        output_partition_sizes=[8192 // tp],
        input_size=4096,
        output_size=8192,
        params_dtype=torch.bfloat16,
        weight_loader=load_column,
    )
    shard_start = rank * (8192 // tp)
    loaded_weight = torch.zeros((8192, 4096), dtype=torch.float8_e4m3fn)
    loaded_weight[shard_start:].fill_(1.0)
    loaded_scale = torch.full((256, 128), 127, dtype=torch.uint8)
    loaded_scale[shard_start // 32 :].fill_(128)

    layer.weight_scale.weight_loader(layer.weight_scale, loaded_scale)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert layer.weight.shape == (8192 // tp, 4096)
    assert layer.weight_scale.shape == (8192 // tp, 128)
    assert torch.all(layer.weight == 1.0)
    assert torch.all(layer.weight_scale == 128)


@pytest.mark.parametrize("scale_first", [False, True])
def test_v41_wo_a_dequantizes_after_name_order_independent_loading(
    monkeypatch, scale_first
):
    method = DeepseekV41WoALinearMethod.__new__(DeepseekV41WoALinearMethod)
    layer = torch.nn.Module()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    def load_column(param, loaded_weight):
        param.load_column_parallel_weight(loaded_weight)

    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=load_column,
    )
    loaded_weight = torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    loaded_scale = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)
    loaders = (
        (
            layer.weight_scale,
            loaded_scale,
        ),
        (layer.weight, loaded_weight),
    )
    if not scale_first:
        loaders = loaders[::-1]
    for param, value in loaders:
        param.weight_loader(param, value)
    expected = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)


def test_v41_wo_a_rejects_missing_checkpoint_scale(monkeypatch):
    method = DeepseekV41WoALinearMethod.__new__(DeepseekV41WoALinearMethod)
    layer = torch.nn.Module()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )

    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, value: param.load_column_parallel_weight(value),
    )
    layer.weight.weight_loader(
        layer.weight, torch.ones((64, 64), dtype=torch.float8_e4m3fn)
    )

    with pytest.raises(ValueError, match="scale was not fully loaded"):
        method.process_weights_after_loading(layer)


def test_v41_wo_a_dummy_loader_uses_finite_neutral_scales(monkeypatch):
    method = DeepseekV41WoALinearMethod.__new__(DeepseekV41WoALinearMethod)
    layer = torch.nn.Module()
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        v41_quant_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(load_config=SimpleNamespace(load_format="dummy")),
    )
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=64,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=lambda param, value: param.load_column_parallel_weight(value),
    )
    layer.weight.data.fill_(1.0)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert torch.all(layer.weight == 1.0)
    assert torch.all(layer.weight_scale == 127)


@pytest.mark.parametrize(
    ("field", "value"),
    [("weight_block_size", [128, 128]), ("scale_fmt", "float32")],
)
def test_v41_rejects_inexact_checkpoint_formats(field, value):
    config = _config_dict()
    config[field] = value
    with pytest.raises(ValueError):
        DeepseekV4FP8Config.from_config(config)
