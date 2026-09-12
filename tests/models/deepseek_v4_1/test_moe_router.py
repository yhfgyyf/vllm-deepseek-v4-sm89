# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.models.deepseek_v4.nvidia import model as base_model
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_SENTINEL_BASE_ID,
)
from vllm.models.deepseek_v4_1.nvidia import dspark as dspark_module
from vllm.models.deepseek_v4_1.nvidia import model as model_module
from vllm.models.deepseek_v4_1.nvidia import vl_model as vl_model_module
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    _make_deepseek_v4_weights_mapper,
)
from vllm.models.deepseek_v4_1.nvidia.router import (
    DeepseekV41TopKBiasRouter,
    deepseek_v41_topk,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


class _FakeGate(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(output_size, input_size), requires_grad=False
        )


class _FakeRunner(torch.nn.Module):
    def __init__(self, router: FusedTopKBiasRouter) -> None:
        super().__init__()
        self.router = router
        self.is_monolithic = False


def _make_vllm_config(*, enable_jit_warmup: bool = False):
    hf_config = DeepseekV41Config(
        text_config={
            "hidden_size": 8,
            "n_routed_experts": 384,
            "num_experts_per_tok": 6,
            "dspark_n_routed_experts": 128,
            "dspark_num_experts_per_tok": 3,
            "num_hidden_layers": 2,
            "num_hash_layers": 7,
            "vocab_size": IMAGE_SENTINEL_BASE_ID + 32,
            "image_token_id": IMAGE_SENTINEL_BASE_ID,
            "topk_method": "noaux_tc",
            "n_shared_experts": None,
            "moe_intermediate_size": 16,
            "swiglu_limit": None,
            "norm_topk_prob": True,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.25,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 1,
            "hc_eps": 1e-6,
        },
        vision_config={"num_hidden_layers": 1},
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        kernel_config=SimpleNamespace(
            moe_backend=None, enable_jit_warmup=enable_jit_warmup
        ),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            enable_eplb=False,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
    )


@pytest.fixture
def lightweight_moe_construction(monkeypatch):
    calls: list[dict] = []

    def fake_factory(**kwargs):
        calls.append(kwargs)
        router = FusedTopKBiasRouter(
            top_k=kwargs["top_k"],
            global_num_experts=kwargs["num_experts"],
            e_score_correction_bias=kwargs["e_score_correction_bias"],
            e_score_correction_bias_vl=kwargs["e_score_correction_bias_vl"],
            input_vocab_size=kwargs["input_vocab_size"],
            renormalize=kwargs["renormalize"],
            scoring_func=kwargs["scoring_func"],
            routed_scaling_factor=kwargs["routed_scaling_factor"],
            hash_indices_table=kwargs["hash_indices_table"],
        )
        return _FakeRunner(router)

    monkeypatch.setattr(base_model, "GateLinear", _FakeGate)
    monkeypatch.setattr(base_model, "FusedMoEFactory", fake_factory)
    monkeypatch.setattr(base_model, "validate_fi_moe_ep_config", lambda config: None)
    monkeypatch.setattr(base_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(base_model, "get_tensor_model_parallel_rank", lambda: 0)
    return calls


@pytest.mark.parametrize(
    ("prefix", "num_experts", "topk"),
    [
        ("model.layers.0.ffn", 384, 6),
        ("model.layers.2.ffn", 128, 3),
    ],
)
def test_v41_moe_inherited_constructor_adapts_target_and_draft_geometry(
    lightweight_moe_construction,
    prefix,
    num_experts,
    topk,
):
    vllm_config = _make_vllm_config()
    original_config = vllm_config.model_config.hf_config

    moe = DeepseekV4MoE(vllm_config, prefix=prefix)

    assert moe.n_routed_experts == num_experts
    assert moe.n_activated_experts == topk
    assert moe.gate.weight.shape == (num_experts, original_config.hidden_size)
    assert moe.gate.e_score_correction_bias.shape == (num_experts,)
    assert moe.gate.e_score_correction_bias_vl.shape == (num_experts,)
    assert moe.gate.tid2eid is None
    assert isinstance(moe.experts.router, DeepseekV41TopKBiasRouter)
    assert moe.experts.router.top_k == topk
    assert lightweight_moe_construction[-1]["num_experts"] == num_experts
    assert lightweight_moe_construction[-1]["top_k"] == topk
    assert original_config.n_routed_experts == 384
    assert original_config.num_experts_per_tok == 6
    assert original_config.num_hash_layers == 7


def test_v41_decoder_constructs_real_inherited_moe(
    monkeypatch, lightweight_moe_construction
):
    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRMSNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(
        model_module, "_select_dsv4_attn_cls", lambda config: FakeAttention
    )
    monkeypatch.setattr(model_module, "RMSNorm", FakeRMSNorm)
    layer = DeepseekV4DecoderLayer(
        _make_vllm_config(),
        prefix="model.layers.0",
    )

    assert isinstance(layer.ffn, DeepseekV4MoE)
    assert isinstance(layer.ffn.experts.router, DeepseekV41TopKBiasRouter)
    assert lightweight_moe_construction[-1]["num_experts"] == 384


def test_v41_decoder_constructs_with_jit_warmup_enabled(
    monkeypatch, lightweight_moe_construction
):
    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeRMSNorm(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    monkeypatch.setattr(
        model_module, "_select_dsv4_attn_cls", lambda config: FakeAttention
    )
    monkeypatch.setattr(model_module, "RMSNorm", FakeRMSNorm)
    monkeypatch.setattr(
        model_module,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )

    layer = DeepseekV4DecoderLayer(
        _make_vllm_config(enable_jit_warmup=True),
        prefix="model.layers.0",
    )

    assert isinstance(layer.ffn, DeepseekV4MoE)


def test_v41_target_weight_mapper_maps_image_router_bias():
    mapper = _make_deepseek_v4_weights_mapper("fp4")

    assert (
        mapper._map_name("layers.0.ffn.gate.bias_vl")
        == "model.layers.0.ffn.gate.e_score_correction_bias_vl"
    )


def test_v41_vl_loader_maps_image_router_bias_through_backbone(monkeypatch):
    def uninitialized(cls):
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        return module

    backbone = uninitialized(model_module.DeepseekV4Model)
    backbone.config = SimpleNamespace(num_attention_heads=1)
    backbone.quant_config = None
    backbone.use_sequence_parallel = False
    backbone.get_expert_mapping = lambda: []
    layer = torch.nn.Module()
    layer.ffn = torch.nn.Module()
    layer.ffn.gate = torch.nn.Module()
    text_bias = torch.full((384,), -1.0)
    layer.ffn.gate.e_score_correction_bias = torch.nn.Parameter(
        text_bias.clone(), requires_grad=False
    )
    layer.ffn.gate.e_score_correction_bias_vl = torch.nn.Parameter(
        torch.zeros(384), requires_grad=False
    )
    backbone.layers = torch.nn.ModuleList([layer])

    language_model = uninitialized(model_module.DeepseekV41LLMForCausalLM)
    language_model.model = backbone
    language_model.lm_head = torch.nn.Linear(8, 8, bias=False)
    language_model.hf_to_vllm_mapper = WeightsMapper()
    language_model.process_weights_after_loading = lambda: None

    model = uninitialized(vl_model_module.DeepseekV41ForCausalLM)
    model.language_model = language_model
    model.hf_to_vllm_mapper = vl_model_module._make_deepseek_v4_vl_weights_mapper(
        "fp4", "weight_scale"
    )
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)

    image_bias = torch.arange(384, dtype=torch.float32)
    head_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    loaded = model.load_weights(
        [
            ("layers.0.ffn.gate.bias_vl", image_bias),
            ("head.weight", head_weight),
        ]
    )

    assert loaded == {
        "language_model.model.layers.0.ffn.gate.e_score_correction_bias_vl",
        "language_model.lm_head.weight",
    }
    assert torch.equal(layer.ffn.gate.e_score_correction_bias_vl, image_bias)
    assert torch.equal(layer.ffn.gate.e_score_correction_bias, text_bias)
    assert torch.equal(language_model.lm_head.weight, head_weight)


def test_v41_dspark_loader_maps_image_router_bias(monkeypatch):
    draft = dspark_module.DSparkDeepseekV4ForCausalLM.__new__(
        dspark_module.DSparkDeepseekV4ForCausalLM
    )
    torch.nn.Module.__init__(draft)
    draft.config = SimpleNamespace(
        num_attention_heads=1,
        n_routed_experts=384,
        dspark_n_routed_experts=128,
    )
    draft.quant_config = None
    draft.linear_scale_name = "weight_scale"
    draft.pad_shared_expert = False
    draft.model = torch.nn.Module()
    draft.model.confidence_head = None
    layer = torch.nn.Module()
    layer.ffn = torch.nn.Module()
    layer.ffn.use_mega_moe = False
    layer.ffn.gate = torch.nn.Module()
    layer.ffn.gate.e_score_correction_bias_vl = torch.nn.Parameter(
        torch.zeros(128), requires_grad=False
    )
    draft.model.layers = torch.nn.ModuleList([layer])
    draft.process_weights_after_loading = lambda: None
    monkeypatch.setattr(
        dspark_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(dspark_module, "get_tensor_model_parallel_rank", lambda: 0)

    loaded = draft.load_weights(
        [("mtp.0.ffn.gate.bias_vl", torch.arange(128, dtype=torch.float32))]
    )

    name = "model.layers.0.ffn.gate.e_score_correction_bias_vl"
    assert loaded == {name}
    assert torch.equal(
        layer.ffn.gate.e_score_correction_bias_vl,
        torch.arange(128, dtype=torch.float32),
    )


def _router_reference(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    image_bias: torch.Tensor,
    input_ids: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softplus(logits.float()).sqrt()
    image_mask = (input_ids >= IMAGE_SENTINEL_BASE_ID) & (
        input_ids < IMAGE_SENTINEL_BASE_ID + 5
    )
    choice_bias = torch.where(
        image_mask.unsqueeze(1), image_bias.unsqueeze(0), text_bias.unsqueeze(0)
    )
    topk_ids = torch.argsort(
        scores + choice_bias, dim=-1, descending=True, stable=True
    )[:, :topk]
    topk_weights = scores.gather(1, topk_ids)
    topk_weights *= routed_scaling_factor / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepSeek V4.1 native routing is CUDA-only."
)
@pytest.mark.parametrize(("num_experts", "topk"), [(384, 6), (128, 3)])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_v41_native_router_matches_reference_at_sentinels_and_ties(
    num_experts,
    topk,
    indices_dtype,
):
    torch.manual_seed(0)
    num_tokens = 8
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    logits[0].zero_()
    logits[1] = torch.linspace(-80.0, -20.0, num_experts, device="cuda")
    logits[2].fill_(-40.0)
    logits[2, 0] = -20.0
    logits[2, 1] = -20.0 + 1e-5
    text_bias = torch.zeros(num_experts, device="cuda")
    image_bias = torch.linspace(-2.0, 2.0, num_experts, device="cuda")
    input_ids = torch.tensor(
        [
            IMAGE_SENTINEL_BASE_ID - 1,
            0,
            1,
            IMAGE_SENTINEL_BASE_ID,
            IMAGE_SENTINEL_BASE_ID + 4,
            IMAGE_SENTINEL_BASE_ID + 5,
            IMAGE_SENTINEL_BASE_ID + 3,
            IMAGE_SENTINEL_BASE_ID + 6,
        ],
        device="cuda",
        dtype=torch.int64,
    )
    scale = 1.25

    expected_weights, expected_ids = _router_reference(
        logits, text_bias, image_bias, input_ids, topk, scale
    )
    actual_weights, actual_ids = deepseek_v41_topk(
        logits,
        text_bias,
        indices_dtype,
        scale,
        topk=topk,
        correction_bias_vl=image_bias,
        input_tokens=input_ids,
    )

    torch.testing.assert_close(
        actual_ids, expected_ids.to(indices_dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(actual_weights, expected_weights, atol=2e-5, rtol=2e-5)


def test_v41_native_router_has_no_cpu_fallback():
    with pytest.raises(ValueError, match="contiguous CUDA FP32"):
        deepseek_v41_topk(
            torch.zeros(1, 128),
            torch.zeros(128),
            torch.int32,
            1.0,
            topk=3,
        )
