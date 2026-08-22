# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 selects the FlashInfer sparse MLA path ported from SM120."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Indexer,
    _requires_wide_eager_attention_region,
)
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLAAttention,
    DeepseekV4FlashInferSM120Attention,
    DeepseekV4FlashInferSparseMetadataBuilder,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import sparse_swa
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops import flashmla as flashmla_ops


def _fake_vllm_config(model_type: str = "deepseek_v4") -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def _make_indexer_for_forward_contract(
    run_indexer_op_in_forward: bool,
) -> DeepseekV4Indexer:
    indexer = object.__new__(DeepseekV4Indexer)
    indexer.compressor = Mock()
    indexer.ln_events = [None, None]
    indexer.aux_stream = None
    indexer.n_head = 2
    indexer.head_dim = 4
    indexer.softmax_scale = 0.5
    indexer.use_fp4_kv = False
    indexer._run_indexer_op_in_forward = run_indexer_op_in_forward
    indexer.indexer_op = Mock()
    indexer.k_cache = SimpleNamespace(prefix="indexer.k_cache")
    indexer.wq_b = Mock(
        return_value=(torch.arange(24, dtype=torch.float32).reshape(3, 8), None)
    )
    return indexer


def _patch_indexer_forward_cpu_ops(monkeypatch) -> None:
    monkeypatch.setattr(
        dsv4_attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )

    def fake_maybe_execute_in_parallel(default_fn, aux_fn, *args, **kwargs):
        return default_fn(), aux_fn()

    def fake_fused_indexer_q_rope_quant(
        positions,
        q,
        cos_sin_cache,
        indexer_weights,
        softmax_scale,
        weight_scale,
        *,
        use_fp4,
    ):
        q_scale = torch.full(q.shape[:-1] + (1,), 2.0)
        return (q, q_scale), indexer_weights + 1

    monkeypatch.setattr(
        dsv4_attention,
        "maybe_execute_in_parallel",
        fake_maybe_execute_in_parallel,
    )
    monkeypatch.setattr(
        dsv4_attention,
        "fused_indexer_q_rope_quant",
        fake_fused_indexer_q_rope_quant,
    )


def test_sm89_capability_accepted(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: True)

    with set_current_vllm_config(_fake_vllm_config()):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 9),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm86_capability_rejected(monkeypatch) -> None:
    with set_current_vllm_config(_fake_vllm_config()):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 6),
            attn_type="decoder",
        )

    assert "compute capability not supported" in invalid_reasons


def test_sm89_dsv4_defaults_to_ported_sm120_attention(monkeypatch) -> None:
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert (
        dsv4_model._select_dsv4_attn_cls(vllm_config)
        is DeepseekV4FlashInferSM120Attention
    )


@pytest.mark.parametrize(
    ("num_heads", "padded_num_heads"),
    [(1, 8), (8, 8), (9, 16), (17, 32), (33, 64), (65, 128)],
)
@pytest.mark.parametrize(
    "attention_cls",
    [DeepseekV4FlashInferMLAAttention, DeepseekV4FlashInferSM120Attention],
)
def test_flashinfer_sparse_native_q_head_counts(
    attention_cls, num_heads, padded_num_heads
) -> None:
    assert attention_cls.get_padded_num_q_heads(num_heads) == padded_num_heads


@pytest.mark.parametrize(
    "attention_cls",
    [DeepseekV4FlashInferMLAAttention, DeepseekV4FlashInferSM120Attention],
)
def test_flashinfer_sparse_rejects_more_than_128_q_heads(attention_cls) -> None:
    with pytest.raises(ValueError, match="does not support 129 heads"):
        attention_cls.get_padded_num_q_heads(129)


def test_sm89_dspark_uses_uniform_cudagraphs(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=7,
        )
    )

    builders = (
        DeepseekV4FlashMLAMetadataBuilder,
        DeepseekV32IndexerMetadataBuilder,
        DeepseekSparseSWAMetadataBuilder,
    )
    for builder in builders:
        assert (
            builder.get_cudagraph_support(vllm_config, None)
            is AttentionCGSupport.UNIFORM_BATCH
        )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (DeviceCapability(8, 9), AttentionCGSupport.UNIFORM_BATCH),
        (DeviceCapability(12, 0), AttentionCGSupport.UNIFORM_BATCH),
    ],
)
def test_flashinfer_sparse_supports_multitoken_cudagraphs(
    monkeypatch, capability, expected
) -> None:
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: capability,
    )

    assert (
        DeepseekV4FlashInferSparseMetadataBuilder.get_cudagraph_support(
            _fake_vllm_config(), None
        )
        is expected
    )


@pytest.mark.parametrize(
    ("capability", "use_v2_model_runner", "backend", "expected"),
    [
        (
            DeviceCapability(8, 9),
            True,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
            True,
        ),
        (
            DeviceCapability(12, 0),
            True,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
            False,
        ),
        (DeviceCapability(8, 9), True, None, True),
        (
            DeviceCapability(8, 9),
            True,
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
            False,
        ),
        (DeviceCapability(12, 0), False, None, True),
    ],
)
def test_deepseek_v4_wide_eager_attention_region(
    monkeypatch, capability, use_v2_model_runner, backend, expected
) -> None:
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda: capability,
    )
    vllm_config = SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        attention_config=SimpleNamespace(backend=backend),
    )

    assert _requires_wide_eager_attention_region(vllm_config) is expected


def test_sm89_indexer_forward_runs_indexer_op_inline(monkeypatch) -> None:
    _patch_indexer_forward_cpu_ops(monkeypatch)
    indexer = _make_indexer_for_forward_contract(run_indexer_op_in_forward=True)
    hidden_states = torch.zeros(3, 5)
    qr = torch.zeros(3, 6)
    compressed_kv_score = torch.zeros(3, 4)
    indexer_weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    positions = torch.arange(3)
    rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(0))

    result = DeepseekV4Indexer.forward(
        indexer,
        hidden_states,
        qr,
        compressed_kv_score,
        indexer_weights,
        positions,
        rotary_emb,
    )

    assert result == (None, None, None)
    indexer.indexer_op.assert_called_once()
    op_hidden_states, op_q_quant, op_k, op_weights = indexer.indexer_op.call_args.args
    assert op_hidden_states is hidden_states
    assert op_k is None
    assert torch.equal(op_q_quant[0], torch.arange(24).reshape(3, 2, 4))
    assert torch.equal(op_q_quant[1], torch.full((3, 2, 1), 2.0))
    assert torch.equal(op_weights, indexer_weights + 1)
    indexer.compressor.assert_called_once_with(
        compressed_kv_score, positions, rotary_emb
    )


def test_non_sm89_indexer_forward_defers_indexer_op(monkeypatch) -> None:
    _patch_indexer_forward_cpu_ops(monkeypatch)
    indexer = _make_indexer_for_forward_contract(run_indexer_op_in_forward=False)
    hidden_states = torch.zeros(3, 5)
    qr = torch.zeros(3, 6)
    compressed_kv_score = torch.zeros(3, 4)
    indexer_weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    positions = torch.arange(3)
    rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(0))

    q, q_scale, weights = DeepseekV4Indexer.forward(
        indexer,
        hidden_states,
        qr,
        compressed_kv_score,
        indexer_weights,
        positions,
        rotary_emb,
    )

    indexer.indexer_op.assert_not_called()
    assert torch.equal(q, torch.arange(24).reshape(3, 2, 4))
    assert torch.equal(q_scale, torch.full((3, 2, 1), 2.0))
    assert torch.equal(weights, indexer_weights + 1)
    indexer.compressor.assert_called_once_with(
        compressed_kv_score, positions, rotary_emb
    )


def test_sm89_swa_metadata_skips_flashmla_scheduler(monkeypatch) -> None:
    sm89 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda _: False,
    )
    monkeypatch.setattr(flashmla_ops, "current_platform", sm89)
    monkeypatch.setattr(flashmla_ops, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla_ops, "_flashmla_extension_C_AVAILABLE", True)
    get_mla_metadata = Mock()
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", get_mla_metadata)
    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder._layer_types = {"swaonly"}

    metadata = builder.build_tile_scheduler(num_decode_tokens=1)

    assert all(value is None for value in metadata.values())
    get_mla_metadata.assert_not_called()


def test_hopper_swa_metadata_keeps_flashmla_scheduler(monkeypatch) -> None:
    hopper = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 90,
    )
    monkeypatch.setattr(flashmla_ops, "current_platform", hopper)
    monkeypatch.setattr(flashmla_ops, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla_ops, "_flashmla_extension_C_AVAILABLE", True)
    scheduler = object()
    get_mla_metadata = Mock(return_value=(scheduler, None))
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", get_mla_metadata)
    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder._layer_types = {"swaonly"}

    metadata = builder.build_tile_scheduler(num_decode_tokens=1)

    assert metadata["swaonly"] is scheduler
    get_mla_metadata.assert_called_once_with()
