# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention import (
    sparse_mla_attention as sparse_attention_module,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
    _flashinfer_sparse_mla_config_error,
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as sparse_module
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
    _kv_scale_format_for_model,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(
    model_type: str,
    *,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    kv_lora_rank: int = 512,
    num_attention_heads: int = 32,
) -> SimpleNamespace:
    class FakeModelConfig(SimpleNamespace):
        def get_num_attention_heads(self, _parallel_config):
            return num_attention_heads

    return SimpleNamespace(
        model_config=FakeModelConfig(
            hf_text_config=SimpleNamespace(
                model_type=model_type,
                index_topk=2048,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                kv_lora_rank=kv_lora_rank,
            ),
        ),
        parallel_config=SimpleNamespace(),
    )


def _mock_single_tp(monkeypatch) -> None:
    monkeypatch.setattr(
        sparse_attention_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm89",
        lambda: True,
    )
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm89_glm_nope",
        lambda: True,
    )


def test_glm_nope_capability_checks_with_kv_cache_mla_signature(monkeypatch) -> None:
    flashinfer_module = ModuleType("flashinfer")
    decode_module = ModuleType("flashinfer.decode")
    mla_module = ModuleType("flashinfer.mla")
    sm120_module = ModuleType("flashinfer.mla._sparse_mla_sm120")
    cache_module = ModuleType("flashinfer.mla._sparse_mla_sm120_cache")

    def trtllm_batch_decode_with_kv_cache_mla(*args, kv_scale_format=None, **kwargs):
        pass

    decode_module.__dict__["trtllm_batch_decode_with_kv_cache_mla"] = (
        trtllm_batch_decode_with_kv_cache_mla
    )
    cache_module.__dict__["glm_nope_gather_and_dequantize"] = lambda *args, **kwargs: (
        None
    )
    cache_module.__dict__["glm_nope_quantize_and_cache"] = lambda *args, **kwargs: None
    sm120_module.__dict__["_DECODE_GLM_NOPE_PAGE_BLOCK_SIZE"] = 64
    sm120_module.__dict__["_DECODE_GLM_NOPE_TOPK2176_DISPATCH"] = frozenset(
        {(32, 2176)}
    )
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer_module)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode_module)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", mla_module)
    monkeypatch.setitem(sys.modules, "flashinfer.mla._sparse_mla_sm120", sm120_module)
    monkeypatch.setitem(
        sys.modules, "flashinfer.mla._sparse_mla_sm120_cache", cache_module
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope.cache_clear()
    fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope()
    assert fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(32, 2176, 64)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(16, 2176, 64)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(32, 2048, 64)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(32, 2176, 256)

    sm120_module.__dict__["_DECODE_GLM_NOPE_PAGE_BLOCK_SIZE"] = 256
    fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config.cache_clear()
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(32, 2176, 64)

    sm120_module.__dict__["_DECODE_GLM_NOPE_PAGE_BLOCK_SIZE"] = 64
    del sm120_module.__dict__["_DECODE_GLM_NOPE_TOPK2176_DISPATCH"]
    fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config.cache_clear()
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_glm_nope_config(32, 2176, 64)


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_sm89_backend_requires_runtime_flashinfer_probe(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: True)
    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        assert (
            FlashInferMLASparseSM120Backend.supports_combination(
                head_size=576,
                dtype=torch.bfloat16,
                kv_cache_dtype="fp8_ds_mla",
                block_size=64,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                use_mm_prefix=False,
                device_capability=DeviceCapability(8, 9),
            )
            is None
        )

    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: False)
    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        reason = FlashInferMLASparseSM120Backend.supports_combination(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=DeviceCapability(8, 9),
        )

    assert reason is not None
    assert "compatible with the current GPU" in reason


def test_sm89_glm_nope_backend_uses_glm_specific_runtime_probe(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: False)
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm89_glm_nope",
        lambda: True,
    )
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm120_glm_nope_config",
        lambda num_heads, top_k, page_block_size: (
            (
                num_heads,
                top_k,
                page_block_size,
            )
            == (32, 2176, 64)
        ),
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        assert (
            FlashInferMLASparseSM120Backend.supports_combination(
                head_size=512,
                dtype=torch.bfloat16,
                kv_cache_dtype="fp8_ds_mla",
                block_size=64,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                use_mm_prefix=False,
                device_capability=DeviceCapability(8, 9),
            )
            is None
        )


def test_sm89_dsv4_backend_selects_packed_flashinfer(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia import model as dsv4_model

    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: True)
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert (
        dsv4_model._select_dsv4_attn_cls(vllm_config)
        is DeepseekV4FlashInferSM120Attention
    )


def test_sm89_dsv4_backend_rejects_unpatched_flashinfer(monkeypatch) -> None:
    import pytest

    from vllm.models.deepseek_v4.nvidia import model as dsv4_model

    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: False)
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    with pytest.raises(RuntimeError, match="native sparse MLA SM89"):
        dsv4_model._select_dsv4_attn_cls(vllm_config)


def test_sm120_kernel_block_sizes_are_glm_config_aware() -> None:
    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        assert FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes() == [
            64
        ]

    with set_current_vllm_config(_fake_vllm_config("deepseek_v3")):
        assert FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes() == [
            64,
            256,
        ]

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=128, qk_rope_head_dim=64)
    ):
        assert FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes() == [
            64,
            256,
        ]


def test_glm_nope_uses_native_scale_format() -> None:
    assert _kv_scale_format_for_model("glm5_next", 256, 0, 512) == "arbitrary_fp32_nope"
    assert _kv_scale_format_for_model("glm5_next_text", 128, 64, 512) == (
        "arbitrary_fp32"
    )
    assert _kv_scale_format_for_model("deepseek_v3", 128, 64, 512) == "pow2_fp32"


def test_sm120_forward_uses_actual_topk_capacity_and_528_byte_geometry(
    monkeypatch,
) -> None:
    _mock_single_tp(monkeypatch)
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: True
    )
    monkeypatch.setattr(
        sparse_module,
        "_get_workspace_buffer",
        lambda device: torch.empty(1024, dtype=torch.uint8, device=device),
    )

    converted_topk = torch.arange(2 * 192, dtype=torch.int32).reshape(2, 192)
    convert_call: dict[str, Any] = {}

    def fake_convert_req_index_to_global_index(
        req_id_per_token, block_table, topk_indices, **kwargs
    ):
        convert_call["req_id_per_token"] = req_id_per_token
        convert_call["block_table"] = block_table
        convert_call["topk_indices"] = topk_indices
        convert_call["kwargs"] = kwargs
        return converted_topk, torch.tensor([192, 191], dtype=torch.int32)

    monkeypatch.setattr(
        sparse_module,
        "triton_convert_req_index_to_global_index",
        fake_convert_req_index_to_global_index,
    )

    decode_call: dict[str, Any] = {}

    def fake_decode(**kwargs):
        decode_call.update(kwargs)
        return torch.zeros(
            (2, 1, 32, 512),
            dtype=kwargs["query"].dtype,
            device=kwargs["query"].device,
        )

    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.decode",
        SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=fake_decode),
    )

    topk_indices = torch.arange(3 * 192, dtype=torch.int32).reshape(3, 192)
    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=32,
            head_size=512,
            scale=0.25,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=topk_indices,
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            qk_head_dim=256,
            v_head_dim=512,
            kv_b_proj=SimpleNamespace(),
        )

    q = torch.zeros((2, 32, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((7, 64, 528), dtype=torch.uint8)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1, 0], dtype=torch.int32),
        block_table=torch.arange(4, dtype=torch.int32).reshape(2, 2),
        block_size=64,
        topk_tokens=128,
        cp_kv_cache_interleave_size=1,
    )

    out, lse = impl.forward_mqa(q, kv_cache, metadata, SimpleNamespace())

    assert lse is None
    assert out.shape == (2, 32, 512)
    assert convert_call["topk_indices"].shape == (2, 192)
    assert convert_call["kwargs"]["NUM_TOPK_TOKENS"] == 192
    assert decode_call["query"].shape == (2, 1, 32, 512)
    assert decode_call["kv_cache"].shape == (7, 1, 64, 528)
    assert decode_call["block_tables"].shape == (2, 1, 192)
    assert decode_call["max_seq_len"] == 192
    assert decode_call["sparse_mla_top_k"] == 192
    assert decode_call["kv_scale_format"] == "arbitrary_fp32_nope"
    assert decode_call["bmm1_scale"] == 0.25
    assert decode_call["bmm2_scale"] == 1.0


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_glm_nope_sm120_accepts_manager_blocks_split_to_page64(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm120_glm_nope_config",
        lambda num_heads, top_k, page_block_size: (
            (
                num_heads,
                top_k,
                page_block_size,
            )
            == (32, 2176, 64)
        ),
    )

    cases = (
        (256, "fp8_ds_mla"),
        (2304, "fp8_ds_mla"),
        (2304, "auto"),
        (2304, "fp8"),
    )
    for manager_block_size, kv_cache_dtype in cases:
        with set_current_vllm_config(
            _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
        ):
            invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
                head_size=512,
                dtype=torch.bfloat16,
                kv_cache_dtype=kv_cache_dtype,
                block_size=manager_block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                use_mm_prefix=False,
                use_per_head_quant_scales=False,
                device_capability=DeviceCapability(12, 0),
                attn_type="decoder",
            )

        assert invalid_reasons == []


def test_non_glm_sm120_backend_keeps_64_and_256_block_sizes_when_glm_probe_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    for block_size in (64, 256):
        with set_current_vllm_config(_fake_vllm_config("deepseek_v3")):
            invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
                head_size=576,
                dtype=torch.bfloat16,
                kv_cache_dtype="fp8_ds_mla",
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=True,
                use_mm_prefix=False,
                use_per_head_quant_scales=False,
                device_capability=DeviceCapability(12, 0),
                attn_type="decoder",
            )

        assert invalid_reasons == []


@torch.no_grad()
def test_glm_nope_sm120_cache_update_uses_pack_helper(monkeypatch) -> None:
    _mock_single_tp(monkeypatch)
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: True
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...], list[int]]] = []
    cache_module = ModuleType("flashinfer.mla._sparse_mla_sm120_cache")

    def fake_quantize_and_cache(kv_c_normed, kv_cache, slot_mapping):
        calls.append(
            (
                tuple(kv_c_normed.shape),
                tuple(kv_cache.shape),
                slot_mapping.tolist(),
            )
        )

    cache_module.__dict__["glm_nope_quantize_and_cache"] = fake_quantize_and_cache
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.mla", ModuleType("flashinfer.mla"))
    monkeypatch.setitem(
        sys.modules, "flashinfer.mla._sparse_mla_sm120_cache", cache_module
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=32,
            head_size=512,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=torch.empty((1, 128), dtype=torch.int32),
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            qk_head_dim=256,
            v_head_dim=512,
            kv_b_proj=SimpleNamespace(),
        )

    impl.do_kv_cache_update(
        kv_c_normed=torch.empty((3, 512), dtype=torch.bfloat16),
        k_pe=torch.empty((3, 0), dtype=torch.bfloat16),
        kv_cache=torch.empty((2, 64, 528), dtype=torch.uint8),
        slot_mapping=torch.tensor([[1], [17], [42]], dtype=torch.int64),
        kv_cache_dtype="fp8_ds_mla",
        k_scale=torch.ones((), dtype=torch.float32),
    )

    assert calls == [((3, 512), (2, 64, 528), [1, 17, 42])]


@torch.no_grad()
def test_glm_nope_cache_update_rejects_rope_payload(monkeypatch) -> None:
    _mock_single_tp(monkeypatch)
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: True
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=32,
            head_size=512,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=torch.empty((1, 128), dtype=torch.int32),
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            qk_head_dim=256,
            v_head_dim=512,
            kv_b_proj=SimpleNamespace(),
        )

    try:
        impl.do_kv_cache_update(
            kv_c_normed=torch.empty((1, 512), dtype=torch.bfloat16),
            k_pe=torch.empty((1, 64), dtype=torch.bfloat16),
            kv_cache=torch.empty((1, 64, 528), dtype=torch.uint8),
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            kv_cache_dtype="fp8_ds_mla",
            k_scale=torch.ones((), dtype=torch.float32),
        )
    except ValueError as exc:
        assert "qk_rope_head_dim=0" in str(exc)
    else:
        raise AssertionError("GLM NoPE cache update accepted RoPE payload")


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192), (32, 256), (32, 1024)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 1024)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_backend_validation_does_not_require_glm_nope_probe(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    invalid_reasons = DeepseekV4FlashInferMLASparseBackend.validate_configuration(
        head_size=512,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8_ds_mla",
        block_size=256,
        use_mla=True,
        has_sink=True,
        use_sparse=True,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(12, 0),
        attn_type="decoder",
    )

    assert invalid_reasons == []


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    vision = SimpleNamespace(
        model_config=SimpleNamespace(
            is_mm_prefix_lm=True,
            hf_config=SimpleNamespace(vision_max_n_token=384),
        ),
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192
    assert _required_sm120_sparse_topk(vision, 128) == 512
    assert _required_sm120_sparse_topk(vision, 512) == 1024


def test_sm89_dsv4_does_not_require_sm120_dispatch_entry(monkeypatch) -> None:
    monkeypatch.setattr(
        fi_utils,
        "has_flashinfer_sparse_mla_sm120_config",
        lambda num_q_heads, top_k: False,
    )

    assert (
        _flashinfer_sparse_mla_config_error(
            DeviceCapability(8, 9),
            num_q_heads=8,
            top_k=128,
        )
        is None
    )
    assert "SM120 requires" in _flashinfer_sparse_mla_config_error(
        DeviceCapability(12, 0),
        num_q_heads=8,
        top_k=128,
    )


@torch.no_grad()
def test_glm_nope_requires_new_flashinfer_helpers(monkeypatch) -> None:
    _mock_single_tp(monkeypatch)
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    with set_current_vllm_config(
        _fake_vllm_config("glm5_next", qk_nope_head_dim=256, qk_rope_head_dim=0)
    ):
        try:
            FlashInferMLASparseSM120Impl(
                num_heads=32,
                head_size=512,
                scale=1.0,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="fp8_ds_mla",
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                topk_indices_buffer=torch.empty((1, 128), dtype=torch.int32),
                q_lora_rank=None,
                kv_lora_rank=512,
                qk_nope_head_dim=256,
                qk_rope_head_dim=0,
                qk_head_dim=256,
                v_head_dim=512,
                kv_b_proj=SimpleNamespace(),
            )
        except RuntimeError as exc:
            assert "GLM NoPE" in str(exc)
        else:
            raise AssertionError("GLM NoPE SM120 init accepted missing helpers")


@torch.no_grad()
def test_non_glm_sparse_mla_does_not_require_glm_helpers(monkeypatch) -> None:
    _mock_single_tp(monkeypatch)
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(
        fi_utils, "has_flashinfer_sparse_mla_sm120_glm_nope", lambda: False
    )

    with set_current_vllm_config(_fake_vllm_config("deepseek_v3")):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=32,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=torch.empty((1, 128), dtype=torch.int32),
            q_lora_rank=None,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=512,
            kv_b_proj=SimpleNamespace(),
        )

    assert impl.kv_scale_format == "pow2_fp32"
