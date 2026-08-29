# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 DeepSeek V4 dispatch guards."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

import vllm.platforms as platforms
from vllm.model_executor.layers import sparse_attn_indexer as indexer_layer
from vllm.models.deepseek_v4.nvidia.ops import (
    sm12x_deep_gemm_fallbacks as portable_mqa,
)
from vllm.models.deepseek_v4.nvidia.ops import sm12x_mqa
from vllm.platforms.interface import DeviceCapability
from vllm.utils import deep_gemm, import_utils
from vllm.v1.attention.backends.mla import indexer as indexer_module

warmup_module = importlib.import_module(
    "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
)


def _load_leaf_module(name: str, relative_path: str):
    path = Path(__file__).parents[3] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_sm89_platform(monkeypatch, module) -> None:
    if hasattr(module, "HAS_TRITON"):
        monkeypatch.setattr(module, "HAS_TRITON", True)
    monkeypatch.setattr(module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        module.current_platform,
        "is_device_capability",
        lambda capability: capability == (8, 9),
    )
    if hasattr(module.current_platform, "is_device_capability_family"):
        monkeypatch.setattr(
            module.current_platform,
            "is_device_capability_family",
            lambda family: False,
        )
    if hasattr(module.current_platform, "get_device_capability"):
        monkeypatch.setattr(
            module.current_platform,
            "get_device_capability",
            lambda: DeviceCapability(8, 9),
        )


def test_has_cutedsl_false_on_sm89(monkeypatch) -> None:
    monkeypatch.setattr(import_utils, "_has_module", lambda _: True)

    class FakePlatform:
        @staticmethod
        def is_cuda() -> bool:
            return True

        @staticmethod
        def is_device_capability(capability: tuple[int, int]) -> bool:
            return capability == (8, 9)

    monkeypatch.setattr(platforms, "current_platform", FakePlatform)

    assert not import_utils.has_cutedsl()


def test_deep_gemm_scheduler_metadata_requires_supported_platform(monkeypatch) -> None:
    monkeypatch.setattr(indexer_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_module, "is_deep_gemm_supported", lambda: False)

    assert not indexer_module._uses_deep_gemm_scheduler_metadata()


def test_deep_gemm_scheduler_metadata_enabled_on_supported_cuda(monkeypatch) -> None:
    monkeypatch.setattr(indexer_module.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_module, "is_deep_gemm_supported", lambda: True)

    assert indexer_module._uses_deep_gemm_scheduler_metadata()


def test_deep_gemm_mqa_wrappers_route_sm89_to_portable_path(monkeypatch) -> None:
    _mock_sm89_platform(monkeypatch, deep_gemm)

    calls = {}

    def fake_mqa(q, kv, weights, starts, ends, clean_logits):
        calls["mqa"] = (q, kv, weights, starts, ends, clean_logits)
        return torch.empty((1, 1), dtype=torch.float32)

    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_sm89", fake_mqa)

    q = torch.empty((1, 1, 1))
    scale = torch.empty((1,))
    result = deep_gemm.fp8_fp4_mqa_logits(
        (q, None),
        (q, scale),
        torch.empty((1, 1)),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        clean_logits=False,
    )

    assert result.shape == (1, 1)
    assert "mqa" in calls


def test_sm89_prefill_mqa_uses_triton_when_clean_logits_is_false(
    monkeypatch,
) -> None:
    expected = torch.empty((1, 1), dtype=torch.float32)
    monkeypatch.setattr(sm12x_mqa, "fp8_mqa_logits_triton", lambda *_args: expected)

    def unexpected_torch(*_args):
        raise AssertionError("unexpected torch path")

    monkeypatch.setattr(portable_mqa, "_fp8_mqa_logits_torch", unexpected_torch)
    q = torch.empty((1, 1, 1))
    k = torch.empty((1, 1))
    scale = torch.empty((1,))

    result = portable_mqa._fp8_mqa_logits_sm12x(
        (q, None),
        (k, scale),
        torch.empty((1, 1)),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        clean_logits=False,
    )

    assert result is expected


def test_sm89_indexer_backend_gate_rejects_fp4_without_deepgemm(monkeypatch) -> None:
    _mock_sm89_platform(monkeypatch, indexer_layer)
    monkeypatch.setattr(indexer_layer, "has_deep_gemm", lambda: False)

    assert indexer_layer._has_cuda_indexer_mqa_backend(use_fp4_cache=False)
    assert not indexer_layer._has_cuda_indexer_mqa_backend(use_fp4_cache=True)


def test_sm89_indexer_backend_gate_requires_triton(monkeypatch) -> None:
    _mock_sm89_platform(monkeypatch, indexer_layer)
    monkeypatch.setattr(indexer_layer, "has_deep_gemm", lambda: False)
    monkeypatch.setattr(indexer_layer, "HAS_TRITON", False)

    assert not indexer_layer._has_cuda_indexer_mqa_backend(use_fp4_cache=False)


def test_sm89_kpool_backend_gate_rejects_fp4_without_deepgemm(monkeypatch) -> None:
    for name in (
        "vllm.models.glm5next",
        "vllm.models.glm5next.nvidia",
        "vllm.models.glm5next.nvidia.ops",
        "vllm.models.glm5next.nvidia.ops.kpool_compress",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    kpool_layer = importlib.import_module(
        "vllm.model_executor.layers.sparse_attn_indexer_kpool"
    )
    _mock_sm89_platform(monkeypatch, kpool_layer)
    monkeypatch.setattr(kpool_layer, "has_deep_gemm", lambda: False)

    assert kpool_layer._has_cuda_indexer_mqa_backend(use_fp4_cache=False)
    assert not kpool_layer._has_cuda_indexer_mqa_backend(use_fp4_cache=True)


def test_flashinfer_sparse_mla_autotune_supports_native_sm89(monkeypatch) -> None:
    _mock_sm89_platform(monkeypatch, warmup_module)
    monkeypatch.setattr(warmup_module, "has_flashinfer", lambda: False)
    monkeypatch.setattr(warmup_module, "has_flashinfer_sparse_mla_sm89", lambda: True)

    assert warmup_module._flashinfer_sparse_mla_decode_autotune_supported()


def test_flashinfer_sparse_mla_autotune_rejects_unpatched_sm89(monkeypatch) -> None:
    _mock_sm89_platform(monkeypatch, warmup_module)
    monkeypatch.setattr(warmup_module, "has_flashinfer", lambda: True)
    monkeypatch.setattr(warmup_module, "has_flashinfer_sparse_mla_sm89", lambda: False)

    assert not warmup_module._flashinfer_sparse_mla_decode_autotune_supported()


def test_deepseek_v4_fp8_einsum_routes_only_sm89_to_triton(monkeypatch) -> None:
    einsum_module = _load_leaf_module(
        "dsv4_sm89_fp8_einsum_dispatch",
        "vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py",
    )
    monkeypatch.setattr(
        einsum_module,
        "_use_deepseek_v4_sm89_triton_fp8_einsum",
        lambda *_: True,
    )

    calls = {}

    def fake_triton(a, a_scale, b, b_scale, out):
        calls["triton"] = (a, a_scale, b, b_scale, out)

    monkeypatch.setattr(einsum_module, "_deepseek_v4_sm89_fp8_einsum", fake_triton)

    a = torch.empty((1, 2, 4))
    a_scale = torch.empty((1, 2, 1))
    b = torch.empty((8, 4))
    b_scale = torch.empty((2, 1))
    out = torch.empty((1, 2, 4))
    einsum_module.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        (1, 128, 128),
    )

    assert "triton" in calls
