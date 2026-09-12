# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v4_1.attention import (
    DeepseekV4IndexerCache,
    DeepseekV41SWACache,
)
from vllm.models.deepseek_v4_1.nvidia import dspark as dspark_module
from vllm.models.deepseek_v4_1.nvidia import flashinfer_sparse
from vllm.models.deepseek_v4_1.nvidia import model as model_module
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV41NativeMixedAttention,
    DeepseekV41NativeSparseBackend,
    DeepseekV41NativeSWABackend,
    DeepseekV41NativeSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def test_v41_native_backends_advertise_native_cache_page_sizes():
    assert DeepseekV41NativeSparseBackend.get_supported_kernel_block_sizes() == [128]
    assert DeepseekV41NativeSWABackend.get_supported_kernel_block_sizes() == [32]
    assert DeepseekV41NativeSWABackend.get_preferred_block_size(64) == 32


@pytest.mark.parametrize(
    ("adaptive", "graph_mode", "capability", "expected_threshold"),
    [
        (True, CUDAGraphMode.FULL, DeviceCapability(8, 9), 2048),
        (True, CUDAGraphMode.FULL, DeviceCapability(12, 0), 2048),
        (False, CUDAGraphMode.FULL, DeviceCapability(12, 0), 7),
        (True, CUDAGraphMode.FULL_AND_PIECEWISE, DeviceCapability(12, 0), 7),
        (True, CUDAGraphMode.FULL_DECODE_ONLY, DeviceCapability(12, 0), 7),
        (True, CUDAGraphMode.FULL, DeviceCapability(10, 0), 7),
    ],
)
def test_v41_adaptive_full_mixed_builder_uses_all_token_decode_capacity(
    monkeypatch,
    adaptive,
    graph_mode,
    capability,
    expected_threshold,
):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(enable_adaptive_verification=adaptive),
        compilation_config=SimpleNamespace(cudagraph_mode=graph_mode),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )
    monkeypatch.setattr(
        flashinfer_sparse.current_platform,
        "get_device_capability",
        lambda: capability,
    )

    def fake_base_init(self, *args, **kwargs):
        self.vllm_config = config
        self.decode_threshold = 7

    monkeypatch.setattr(
        flashinfer_sparse.DeepseekV41SparseSWAMetadataBuilder,
        "__init__",
        fake_base_init,
    )

    builder = DeepseekV41NativeSWAMetadataBuilder()

    assert builder.decode_threshold == expected_threshold


def test_v41_cache_specs_match_native_row_layouts():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))

    swa = DeepseekV41SWACache.__new__(DeepseekV41SWACache)
    swa.block_size = 32
    swa.head_dim = 512
    swa.window_size = 128
    swa.cache_config = SimpleNamespace(cache_dtype="fp8_ds_mla")
    swa_spec = swa.get_kv_cache_spec(vllm_config)
    assert swa_spec.block_size == 32
    assert swa_spec.state_content_bytes == 528
    assert swa_spec.alignment == 16

    indexer = DeepseekV4IndexerCache.__new__(DeepseekV4IndexerCache)
    indexer.head_dim = 68
    indexer.dtype = torch.uint8
    indexer.compress_ratio = 2
    indexer.cache_config = vllm_config.cache_config
    indexer_spec = indexer.get_kv_cache_spec(vllm_config)
    assert indexer_spec.block_size == 128
    assert indexer_spec.tokens_per_state == 2
    assert indexer_spec.state_content_bytes == 68
    assert indexer_spec.alignment == 128

    global_cache = DeepseekV41NativeMixedAttention.__new__(
        DeepseekV41NativeMixedAttention
    )
    global_cache.is_kv_source = True
    global_cache.head_dim = 512
    global_cache.compress_ratio = 2
    global_cache.kv_cache_dtype = "fp8_ds_mla"
    global_spec = global_cache.get_kv_cache_spec(vllm_config)
    assert global_spec is not None
    assert global_spec.block_size == 128
    assert global_spec.tokens_per_state == 2
    assert global_spec.state_content_bytes == 288
    assert global_spec.alignment == 32

    merged_swa = SlidingWindowMLASpec.merge([swa_spec, swa_spec])
    merged_global = MLAAttentionSpec.merge([global_spec, global_spec])
    assert merged_swa.model_version == "deepseek_v4_1"
    assert merged_swa.state_content_bytes == 528
    assert merged_global.model_version == "deepseek_v4_1"
    assert merged_global.state_content_bytes == 288


@pytest.mark.parametrize(
    "capability", [DeviceCapability(8, 9), DeviceCapability(12, 0)]
)
def test_v41_attention_selector_accepts_only_native_sm_targets(capability, monkeypatch):
    platform = SimpleNamespace(
        is_rocm=lambda: False,
        get_device_capability=lambda: capability,
    )
    monkeypatch.setattr(model_module, "current_platform", platform)
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert model_module._select_dsv4_attn_cls(config) is DeepseekV41NativeMixedAttention


def test_v41_attention_selector_rejects_fallback_backend(monkeypatch):
    platform = SimpleNamespace(
        is_rocm=lambda: False,
        get_device_capability=lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(model_module, "current_platform", platform)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV4
        )
    )

    with pytest.raises(ValueError, match="does not fall back"):
        model_module._select_dsv4_attn_cls(config)


def test_v41_attention_selector_rejects_unsupported_hardware(monkeypatch):
    platform = SimpleNamespace(
        is_rocm=lambda: False,
        get_device_capability=lambda: DeviceCapability(9, 0),
    )
    monkeypatch.setattr(model_module, "current_platform", platform)
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    with pytest.raises(NotImplementedError, match="SM89 and SM12x"):
        model_module._select_dsv4_attn_cls(config)


def test_v41_native_attention_passes_separate_cache_strides(monkeypatch):
    seen: dict[str, Any] = {}
    workspace = torch.empty(4096, dtype=torch.uint8)

    def fake_native(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        kwargs["out"].fill_(2)
        return None

    monkeypatch.setattr(flashinfer_sparse, "_native_mixed_attention", fake_native)
    monkeypatch.setattr(
        flashinfer_sparse,
        "_get_native_workspace",
        lambda q, swa_slots, global_slots: (
            seen.setdefault("workspace_slots", swa_slots),
            workspace,
        )[1],
    )
    layer = DeepseekV41NativeMixedAttention.__new__(DeepseekV41NativeMixedAttention)
    layer.scale = 0.125
    layer.attn_sink = None
    layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(32, 64))
    q = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    swa_cache = torch.empty(3, 32, 528, dtype=torch.uint8)
    global_cache = torch.empty(4, 64, 288, dtype=torch.uint8)
    swa_slots = torch.tensor([[[0, 1]], [[32, -1]]], dtype=torch.int32)
    global_slots = torch.tensor([[0, 2], [64, -1]], dtype=torch.int32)
    output = torch.empty_like(q)
    positions = torch.arange(2, dtype=torch.int64)
    grouped_output = output.view(1, 2, 4096)

    layer._run_native(
        q,
        swa_cache,
        global_cache,
        swa_slots,
        global_slots,
        output,
        positions,
        grouped_output,
        0,
    )

    native_args = seen["args"]
    assert native_args[0] is q
    assert native_args[1] is swa_cache
    assert native_args[2] is global_cache
    assert torch.equal(native_args[3], swa_slots[:, 0])
    assert native_args[4] is global_slots
    assert native_args[5] == layer.scale
    assert seen["kwargs"]["swa_page_stride_bytes"] == swa_cache.stride(0)
    assert seen["kwargs"]["swa_row_stride_bytes"] == swa_cache.stride(1)
    assert seen["kwargs"]["global_page_stride_bytes"] == global_cache.stride(0)
    assert seen["kwargs"]["global_row_stride_bytes"] == global_cache.stride(1)
    assert seen["kwargs"]["workspace"] is workspace
    assert seen["kwargs"]["positions"] is positions
    assert seen["kwargs"]["cos_sin"] is layer.rotary_emb.cos_sin_cache
    assert seen["kwargs"]["grouped_out"] is grouped_output
    assert seen["kwargs"]["token_offset"] == 0
    assert seen["workspace_slots"].shape == (2, 2)
    assert torch.all(output == 2)


def test_v41_mixed_batch_epilogue_uses_full_group_stride(monkeypatch):
    """Decode and prefill slices must share one grouped output allocation."""
    seen = []
    layer = DeepseekV41NativeMixedAttention.__new__(DeepseekV41NativeMixedAttention)
    layer.n_local_groups = 2
    layer.compress_ratio = 0
    layer.compressed_cache_prefix = None
    layer.swa_cache_layer = SimpleNamespace(
        prefix="swa", kv_cache=torch.empty(1, 32, 528, dtype=torch.uint8)
    )
    layer._run_native = lambda *args: seen.append(args)
    metadata = SimpleNamespace(
        num_decode_tokens=2,
        num_prefill_tokens=3,
        decode_swa_indices=torch.zeros(2, 128, dtype=torch.int32),
        prefill_swa_indices=torch.zeros(3, 128, dtype=torch.int32),
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"swa": metadata}),
    )
    q = torch.empty(5, 16, 512, dtype=torch.bfloat16)
    positions = torch.tensor([17, 18, 64, 65, 66])
    output = torch.empty_like(q)
    layer.forward_mqa(q, q[:, 0], positions, output)

    assert len(seen) == 2
    for args, start, end in zip(seen, (0, 2), (2, 5)):
        torch.testing.assert_close(args[6], positions[start:end])
        assert args[7].shape == (2, 5, 4096)
        assert args[7].data_ptr() == output.data_ptr()
        assert args[8] == start


def test_v41_output_projection_consumes_grouped_epilogue_without_second_rope():
    torch.manual_seed(43)
    layer = DeepseekV41NativeMixedAttention.__new__(DeepseekV41NativeMixedAttention)
    layer.n_local_groups = 2
    layer.o_lora_rank = 3
    weight = torch.randn(2, 3, 4096)
    layer.wo_a = SimpleNamespace(weight=weight)
    layer.wo_b = lambda value: value
    grouped = torch.randn(2, 5, 4096)
    buffer = grouped.view(5, 16, 512)
    expected = torch.einsum("gtd,grd->tgr", grouped, weight).flatten(1)

    actual = layer._o_proj(buffer, torch.tensor([17, 18, 64, 65, 66]))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("max_model_len", "expected_max_swa_slots"), [(2048, 2048), (64, 128)]
)
def test_v41_native_attention_constructor_reserves_worst_case_workspace(
    monkeypatch, max_model_len, expected_max_swa_slots
):
    queries = []
    reservations = []

    def fake_base_init(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.n_local_heads = 8
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = 2048
        self.window_size = 128
        self.index_topk = 512
        self.compress_ratio = 1

    def fake_upper_bound(
        max_num_tokens,
        num_heads,
        min_swa_slots,
        max_swa_slots,
        num_global_slots,
    ):
        queries.append(
            (
                max_num_tokens,
                num_heads,
                min_swa_slots,
                max_swa_slots,
                num_global_slots,
            )
        )
        return 7000

    class FakeWorkspaceManager:
        def get_simultaneous(self, reservation):
            reservations.append(reservation)
            return [torch.empty(reservation[0], dtype=reservation[1])]

    monkeypatch.setattr(
        flashinfer_sparse.DeepseekV4Attention, "__init__", fake_base_init
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "_native_workspace_size_upper_bound",
        fake_upper_bound,
    )
    monkeypatch.setattr(
        flashinfer_sparse, "is_workspace_manager_initialized", lambda: True
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "current_workspace_manager",
        lambda: FakeWorkspaceManager(),
    )

    DeepseekV41NativeMixedAttention(None, prefix="model.layers.0.attn")

    assert queries == [(2048, 8, 128, expected_max_swa_slots, 512)]
    assert reservations == [((7000,), torch.uint8)]


@pytest.mark.parametrize("num_heads", [8, 16, 32])
@pytest.mark.parametrize("max_num_tokens", [1, 16, 32, 33, 2048, 8192])
@pytest.mark.parametrize(
    ("num_swa_slots", "num_global_slots"),
    [(128, 512), (128, 0), (2048, 0)],
)
def test_v41_workspace_reservation_covers_flashinfer_formula(
    num_heads,
    max_num_tokens,
    num_swa_slots,
    num_global_slots,
):
    from flashinfer.mla import deepseek_v41_mixed_sparse_workspace_size

    expected = max(
        deepseek_v41_mixed_sparse_workspace_size(
            num_tokens,
            num_heads,
            num_swa_slots,
            num_global_slots,
        )
        for num_tokens in range(1, max_num_tokens + 1)
    )

    assert (
        flashinfer_sparse._max_native_workspace_size(
            num_heads,
            max_num_tokens,
            num_swa_slots,
            num_global_slots,
        )
        == expected
    )


def test_v41_wide_swa_only_workspace_peaks_inside_decode_range():
    from flashinfer.mla import deepseek_v41_mixed_sparse_workspace_size

    peak = deepseek_v41_mixed_sparse_workspace_size(31, 8, 2048, 0)
    endpoints = max(
        deepseek_v41_mixed_sparse_workspace_size(1, 8, 2048, 0),
        deepseek_v41_mixed_sparse_workspace_size(2048, 8, 2048, 0),
    )

    assert peak == 31 * 5 * 8 * (512 * 2 + 4)
    assert peak > endpoints


@pytest.mark.parametrize(("max_num_tokens", "expected"), [(1, 526336), (32, 1274720)])
def test_v41_workspace_reservation_covers_unified_swa_width_peak(
    max_num_tokens, expected
):
    from flashinfer.mla import deepseek_v41_mixed_sparse_workspace_size

    reserved = flashinfer_sparse._native_workspace_size_upper_bound(
        max_num_tokens=max_num_tokens,
        num_heads=8,
        min_swa_slots=128,
        max_swa_slots=2048,
        num_global_slots=512,
    )
    exhaustive = max(
        deepseek_v41_mixed_sparse_workspace_size(num_tokens, 8, slots, 512)
        for num_tokens in range(1, max_num_tokens + 1)
        for slots in range(128, 2049)
    )

    assert reserved == exhaustive == expected


def test_v41_dspark_context_uses_native_swa_writer(monkeypatch):
    seen = {}

    def fake_writer(q, kv, positions, cos_sin_cache, cache, slots):
        seen["args"] = (q, kv, positions, cos_sin_cache, cache, slots)
        return q

    monkeypatch.setattr(dspark_module, "fused_q_rope_swa_insert", fake_writer)
    cache = torch.empty(2, 32, 528, dtype=torch.uint8)
    cos_sin_cache = torch.empty(8, 64)
    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=cache),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        n_local_heads=8,
        head_dim=512,
    )
    kv = torch.empty(3, 512, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    slots = torch.tensor([0, 1, 32], dtype=torch.int64)

    dspark_module._insert_context_kv(attn, kv, positions, slots)

    q, seen_kv, seen_positions, seen_cos_sin, seen_cache, seen_slots = seen["args"]
    assert q.shape == (3, 8, 512)
    assert q.dtype == torch.bfloat16
    assert torch.count_nonzero(q) == 0
    assert seen_kv is kv
    assert seen_positions is positions
    assert seen_cos_sin is cos_sin_cache
    assert seen_cache is cache
    assert seen_slots is slots
