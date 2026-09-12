# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.sparse_attn_indexer import (
    _dsv41_expand_decode_seq_lens,
    _dsv41_rows_per_chunk,
    _dsv41_workspace_specs,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla import indexer as indexer_backend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    DeepseekV4IndexerMetadataBuilder,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
    DeepseekV41IndexerMetadataBuilder,
    _dsv41_decode_max_indexer_kv_len,
    dsv41_indexer_uses_fp4,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.workspace import WorkspaceManager


class _AttentionConfig:
    def __init__(self, dtype: str):
        self.dtype = dtype

    def resolve_indexer_kv_dtype(self, default: str) -> str:
        return default if self.dtype == "auto" else self.dtype


def _config(dtype: str = "auto"):
    return SimpleNamespace(attention_config=_AttentionConfig(dtype))


def _builder_config(
    max_model_len: int = 128,
    *,
    adaptive: bool = False,
    num_speculative_tokens: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL_AND_PIECEWISE,
    max_num_batched_tokens: int = 8,
):
    speculative_config = None
    if adaptive or num_speculative_tokens:
        speculative_config = SimpleNamespace(
            enable_adaptive_verification=adaptive,
            num_speculative_tokens=num_speculative_tokens,
        )
    return SimpleNamespace(
        attention_config=_AttentionConfig("auto"),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=4,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=max_num_batched_tokens,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
    )


def _set_cuda_capability(monkeypatch, capability, *, has_deep_gemm: bool) -> None:
    monkeypatch.setattr(indexer_backend.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer_backend.current_platform,
        "is_device_capability",
        lambda target: capability == target,
    )
    monkeypatch.setattr(
        indexer_backend.current_platform,
        "is_device_capability_family",
        lambda family: capability[0] == family // 10,
    )
    monkeypatch.setattr(indexer_backend, "has_deep_gemm", lambda: has_deep_gemm)


def _build_single_decode(
    builder_cls,
    monkeypatch,
    *,
    compress_ratio: int = 1,
    seq_len: int = 17,
    max_model_len: int = 128,
    for_cudagraph_capture: bool = False,
):
    monkeypatch.setattr(indexer_backend, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(indexer_backend, "_use_flattening", lambda _: False)
    monkeypatch.setattr(
        indexer_backend, "_supports_varlen_paged_mqa_logits", lambda: False
    )
    monkeypatch.setattr(indexer_backend, "dsv41_indexer_uses_fp4", lambda _: True)
    monkeypatch.setattr(
        indexer_backend,
        "get_compressed_slot_mapping",
        lambda num_tokens, *_args, out, **_kwargs: out[:num_tokens],
    )
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
        state_content_bytes=68,
        alignment=128,
        model_version="deepseek_v4_1",
    )
    builder = builder_cls(
        kv_cache_spec=spec,
        layer_names=["model.layers.20.attn.indexer.k_cache"],
        vllm_config=_builder_config(max_model_len),
        device=torch.device("cpu"),
        block_table_width=1,
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=seq_len,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([seq_len], dtype=torch.int32),
    )
    metadata = (
        builder.build_for_cudagraph_capture(common)
        if for_cudagraph_capture
        else builder.build(0, common)
    )
    return builder, metadata


def test_v41_backend_isolated_interface() -> None:
    assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [256]
    assert (
        DeepseekV4IndexerBackend.get_builder_cls() is DeepseekV4IndexerMetadataBuilder
    )
    assert DeepseekV41IndexerBackend.get_name() == "DEEPSEEK_V41_INDEXER"
    assert not DeepseekV41IndexerBackend.supports_pcp()
    assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [128]
    assert (
        DeepseekV41IndexerBackend.get_builder_cls() is DeepseekV41IndexerMetadataBuilder
    )


@pytest.mark.parametrize(
    (
        "capability",
        "has_deep_gemm",
        "backend_cls",
        "builder_cls",
        "flattened",
        "varlen",
    ),
    [
        (
            (8, 9),
            False,
            DeepseekV4IndexerBackend,
            DeepseekV4IndexerMetadataBuilder,
            True,
            False,
        ),
        (
            (12, 0),
            True,
            DeepseekV4IndexerBackend,
            DeepseekV4IndexerMetadataBuilder,
            False,
            True,
        ),
        (
            (8, 9),
            False,
            DeepseekV41IndexerBackend,
            DeepseekV41IndexerMetadataBuilder,
            True,
            False,
        ),
        (
            (12, 0),
            True,
            DeepseekV41IndexerBackend,
            DeepseekV41IndexerMetadataBuilder,
            True,
            False,
        ),
    ],
)
def test_adaptive_model_specific_metadata_capabilities(
    monkeypatch,
    capability,
    has_deep_gemm,
    backend_cls,
    builder_cls,
    flattened,
    varlen,
) -> None:
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=has_deep_gemm)
    config = _builder_config(adaptive=True)

    assert backend_cls.supports_device_cpu_query_lens_mismatch()
    assert builder_cls.get_cudagraph_support(config, None) is AttentionCGSupport.ALWAYS
    assert builder_cls._use_flattened_decode(config) is flattened
    assert builder_cls._supports_varlen_decode(config) is varlen


@pytest.mark.parametrize(
    ("capability", "has_deep_gemm", "builder_cls"),
    [
        ((8, 9), False, DeepseekV32IndexerMetadataBuilder),
        ((8, 9), False, DeepseekV4IndexerMetadataBuilder),
        ((12, 0), True, DeepseekV4IndexerMetadataBuilder),
        ((8, 9), False, DeepseekV41IndexerMetadataBuilder),
        ((12, 0), True, DeepseekV41IndexerMetadataBuilder),
    ],
)
def test_nonadaptive_k1_metadata_paths_remain_uniform_batch(
    monkeypatch,
    capability,
    has_deep_gemm,
    builder_cls,
) -> None:
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=has_deep_gemm)
    config = _builder_config()

    assert (
        builder_cls.get_cudagraph_support(config, None)
        is AttentionCGSupport.UNIFORM_BATCH
    )
    assert not builder_cls._use_flattened_decode(config)
    assert not builder_cls._supports_varlen_decode(config)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_v4_adaptive_full_preserves_sm90_sm100_split(monkeypatch, capability) -> None:
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=True)
    config = _builder_config(
        adaptive=True,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL,
    )

    assert DeepseekV4IndexerMetadataBuilder._use_flattened_decode(
        config
    ) is DeepseekV32IndexerMetadataBuilder._use_flattened_decode(config)
    assert DeepseekV4IndexerMetadataBuilder._supports_varlen_decode(
        config
    ) is DeepseekV32IndexerMetadataBuilder._supports_varlen_decode(config)
    assert not DeepseekV4IndexerMetadataBuilder._supports_adaptive_full_mixed_decode()


@pytest.mark.parametrize(
    ("capability", "has_deep_gemm"),
    [((8, 9), False), ((12, 0), True)],
)
def test_v4_adaptive_builds_flattened_device_metadata(
    monkeypatch, capability, has_deep_gemm
) -> None:
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=has_deep_gemm)
    monkeypatch.setattr(indexer_backend, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(indexer_backend, "dsa_indexer_uses_fp4", lambda _: False)
    monkeypatch.setattr(
        indexer_backend,
        "get_compressed_slot_mapping",
        lambda num_tokens, *_args, out, **_kwargs: out[:num_tokens],
    )
    monkeypatch.setattr(
        indexer_backend,
        "_uses_deep_gemm_scheduler_metadata",
        lambda: has_deep_gemm,
    )
    calls = []

    def fake_metadata(seq_lens, block_size, num_sms, *, indices):
        calls.append((seq_lens.clone(), block_size, num_sms, indices.clone()))
        return torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    monkeypatch.setattr(indexer_backend, "get_paged_mqa_logits_metadata", fake_metadata)
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=4,
        alignment=512,
        model_version="deepseek_v4",
    )
    builder = DeepseekV4IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["model.layers.20.attn.indexer.k_cache"],
        vllm_config=_builder_config(
            adaptive=True,
            num_speculative_tokens=3,
            cudagraph_mode=CUDAGraphMode.FULL,
            max_num_batched_tokens=16,
        ),
        device=torch.device("cpu"),
        block_table_width=2,
    )
    capture_common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        seq_lens=torch.tensor([40, 44, 52, 80], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=9,
        max_query_len=1,
        max_seq_len=80,
        block_table_tensor=torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32
        ),
        slot_mapping=torch.zeros(9, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([40, 44, 52, 80], dtype=torch.int32),
        is_prefilling=torch.zeros(4, dtype=torch.bool),
    )
    capture_metadata = builder.build_for_cudagraph_capture(capture_common)
    assert capture_metadata.decode is not None
    capture_seq_lens_ptr = capture_metadata.decode.seq_lens.data_ptr()
    assert capture_metadata.decode.seq_lens.squeeze(-1).tolist() == [
        10,
        11,
        13,
        20,
        0,
        0,
        0,
        0,
        0,
    ]
    if has_deep_gemm:
        assert capture_metadata.decode.indices is not None
        assert capture_metadata.decode.indices.tolist() == list(range(9))
    else:
        assert capture_metadata.decode.indices is None
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 4, 4, 9], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4, 4, 9], dtype=torch.int32),
        seq_lens=torch.tensor([40, 44, 0, 80], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=9,
        max_query_len=5,
        max_seq_len=80,
        block_table_tensor=torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int32
        ),
        slot_mapping=torch.zeros(9, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([40, 44, 0, 80], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False, False, True]),
    )

    metadata = builder.build(0, common)

    assert metadata.decode is not None
    assert metadata.decode.seq_lens.data_ptr() == capture_seq_lens_ptr
    assert (metadata.num_decodes, metadata.num_prefills) == (4, 0)
    if has_deep_gemm:
        assert metadata.decode.indices is not None
        assert metadata.decode.indices.tolist() == [0, 0, 0, 1, 3, 3, 3, 3, 3]
    else:
        assert metadata.decode.indices is None
    assert metadata.decode.seq_lens.squeeze(-1).tolist() == [
        9,
        9,
        10,
        11,
        19,
        19,
        19,
        19,
        20,
    ]
    assert metadata.decode.block_table.tolist() == [
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 4],
        [7, 8],
        [7, 8],
        [7, 8],
        [7, 8],
        [7, 8],
    ]
    assert metadata.decode.per_req_decode_lens is not None
    assert metadata.decode.per_req_decode_lens.tolist() == [3, 1, 0, 5]
    if has_deep_gemm:
        assert len(calls) == 2
        seq_lens, block_size, num_sms, indices = calls[1]
        torch.testing.assert_close(seq_lens, metadata.decode.seq_lens)
        assert block_size == 64
        assert num_sms == 4
        assert indices.tolist() == [0, 0, 0, 1, 3, 3, 3, 3, 3]
    else:
        assert not calls


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
@pytest.mark.parametrize("prefill_lens", [[5], [6], [20], [5, 5]])
def test_v4_adaptive_prefill_outside_capture_uses_bounded_chunks(
    monkeypatch, capability, prefill_lens
) -> None:
    """Only graph-sized batches may bypass the prefill logits budget."""
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=capability == (12, 0))
    monkeypatch.setattr(indexer_backend, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(indexer_backend, "dsa_indexer_uses_fp4", lambda _: False)
    monkeypatch.setattr(
        indexer_backend,
        "get_compressed_slot_mapping",
        lambda num_tokens, *_args, out, **_kwargs: out[:num_tokens],
    )
    monkeypatch.setattr(
        indexer_backend, "_uses_deep_gemm_scheduler_metadata", lambda: False
    )
    monkeypatch.setattr(indexer_backend.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    chunks: list[slice] = []

    def record_chunk(*args, query_slice, **kwargs):
        chunks.append(query_slice)
        return None

    monkeypatch.setattr(indexer_backend, "build_prefill_chunk_metadata", record_chunk)
    config = _builder_config(
        max_model_len=1024 * 1024,
        adaptive=True,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL,
        max_num_batched_tokens=32,
    )
    config.compilation_config.max_cudagraph_capture_size = 9
    builder = DeepseekV4IndexerMetadataBuilder(
        kv_cache_spec=MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.uint8,
            tokens_per_state=4,
            alignment=512,
            model_version="deepseek_v4",
        ),
        layer_names=["model.layers.20.attn.indexer.k_cache"],
        vllm_config=config,
        device=torch.device("cpu"),
        block_table_width=1,
    )
    device_lens = [3, 1, *prefill_lens]
    cpu_lens = [2, 2, *prefill_lens]
    num_tokens = sum(device_lens)
    num_reqs = len(device_lens)
    seq_lens = torch.tensor(
        [40, 44, *([config.model_config.max_model_len] * len(prefill_lens))],
        dtype=torch.int32,
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, *device_lens], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        ),
        query_start_loc_cpu=torch.tensor([0, *cpu_lens], dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        ),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(cpu_lens),
        max_seq_len=config.model_config.max_model_len,
        block_table_tensor=torch.ones((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        is_prefilling=torch.tensor([False, False, *([True] * len(prefill_lens))]),
    )

    metadata = builder.build(0, common)

    assert builder.decode_threshold == 32
    assert builder.adaptive_full_mixed_decode
    assert metadata.decode is not None
    assert metadata.decode.seq_lens[:4].flatten().tolist() == [9, 9, 10, 11]
    if num_tokens <= 9:
        assert (metadata.num_decodes, metadata.num_decode_tokens) == (
            num_reqs,
            num_tokens,
        )
        assert metadata.prefill is None
        assert not chunks
    else:
        assert (metadata.num_decodes, metadata.num_decode_tokens) == (2, 4)
        assert (metadata.num_prefills, metadata.num_prefill_tokens) == (
            len(prefill_lens),
            sum(prefill_lens),
        )
        assert metadata.prefill is not None
        assert len(chunks) == sum(prefill_lens)
        assert all(chunk.stop - chunk.start == 1 for chunk in chunks)


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL],
)
def test_v41_adaptive_flattens_device_lens_with_padding_and_prefill(
    monkeypatch, capability, cudagraph_mode
) -> None:
    _set_cuda_capability(monkeypatch, capability, has_deep_gemm=capability == (12, 0))
    monkeypatch.setattr(indexer_backend, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(indexer_backend, "dsv41_indexer_uses_fp4", lambda _: True)
    monkeypatch.setattr(
        indexer_backend,
        "get_compressed_slot_mapping",
        lambda num_tokens, *_args, out, **_kwargs: out[:num_tokens],
    )
    monkeypatch.setattr(
        indexer_backend,
        "get_paged_mqa_logits_metadata",
        lambda *_args, **_kwargs: pytest.fail("V4.1 requested DeepGEMM metadata"),
    )
    prefill_calls = []

    def fake_prefill(*args, **kwargs):
        prefill_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(indexer_backend, "build_prefill_chunk_metadata", fake_prefill)
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
        tokens_per_state=1,
        state_content_bytes=68,
        alignment=128,
        model_version="deepseek_v4_1",
    )
    builder = DeepseekV41IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["model.layers.20.attn.indexer.k_cache"],
        vllm_config=_builder_config(
            adaptive=True,
            num_speculative_tokens=3,
            cudagraph_mode=cudagraph_mode,
            max_num_batched_tokens=16,
        ),
        device=torch.device("cpu"),
        block_table_width=1,
    )
    common = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 3, 4, 4, 9], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4, 4, 9], dtype=torch.int32),
        seq_lens=torch.tensor([13, 21, 0, 50], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=9,
        max_query_len=5,
        max_seq_len=50,
        block_table_tensor=torch.tensor([[1], [2], [0], [3]], dtype=torch.int32),
        slot_mapping=torch.zeros(9, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor([16, 24, 0, 50], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False, False, True]),
    )

    metadata = builder.build(0, common)

    assert metadata.decode is not None
    assert metadata.decode.indices is None
    assert metadata.decode.per_req_decode_lens is not None
    if cudagraph_mode == CUDAGraphMode.FULL:
        assert (metadata.num_decodes, metadata.num_decode_tokens) == (4, 9)
        assert (metadata.num_prefills, metadata.num_prefill_tokens) == (0, 0)
        assert metadata.decode.seq_lens.squeeze(-1).tolist() == [
            11,
            12,
            13,
            21,
            46,
            47,
            48,
            49,
            50,
        ]
        assert metadata.decode.block_table.tolist() == [
            [1],
            [1],
            [1],
            [2],
            [3],
            [3],
            [3],
            [3],
            [3],
        ]
        assert metadata.decode.decode_lens.tolist() == [1] * 9
        assert metadata.decode.per_req_decode_lens.tolist() == [3, 1, 0, 5]
        assert metadata.decode.max_indexer_kv_len == 50
        assert not prefill_calls
    else:
        assert (metadata.num_decodes, metadata.num_decode_tokens) == (3, 4)
        assert (metadata.num_prefills, metadata.num_prefill_tokens) == (1, 5)
        assert metadata.decode.seq_lens.squeeze(-1).tolist() == [11, 12, 13, 21]
        assert metadata.decode.block_table.tolist() == [[1], [1], [1], [2]]
        assert metadata.decode.decode_lens.tolist() == [1, 1, 1, 1]
        assert metadata.decode.per_req_decode_lens.tolist() == [3, 1, 0]
        assert metadata.decode.max_indexer_kv_len == 24
        assert len(prefill_calls) == 1
        args, _ = prefill_calls[0]
        assert args[:2] == (3, 4)


def test_v41_builder_skips_deep_gemm_scheduler_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        indexer_backend, "_uses_deep_gemm_scheduler_metadata", lambda: True
    )
    monkeypatch.setattr(
        indexer_backend,
        "get_paged_mqa_logits_metadata",
        lambda *_args, **_kwargs: pytest.fail("V4.1 requested DeepGEMM metadata"),
    )

    builder, metadata = _build_single_decode(
        DeepseekV41IndexerMetadataBuilder, monkeypatch
    )

    assert metadata.decode is not None
    assert (
        metadata.decode.schedule_metadata.data_ptr()
        == builder.scheduler_metadata_buffer.data_ptr()
    )


@pytest.mark.parametrize(
    ("compress_ratio", "expected_active", "expected_capture"),
    [(1, 257, 1024), (2, 128, 512)],
)
def test_v41_builder_uses_active_width_but_capture_uses_configured_max(
    monkeypatch,
    compress_ratio: int,
    expected_active: int,
    expected_capture: int,
) -> None:
    _, active = _build_single_decode(
        DeepseekV41IndexerMetadataBuilder,
        monkeypatch,
        compress_ratio=compress_ratio,
        seq_len=257,
        max_model_len=1024,
    )
    _, capture = _build_single_decode(
        DeepseekV41IndexerMetadataBuilder,
        monkeypatch,
        compress_ratio=compress_ratio,
        seq_len=257,
        max_model_len=1024,
        for_cudagraph_capture=True,
    )

    assert active.decode is not None
    assert capture.decode is not None
    assert active.decode.max_indexer_kv_len == expected_active
    assert capture.decode.max_indexer_kv_len == expected_capture


@pytest.mark.parametrize(("compress_ratio", "expected"), [(1, 257), (2, 128)])
def test_v41_active_width_ignores_prefill_and_graph_padding_rows(
    compress_ratio: int, expected: int
) -> None:
    seq_lens_cpu_upper_bound = torch.tensor([130, 257, 0, 0, 8192], dtype=torch.int32)

    actual = _dsv41_decode_max_indexer_kv_len(
        seq_lens_cpu_upper_bound,
        num_decodes=4,
        compress_ratio=compress_ratio,
    )

    assert actual == expected


def test_legacy_builder_keeps_deep_gemm_scheduler_metadata(monkeypatch) -> None:
    calls = []
    expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    monkeypatch.setattr(
        indexer_backend, "_uses_deep_gemm_scheduler_metadata", lambda: True
    )

    def fake_metadata(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(indexer_backend, "get_paged_mqa_logits_metadata", fake_metadata)

    _, metadata = _build_single_decode(DeepseekV32IndexerMetadataBuilder, monkeypatch)

    assert len(calls) == 1
    assert metadata.decode is not None
    assert metadata.decode.max_indexer_kv_len is None
    torch.testing.assert_close(metadata.decode.schedule_metadata, expected)


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
def test_v41_fp4_contract_accepts_sm89_and_sm120(monkeypatch, capability) -> None:
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability",
        lambda target: capability == target,
    )
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda family: capability[0] == family // 10,
    )

    assert dsv41_indexer_uses_fp4(_config())


def test_v41_fp4_contract_rejects_precision_override(monkeypatch) -> None:
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer.current_platform, "is_device_capability", lambda _: True
    )

    with pytest.raises(ValueError, match="requires indexer_kv_dtype='mxfp4'"):
        dsv41_indexer_uses_fp4(_config("fp8"))


@pytest.mark.parametrize(
    ("seq_lens", "expected"),
    [
        ([10, 20], [10, 10, 10, 20, 20, 20]),
        ([10, 11, 12, 20, 21, 22], [10, 11, 12, 20, 21, 22]),
        ([[10, 11, 12], [20, 21, 22]], [10, 11, 12, 20, 21, 22]),
    ],
)
def test_v41_decode_seq_lens_expand_per_request_or_query(seq_lens, expected) -> None:
    actual = _dsv41_expand_decode_seq_lens(torch.tensor(seq_lens), 2, 3)
    assert actual.tolist() == expected


def test_v41_short_context_chunk_is_capped_by_block_score_scratch() -> None:
    max_logits_elements = 512 * 1024 * 1024 // 4
    assert (
        _dsv41_rows_per_chunk(1, max_logits_elements, needs_block_scores=True) == 8192
    )
    assert (
        _dsv41_rows_per_chunk(1, max_logits_elements, needs_block_scores=False) == 8192
    )
    assert (
        _dsv41_rows_per_chunk(2048 * 8, max_logits_elements, needs_block_scores=False)
        == 8192
    )


def test_v41_workspace_specs_cover_runtime_scratch(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    specs = _dsv41_workspace_specs(1024 * 1024)
    max_logits_elements = 512 * 1024 * 1024 // 4
    assert specs == (
        ((max_logits_elements,), torch.float32),
        ((max_logits_elements // 8,), torch.float32),
        ((max_logits_elements // (2048 * 8),), torch.int32),
        ((1024 * 1024,), torch.uint8),
    )


def test_v41_workspace_views_are_distinct_and_stable(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    specs = _dsv41_workspace_specs(64 * 1024)
    manager = WorkspaceManager(torch.device("cpu"))

    first = manager.get_simultaneous(*specs)
    byte_ranges = sorted(
        (tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in first
    )
    assert all(
        left_end <= right_start
        for (_, left_end), (right_start, _) in zip(byte_ranges, byte_ranges[1:])
    )

    manager.lock()
    second = manager.get_simultaneous(*specs)
    assert [tensor.data_ptr() for tensor in second] == [
        tensor.data_ptr() for tensor in first
    ]


def test_v41_candidate_source_requires_exact_output_shape() -> None:
    args = (
        None,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        512,
        1024,
    )
    with pytest.raises(ValueError, match="requires a candidate output buffer"):
        sparse_indexer._dsv41_native_indexer(*args, None, 8, True)
    with pytest.raises(ValueError, match="2048 blocks per row"):
        sparse_indexer._dsv41_native_indexer(
            *args, torch.empty(1, 2047, dtype=torch.int32), 8, False
        )


def test_v41_profile_reserves_runtime_workspace_without_legacy_dummy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
    expected = _dsv41_workspace_specs(1024 * 1024)
    calls = []

    class _Workspace:
        def get_simultaneous(self, *specs):
            calls.append(specs)
            return []

    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.float16
    )
    monkeypatch.setattr(
        sparse_indexer, "current_workspace_manager", lambda: _Workspace()
    )
    hidden = torch.zeros(1, 1)
    q_values = torch.zeros(1, 32, 64, dtype=torch.uint8)
    q_scales = torch.zeros(1, 32, 4, dtype=torch.uint8)
    weights = torch.zeros(1, 32)
    topk = torch.empty(1, 512, dtype=torch.int32)
    monkeypatch.setattr(
        sparse_indexer.torch,
        "empty",
        lambda *args, **kwargs: pytest.fail("legacy dummy allocation reached"),
    )

    result = sparse_indexer.sparse_attn_indexer(
        hidden,
        "model.layers.0.self_attn.indexer.k_cache",
        torch.zeros(1, 64, 68, dtype=torch.uint8),
        q_values,
        q_scales,
        None,
        weights,
        128,
        "ue8m0",
        512,
        128,
        1024 * 1024,
        1024 * 1024,
        topk,
        True,
        False,
        "",
        use_fp4_cache=True,
        use_v41_native=True,
    )

    assert result is topk
    assert calls == [expected]


@pytest.mark.parametrize("batch_size", [16, 32])
@pytest.mark.parametrize("next_n", [5, 7, 8])
@pytest.mark.parametrize("max_model_len", [128 * 1024, 1024 * 1024])
def test_v41_decode_chunks_preserve_whole_query_groups(
    batch_size: int, next_n: int, max_model_len: int
) -> None:
    max_logits_elements = 512 * 1024 * 1024 // 4
    rows = _dsv41_rows_per_chunk(
        max_model_len,
        max_logits_elements,
        needs_block_scores=True,
        group_size=next_n,
    )
    assert rows >= next_n
    assert rows % next_n == 0
    assert rows * max_model_len <= max_logits_elements
    assert rows <= max_logits_elements // (2048 * 8)
    score_width = max(2048, (max_model_len + 7) // 8)
    assert rows * score_width <= max_logits_elements // 8
    requests_per_chunk = rows // next_n
    chunk_rows = [
        (min(start + requests_per_chunk, batch_size) - start) * next_n
        for start in range(0, batch_size, requests_per_chunk)
    ]
    assert sum(chunk_rows) == batch_size * next_n
    assert all(0 < size <= rows and size % next_n == 0 for size in chunk_rows)
