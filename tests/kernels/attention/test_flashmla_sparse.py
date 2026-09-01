# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch


def test_deepseek_v4_c128a_adaptive_width_has_capture_stable_stride():
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    device = torch.device("cuda")
    capacity_width = 512
    global_decode_buffer = torch.empty(
        (2, capacity_width), dtype=torch.int32, device=device
    )
    prefill_buffer = torch.empty_like(global_decode_buffer)
    kwargs = dict(
        positions=torch.tensor([255, 511, 383, 639], device=device),
        compress_ratio=128,
        num_decode_tokens=2,
        token_to_req_indices=torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32, device=device
        ),
        block_table=torch.tensor([[3], [5]], dtype=torch.int32, device=device),
        block_size=capacity_width,
        slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
        global_decode_buffer=global_decode_buffer,
        decode_lens_buffer=torch.empty(2, dtype=torch.int32, device=device),
        prefill_buffer=prefill_buffer,
    )
    captured_decode, _, captured_prefill = build_c128a_topk_metadata(
        max_compressed_tokens=256,
        **kwargs,
    )
    assert captured_decode.shape == captured_prefill.shape == (2, 256)
    assert captured_decode.stride(0) == captured_prefill.stride(0) == capacity_width

    captured_rows = torch.empty((4, 4), dtype=torch.int32, device=device)
    captured_rows[:2].copy_(captured_decode[:, :4])
    captured_rows[2:].copy_(captured_prefill[:, :4])
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_rows[:2].copy_(captured_decode[:, :4])
        captured_rows[2:].copy_(captured_prefill[:, :4])

    global_decode_buffer.fill_(-99)
    prefill_buffer.fill_(-99)
    build_c128a_topk_metadata(
        max_compressed_tokens=128,
        **kwargs,
    )
    graph.replay()

    assert captured_rows.cpu().tolist() == [
        [1536, 1537, -1, -1],
        [2560, 2561, 2562, 2563],
        [0, 1, 2, -1],
        [0, 1, 2, 3],
    ]
    assert torch.all(global_decode_buffer[:, 128:] == -99)
    assert torch.all(prefill_buffer[:, 128:] == -99)


def test_dsv4_vision_warmup_covers_fixed_width_and_mm_prefix_variants():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        CombineTopkSwaIndicesKernel,
    )

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        model_config=SimpleNamespace(
            is_mm_prefix_lm=True,
            hf_config=SimpleNamespace(
                sliding_window=128,
                vision_max_n_token=384,
            ),
        ),
    )

    keys = CombineTopkSwaIndicesKernel().get_warmup_keys(config)

    assert {key.SWA_INDEX_WIDTH for key in keys} == {512}
    assert {key.HAS_MM_PREFIX for key in keys} == {False, True}


def test_sparse_flashmla_metadata_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 128
    num_heads_k = 1
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    topk = 128

    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )
    assert isinstance(tile_md, fm.FlashMLASchedMeta)
    assert tile_md.tile_scheduler_metadata is None
    assert tile_md.num_splits is None
    assert num_splits is None


def test_sparse_flashmla_decode_smoke():
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    batch_size = 1
    seqlen_q = 1
    num_heads_q = 64
    head_dim_k = 576
    head_dim_v = 512
    num_heads_k = 1
    page_block_size = 64
    bytes_per_token = 656
    topk = 128

    # Metadata
    q_seq_per_hk = seqlen_q * num_heads_q // num_heads_k
    # q_heads_per_hk = num_heads_q // num_heads_k
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tile_md, num_splits = fm.get_mla_metadata(
        cache_seqlens,
        q_seq_per_hk,
        num_heads_k,
        num_heads_q=num_heads_q,
        topk=topk,
        is_fp8_kvcache=True,
    )

    # Inputs
    q = torch.zeros(
        (batch_size, seqlen_q, num_heads_q, head_dim_k),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = torch.zeros(
        (1, page_block_size, num_heads_k, bytes_per_token),
        dtype=torch.uint8,
        device=device,
    )
    indices = torch.zeros(
        (batch_size, seqlen_q, topk), dtype=torch.int32, device=device
    )

    block_table = torch.zeros((batch_size, 128), dtype=torch.int32, device=device)
    out, lse = fm.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_md,
        num_splits,
        indices=indices,
        is_fp8_kvcache=True,
    )
    assert out.shape[0] == batch_size
    assert out.shape[-1] == head_dim_v
    assert lse.shape[0] == batch_size


@pytest.mark.parametrize("h_q", [64, 128])
def test_sparse_flashmla_prefill_smoke(h_q: int):
    import vllm.v1.attention.ops.flashmla as fm

    ok, reason = fm.is_flashmla_sparse_supported()
    if not ok:
        pytest.skip(reason)

    device = torch.device("cuda")
    torch.manual_seed(0)
    s_q = 1
    s_kv = 8
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128
    q = torch.randn((s_q, h_q, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=torch.bfloat16, device=device)
    indices = torch.randint(s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device)
    reference_indices = indices.clone()
    reference_indices[..., 1:] = -1
    kwargs = {"topk_length": torch.ones(1, dtype=torch.int32, device=device)}
    reference = fm.flash_mla_sparse_fwd(q, kv, reference_indices, 1.0, d_v, **kwargs)
    actual = fm.flash_mla_sparse_fwd(q, kv, indices, 1.0, d_v, **kwargs)

    for actual_tensor, reference_tensor in zip(actual, reference):
        torch.testing.assert_close(actual_tensor, reference_tensor, rtol=0, atol=0)
    assert actual[0].shape == (s_q, h_q, d_v)


def test_deepseek_v4_prefill_chunk_planning_expands_for_short_sequences():
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    metadata = DeepseekSparseSWAMetadata(
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        block_size=64,
        num_prefills=5,
        prefill_seq_lens_cpu=torch.tensor([80, 96, 112, 128, 144], dtype=torch.int32),
        prefill_query_lens_cpu=torch.tensor([4, 4, 4, 4, 4], dtype=torch.int32),
        prefill_window_size=64,
        prefill_max_model_len=1024,
        prefill_max_num_batched_tokens=128,
    )

    chunk_plan = metadata.get_prefill_chunk_plan(compress_ratio=4, prefill_chunk_size=4)

    # the adaptive plan keeps all 5 in one chunk
    assert chunk_plan == [(0, 5, 36, 103)]


def test_flashinfer_sparse_indices_cache(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    builder_calls = 0

    def fake_build(*args, **kwargs):
        nonlocal builder_calls
        builder_calls += 1
        return (
            torch.tensor([[builder_calls]], dtype=torch.int32),
            torch.tensor([builder_calls], dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    def make_attn(compress_ratio: int, topk_width: int):
        attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
        attn.compress_ratio = compress_ratio
        attn.window_size = 4
        attn.topk_indices_buffer = torch.tensor(
            [[0, 1], [2, 3], [4, 5]], dtype=torch.int32
        )[:, :topk_width]
        return attn

    def make_swa_metadata():
        return DeepseekSparseSWAMetadata(
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            block_size=64,
            seq_lens=torch.tensor([8, 10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
            token_to_req_indices=torch.tensor([0, 1, 1], dtype=torch.int32),
            decode_swa_indices=torch.tensor([[5, 6, -1, -1]], dtype=torch.int32),
            decode_swa_lens=torch.tensor([2], dtype=torch.int32),
            decode_swa_width=4,
            is_valid_token=torch.tensor([True], dtype=torch.bool),
            num_decodes=1,
            num_prefills=1,
            num_decode_tokens=1,
            num_prefill_tokens=2,
        )

    def make_flashmla_metadata():
        return DeepseekV4FlashMLAMetadata(
            num_reqs=2,
            max_query_len=2,
            max_seq_len=10,
            num_actual_tokens=3,
            query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
            block_size=256,
            topk_tokens=2,
            c128a_global_decode_topk_indices=torch.tensor(
                [[[9, 10]]], dtype=torch.int32
            ),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            c128a_prefill_topk_indices=torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.int32
            ),
        )

    swa_attn = make_attn(1, 0)
    swa_metadata = make_swa_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        swa_attn._build_sparse_index_metadata(
            kv_cache=None,
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
        )
    )
    assert builder_calls == 1
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c128a_attn = make_attn(128, 2)
    c128a_metadata = make_swa_metadata()
    c128a_flashmla_md = make_flashmla_metadata()
    _, _, sparse_indices_first, sparse_lens_first = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_second, sparse_lens_second = (
        c128a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c128a_metadata,
            attn_metadata=c128a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 2
    assert sparse_indices_first is sparse_indices_second
    assert sparse_lens_first is sparse_lens_second

    c4a_attn = make_attn(4, 2)
    c4a_metadata = make_swa_metadata()
    c4a_flashmla_md = make_flashmla_metadata()
    c4a_flashmla_md.c128a_global_decode_topk_indices = None
    c4a_flashmla_md.c128a_decode_topk_lens = None
    c4a_flashmla_md.c128a_prefill_topk_indices = None
    _, _, sparse_indices_third, sparse_lens_third = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )
    _, _, sparse_indices_fourth, sparse_lens_fourth = (
        c4a_attn._build_sparse_index_metadata(
            kv_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
            swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
            swa_metadata=c4a_metadata,
            attn_metadata=c4a_flashmla_md,
            swa_only=False,
        )
    )

    assert builder_calls == 4
    assert sparse_indices_third is not sparse_indices_fourth
    assert sparse_lens_third is not sparse_lens_fourth


def test_flashinfer_sparse_index_preserves_logical_window(monkeypatch):
    from vllm.models.deepseek_v4.nvidia import flashinfer_sparse as flashinfer_mod
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

    captured_shapes_and_windows: list[tuple[int, int]] = []

    def fake_build(*args, **kwargs):
        # window_size is the 12th positional arg of
        # build_flashinfer_mixed_sparse_indices.
        captured_shapes_and_windows.append((args[0].shape[-1], args[11]))
        num_tokens = args[0].shape[0] + args[3].shape[0]
        return (
            torch.zeros((num_tokens, 1), dtype=torch.int32),
            torch.zeros((num_tokens,), dtype=torch.int32),
        )

    monkeypatch.setattr(
        flashinfer_mod, "build_flashinfer_mixed_sparse_indices", fake_build
    )

    attn = object.__new__(flashinfer_mod.DeepseekV4FlashInferMLAAttention)
    attn.compress_ratio = 1
    attn.window_size = 4
    attn.topk_indices_buffer = torch.zeros((4, 0), dtype=torch.int32)

    wide_width = 8
    wide_indices = torch.full((1, wide_width), -1, dtype=torch.int32)
    wide_indices[0, :2] = torch.tensor([5, 6], dtype=torch.int32)
    wide_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_size=64,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        decode_swa_indices=wide_indices,
        decode_swa_lens=torch.tensor([2], dtype=torch.int32),
        decode_swa_width=wide_width,
        is_valid_token=torch.tensor([True], dtype=torch.bool),
        num_decodes=1,
        num_prefills=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
    )
    attn._build_sparse_index_metadata(
        kv_cache=None,
        swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
        swa_metadata=wide_metadata,
        attn_metadata=None,
        swa_only=True,
    )
    assert captured_shapes_and_windows == [(wide_width, attn.window_size)]

    empty_width = 8
    empty_metadata = DeepseekSparseSWAMetadata(
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_size=64,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32),
        decode_swa_indices=torch.empty((0, 1, empty_width), dtype=torch.int32),
        decode_swa_lens=torch.empty((0,), dtype=torch.int32),
        decode_swa_width=empty_width,
        is_valid_token=torch.tensor([True, True], dtype=torch.bool),
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
        num_prefill_tokens=2,
    )
    attn._build_sparse_index_metadata(
        kv_cache=None,
        swa_k_cache=torch.empty((1, 64, 512), dtype=torch.bfloat16),
        swa_metadata=empty_metadata,
        attn_metadata=None,
        swa_only=True,
    )
    assert captured_shapes_and_windows == [
        (wide_width, attn.window_size),
        (empty_width, attn.window_size),
    ]


def test_flashinfer_mixed_sparse_indices_separates_window_and_padded_width():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
    )

    device = torch.device("cuda")
    padded_width = 8
    logical_window = 4
    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=torch.empty(
            (0, padded_width), dtype=torch.int32, device=device
        ),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((1, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([logical_window], dtype=torch.int32, device=device),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32, device=device),
        swa_block_table=torch.tensor([[0]], dtype=torch.int32, device=device),
        swa_block_size=64,
        compressed_block_table=None,
        compressed_block_size=64,
        window_size=logical_window,
        compress_ratio=1,
        topk=0,
    )

    assert sparse_indices.shape == (1, padded_width)
    assert sparse_indices[0].cpu().tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert sparse_lens.cpu().tolist() == [padded_width]


def _golden_swa_rows(
    *,
    seq_lens: list[int],
    query_start_loc: list[int],
    block_table: list[list[int]],
    token_to_req: list[int],
    block_size: int,
    window_size: int,
    index_width: int | None = None,
    mm_query_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[list[int]], list[int]]:
    if index_width is None:
        index_width = window_size
    rows: list[list[int]] = []
    lens: list[int] = []
    for token_idx, req_idx in enumerate(token_to_req):
        query_start = query_start_loc[req_idx]
        query_end = query_start_loc[req_idx + 1]
        query_len = query_end - query_start
        prefix_len = seq_lens[req_idx] - query_len
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - window_size + 1, 0)
        end_pos = pos + 1
        if mm_query_ranges is not None:
            mm_start, mm_end = mm_query_ranges[token_idx]
            if mm_start >= 0 and mm_start <= mm_end:
                start_pos = min(start_pos, mm_start)
                end_pos = mm_end + 1
        row = []
        for pos_offset in range(start_pos, end_pos):
            block_number = block_table[req_idx][pos_offset // block_size]
            row.append(block_number * block_size + pos_offset % block_size)
        lens.append(len(row))
        rows.append(row + [-1] * (index_width - len(row)))
    return rows, lens


def test_dsv4_swa_indices_use_bidirectional_image_span_golden():
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _compute_swa_indices_and_lens_kernel,
    )

    device = torch.device("cuda")
    window_size = 4
    index_width = 8
    block_size = 4
    seq_lens = [5, 3]
    query_start_loc = [0, 5, 8]
    block_table = [[10, 11, 12], [20, 21, 22]]
    token_to_req = [0, 0, 0, 0, 0, 1, 1, 1]
    mm_ranges = [
        (-1, -1),
        (1, 3),
        (1, 3),
        (1, 3),
        (-1, -1),
        (0, 1),
        (0, 1),
        (-1, -1),
    ]

    swa_indices = torch.empty((8, index_width), dtype=torch.int32, device=device)
    swa_lens = torch.empty((8,), dtype=torch.int32, device=device)
    _compute_swa_indices_and_lens_kernel[(8,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        window_size,
        index_width,
        torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        torch.tensor(token_to_req, dtype=torch.int32, device=device),
        torch.ones((8,), dtype=torch.bool, device=device),
        torch.tensor(block_table, dtype=torch.int32, device=device),
        len(block_table[0]),
        block_size,
        torch.tensor(mm_ranges, dtype=torch.int32, device=device),
        True,
        token_offset=0,
        TRITON_BLOCK_SIZE=8,
    )

    expected_rows, expected_lens = _golden_swa_rows(
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table=block_table,
        token_to_req=token_to_req,
        block_size=block_size,
        window_size=window_size,
        index_width=index_width,
        mm_query_ranges=mm_ranges,
    )
    assert swa_indices.cpu().tolist() == expected_rows
    assert swa_lens.cpu().tolist() == expected_lens
    assert expected_rows[1] == [40, 41, 42, 43, -1, -1, -1, -1]
    assert expected_rows[2] == expected_rows[3] == expected_rows[1]
    assert expected_rows[5] == [80, 81, -1, -1, -1, -1, -1, -1]
    assert expected_rows[6] == expected_rows[5]
    assert expected_rows[1] != expected_rows[5]


def test_flashinfer_mixed_sparse_indices_use_prefill_image_span():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
    )

    device = torch.device("cuda")
    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=torch.empty((0, 8), dtype=torch.int32, device=device),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((5, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([5], dtype=torch.int32, device=device),
        token_to_req_indices=torch.zeros((5,), dtype=torch.int32, device=device),
        swa_block_table=torch.tensor([[10, 11]], dtype=torch.int32, device=device),
        swa_block_size=4,
        compressed_block_table=None,
        compressed_block_size=4,
        window_size=4,
        compress_ratio=1,
        topk=0,
        mm_prefix_query_ranges=torch.tensor(
            [(-1, -1), (1, 3), (1, 3), (1, 3), (-1, -1)],
            dtype=torch.int32,
            device=device,
        ),
    )

    assert sparse_indices.cpu().tolist() == [
        [40, -1, -1, -1, -1, -1, -1, -1],
        [40, 41, 42, 43, -1, -1, -1, -1],
        [40, 41, 42, 43, -1, -1, -1, -1],
        [40, 41, 42, 43, -1, -1, -1, -1],
        [41, 42, 43, 44, -1, -1, -1, -1],
    ]
    assert sparse_lens.cpu().tolist() == [8, 8, 8, 8, 8]


def test_flashinfer_mixed_sparse_indices_keep_past_window_and_future_image():
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices,
    )

    device = torch.device("cuda")
    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=torch.empty((0, 8), dtype=torch.int32, device=device),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((8, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
        token_to_req_indices=torch.zeros((8,), dtype=torch.int32, device=device),
        swa_block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        swa_block_size=4,
        compressed_block_table=None,
        compressed_block_size=4,
        window_size=4,
        compress_ratio=1,
        topk=0,
        mm_prefix_query_ranges=torch.tensor(
            [(-1, -1), (-1, -1), (2, 6), (2, 6), (2, 6), (2, 6), (2, 6), (-1, -1)],
            dtype=torch.int32,
            device=device,
        ),
    )

    assert sparse_indices[2].cpu().tolist() == [0, 1, 2, 3, 4, 5, 6, -1]
    assert sparse_lens[2].item() == 8


def test_flashmla_combine_topk_swa_indices_use_prefill_image_span():
    from vllm.models.deepseek_v4.common.ops.cache_utils import combine_topk_swa_indices

    device = torch.device("cuda")
    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices=torch.empty((5, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([5], dtype=torch.int32, device=device),
        gather_lens=torch.tensor([5], dtype=torch.int32, device=device),
        window_size=4,
        compress_ratio=1,
        topk=0,
        M=5,
        N=0,
        mm_prefix_query_ranges=torch.tensor(
            [(-1, -1), (1, 3), (1, 3), (1, 3), (-1, -1)],
            dtype=torch.int32,
            device=device,
        ),
        swa_index_width=8,
    )

    assert combined_indices[:, :8].cpu().tolist() == [
        [0, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 2, 3, -1, -1, -1, -1],
        [0, 1, 2, 3, -1, -1, -1, -1],
        [0, 1, 2, 3, -1, -1, -1, -1],
        [1, 2, 3, 4, -1, -1, -1, -1],
    ]
    assert combined_lens.cpu().tolist() == [1, 4, 4, 4, 4]


def test_flashmla_combine_keeps_past_window_and_future_image():
    from vllm.models.deepseek_v4.common.ops.cache_utils import combine_topk_swa_indices

    device = torch.device("cuda")
    combined_indices, combined_lens = combine_topk_swa_indices(
        topk_indices=torch.empty((8, 0), dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
        gather_lens=torch.tensor([8], dtype=torch.int32, device=device),
        window_size=4,
        compress_ratio=1,
        topk=0,
        M=8,
        N=0,
        mm_prefix_query_ranges=torch.tensor(
            [(-1, -1), (-1, -1), (2, 6), (2, 6), (2, 6), (2, 6), (2, 6), (-1, -1)],
            dtype=torch.int32,
            device=device,
        ),
        swa_index_width=8,
    )

    assert combined_indices[2, :8].cpu().tolist() == [0, 1, 2, 3, 4, 5, 6, -1]
    assert combined_lens[2].item() == 7


def test_mm_prefix_fill_ranges_are_inclusive_at_end_boundary():
    from vllm.v1.attention.backends.utils import fill_mm_prefix_query_ranges

    out = torch.empty((5, 2), dtype=torch.int32).numpy()
    num_rows = fill_mm_prefix_query_ranges(
        out,
        {0: [(1, 3), (4, 4)]},
        torch.tensor([0, 5], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
    )

    assert num_rows == 5
    assert out.tolist() == [[-1, -1], [1, 3], [1, 3], [1, 3], [4, 4]]
