# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode, ParallelConfig
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekSparseSWAFlashInferMetadataBuilder,
    DeepseekV4FlashInferSM120Attention,
    DeepseekV4FlashInferSparseMLAMetadataBuilder,
)
from vllm.models.deepseek_v4_1.nvidia import flashinfer_sparse as v41_sparse
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV41NativeMetadataBuilder,
    DeepseekV41NativeMixedAttention,
    DeepseekV41NativeSWAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.workspace import WorkspaceManager

_CAPTURE_TOKENS = 12
_CAPTURE_REQUESTS = 6
_HEADS = 32
_GROUPS = 4


def _adaptive_full_config():
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=[_CAPTURE_TOKENS],
    )
    compilation_config.max_cudagraph_capture_size = _CAPTURE_TOKENS
    compilation_config.post_init_cudagraph_sizes()
    speculative_config = SimpleNamespace(
        enable_adaptive_verification=True,
        num_speculative_tokens=2,
        parallel_drafting=False,
        use_dspark=lambda: False,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    return SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=_CAPTURE_TOKENS,
            max_num_seqs=_CAPTURE_REQUESTS,
        ),
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
        num_speculative_tokens=2,
        model_config=SimpleNamespace(
            max_model_len=256,
            is_mm_prefix_lm=False,
            hf_config=SimpleNamespace(
                index_topk=128,
                sliding_window=128,
                compress_ratios=[1, 128],
            ),
        ),
    )


def _swa_spec(model_version: str) -> SlidingWindowMLASpec:
    is_v4 = model_version == "deepseek_v4"
    block_size = 64 if is_v4 else 32
    row_bytes = 584 if is_v4 else 528
    return SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=row_bytes,
        dtype=torch.uint8,
        tokens_per_state=1,
        state_content_bytes=row_bytes,
        sliding_window=128,
        model_version=model_version,
    )


def _fill_dsv4_cache(cache: torch.Tensor, values: torch.Tensor) -> None:
    pages, page_size, _ = cache.shape
    page_storage = cache.view(pages, -1)
    data = page_storage[:, : page_size * 576].view(pages, page_size, 576)
    scales = page_storage[:, page_size * 576 :].view(pages, page_size, 8)
    values = values.view(pages, page_size, 1)
    data[:, :, :448] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = values.expand(pages, page_size, 64).to(torch.bfloat16).contiguous()
    data[:, :, 448:] = rope.view(torch.uint8)
    scales.fill_(127)


class _AdaptiveFullTargetHarness:
    def __init__(self, config, device: torch.device):
        from flashinfer.mla.deepseek_v41 import (
            deepseek_v41_pack_global_cache,
            deepseek_v41_pack_swa_cache,
        )

        self.device = device
        self.query_start_loc = torch.zeros(
            _CAPTURE_REQUESTS + 1, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(_CAPTURE_REQUESTS, dtype=torch.int32, device=device)
        self.positions = torch.zeros(_CAPTURE_TOKENS, dtype=torch.int64, device=device)
        self.slot_mapping = torch.full(
            (_CAPTURE_TOKENS,), -1, dtype=torch.int64, device=device
        )
        self.swa_block_table = torch.arange(
            _CAPTURE_REQUESTS * 8, dtype=torch.int32, device=device
        ).view(_CAPTURE_REQUESTS, 8)
        self.v4_global_block_table = (
            torch.arange(_CAPTURE_REQUESTS, dtype=torch.int32, device=device) % 2
        )[:, None].repeat(1, 2)
        self.v41_global_block_table = torch.arange(
            _CAPTURE_REQUESTS, dtype=torch.int32, device=device
        )[:, None].repeat(1, 2)

        self.v4_swa_builder = DeepseekSparseSWAFlashInferMetadataBuilder(
            _swa_spec("deepseek_v4"), ["v4_swa"], config, device
        )
        v41_config = SimpleNamespace(**vars(config))
        v41_config.model_config = SimpleNamespace(**vars(config.model_config))
        v41_config.model_config.hf_config = SimpleNamespace(
            **vars(config.model_config.hf_config)
        )
        v41_config.model_config.hf_config.compress_ratios = [0, 1, 2]
        self.v41_swa_builder = DeepseekV41NativeSWAMetadataBuilder(
            _swa_spec("deepseek_v4_1"), ["v41_swa"], v41_config, device
        )
        for builder in (self.v4_swa_builder, self.v41_swa_builder):
            builder.build_tile_scheduler = lambda _: {
                "swaonly": None,
                "c4a": None,
                "c128a": None,
                "c1a": None,
                "c2a": None,
            }
        mla_spec = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=584,
            dtype=torch.uint8,
            tokens_per_state=128,
            state_content_bytes=584,
            model_version="deepseek_v4",
        )
        self.v4_mla_builder = DeepseekV4FlashInferSparseMLAMetadataBuilder(
            mla_spec, ["v4_mla"], config, device
        )
        self.v41_global_builders = {
            ratio: DeepseekV41NativeMetadataBuilder(
                MLAAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=512,
                    dtype=torch.uint8,
                    tokens_per_state=ratio,
                    state_content_bytes=288,
                    model_version="deepseek_v4_1",
                ),
                [f"v41_c{ratio}"],
                v41_config,
                device,
            )
            for ratio in (1, 2)
        }

        self.v4_q = torch.zeros(
            _CAPTURE_TOKENS, _HEADS, 512, dtype=torch.bfloat16, device=device
        )
        self.v4_output = torch.empty_like(self.v4_q)
        self.v4_swa_cache = torch.empty(
            _CAPTURE_REQUESTS * 8, 64, 584, dtype=torch.uint8, device=device
        )
        _fill_dsv4_cache(
            self.v4_swa_cache,
            torch.zeros(_CAPTURE_REQUESTS * 8, 64, device=device),
        )
        self.v4_global_cache = torch.empty(1, 2, 584, dtype=torch.uint8, device=device)
        _fill_dsv4_cache(
            self.v4_global_cache,
            torch.tensor([[-1.0, 1.0]], device=device),
        )
        self.v4_workspace = torch.empty(32 << 20, dtype=torch.uint8, device=device)
        self.v4_attention = SimpleNamespace(
            compress_ratio=128,
            kv_cache_torch_dtype=torch.uint8,
            swa_cache_layer=SimpleNamespace(kv_cache=self.v4_swa_cache),
            _as_sparse_cache=DeepseekV4FlashInferSM120Attention._as_sparse_cache,
            _get_workspace=lambda _: self.v4_workspace,
            scale=512**-0.5,
            attn_sink=None,
            topk_indices_buffer=None,
        )
        self.v4_attention._prepare_query = lambda query, output: (
            DeepseekV4FlashInferSM120Attention._prepare_query(
                self.v4_attention, query, output
            )
        )
        self.v4_attention._forward_decode = lambda **kwargs: (
            DeepseekV4FlashInferSM120Attention._forward_decode(
                self.v4_attention, **kwargs
            )
        )

        self.v41_q = torch.zeros_like(self.v4_q)
        self.v41_outputs = {
            ratio: torch.empty_like(self.v41_q) for ratio in self.v41_global_builders
        }
        self.v41_swa_cache = torch.empty(
            _CAPTURE_REQUESTS * 8, 32, 528, dtype=torch.uint8, device=device
        )
        slots = torch.arange(self.v41_swa_cache.numel() // 528, device=device)
        slot_values = ((slots % 7).float() - 3) / 4
        kv = slot_values[:, None].expand(-1, 512).to(torch.bfloat16).contiguous()
        deepseek_v41_pack_swa_cache(kv, slots, self.v41_swa_cache)
        self.v41_topk_indices = torch.full(
            (_CAPTURE_TOKENS, 128), -1, dtype=torch.int32, device=device
        )
        self.v41_topk_indices[:, :4] = torch.arange(4, dtype=torch.int32, device=device)
        self.v41_global_caches = {}
        self.v41_attentions = {}
        cos_sin_cache = torch.cat(
            (
                torch.ones(512, 32, device=device),
                torch.zeros(512, 32, device=device),
            ),
            dim=-1,
        )
        for ratio in self.v41_global_builders:
            page_size = 128 // ratio
            cache = torch.empty(
                _CAPTURE_REQUESTS,
                page_size,
                288,
                dtype=torch.uint8,
                device=device,
            )
            global_slots = torch.arange(cache.shape[0] * cache.shape[1], device=device)
            global_values = ((global_slots % 11).float() - 5) / 4
            global_kv = (
                global_values[:, None].expand(-1, 512).to(torch.bfloat16).contiguous()
            )
            deepseek_v41_pack_global_cache(global_kv, global_slots, cache)
            self.v41_global_caches[ratio] = cache

            attention = DeepseekV41NativeMixedAttention.__new__(
                DeepseekV41NativeMixedAttention
            )
            attention.n_local_groups = _GROUPS
            attention.compress_ratio = ratio
            attention.compressed_cache_prefix = f"v41_c{ratio}"
            attention.swa_cache_layer = SimpleNamespace(
                prefix="v41_swa", kv_cache=self.v41_swa_cache
            )
            attention.scale = 512**-0.5
            attention.attn_sink = None
            attention.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin_cache)
            attention.topk_indices_buffer = self.v41_topk_indices
            attention.is_kv_source = True
            attention.kv_cache = cache
            self.v41_attentions[ratio] = attention
        self.v41_context = SimpleNamespace(attn_metadata=None)

    def _common(
        self,
        block_table: torch.Tensor,
        cpu_starts: torch.Tensor,
        cpu_seq_lens: list[int],
        num_reqs: int,
    ) -> CommonAttentionMetadata:
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc,
            query_start_loc_cpu=cpu_starts,
            seq_lens=self.seq_lens,
            seq_lens_cpu_upper_bound=torch.tensor(cpu_seq_lens, dtype=torch.int32),
            num_reqs=_CAPTURE_REQUESTS,
            num_actual_tokens=_CAPTURE_TOKENS,
            max_query_len=int((cpu_starts[1:] - cpu_starts[:-1]).max()),
            max_seq_len=max(cpu_seq_lens),
            block_table_tensor=block_table,
            slot_mapping=self.slot_mapping,
            positions=self.positions,
            is_prefilling=torch.tensor(
                [False] * max(num_reqs - 1, 0)
                + ([True] if num_reqs else [])
                + [False] * (_CAPTURE_REQUESTS - num_reqs),
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def stage(
        self,
        device_query_lens: list[int],
        cpu_query_lens: list[int],
        context_lens: list[int],
    ):
        num_reqs = len(device_query_lens)
        assert len(cpu_query_lens) == len(context_lens) == num_reqs
        assert sum(device_query_lens) == sum(cpu_query_lens)
        live_tokens = sum(device_query_lens)
        padded_device_lens = device_query_lens + [0] * (_CAPTURE_REQUESTS - num_reqs)
        padded_cpu_lens = cpu_query_lens + [0] * (_CAPTURE_REQUESTS - num_reqs)
        device_starts = torch.tensor(
            [0, *padded_device_lens], dtype=torch.int32, device=self.device
        ).cumsum(0)
        cpu_starts = torch.tensor([0, *padded_cpu_lens], dtype=torch.int32).cumsum(0)
        self.query_start_loc.copy_(device_starts)

        padded_contexts = context_lens + [0] * (_CAPTURE_REQUESTS - num_reqs)
        seq_lens = [
            context + query
            for context, query in zip(padded_contexts, padded_device_lens)
        ]
        cpu_seq_lens = [
            context + max(device_query, cpu_query)
            for context, device_query, cpu_query in zip(
                padded_contexts, padded_device_lens, padded_cpu_lens
            )
        ]
        self.seq_lens.copy_(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        )
        positions = [
            context + offset
            for context, query_len in zip(context_lens, device_query_lens)
            for offset in range(query_len)
        ]
        self.positions.zero_()
        self.positions[:live_tokens].copy_(
            torch.tensor(positions, dtype=torch.int64, device=self.device)
        )
        self.slot_mapping.fill_(-1)
        self.slot_mapping[:live_tokens] = 0

        swa_common = self._common(
            self.swa_block_table, cpu_starts, cpu_seq_lens, num_reqs
        )
        v4_global_common = self._common(
            self.v4_global_block_table, cpu_starts, cpu_seq_lens, num_reqs
        )
        v41_global_common = self._common(
            self.v41_global_block_table, cpu_starts, cpu_seq_lens, num_reqs
        )
        return (
            self.v4_mla_builder.build(0, v4_global_common),
            self.v4_swa_builder.build(0, swa_common),
            self.v41_swa_builder.build(0, swa_common),
            {
                ratio: builder.build(0, v41_global_common)
                for ratio, builder in self.v41_global_builders.items()
            },
        )

    def run(self, v4_mla, v4_swa, v41_swa, v41_globals) -> None:
        DeepseekV4FlashInferSM120Attention._forward_sparse_impl(
            self.v4_attention,
            self.v4_q,
            self.v4_output,
            v4_mla,
            v4_swa,
            self.v4_global_cache,
            self.v4_swa_cache,
            False,
        )
        self.v41_context.attn_metadata = {"v41_swa": v41_swa} | {
            f"v41_c{ratio}": metadata for ratio, metadata in v41_globals.items()
        }
        for ratio, attention in self.v41_attentions.items():
            attention.forward_mqa(
                self.v41_q,
                self.v41_q[:, 0],
                self.positions,
                self.v41_outputs[ratio],
            )

    def v4_reference(self, v4_mla, v4_swa) -> torch.Tensor:
        expected = torch.zeros_like(self.v4_output)
        global_indices = v4_mla.c128a_global_decode_topk_indices[:, 0]
        for token in range(_CAPTURE_TOKENS):
            swa_len = int(v4_swa.decode_swa_lens[token])
            global_len = int(v4_mla.c128a_decode_topk_lens[token])
            if swa_len + global_len == 0:
                continue
            selected = global_indices[token, :global_len]
            global_sum = (2 * selected.float() - 1).sum()
            expected[token].fill_(global_sum / (swa_len + global_len))
        return expected

    def _decode_v41_swa_cache(self) -> torch.Tensor:
        cache_rows = self.v41_swa_cache.flatten(0, 1)
        values = cache_rows[:, :512].view(torch.float8_e4m3fn).float()
        values = values.view(-1, 16, 32)
        scales = cache_rows[:, 512:].view(torch.float8_e8m0fnu).float()
        return (values * scales[:, :, None]).flatten(1)

    def _decode_v41_global_cache(self, ratio: int) -> torch.Tensor:
        cache_rows = self.v41_global_caches[ratio].flatten(0, 1)
        packed = cache_rows[:, :256]
        codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(1)
        magnitudes = codes.new_tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32
        )[(codes & 7).long()]
        values = torch.where((codes & 8).bool(), -magnitudes, magnitudes)
        scales = cache_rows[:, 256:].view(torch.float8_e4m3fn).float()
        return values * scales.repeat_interleave(16, dim=1)

    def _v41_global_slots(self, token: int, ratio: int, v41_swa, v41_global):
        if not bool(v41_swa.is_valid_token[token]):
            return self.v41_topk_indices.new_empty(0, dtype=torch.int64)
        req = int(v41_swa.token_to_req_indices[token])
        local = self.v41_topk_indices[token]
        page_size = v41_global.block_size // ratio
        page_indices = local // page_size
        valid = (local >= 0) & (page_indices < v41_global.block_table.shape[1])
        page_indices = page_indices.clamp(
            min=0, max=v41_global.block_table.shape[1] - 1
        )
        pages = v41_global.block_table[req, page_indices.long()]
        valid &= pages >= 0
        return (pages * page_size + local % page_size)[valid].long()

    def v41_reference(self, ratio: int, v41_swa, v41_global) -> torch.Tensor:
        swa_cache = self._decode_v41_swa_cache()
        global_cache = self._decode_v41_global_cache(ratio)
        expected = torch.zeros_like(self.v41_outputs[ratio])
        grouped = expected.view(_GROUPS, _CAPTURE_TOKENS, -1)
        for token, row in enumerate(v41_swa.decode_swa_indices[:, 0]):
            swa_slots = row[row >= 0].long()
            global_slots = self._v41_global_slots(token, ratio, v41_swa, v41_global)
            selected = torch.cat((swa_cache[swa_slots], global_cache[global_slots]))
            if selected.shape[0]:
                per_head = selected.mean(0).to(torch.bfloat16)
                grouped[:, token] = per_head.repeat(_HEADS // _GROUPS)
        return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_dsv4_adaptive_fullgraph_replays_real_target_attention(monkeypatch):
    if torch.cuda.get_device_capability() not in ((8, 9), (12, 0), (12, 1)):
        pytest.skip("Requires native SM89/SM12x DeepSeek V4 target support.")
    pytest.importorskip("flashinfer")

    device = torch.device("cuda")
    config = _adaptive_full_config()
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    graph_pool = torch.cuda.graph_pool_handle()
    monkeypatch.setattr(
        cudagraph_utils.current_platform, "get_global_graph_pool", lambda: graph_pool
    )
    monkeypatch.setattr(cudagraph_utils, "set_graph_pool_id", lambda _: None)
    monkeypatch.setattr(cudagraph_utils, "is_global_first_rank", lambda: False)
    offloader = SimpleNamespace(
        sync_prev_onload=lambda: None,
        join_after_forward=lambda: None,
    )
    monkeypatch.setattr(cudagraph_utils, "get_offloader", lambda: offloader)

    @contextmanager
    def local_graph_capture(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            yield SimpleNamespace(stream=stream)
        torch.cuda.current_stream(device).wait_stream(stream)

    monkeypatch.setattr(cudagraph_utils, "graph_capture", local_graph_capture)
    manager = cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=device,
        cudagraph_mode=CUDAGraphMode.FULL,
        decode_query_len=3,
        varlen_decode=True,
    )
    harness = _AdaptiveFullTargetHarness(config, device)
    monkeypatch.setattr(v41_sparse, "get_forward_context", lambda: harness.v41_context)
    workspace_manager = WorkspaceManager(device)
    monkeypatch.setattr(
        v41_sparse, "current_workspace_manager", lambda: workspace_manager
    )
    captured_metadata = None

    def create_forward_fn(desc, warmup):
        nonlocal captured_metadata
        assert desc.cg_mode == CUDAGraphMode.FULL
        metadata = harness.stage([2] * 6, [2] * 6, [128] * 6)
        if not warmup:
            captured_metadata = metadata
        return lambda _mode: harness.run(*metadata)

    manager.capture(create_forward_fn, progress_bar_desc="test capture")
    assert captured_metadata is not None
    (
        captured_v4_mla,
        captured_v4_swa,
        captured_v41_swa,
        captured_v41_globals,
    ) = captured_metadata
    captured_ptrs = (
        captured_v4_swa.decode_swa_indices.data_ptr(),
        captured_v4_mla.c128a_global_decode_topk_indices.data_ptr(),
        captured_v41_swa.decode_swa_indices.data_ptr(),
    )
    captured_global_ptrs = {
        ratio: (
            metadata.req_id_per_token.data_ptr(),
            metadata.slot_mapping.data_ptr(),
        )
        for ratio, metadata in captured_v41_globals.items()
    }

    # Verification rows stay within K + 1 = 3. The first case redistributes
    # CPU budget [2, 2] to device [3, 1]; the second has zero drafts on both.
    # A four-token prefill crosses the ordinary threshold, while FULL padding
    # keeps both cases on the captured 12-token/6-request descriptor.
    cases = [
        ([3, 1, 4], [2, 2, 4], [128, 129, 128]),
        ([1, 1, 4], [1, 1, 4], [128, 129, 128]),
    ]
    for device_lens, cpu_lens, contexts in cases:
        (
            replay_v4_mla,
            replay_v4_swa,
            replay_v41_swa,
            replay_v41_globals,
        ) = harness.stage(device_lens, cpu_lens, contexts)
        assert captured_global_ptrs == {
            ratio: (
                metadata.req_id_per_token.data_ptr(),
                metadata.slot_mapping.data_ptr(),
            )
            for ratio, metadata in replay_v41_globals.items()
        }
        assert set(replay_v41_globals) == {1, 2}
        assert all(
            metadata.num_actual_tokens == _CAPTURE_TOKENS
            for metadata in replay_v41_globals.values()
        )
        for metadata in (replay_v4_swa, replay_v41_swa):
            assert (metadata.num_decodes, metadata.num_prefills) == (6, 0)
            assert metadata.num_decode_tokens == _CAPTURE_TOKENS
            assert metadata.num_prefill_tokens == 0
        assert replay_v4_mla.c128a_prefill_topk_indices is None
        assert captured_ptrs == (
            replay_v4_swa.decode_swa_indices.data_ptr(),
            replay_v4_mla.c128a_global_decode_topk_indices.data_ptr(),
            replay_v41_swa.decode_swa_indices.data_ptr(),
        )

        harness.run(
            captured_v4_mla,
            captured_v4_swa,
            captured_v41_swa,
            captured_v41_globals,
        )
        eager_v4 = harness.v4_output.clone()
        eager_v41 = {
            ratio: output.clone() for ratio, output in harness.v41_outputs.items()
        }
        v4_reference = harness.v4_reference(replay_v4_mla, replay_v4_swa)
        assert v4_reference.abs().amax() > 1e-3
        torch.testing.assert_close(
            eager_v4,
            v4_reference,
            rtol=0.01,
            atol=1e-4,
        )
        for ratio, output in eager_v41.items():
            reference = harness.v41_reference(
                ratio, replay_v41_swa, replay_v41_globals[ratio]
            )
            assert reference.abs().amax() > 1e-2
            torch.testing.assert_close(output, reference, rtol=0.03, atol=0.005)

        harness.v4_output.fill_(37)
        for output in harness.v41_outputs.values():
            output.fill_(37)
        live_tokens = sum(device_lens)
        desc = manager.dispatch(
            num_reqs=3,
            num_tokens=live_tokens,
            uniform_token_count=None,
            num_active_loras=0,
            max_query_len=max(cpu_lens),
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert (desc.num_reqs, desc.num_tokens) == (6, 12)
        manager.run_fullgraph(desc)
        torch.testing.assert_close(harness.v4_output, eager_v4, rtol=0, atol=0)
        for ratio, output in harness.v41_outputs.items():
            torch.testing.assert_close(output, eager_v41[ratio], rtol=0, atol=0)
        torch.testing.assert_close(
            harness.v4_output[live_tokens:],
            torch.zeros_like(harness.v4_output[live_tokens:]),
        )
        for output in harness.v41_outputs.values():
            grouped = output.view(_GROUPS, _CAPTURE_TOKENS, -1)
            torch.testing.assert_close(
                grouped[:, live_tokens:], torch.zeros_like(grouped[:, live_tokens:])
            )
