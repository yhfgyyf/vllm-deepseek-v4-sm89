# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness checks for the native DeepSeek V4.1 MXFP4 indexer."""

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.kernels.attention.dsa import dsv41_indexer
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    select_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    _decode_e2m1,
    dsv41_mxfp4_candidate_logits,
    dsv41_mxfp4_dense_logits,
    map_candidate_topk_,
)
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)

HEADS = 32
HEAD_DIM = 128

requires_cuda_and_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON,
    reason="requires CUDA and Triton",
)


@triton.jit
def _decode_nibbles_kernel(codes, values):
    offsets = tl.arange(0, 16)
    decoded = _decode_e2m1(tl.load(codes + offsets))
    tl.store(values + offsets, decoded)


@pytest.mark.parametrize(
    ("width", "expected_grid", "expected_swap"),
    [
        (65_535 * 16, (3, 65_535), False),
        (65_535 * 16 + 1, (65_536, 3), True),
        (1024 * 1024, (65_536, 3), True),
    ],
)
def test_dense_launcher_keeps_grid_y_within_cuda_limit(
    monkeypatch, width: int, expected_grid: tuple[int, int], expected_swap: bool
) -> None:
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*_args, **kwargs):
                launches.append((grid, kwargs))

            return launch

    monkeypatch.setattr(dsv41_indexer, "_validate_inputs", lambda *_: None)
    monkeypatch.setattr(dsv41_indexer, "_mxfp4_indexer_logits_kernel", FakeKernel())
    rows = 3
    out = torch.full((rows, width), 17.0)
    actual = dsv41_indexer.dsv41_mxfp4_dense_logits(
        torch.zeros(rows, HEADS, 64, dtype=torch.uint8),
        torch.zeros(rows, HEADS, 4, dtype=torch.uint8),
        torch.zeros(1, 128, 68, dtype=torch.uint8),
        torch.zeros(rows, HEADS, dtype=torch.float32),
        torch.zeros(rows, 1, dtype=torch.int32),
        torch.ones(rows, dtype=torch.int32),
        width,
        out=out,
    )

    assert actual is out
    assert torch.all(out == 17)
    assert len(launches) == 1
    grid, kwargs = launches[0]
    assert grid == expected_grid
    assert kwargs["SWAP_GRID_AXES"] is expected_swap


@requires_cuda_and_triton
def test_e2m1_decoder_preserves_all_codes_and_signed_zero() -> None:
    codes = torch.arange(16, device="cuda", dtype=torch.uint8)
    actual = torch.empty(16, device="cuda", dtype=torch.float32)
    _decode_nibbles_kernel[(1,)](codes, actual, num_warps=1)
    positive = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    expected = torch.cat((positive, -positive))
    torch.testing.assert_close(
        actual.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0
    )


def _quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], HEAD_DIM // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=6 * 2**-126)
    exponent = (amax / 6).log2().ceil().clamp(-127, 127)
    scaled = (blocks / exponent.exp2()).clamp(-6, 6)
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold, value, inclusive in (
        (0.25, 1, False),
        (0.75, 2, True),
        (1.25, 3, False),
        (1.75, 4, True),
        (2.50, 5, False),
        (3.50, 6, True),
        (5.00, 7, False),
    ):
        condition = magnitude >= threshold if inclusive else magnitude > threshold
        code = torch.where(condition, value, code)
    sign = ((scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = (code | (sign << 3)).reshape(*shape)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    scales = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


def _dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    low, high = packed & 0xF, packed >> 4
    nibble = torch.stack((low, high), dim=-1).flatten(-2)
    magnitude = table[(nibble & 0x7).long()]
    values = torch.where((nibble & 0x8) != 0, -magnitude, magnitude)
    scale = (scales.float() - 127).exp2().repeat_interleave(32, dim=-1)
    return values * scale


def _make_cache(
    rows: torch.Tensor, page_size: int, block_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    packed, scales = _quantize_mxfp4(rows)
    num_blocks = int(block_table.max().item()) + 1
    cache = torch.zeros(
        (num_blocks, page_size, 68), device=rows.device, dtype=torch.uint8
    )
    raw = cache.view(num_blocks, -1)
    for req in range(block_table.shape[0]):
        for logical, physical in enumerate(block_table[req].tolist()):
            start = logical * page_size
            stop = min(start + page_size, rows.shape[1])
            count = stop - start
            raw[physical, : count * 64] = packed[req, start:stop].reshape(-1)
            scale_base = page_size * 64
            raw[physical, scale_base : scale_base + count * 4] = scales[
                req, start:stop
            ].reshape(-1)
    return cache, _dequantize_mxfp4(packed, scales)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (torch.einsum("qhd,qtd->qht", q, k).relu() * weights[:, :, None]).sum(dim=1)


@pytest.mark.parametrize("page_size", [64, 128])
@requires_cuda_and_triton
def test_dense_indexer_reads_segregated_pages(page_size: int) -> None:
    torch.manual_seed(0)
    batch, length = 2, page_size + 3
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scales = _quantize_mxfp4(q)
    block_table = torch.tensor([[1, 0], [2, 3]], device="cuda", dtype=torch.int32)
    cache, k_dequant = _make_cache(k, page_size, block_table)
    q_dequant = _dequantize_mxfp4(q_packed, q_scales)
    weights = torch.randn(batch, HEADS, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([length, page_size - 1], device="cuda", dtype=torch.int32)

    width = length + 13
    out = torch.empty(batch, width, device="cuda", dtype=torch.float32)
    actual = dsv41_mxfp4_dense_logits(
        q_packed,
        q_scales,
        cache,
        weights,
        block_table,
        lengths,
        width,
        out=out,
    )
    assert actual is out
    expected = _reference(q_dequant, k_dequant, weights)
    expected = torch.nn.functional.pad(expected, (0, width - length), value=-torch.inf)
    positions = torch.arange(width, device="cuda")
    expected.masked_fill_(positions[None] >= lengths[:, None], -torch.inf)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_cuda_and_triton
def test_breakable_replay_reads_fresh_v41_decode_width(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    layer_name = "model.layers.20.attn.indexer.k_cache"
    observed_widths = []

    class _Workspace:
        def __init__(self) -> None:
            self.tensors = (
                torch.empty(16_384, device="cuda", dtype=torch.float32),
                torch.empty(2_048, device="cuda", dtype=torch.float32),
                torch.empty(1, device="cuda", dtype=torch.int32),
                torch.empty(1, device="cuda", dtype=torch.uint8),
            )

        def get_simultaneous(self, *_specs):
            return self.tensors

    def fake_dense(*args, out, **kwargs):
        width = args[6]
        observed_widths.append(width)
        out.fill_(-torch.inf)
        return out

    workspace = _Workspace()
    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(sparse_indexer, "dsv41_mxfp4_dense_logits", fake_dense)
    monkeypatch.setattr(
        sparse_indexer.torch.ops._C,
        "persistent_topk",
        lambda *_args, **_kwargs: None,
    )

    block_table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    def metadata(width: int):
        decode = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=torch.tensor([[width]], device="cuda", dtype=torch.int32),
            decode_lens=torch.ones(1, device="cuda", dtype=torch.int32),
            requires_padding=False,
            schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
            max_indexer_kv_len=width,
        )
        return DeepseekV32IndexerMetadata(
            seq_lens=decode.seq_lens,
            max_seq_len=width,
            slot_mapping=torch.zeros(1, device="cuda", dtype=torch.int64),
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=decode,
        )

    hidden = torch.zeros((1, 1), device="cuda")
    q_values = torch.zeros((1, HEADS, 64), device="cuda", dtype=torch.uint8)
    q_scales = torch.zeros((1, HEADS, 4), device="cuda", dtype=torch.uint8)
    cache = torch.zeros((1, 128, 68), device="cuda", dtype=torch.uint8)
    weights = torch.zeros((1, HEADS), device="cuda")
    topk = torch.full((1, 1), -1, device="cuda", dtype=torch.int32)

    def context(width: int) -> ForwardContext:
        return ForwardContext(
            no_compile_layers={},
            attn_metadata={layer_name: metadata(width)},
            slot_mapping={},
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )

    def run_indexer() -> None:
        sparse_indexer.sparse_attn_indexer(
            hidden,
            layer_name,
            cache,
            q_values,
            q_scales,
            None,
            weights,
            128,
            "ue8m0",
            1,
            128,
            128,
            128,
            topk,
            True,
            False,
            "",
            use_fp4_cache=True,
            skip_topk_buffer_clear=True,
            use_v41_native=True,
        )

    marker = torch.zeros(1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = BreakableCUDAGraphCapture()
    with torch.cuda.stream(stream), override_forward_context(context(17)), graph:
        marker.add_(1)
        run_indexer()
        marker.add_(1)
    torch.cuda.current_stream().wait_stream(stream)

    assert observed_widths == [17]
    assert graph.num_eager_breaks == 1
    assert graph.num_graphs == 2
    for width in (65, 9):
        marker.zero_()
        with override_forward_context(context(width)):
            graph.replay()
        torch.cuda.synchronize()
        assert marker.item() == 2

    assert observed_widths == [17, 65, 9]


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
@requires_cuda_and_triton
def test_dense_indexer_one_million_width_eager_and_cudagraph(
    prefill: bool,
) -> None:
    torch.manual_seed(4)
    width, page_size = 1024 * 1024, 128
    valid_tokens = (5, 4)
    q = torch.randn(2, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, max(valid_tokens), HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scales = _quantize_mxfp4(q)
    cache, k_dequant = _make_cache(
        k,
        page_size,
        torch.arange(2, device="cuda", dtype=torch.int32).view(2, 1),
    )
    q_dequant = _dequantize_mxfp4(q_packed, q_scales)
    weights = torch.randn(2, HEADS, device="cuda", dtype=torch.float32)
    expected = _reference(q_dequant, k_dequant, weights)
    prefill_kwargs = {}
    if prefill:
        valid_ranges = ((3, 8), (8, 12))
        block_table = torch.full(
            (3, width // page_size), -1, device="cuda", dtype=torch.int32
        )
        block_table[1:, 0] = torch.arange(2, device="cuda", dtype=torch.int32)
        prefill_kwargs = {
            "row_starts": torch.tensor(
                [start for start, _ in valid_ranges],
                device="cuda",
                dtype=torch.int32,
            ),
            "cu_seq_lens": torch.tensor(
                [0, 3, 8, 12], device="cuda", dtype=torch.int32
            ),
            "token_to_seq": torch.tensor(
                [0] * 3 + [1] * valid_tokens[0] + [2] * valid_tokens[1],
                device="cuda",
                dtype=torch.int32,
            ),
        }
    else:
        valid_ranges = ((0, valid_tokens[0]), (0, valid_tokens[1]))
        block_table = torch.full(
            (2, width // page_size), -1, device="cuda", dtype=torch.int32
        )
        block_table[:, 0] = torch.arange(2, device="cuda", dtype=torch.int32)
    row_ends = torch.tensor(
        [end for _, end in valid_ranges], device="cuda", dtype=torch.int32
    )

    def check(actual: torch.Tensor) -> None:
        for row, (start, end) in enumerate(valid_ranges):
            torch.testing.assert_close(
                actual[row, start:end],
                expected[row, : end - start],
                rtol=2e-2,
                atol=2e-2,
            )
            assert actual[row, :start].isneginf().all()
            assert actual[row, end:].isneginf().all()

    eager_out = torch.full((2, width), 17.0, device="cuda")
    actual = dsv41_mxfp4_dense_logits(
        q_packed,
        q_scales,
        cache,
        weights,
        block_table,
        row_ends,
        width,
        out=eager_out,
        **prefill_kwargs,
    )
    assert actual is eager_out
    check(actual)

    graph_out = torch.full((2, width), 17.0, device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        actual = dsv41_mxfp4_dense_logits(
            q_packed,
            q_scales,
            cache,
            weights,
            block_table,
            row_ends,
            width,
            out=graph_out,
            **prefill_kwargs,
        )
    assert actual is graph_out
    graph_out.fill_(17.0)
    graph.replay()
    torch.cuda.synchronize()
    check(graph_out)


@requires_cuda_and_triton
def test_candidate_indexer_reads_only_valid_candidate_tokens() -> None:
    torch.manual_seed(1)
    length, page_size = 37, 64
    q = torch.randn(2, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scales = _quantize_mxfp4(q)
    block_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    cache, k_dequant = _make_cache(k, page_size, block_table)
    q_dequant = _dequantize_mxfp4(q_packed, q_scales)
    weights = torch.randn(2, HEADS, device="cuda", dtype=torch.float32)
    candidates = torch.tensor(
        [[0, 4, 1 << 29], [1, 5, -1]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([37, 17], device="cuda", dtype=torch.int32)

    actual = dsv41_mxfp4_candidate_logits(
        q_packed,
        q_scales,
        cache,
        weights,
        block_table,
        lengths,
        candidates,
    )
    dense = _reference(q_dequant, k_dequant, weights)
    expected = torch.full_like(actual, -torch.inf)
    for row in range(2):
        for candidate_slot, block in enumerate(candidates[row].tolist()):
            for offset in range(8):
                token = block * 8 + offset
                if block >= 0 and token < int(lengths[row]):
                    expected[row, candidate_slot * 8 + offset] = dense[row, token]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_cuda_and_triton
def test_candidate_indexer_masks_invalid_physical_pages() -> None:
    q = torch.zeros(1, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scales = _quantize_mxfp4(q)
    cache = torch.zeros(1, 64, 68, device="cuda", dtype=torch.uint8)
    weights = torch.ones(1, HEADS, device="cuda", dtype=torch.float32)
    block_table = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([8], device="cuda", dtype=torch.int32)
    candidates = torch.tensor([[0]], device="cuda", dtype=torch.int32)

    actual = dsv41_mxfp4_candidate_logits(
        q_packed,
        q_scales,
        cache,
        weights,
        block_table,
        lengths,
        candidates,
    )

    assert actual.isneginf().all()


@requires_cuda_and_triton
def test_candidate_prefill_uses_packed_bounds_and_request_local_blocks() -> None:
    torch.manual_seed(2)
    q = torch.randn(2, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 5, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_packed, q_scales = _quantize_mxfp4(q)
    block_table = torch.tensor([[0], [1]], device="cuda", dtype=torch.int32)
    cache, k_dequant = _make_cache(k, 64, block_table)
    q_dequant = _dequantize_mxfp4(q_packed, q_scales)
    weights = torch.randn(2, HEADS, device="cuda", dtype=torch.float32)
    candidates = torch.zeros(2, 1, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    ends = torch.tensor([5, 9], device="cuda", dtype=torch.int32)
    cu_seq_lens = torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32)
    token_to_seq = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1], device="cuda", dtype=torch.int32
    )

    actual = dsv41_mxfp4_candidate_logits(
        q_packed,
        q_scales,
        cache,
        weights,
        block_table,
        ends,
        candidates,
        row_starts=starts,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
    )
    dense = _reference(q_dequant, k_dequant, weights)
    expected = torch.full_like(actual, -torch.inf)
    expected[0, :5] = dense[0, :5]
    expected[1, :4] = dense[1, :4]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_cuda_and_triton
def test_candidate_selection_pins_partial_block_and_pads_empty_rows() -> None:
    logits = torch.zeros(2, 24, device="cuda")
    logits[0, :8] = 3
    logits[0, 8:16] = 3  # exact tie between two full blocks
    starts = torch.zeros(2, device="cuda", dtype=torch.int32)
    ends = torch.tensor([17, 0], device="cuda", dtype=torch.int32)
    out = torch.empty(2, 2048, device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    scores = torch.empty(2, 2048, device="cuda", dtype=torch.float32)
    block_lens = torch.empty(2, device="cuda", dtype=torch.int32)

    select_candidate_blocks(
        logits,
        starts,
        ends,
        2048,
        8,
        out,
        workspace,
        scores_out=scores,
        block_lens_out=block_lens,
    )

    assert set(out[0].tolist()) == {-1, 0, 1, 2}
    assert out[1].tolist() == [-1] * 2048


@requires_cuda_and_triton
def test_candidate_topk_maps_compact_offsets_to_local_tokens() -> None:
    compact_logits = torch.zeros(1, 24, device="cuda", dtype=torch.float32)
    compact_logits[0, 17] = -torch.inf
    candidates = torch.tensor([[4, -1, 2]], device="cuda", dtype=torch.int32)
    indices = torch.tensor(
        [[0, 7, 8, 9, 16, 17, 24, -1]], device="cuda", dtype=torch.int32
    )

    map_candidate_topk_(indices, compact_logits, candidates)

    assert indices.tolist() == [[32, 39, -1, -1, 16, -1, -1, -1]]
