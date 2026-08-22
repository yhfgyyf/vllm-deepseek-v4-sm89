# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _COMBINE_SWA_ONLY_CUTEDSL_MIN_TOKENS,
    _COMBINE_TOPK_CUTEDSL_MIN_TOKENS,
    _combine_topk_swa_indices_triton_baseline,
    combine_topk_swa_indices_fused_triton,
)
from vllm.platforms import current_platform

cache_utils_module = importlib.import_module(
    "vllm.models.deepseek_v4.common.ops.cache_utils"
)

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)


def _make_case(
    topk: int,
    compress_ratio: int,
    *,
    query_lens_host: list[int] | None = None,
    source_width: int | None = None,
) -> tuple[tuple[object, ...], torch.Tensor, torch.Tensor]:
    if query_lens_host is None:
        query_lens_host = [1, 3, 5, 7]
    if len(query_lens_host) != 4 or any(length < 0 for length in query_lens_host):
        raise ValueError("test query lengths must describe four requests")
    seq_lens_host = [8192, 7901, 7603, 7331]
    window_size = 128
    query_base = 37
    num_tokens = sum(query_lens_host)
    if source_width is None:
        source_width = max(topk, 32)
    if source_width < topk:
        raise ValueError("test source width must fit topk")
    device = torch.device("cuda")

    query_lens = torch.tensor(query_lens_host, dtype=torch.int32, device=device)
    query_start_values = torch.empty(5, dtype=torch.int32, device=device)
    query_start_values[0] = query_base
    torch.cumsum(query_lens, dim=0, out=query_start_values[1:])
    query_start_values[1:].add_(query_base)
    seq_lens_values = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    gather_lens_values = torch.tensor(
        [
            query_len + min(seq_len - query_len, window_size - 1)
            for query_len, seq_len in zip(query_lens_host, seq_lens_host, strict=True)
        ],
        dtype=torch.int32,
        device=device,
    )

    def offset(values: torch.Tensor) -> torch.Tensor:
        storage = torch.full(
            (values.numel() + 2,),
            -777,
            dtype=torch.int32,
            device=device,
        )
        storage[1:-1].copy_(values)
        return storage[1:-1]

    query_start_loc = offset(query_start_values)
    seq_lens = offset(seq_lens_values)
    gather_lens = offset(gather_lens_values)

    request_indices = torch.arange(4, dtype=torch.int64, device=device)
    request_indices = request_indices.repeat_interleave(query_lens.to(torch.int64))
    request_starts = torch.tensor(
        [sum(query_lens_host[:index]) for index in range(4)],
        dtype=torch.int64,
        device=device,
    )
    token_indices = torch.arange(num_tokens, dtype=torch.int64, device=device)
    local_token_indices = token_indices - request_starts[request_indices]
    positions = (
        seq_lens_values.to(torch.int64)[request_indices]
        - query_lens.to(torch.int64)[request_indices]
        + local_token_indices
    )
    topk_lens = torch.minimum(
        torch.div(positions + 1, compress_ratio, rounding_mode="floor"),
        torch.full_like(positions, topk),
    )
    swa_lens = torch.minimum(
        positions + 1,
        torch.full_like(positions, window_size),
    )

    n = 0 if topk == 0 else max(seq_lens_host) // compress_ratio
    m = n + int(gather_lens_values.max().item())
    topk_storage = torch.full(
        (num_tokens + 2, source_width + 3),
        -777,
        dtype=torch.int32,
        device=device,
    )
    topk_indices = topk_storage[1:-1, :source_width]
    columns = torch.arange(source_width, dtype=torch.int64, device=device)
    source_values = (columns[None, :] * 67 + token_indices[:, None] * 131) % max(n, 1)
    topk_indices.copy_(
        torch.where(
            columns[None, :] < topk_lens[:, None],
            source_values,
            -777,
        ).to(torch.int32)
    )
    if topk > 1:
        duplicate_rows = topk_lens > 1
        topk_indices[duplicate_rows, 1] = topk_indices[duplicate_rows, 0]

    combined_width = ((topk + window_size + 127) // 128) * 128
    output_columns = torch.arange(combined_width, dtype=torch.int64, device=device)
    expected = torch.full(
        (num_tokens, combined_width), -1, dtype=torch.int32, device=device
    )
    request_offsets = request_indices[:, None] * m
    if source_width > 0:
        safe_topk_columns = output_columns.clamp_max(source_width - 1)
        topk_values = topk_indices[:, safe_topk_columns]
        expected = torch.where(
            output_columns[None, :] < topk_lens[:, None],
            topk_values + request_offsets,
            expected,
        )
    swa_offsets = output_columns[None, :] - topk_lens[:, None]
    gather_starts = seq_lens_values.to(torch.int64) - gather_lens_values.to(torch.int64)
    swa_values = (
        request_offsets
        + n
        + swa_offsets
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[request_indices, None]
    )
    expected = torch.where(
        (swa_offsets >= 0) & (swa_offsets < swa_lens[:, None]),
        swa_values,
        expected,
    ).to(torch.int32)
    expected_lens = (topk_lens + swa_lens).to(torch.int32)

    args: tuple[object, ...] = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        m,
        n,
    )
    return args, expected, expected_lens


def test_combine_topk_swa_cutedsl_dispatch_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: capability == 120,
    )
    monkeypatch.setattr(cache_utils_module, "current_platform", sm120)
    monkeypatch.setattr(cache_utils_module, "has_cutedsl", lambda: True)
    can_use = cache_utils_module._can_use_combine_topk_swa_cutedsl

    for topk in (512, 1024):
        assert can_use(topk=topk, num_tokens=_COMBINE_TOPK_CUTEDSL_MIN_TOKENS)
        assert not can_use(topk=topk, num_tokens=_COMBINE_TOPK_CUTEDSL_MIN_TOKENS - 1)
    assert can_use(topk=0, num_tokens=_COMBINE_SWA_ONLY_CUTEDSL_MIN_TOKENS)
    assert not can_use(topk=0, num_tokens=_COMBINE_SWA_ONLY_CUTEDSL_MIN_TOKENS - 1)
    assert not can_use(topk=8192, num_tokens=8192)

    monkeypatch.setattr(cache_utils_module, "has_cutedsl", lambda: False)
    assert not can_use(topk=512, num_tokens=8192)
    monkeypatch.setattr(
        cache_utils_module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: False,
        ),
    )
    assert not can_use(topk=512, num_tokens=8192)


@requires_sm120
@pytest.mark.parametrize(
    ("topk", "compress_ratio"),
    [(0, 1), (512, 4), (1024, 128)],
)
@torch.inference_mode()
def test_combine_topk_swa_cutedsl_matches_reference(
    topk: int,
    compress_ratio: int,
):
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v4.nvidia.ops.combine_topk_swa_cutedsl import (
        combine_topk_swa_indices_cutedsl,
    )

    args, expected, expected_lens = _make_case(
        topk,
        compress_ratio,
        source_width=0 if topk == 0 else None,
    )
    output, output_lens = combine_topk_swa_indices_cutedsl(*args)
    assert torch.equal(output, expected)
    assert torch.equal(output_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_combine_topk_swa_empty_requests_match_reference():
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v4.nvidia.ops.combine_topk_swa_cutedsl import (
        combine_topk_swa_indices_cutedsl,
    )

    args, expected, expected_lens = _make_case(
        512,
        4,
        query_lens_host=[0, 3, 0, 13],
    )
    for implementation in (
        combine_topk_swa_indices_cutedsl,
        combine_topk_swa_indices_fused_triton,
    ):
        output, output_lens = implementation(*args)
        assert torch.equal(output, expected)
        assert torch.equal(output_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_combine_topk_swa_public_sm120_dispatch_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("cutlass")
    monkeypatch.setattr(cache_utils_module, "_COMBINE_TOPK_CUTEDSL_MIN_TOKENS", 0)
    args, expected, expected_lens = _make_case(512, 4)
    output, output_lens = cache_utils_module.combine_topk_swa_indices(*args)
    assert torch.equal(output, expected)
    assert torch.equal(output_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_combine_topk_swa_public_zero_width_swa_only_matches_reference():
    args, expected, expected_lens = _make_case(0, 1, source_width=0)
    output, output_lens = cache_utils_module.combine_topk_swa_indices(*args)
    assert torch.equal(output, expected)
    assert torch.equal(output_lens, expected_lens)


@requires_sm120
@pytest.mark.parametrize(
    ("topk", "compress_ratio"),
    [(0, 1), (512, 4), (1024, 128), (8192, 128)],
)
@torch.inference_mode()
def test_combine_topk_swa_fused_triton_matches_reference(
    topk: int,
    compress_ratio: int,
):
    args, expected, expected_lens = _make_case(
        topk,
        compress_ratio,
        source_width=0 if topk == 0 else None,
    )
    output, output_lens = combine_topk_swa_indices_fused_triton(*args)
    assert torch.equal(output, expected)
    assert torch.equal(output_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_combine_topk_swa_baseline_reinitializes_preallocated_padding():
    args, expected, expected_lens = _make_case(512, 4)
    output = torch.full_like(expected, -777)
    output_lens = torch.full_like(expected_lens, -777)

    actual, actual_lens = _combine_topk_swa_indices_triton_baseline(
        *args, out=(output, output_lens)
    )

    assert actual.data_ptr() == output.data_ptr()
    assert actual_lens.data_ptr() == output_lens.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_combine_topk_swa_cutedsl_cuda_graph():
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v4.nvidia.ops.combine_topk_swa_cutedsl import (
        combine_topk_swa_indices_cutedsl,
    )

    args, expected, expected_lens = _make_case(512, 4)
    combine_topk_swa_indices_cutedsl(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, output_lens = combine_topk_swa_indices_cutedsl(*args)
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.equal(output, expected)
    assert torch.equal(output_lens, expected_lens)
