# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure adaptive-verification indexer operator cost on NVIDIA GPUs.

This benchmark intentionally stops at the indexer boundary: production logits
followed by production top-k. It does not model draft generation, target-model
execution, acceptance, scheduling, or end-to-end serving latency, so its output
is not a model-budget cost curve.

Examples:
    .venv/bin/python benchmarks/kernels/benchmark_dsv4_adaptive_indexer.py \
        --output /tmp/indexer-cost.json
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    dsv41_mxfp4_dense_logits,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    fused_indexer_q_rope_quant as fused_v4_q_quant,
)
from vllm.models.deepseek_v4_1.common.ops.fused_indexer_q import (
    fused_indexer_q_rope_quant as fused_v41_q_quant,
)
from vllm.models.deepseek_v4_1.common.ops.indexer_k_store import (
    indexer_k_norm_rope_store,
)
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
)

DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_CONTEXTS = (1024, 8192)
HEADS = 32
HEAD_DIM = 128
V4_PAGE_SIZE = 64
V41_PAGE_SIZE = 128
SEED = 20260912
SCOPE_NOTICE = (
    "Operator-only logits + production top-k cost. This is not a full-model "
    "adaptive target+draft profile and is not model-budget-ready."
)


@dataclass
class BenchmarkCase:
    model: str
    implementation: str
    tokens: int
    context: int
    requests: int
    request_token_counts: list[int]
    page_size: int
    topk: int
    score: Callable[[], torch.Tensor]
    run: Callable[[], None]
    correctness: dict[str, Any]
    tensors: list[torch.Tensor]


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _require_cupti() -> str:
    cupti_version = _package_version("cupti-python")
    if cupti_version is None or int(cupti_version.split(".")[0]) < 13:
        raise RuntimeError(
            "CUPTI timing requires cupti-python>=13; refusing CUDA-event fallback"
        )
    from cupti import cupti  # noqa: F401

    return cupti_version


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(obj: object) -> dict[str, str]:
    path = Path(inspect.getsourcefile(obj) or inspect.getfile(obj)).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _binary_record(path: str) -> dict[str, str]:
    version_output = subprocess.run(
        [path, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "path": path,
        "resolved_path": str(Path(path).resolve()),
        "version_output": version_output,
    }


def _toolchain_metadata(cuda_arch: int) -> dict[str, Any]:
    from triton.backends.nvidia.compiler import get_ptxas

    triton_ptxas = get_ptxas(cuda_arch)
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is unavailable on PATH")
    return {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "torch_cuda_build": torch.version.cuda,
        "nvcc": _binary_record(nvcc),
        "triton_nvidia_tool": {
            **_binary_record(triton_ptxas.path),
            "parsed_version": triton_ptxas.version,
            "selector": "ptxas_blackwell" if cuda_arch >= 100 else "ptxas",
            "configured_blackwell_path": os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
            "configured_generic_path": os.environ.get("TRITON_PTXAS_PATH"),
        },
    }


def _git_metadata(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--short").splitlines()
    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }


def _identity_rope(context: int) -> torch.Tensor:
    result = torch.zeros((context, 64), device="cuda", dtype=torch.bfloat16)
    result[:, :32] = 1
    return result


def _request_layout(
    tokens: int, device: str | torch.device = "cuda"
) -> tuple[int, list[int], torch.Tensor]:
    requests = min(tokens, 16)
    per_request, remainder = divmod(tokens, requests)
    counts = [per_request + (request < remainder) for request in range(requests)]
    if remainder == 0 and per_request > 1:
        counts[0] -= 1
        counts[1] += 1
    row_requests = torch.repeat_interleave(
        torch.arange(requests, device=device, dtype=torch.int32),
        torch.tensor(counts, device=device, dtype=torch.int64),
    )
    return requests, counts, row_requests


def _check_request_layouts() -> None:
    for tokens in (1, 7, 17, 24, 33, 127):
        requests, counts, row_requests = _request_layout(tokens, device="cpu")
        observed = torch.bincount(
            row_requests.to(torch.int64), minlength=requests
        ).tolist()
        if sum(counts) != tokens or row_requests.numel() != tokens:
            raise AssertionError(
                f"invalid request layout for {tokens=}: {counts=}, "
                f"rows={row_requests.numel()}"
            )
        if observed != counts:
            raise AssertionError(
                f"request ids do not match counts for {tokens=}: "
                f"{observed=} != {counts=}"
            )
        request_blocks = torch.arange(requests * 3).view(requests, 3)
        expanded = _expand_request_block_tables(request_blocks, row_requests)
        if expanded.shape != (tokens, 3):
            raise AssertionError(
                f"flattened block tables have wrong shape for {tokens=}: "
                f"{tuple(expanded.shape)}"
            )
        if not torch.equal(expanded[:, 0] // 3, row_requests.to(torch.int64)):
            raise AssertionError(
                f"flattened block tables map wrong requests for {tokens=}"
            )


def _expand_request_block_tables(
    request_blocks: torch.Tensor, row_requests: torch.Tensor
) -> torch.Tensor:
    return request_blocks.index_select(0, row_requests.long()).contiguous()


def _block_tables(requests: int, context: int, page_size: int) -> torch.Tensor:
    blocks_per_request = math.ceil(context / page_size)
    num_blocks = requests * blocks_per_request
    physical = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
    physical = physical.roll(shifts=max(1, num_blocks // 3))
    return physical.view(requests, blocks_per_request).contiguous()


def _cache_slots(
    block_tables: torch.Tensor, context: int, page_size: int
) -> torch.Tensor:
    positions = torch.arange(context, device="cuda", dtype=torch.int64)
    logical_blocks = positions // page_size
    offsets = positions % page_size
    physical = block_tables.to(torch.int64).index_select(1, logical_blocks)
    return (physical * page_size + offsets).reshape(-1).contiguous()


def _dequantize_fp8_cache(
    cache: torch.Tensor, block_tables: torch.Tensor, context: int
) -> torch.Tensor:
    num_blocks = cache.shape[0]
    page_size = cache.shape[1]
    raw = cache.view(num_blocks, -1)
    values = raw[:, : page_size * HEAD_DIM].view(num_blocks, page_size, HEAD_DIM)
    values = values.view(torch.float8_e4m3fn).float()
    scales = (
        raw[:, page_size * HEAD_DIM :]
        .contiguous()
        .view(torch.float32)
        .view(num_blocks, page_size, 1)
    )
    physical = block_tables.long().flatten()
    requests = block_tables.shape[0]
    rows = (values * scales).index_select(0, physical)
    return rows.view(requests, -1, HEAD_DIM)[:, :context]


def _decode_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    low, high = packed & 0xF, packed >> 4
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    magnitude = table[(nibbles & 0x7).long()]
    values = torch.where((nibbles & 0x8) != 0, -magnitude, magnitude)
    scale = (scales.float() - 127).exp2().repeat_interleave(32, dim=-1)
    return values * scale


def _dequantize_mxfp4_cache(
    cache: torch.Tensor, block_tables: torch.Tensor, context: int
) -> torch.Tensor:
    num_blocks = cache.shape[0]
    page_size = cache.shape[1]
    raw = cache.view(num_blocks, -1)
    packed = raw[:, : page_size * 64].view(num_blocks, page_size, 64)
    scales = raw[:, page_size * 64 :].view(num_blocks, page_size, 4)
    physical = block_tables.long().flatten()
    rows = _decode_mxfp4(packed, scales).index_select(0, physical)
    return rows.view(block_tables.shape[0], -1, HEAD_DIM)[:, :context]


def _run_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    topk: int,
) -> None:
    torch.ops._C.persistent_topk(
        logits,
        lengths,
        output,
        workspace,
        topk,
        logits.shape[1],
    )


def _check_topk(
    logits: torch.Tensor,
    actual: torch.Tensor,
    topk: int,
) -> None:
    expected = torch.topk(logits, topk, dim=1).indices.to(torch.int32)
    torch.testing.assert_close(
        actual.sort(dim=1).values,
        expected.sort(dim=1).values,
        rtol=0,
        atol=0,
    )


def _cosine_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    similarity = torch.dot(actual, expected) / (
        torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(expected)
    )
    return float(1 - similarity)


def _representative_rows(tokens: int) -> torch.Tensor:
    rows = sorted({0, tokens // 2, tokens - 1})
    return torch.tensor(rows, device="cuda", dtype=torch.int64)


def _make_v4_case(tokens: int, context: int, cc: tuple[int, int]) -> BenchmarkCase:
    requests, counts, row_requests = _request_layout(tokens)
    request_blocks = _block_tables(requests, context, V4_PAGE_SIZE)
    expanded_blocks = _expand_request_block_tables(request_blocks, row_requests)
    num_blocks = request_blocks.numel()

    k_source = torch.randn(
        (requests * context, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    cache = torch.zeros(
        (num_blocks, V4_PAGE_SIZE, HEAD_DIM + 4),
        device="cuda",
        dtype=torch.uint8,
    )
    slots = _cache_slots(request_blocks, context, V4_PAGE_SIZE)
    ops.indexer_k_quant_and_cache(k_source, cache, slots, HEAD_DIM, "ue8m0")
    cache_view = cache.unsqueeze(-2)

    q_source = torch.randn(
        (tokens, HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    input_weights = torch.randn((tokens, HEADS), device="cuda", dtype=torch.float32)
    positions = torch.full((tokens,), context - 1, device="cuda", dtype=torch.int64)
    q, weights = fused_v4_q_quant(
        positions,
        q_source,
        _identity_rope(context),
        input_weights,
        1.0,
        1.0,
        False,
    )
    q = q.reshape(tokens, 1, HEADS, HEAD_DIM)
    lengths = torch.full((tokens, 1), context, device="cuda", dtype=torch.int32)
    if cc == (8, 9):
        schedule = torch.empty(0, device="cuda", dtype=torch.int32)
        implementation = "portable_triton_sm89"
    elif cc[0] == 12:
        if not has_deep_gemm():
            raise RuntimeError("SM120 V4 benchmark requires DeepGEMM")
        schedule = get_paged_mqa_logits_metadata(
            lengths,
            V4_PAGE_SIZE,
            get_num_sms(),
            indices=row_requests,
        )
        implementation = "native_deepgemm_sm120_varlen_indices"
    else:
        raise RuntimeError(f"V4 adaptive benchmark supports SM89/SM120, got SM{cc}")

    topk = min(2048, context)
    topk_output = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8)

    def score() -> torch.Tensor:
        return fp8_fp4_paged_mqa_logits(
            (q, None),
            cache_view,
            weights,
            lengths,
            expanded_blocks,
            schedule,
            context,
            clean_logits=False,
            indices=row_requests if cc[0] == 12 else None,
        )

    def run() -> None:
        logits = score()
        _run_topk(logits, lengths, topk_output, workspace, topk)

    actual = score()
    _run_topk(actual, lengths, topk_output, workspace, topk)
    rows = _representative_rows(tokens)
    logical_k = _dequantize_fp8_cache(cache, request_blocks, context)
    q_ref = q.index_select(0, rows)[:, 0].float()
    k_ref = logical_k.index_select(0, row_requests.index_select(0, rows).long())
    w_ref = weights.index_select(0, rows)
    expected = (
        torch.einsum("rhd,rtd->rht", q_ref, k_ref).relu() * w_ref[:, :, None]
    ).sum(dim=1)
    diff = _cosine_diff(actual.index_select(0, rows), expected)
    if not math.isfinite(diff) or diff >= 1e-3:
        raise AssertionError(f"V4 quantized-reference cosine diff {diff} >= 1e-3")
    _check_topk(actual, topk_output, topk)
    correctness = {
        "representative_rows": rows.tolist(),
        "reference": "independent torch math over dequantized production inputs",
        "metric": "one_minus_cosine_similarity",
        "value": diff,
        "threshold": 1e-3,
        "topk_exact_set_match": True,
    }
    tensors = [
        k_source,
        cache,
        cache_view,
        slots,
        q_source,
        input_weights,
        positions,
        q,
        weights,
        lengths,
        schedule,
        request_blocks,
        expanded_blocks,
        row_requests,
        topk_output,
        workspace,
        logical_k,
        actual,
        expected,
    ]
    return BenchmarkCase(
        "deepseek_v4",
        implementation,
        tokens,
        context,
        requests,
        counts,
        V4_PAGE_SIZE,
        topk,
        score,
        run,
        correctness,
        tensors,
    )


def _make_v41_case(tokens: int, context: int, cc: tuple[int, int]) -> BenchmarkCase:
    requests, counts, row_requests = _request_layout(tokens)
    request_blocks = _block_tables(requests, context, V41_PAGE_SIZE)
    expanded_blocks = _expand_request_block_tables(request_blocks, row_requests)
    num_blocks = request_blocks.numel()

    k_source = torch.randn(
        (requests * context, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(context, device="cuda", dtype=torch.int64).repeat(requests)
    cache = torch.zeros(
        (num_blocks, V41_PAGE_SIZE, 68), device="cuda", dtype=torch.uint8
    )
    slots = _cache_slots(request_blocks, context, V41_PAGE_SIZE)
    norm_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rope = _identity_rope(context)
    indexer_k_norm_rope_store(
        k_source,
        positions,
        rope,
        norm_weight,
        1e-6,
        cache,
        slots,
        1,
        True,
    )

    q_source = torch.randn(
        (tokens, HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    input_weights = torch.randn((tokens, HEADS), device="cuda", dtype=torch.float32)
    q_positions = torch.full((tokens,), context - 1, device="cuda", dtype=torch.int64)
    (q_values, q_scales), weights = fused_v41_q_quant(
        q_positions,
        q_source,
        rope,
        input_weights,
        1.0,
        1.0,
        use_fp4=True,
    )
    lengths = torch.full((tokens,), context, device="cuda", dtype=torch.int32)
    logits_out = torch.empty((tokens, context), device="cuda", dtype=torch.float32)
    topk = min(2048, context)
    topk_output = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    workspace = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8)

    def score() -> torch.Tensor:
        return dsv41_mxfp4_dense_logits(
            q_values,
            q_scales,
            cache,
            weights,
            expanded_blocks,
            lengths,
            context,
            row_repeat=1,
            out=logits_out,
        )

    def run() -> None:
        logits = score()
        _run_topk(logits, lengths, topk_output, workspace, topk)

    actual = score()
    _run_topk(actual, lengths, topk_output, workspace, topk)
    rows = _representative_rows(tokens)
    logical_k = _dequantize_mxfp4_cache(cache, request_blocks, context)
    q_ref = _decode_mxfp4(
        q_values.index_select(0, rows), q_scales.index_select(0, rows)
    )
    requests_for_rows = row_requests.index_select(0, rows).long()
    k_ref = logical_k.index_select(0, requests_for_rows)
    w_ref = weights.index_select(0, rows)
    expected = (
        torch.einsum("rhd,rtd->rht", q_ref, k_ref).relu() * w_ref[:, :, None]
    ).sum(dim=1)
    checked = actual.index_select(0, rows)
    torch.testing.assert_close(checked, expected, rtol=2e-2, atol=2e-2)
    max_abs = float((checked - expected).abs().max())
    _check_topk(actual, topk_output, topk)
    correctness = {
        "representative_rows": rows.tolist(),
        "reference": "independent torch math over dequantized production inputs",
        "metric": "torch_assert_close",
        "rtol": 2e-2,
        "atol": 2e-2,
        "max_abs_error": max_abs,
        "topk_exact_set_match": True,
    }
    implementation = (
        f"custom_mxfp4_dense_logits_persistent_topk_flat_token_rows_sm{cc[0]}{cc[1]}"
    )
    tensors = [
        k_source,
        positions,
        cache,
        slots,
        norm_weight,
        rope,
        q_source,
        input_weights,
        q_positions,
        q_values,
        q_scales,
        weights,
        lengths,
        logits_out,
        request_blocks,
        expanded_blocks,
        row_requests,
        topk_output,
        workspace,
        logical_k,
        actual,
        expected,
    ]
    return BenchmarkCase(
        "deepseek_v4_1",
        implementation,
        tokens,
        context,
        requests,
        counts,
        V41_PAGE_SIZE,
        topk,
        score,
        run,
        correctness,
        tensors,
    )


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _measure(case: BenchmarkCase, samples: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        case.run()
    torch.accelerator.synchronize()
    times_ms = bench_gpu_time_with_cupti(
        case.run,
        dry_run_iters=warmup,
        repeat_iters=samples,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    times_us = [float(item) * 1000 for item in times_ms]
    return {
        "model": case.model,
        "implementation": case.implementation,
        "tokens": case.tokens,
        "context_tokens": case.context,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "requests": case.requests,
        "request_token_counts": case.request_token_counts,
        "query_shape": [case.tokens, HEADS, HEAD_DIM],
        "operator_query_shape": (
            [case.tokens, 1, HEADS, HEAD_DIM]
            if case.model == "deepseek_v4"
            else [case.tokens, HEADS, HEAD_DIM]
        ),
        "cache_page_size": case.page_size,
        "topk": case.topk,
        "input_dtype": "FP8 Q/K" if case.model == "deepseek_v4" else "MXFP4 Q/K",
        "output_dtype": "float32 logits + int32 indices",
        "samples": len(times_us),
        "median_us": _nearest_rank(times_us, 0.5),
        "p90_us": _nearest_rank(times_us, 0.9),
        "min_us": min(times_us),
        "max_us": max(times_us),
        "percentile_method": "nearest_rank",
        "correctness": case.correctness,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=_parse_ints, default=DEFAULT_TOKENS)
    parser.add_argument("--contexts", type=_parse_ints, default=DEFAULT_CONTEXTS)
    parser.add_argument("--modes", default="v4,v41")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    args.modes = tuple(item for item in args.modes.split(",") if item)
    if not args.modes or set(args.modes) - {"v4", "v41"}:
        parser.error("--modes must contain v4 and/or v41")
    if args.samples < 1 or args.warmup < 1:
        parser.error("--samples and --warmup must be positive")
    return args


def main() -> None:
    args = _parse_args()
    _check_request_layouts()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cupti_version = _require_cupti()
    torch.manual_seed(SEED)
    device = torch.accelerator.current_device_index()
    properties = torch.cuda.get_device_properties(device)
    cc = (properties.major, properties.minor)
    if cc not in ((8, 9), (12, 0)):
        raise RuntimeError(f"benchmark is scoped to SM89/SM120, got SM{cc[0]}{cc[1]}")

    repo = Path(__file__).resolve().parents[2]
    toolchain = _toolchain_metadata(properties.major * 10 + properties.minor)
    builders = {"v4": _make_v4_case, "v41": _make_v41_case}
    started = time.time()
    results: list[dict[str, Any]] = []
    for context in args.contexts:
        for tokens in args.tokens:
            for mode in args.modes:
                print(
                    f"checking and measuring {mode}: tokens={tokens}, "
                    f"context={context}",
                    file=sys.stderr,
                    flush=True,
                )
                case = builders[mode](tokens, context, cc)
                results.append(_measure(case, args.samples, args.warmup))
                del case
                torch.accelerator.empty_cache()

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "scope": "operator_only",
        "scope_notice": SCOPE_NOTICE,
        "comparison_notice": (
            "V4 FP8 and V4.1 MXFP4 results use different operators and layouts; "
            "do not interpret their ratio as a model or kernel speedup."
        ),
        "v41_coverage_notice": (
            "V4.1 measures dense logits + persistent top-k over flattened token "
            "rows. It excludes candidate-block selection and the candidate-consumer "
            "branch of the full production native indexer."
        ),
        "timed_boundary": (
            "CUPTI GPU-kernel span for production indexer logits immediately "
            "followed by torch.ops._C.persistent_topk"
        ),
        "excluded_from_timing": [
            "Q/K quantization",
            "paged-cache construction",
            "DeepGEMM scheduling metadata preparation",
            "host-side wrapper and allocation overhead",
            "compilation and explicit warmup",
        ],
        "timing": {
            "backend": "flashinfer.testing.bench_gpu_time_with_cupti",
            "cupti_python": cupti_version,
            "cuda_graph": True,
            "cold_l2_cache": True,
            "samples_per_point": args.samples,
            "explicit_warmup_calls": args.warmup,
        },
        "seed": SEED,
        "command": sys.argv,
        "git": _git_metadata(repo),
        "gpu": {
            "device_index": device,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": getattr(vllm, "__version__", None),
            "flashinfer": _package_version("flashinfer-python"),
            "toolchain": toolchain,
        },
        "production_sources": {
            "v4_wrapper": _source_record(fp8_fp4_paged_mqa_logits),
            "v4_q_quantizer": _source_record(fused_v4_q_quant),
            "v41_logits": _source_record(dsv41_mxfp4_dense_logits),
            "v41_q_quantizer": _source_record(fused_v41_q_quant),
            "v41_k_quantizer": _source_record(indexer_k_norm_rope_store),
            "benchmark": _source_record(main),
        },
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
