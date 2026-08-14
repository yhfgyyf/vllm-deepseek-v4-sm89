# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4 direct MQA logits tiles on SM80."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch
import triton

from vllm.models.deepseek_v4.nvidia.ops import sm12x_mqa
from vllm.utils.argparse_utils import FlexibleArgumentParser

NUM_HEADS = 64
HEAD_DIM = 128
BLOCK_N = 128
BLOCK_D = 128
LOGITS_BUDGET_BYTES = 256 * 1024 * 1024
DEFAULT_CONTEXTS = (98_304, 131_072, 155_648)


def make_inputs(m: int, n: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn(
            (m, NUM_HEADS, HEAD_DIM),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        .clamp_(-2.0, 2.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    k = (
        torch.randn(
            (n, HEAD_DIM),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        .clamp_(-2.0, 2.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scale = (
        torch.rand(n, device="cuda", dtype=torch.float32, generator=generator) * 0.02
        + 0.01
    )
    weights = torch.rand(
        (m, NUM_HEADS), device="cuda", dtype=torch.float32, generator=generator
    )
    weights /= weights.sum(dim=1, keepdim=True)
    row_starts = torch.zeros(m, device="cuda", dtype=torch.int32)
    row_ends = torch.full((m,), n, device="cuda", dtype=torch.int32)
    return q, k, scale, weights, row_starts, row_ends


def launch(
    inputs: tuple[torch.Tensor, ...],
    logits: torch.Tensor,
    block_m: int,
) -> None:
    q, k, scale, weights, row_starts, row_ends = inputs
    m, n = logits.shape
    grid = (math.ceil(m / block_m), math.ceil(n / BLOCK_N))
    sm12x_mqa._fp8_mqa_logits_kernel[grid](
        q,
        k,
        scale,
        weights,
        row_starts,
        row_ends,
        logits,
        m,
        n,
        NUM_HEADS,
        HEAD_DIM,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )


def time_kernel(
    inputs: tuple[torch.Tensor, ...],
    logits: torch.Tensor,
    block_m: int,
    repeats: int,
) -> tuple[float, list[float]]:
    for _ in range(2):
        launch(inputs, logits, block_m)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch(inputs, logits, block_m)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), samples


def benchmark_context(context: int, repeats: int, seed: int) -> dict[str, object]:
    n = context // 4
    m = LOGITS_BUDGET_BYTES // (n * torch.float32.itemsize)
    inputs = make_inputs(m, n, seed)
    outputs: dict[int, torch.Tensor] = {}
    medians: dict[int, float] = {}
    samples: dict[int, list[float]] = {}

    for block_m in (16, 64):
        logits = torch.empty((m, n), device="cuda", dtype=torch.float32)
        median_ms, samples_ms = time_kernel(inputs, logits, block_m, repeats)
        outputs[block_m] = logits
        medians[block_m] = median_ms
        samples[block_m] = samples_ms

    exact_equal = bool(torch.equal(outputs[16], outputs[64]))
    if not exact_equal:
        raise AssertionError(f"M16 and M64 logits differ for context {context}")

    return {
        "context_tokens": context,
        "compressed_n": n,
        "num_q_m": m,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "logits_bytes": m * n * torch.float32.itemsize,
        "block_m16": {"median_ms": medians[16], "samples_ms": samples[16]},
        "block_m64": {"median_ms": medians[64], "samples_ms": samples[64]},
        "m16_speedup": medians[64] / medians[16],
        "exact_equal": exact_equal,
    }


def main() -> None:
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an SM80 CUDA device")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if any(context <= 0 or context % 4 for context in args.contexts):
        parser.error("--contexts values must be positive and divisible by 4")

    results = [
        benchmark_context(context, args.repeats, args.seed) for context in args.contexts
    ]
    report = {
        "kind": "sm80_mqa_kernel_latency",
        "device": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "untimed_warmups": 2,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
