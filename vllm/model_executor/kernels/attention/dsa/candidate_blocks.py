# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native candidate-block selection for the DeepSeek V4.1 indexer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "num_blocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    block_lens,
    scores_stride_row,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    bound_row = row // ROW_REPEAT
    start = tl.load(starts + bound_row * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + bound_row * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    valid_blocks = blocks < num_blocks
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        mask=(
            valid_blocks[:, None]
            & (offsets[None, :] < BLOCK_SIZE)
            & (cols < end)
            & (cols < width)
        ),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    block_len = tl.cdiv(tl.maximum(end - start, 0), BLOCK_SIZE)
    # Always retain the newest (possibly partial) causal block.
    reduced = tl.where(
        (block_len > 0) & (blocks == block_len - 1), float("inf"), reduced
    )
    tl.store(scores + row * scores_stride_row + blocks, reduced, valid_blocks)
    if tl.program_id(1) == 0:
        tl.store(block_lens + row, block_len)


def select_candidate_blocks(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    workspace: torch.Tensor,
    row_repeat: int = 1,
    scores_out: torch.Tensor | None = None,
    block_lens_out: torch.Tensor | None = None,
) -> None:
    """Select request-local blocks using max token score per block.

    The newest causal block is pinned, invalid output entries are ``-1``, and
    ties use vLLM's native persistent-top-k policy.
    """
    if not logits.is_cuda:
        raise RuntimeError("candidate-block selection requires CUDA tensors")
    rows, width = logits.shape
    if out.shape != (rows, topk_blocks) or out.dtype != torch.int32:
        raise ValueError(
            "candidate output must be int32 with shape "
            f"({rows}, {topk_blocks}), got {out.dtype} {tuple(out.shape)}"
        )
    if block_size <= 0 or row_repeat <= 0:
        raise ValueError("block_size and row_repeat must be positive")
    if topk_blocks not in (512, 1024, 2048):
        raise ValueError(
            "native candidate selection requires 512, 1024, or 2048 blocks"
        )
    if rows == 0:
        return
    if width == 0:
        out.fill_(-1)
        return

    num_blocks = triton.cdiv(width, block_size)
    # persistent_topk handles short rows, but its physical width still has to
    # cover K. Pad the score matrix to K without changing valid lengths.
    score_width = topk_blocks if topk_blocks > num_blocks else num_blocks
    if scores_out is None:
        scores = logits.new_full((rows, score_width), -float("inf"))
    else:
        if (
            scores_out.shape != (rows, score_width)
            or scores_out.dtype != torch.float32
            or scores_out.device != logits.device
            or scores_out.stride(1) != 1
        ):
            raise ValueError(
                "candidate score output must be contiguous fp32 with shape "
                f"({rows}, {score_width})"
            )
        scores = scores_out
        scores.fill_(-float("inf"))
    if block_lens_out is None:
        block_lens = torch.empty(rows, device=logits.device, dtype=torch.int32)
    else:
        if (
            block_lens_out.shape != (rows,)
            or block_lens_out.dtype != torch.int32
            or block_lens_out.device != logits.device
            or not block_lens_out.is_contiguous()
        ):
            raise ValueError(
                f"block lengths output must be contiguous int32 with shape ({rows},)"
            )
        block_lens = block_lens_out
    _block_scores_kernel[(rows, triton.cdiv(num_blocks, 128))](
        logits,
        row_starts,
        row_ends,
        scores,
        block_lens,
        scores.stride(0),
        *logits.stride(),
        row_starts.stride(0) if row_starts is not None else 0,
        row_ends.stride(0),
        width,
        num_blocks,
        BLOCK_SIZE=block_size,
        HAS_STARTS=row_starts is not None,
        ROW_REPEAT=row_repeat,
        BLOCKS_PER_PROGRAM=128,
        num_warps=4,
    )
    torch.ops._C.persistent_topk(
        scores,
        block_lens,
        out,
        workspace,
        topk_blocks,
        score_width,
    )


__all__ = ["select_candidate_blocks"]
