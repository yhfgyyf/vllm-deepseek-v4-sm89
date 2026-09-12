# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable SM89/SM120 MXFP4 kernels for the DeepSeek V4.1 indexer."""

import torch

from vllm.triton_utils import tl, triton

INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
PACKED_HEAD_DIM = 64
SCALE_GROUP_SIZE = 32
SCALES_PER_HEAD = 4
CANDIDATE_BLOCK_SIZE = 8
_LOGITS_BLOCK_N = 16
_CUDA_GRID_Y_LIMIT = 65_535

_INDEX_HEADS = tl.constexpr(32)
_INDEX_HEAD_DIM = tl.constexpr(128)
_PACKED_HEAD_DIM = tl.constexpr(64)
_SCALE_GROUP_SIZE = tl.constexpr(32)
_SCALES_PER_HEAD = tl.constexpr(4)
_CANDIDATE_BLOCK_SIZE = tl.constexpr(8)


@triton.jit
def _decode_e2m1(nibble):
    nibble = nibble.to(tl.uint32)
    code = nibble & 7
    normalized = (((code >> 1) + 126) << 23) | ((code & 1) << 22)
    magnitude_bits = tl.where(code == 0, 0, tl.where(code == 1, 0x3F000000, normalized))
    bits = magnitude_bits | ((nibble & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp4_indexer_logits_kernel(
    q_values,
    q_scales,
    k_cache,
    weights,
    block_tables,
    row_starts,
    row_ends,
    cu_seq_lens,
    token_to_seq,
    candidates,
    logits,
    qv_stride_row,
    qv_stride_head,
    qs_stride_row,
    qs_stride_head,
    weight_stride_row,
    block_table_stride,
    cache_block_stride,
    logits_stride_row,
    block_table_width,
    num_cache_blocks,
    width,
    PAGE_SIZE: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    PREFILL: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_STRIDE_ROW: tl.constexpr,
    SWAP_GRID_AXES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if SWAP_GRID_AXES:
        row = tl.program_id(1).to(tl.int64)
        col_tile = tl.program_id(0)
    else:
        row = tl.program_id(0).to(tl.int64)
        col_tile = tl.program_id(1)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, _INDEX_HEAD_DIM)
    heads = tl.arange(0, _INDEX_HEADS)

    if PREFILL:
        start = tl.load(row_starts + row)
        end = tl.load(row_ends + row)
        has_context = end > start
        req = tl.load(token_to_seq + start, mask=has_context, other=0).to(tl.int64)
        if USE_CANDIDATES:
            candidate_slot = cols // _CANDIDATE_BLOCK_SIZE
            candidate_offset = cols % _CANDIDATE_BLOCK_SIZE
            candidate_block = tl.load(
                candidates + row * CANDIDATE_STRIDE_ROW + candidate_slot,
                mask=cols < width,
                other=-1,
            ).to(tl.int64)
            token = candidate_block * _CANDIDATE_BLOCK_SIZE + candidate_offset
            valid = (
                (cols < width)
                & has_context
                & (candidate_block >= 0)
                & (token < end - start)
            )
        else:
            packed_col = cols
            valid = (packed_col >= start) & (packed_col < end) & (packed_col < width)
            req = tl.load(token_to_seq + packed_col, mask=valid, other=0).to(tl.int64)
            req_start = tl.load(cu_seq_lens + req, mask=valid, other=0)
            token = packed_col - req_start
    else:
        req = row // ROW_REPEAT
        end = tl.load(row_ends + row)
        if USE_CANDIDATES:
            candidate_slot = cols // _CANDIDATE_BLOCK_SIZE
            candidate_offset = cols % _CANDIDATE_BLOCK_SIZE
            candidate_block = tl.load(
                candidates + row * CANDIDATE_STRIDE_ROW + candidate_slot,
                mask=cols < width,
                other=-1,
            ).to(tl.int64)
            token = candidate_block * _CANDIDATE_BLOCK_SIZE + candidate_offset
            valid = (cols < width) & (candidate_block >= 0) & (token < end)
        else:
            token = cols
            valid = (cols < width) & (token < end)

    logical_block = token // PAGE_SIZE
    block_offset = token % PAGE_SIZE
    valid = valid & (token >= 0) & (logical_block < block_table_width)
    physical_block = tl.load(
        block_tables + req * block_table_stride + logical_block,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid = valid & (physical_block >= 0) & (physical_block < num_cache_blocks)

    packed_dim = dims // 2
    q_bytes = tl.load(
        q_values
        + row * qv_stride_row
        + heads[:, None] * qv_stride_head
        + packed_dim[None, :]
    )
    q_nibbles = tl.where((dims[None, :] & 1) == 0, q_bytes & 0xF, q_bytes >> 4)
    q_scale_bytes = tl.load(
        q_scales
        + row * qs_stride_row
        + heads[:, None] * qs_stride_head
        + (dims[None, :] // _SCALE_GROUP_SIZE)
    )
    q = _decode_e2m1(q_nibbles) * tl.exp2(q_scale_bytes.to(tl.float32) - 127.0)

    value_offsets = (
        physical_block[None, :] * cache_block_stride
        + block_offset[None, :] * _PACKED_HEAD_DIM
        + packed_dim[:, None]
    )
    k_bytes = tl.load(k_cache + value_offsets, mask=valid[None, :], other=0)
    k_nibbles = tl.where((dims[:, None] & 1) == 0, k_bytes & 0xF, k_bytes >> 4)
    scale_offsets = (
        physical_block[None, :] * cache_block_stride
        + PAGE_SIZE * _PACKED_HEAD_DIM
        + block_offset[None, :] * _SCALES_PER_HEAD
        + (dims[:, None] // _SCALE_GROUP_SIZE)
    )
    k_scale_bytes = tl.load(k_cache + scale_offsets, mask=valid[None, :], other=0)
    k = _decode_e2m1(k_nibbles) * tl.exp2(k_scale_bytes.to(tl.float32) - 127.0)

    head_scores = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16))
    head_weights = tl.load(weights + row * weight_stride_row + heads)
    scores = tl.sum(tl.maximum(head_scores, 0.0) * head_weights[:, None], axis=0)
    tl.store(
        logits + row * logits_stride_row + cols,
        tl.where(valid, scores, -float("inf")),
        mask=cols < width,
    )


@triton.jit
def _map_candidate_topk_kernel(
    compact_indices,
    compact_logits,
    candidates,
    stride_indices_row,
    stride_logits_row,
    stride_candidates_row,
    compact_width,
    candidate_width,
    TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, PADDED_TOPK)
    compact = tl.load(
        compact_indices + row * stride_indices_row + offsets,
        mask=offsets < TOPK,
        other=-1,
    )
    valid_compact = (compact >= 0) & (compact < compact_width)
    score = tl.load(
        compact_logits + row * stride_logits_row + compact,
        mask=(offsets < TOPK) & valid_compact,
        other=-float("inf"),
    )
    candidate_slot = compact // _CANDIDATE_BLOCK_SIZE
    candidate_offset = compact % _CANDIDATE_BLOCK_SIZE
    block = tl.load(
        candidates + row * stride_candidates_row + candidate_slot,
        mask=(offsets < TOPK) & valid_compact & (candidate_slot < candidate_width),
        other=-1,
    ).to(tl.int64)
    token = block * _CANDIDATE_BLOCK_SIZE + candidate_offset
    token = tl.where(valid_compact & (block >= 0) & (score > -float("inf")), token, -1)
    tl.store(
        compact_indices + row * stride_indices_row + offsets,
        token,
        mask=offsets < TOPK,
    )


def _validate_inputs(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
) -> None:
    if not q_values.is_cuda:
        raise RuntimeError("DeepSeek V4.1 native indexer requires CUDA tensors")
    if q_values.dtype != torch.uint8 or q_values.shape[-2:] != (
        INDEX_HEADS,
        PACKED_HEAD_DIM,
    ):
        raise ValueError("q_values must be uint8 [..., 32, 64]")
    if q_values.stride(-1) != 1:
        raise ValueError("q_values packed dimension must be contiguous")
    if q_scales.dtype != torch.uint8 or q_scales.shape[-2:] != (
        INDEX_HEADS,
        SCALES_PER_HEAD,
    ):
        raise ValueError("q_scales must be uint8 [..., 32, 4]")
    if q_scales.stride(-1) != 1:
        raise ValueError("q_scales group dimension must be contiguous")
    if k_cache.dtype != torch.uint8 or k_cache.ndim != 3 or k_cache.shape[2] != 68:
        raise ValueError("k_cache must be uint8 [blocks, page_size, 68]")
    if k_cache.shape[1] not in (64, 128):
        raise ValueError("DeepSeek V4.1 indexer page size must be 64 or 128")
    if k_cache.stride(1) != 68 or k_cache.stride(2) != 1:
        raise ValueError("k_cache rows must be contiguous within each page")
    if weights.dtype != torch.float32 or weights.shape[-1] != INDEX_HEADS:
        raise ValueError("weights must be float32 [..., 32]")
    if weights.stride(-1) != 1:
        raise ValueError("weights head dimension must be contiguous")
    if block_tables.dtype != torch.int32 or block_tables.ndim != 2:
        raise ValueError("block_tables must be int32 [requests, blocks]")
    if block_tables.stride(1) != 1:
        raise ValueError("block_tables rows must be contiguous")


def _prepare_logits_out(
    q_values: torch.Tensor,
    rows: int,
    width: int,
    out: torch.Tensor | None,
) -> torch.Tensor:
    if out is None:
        return torch.empty((rows, width), device=q_values.device, dtype=torch.float32)
    if (
        out.device != q_values.device
        or out.dtype != torch.float32
        or out.shape != (rows, width)
        or out.stride(1) != 1
    ):
        raise ValueError(
            "logits output must be contiguous fp32 with shape "
            f"({rows}, {width}) on {q_values.device}"
        )
    return out


def _logits_launch_grid(rows: int, width: int) -> tuple[tuple[int, int], bool]:
    col_tiles = triton.cdiv(width, _LOGITS_BLOCK_N)
    swap_grid_axes = col_tiles > _CUDA_GRID_Y_LIMIT
    if swap_grid_axes and rows > _CUDA_GRID_Y_LIMIT:
        raise ValueError(
            "DeepSeek V4.1 native indexer cannot launch with both "
            f"rows={rows} and column tiles={col_tiles} above "
            f"the CUDA grid-y limit {_CUDA_GRID_Y_LIMIT}"
        )
    grid = (col_tiles, rows) if swap_grid_axes else (rows, col_tiles)
    return grid, swap_grid_axes


def dsv41_mxfp4_dense_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    row_ends: torch.Tensor,
    width: int,
    *,
    row_starts: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    token_to_seq: torch.Tensor | None = None,
    row_repeat: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute dense V4.1 index scores directly from segregated MXFP4 pages."""
    _validate_inputs(q_values, q_scales, k_cache, weights, block_tables)
    prefill = row_starts is not None
    if prefill != (cu_seq_lens is not None and token_to_seq is not None):
        raise ValueError(
            "prefill requires row_starts, cu_seq_lens, and token_to_seq together"
        )
    q_values = q_values.reshape(-1, INDEX_HEADS, PACKED_HEAD_DIM)
    q_scales = q_scales.reshape(-1, INDEX_HEADS, SCALES_PER_HEAD)
    weights = weights.reshape(-1, INDEX_HEADS)
    rows = q_values.shape[0]
    if row_ends.numel() != rows:
        raise ValueError(f"row_ends has {row_ends.numel()} entries for {rows} rows")
    if row_ends.stride(0) != 1 or (
        row_starts is not None and row_starts.stride(0) != 1
    ):
        raise ValueError("row bounds must be contiguous")
    logits = _prepare_logits_out(q_values, rows, width, out)
    if rows == 0 or width == 0:
        return logits
    grid, swap_grid_axes = _logits_launch_grid(rows, width)
    _mxfp4_indexer_logits_kernel[grid](
        q_values,
        q_scales,
        k_cache,
        weights,
        block_tables,
        row_starts,
        row_ends,
        cu_seq_lens,
        token_to_seq,
        None,
        logits,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        q_scales.stride(1),
        weights.stride(0),
        block_tables.stride(0),
        k_cache.stride(0),
        logits.stride(0),
        block_tables.shape[1],
        k_cache.shape[0],
        width,
        PAGE_SIZE=k_cache.shape[1],
        ROW_REPEAT=row_repeat,
        PREFILL=prefill,
        USE_CANDIDATES=False,
        CANDIDATE_STRIDE_ROW=0,
        SWAP_GRID_AXES=swap_grid_axes,
        BLOCK_N=_LOGITS_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return logits


def dsv41_mxfp4_candidate_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    row_ends: torch.Tensor,
    candidates: torch.Tensor,
    *,
    row_starts: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    token_to_seq: torch.Tensor | None = None,
    row_repeat: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score only the token positions named by request-local candidate blocks."""
    _validate_inputs(q_values, q_scales, k_cache, weights, block_tables)
    if candidates.dtype != torch.int32 or candidates.ndim != 2:
        raise ValueError("candidates must be int32 [rows, blocks]")
    if candidates.stride(1) != 1:
        raise ValueError("candidate rows must be contiguous")
    q_values = q_values.reshape(-1, INDEX_HEADS, PACKED_HEAD_DIM)
    q_scales = q_scales.reshape(-1, INDEX_HEADS, SCALES_PER_HEAD)
    weights = weights.reshape(-1, INDEX_HEADS)
    rows = q_values.shape[0]
    if candidates.shape[0] != rows or row_ends.numel() != rows:
        raise ValueError("candidate and length rows must match query rows")
    if row_ends.stride(0) != 1 or (
        row_starts is not None and row_starts.stride(0) != 1
    ):
        raise ValueError("row bounds must be contiguous")
    prefill = row_starts is not None
    if prefill != (cu_seq_lens is not None and token_to_seq is not None):
        raise ValueError(
            "prefill requires row_starts, cu_seq_lens, and token_to_seq together"
        )
    width = candidates.shape[1] * CANDIDATE_BLOCK_SIZE
    logits = _prepare_logits_out(q_values, rows, width, out)
    if rows == 0 or width == 0:
        return logits
    grid, swap_grid_axes = _logits_launch_grid(rows, width)
    _mxfp4_indexer_logits_kernel[grid](
        q_values,
        q_scales,
        k_cache,
        weights,
        block_tables,
        row_starts,
        row_ends,
        cu_seq_lens,
        token_to_seq,
        candidates,
        logits,
        q_values.stride(0),
        q_values.stride(1),
        q_scales.stride(0),
        q_scales.stride(1),
        weights.stride(0),
        block_tables.stride(0),
        k_cache.stride(0),
        logits.stride(0),
        block_tables.shape[1],
        k_cache.shape[0],
        width,
        PAGE_SIZE=k_cache.shape[1],
        ROW_REPEAT=row_repeat,
        PREFILL=prefill,
        USE_CANDIDATES=True,
        CANDIDATE_STRIDE_ROW=candidates.stride(0),
        SWAP_GRID_AXES=swap_grid_axes,
        BLOCK_N=_LOGITS_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return logits


def map_candidate_topk_(
    compact_indices: torch.Tensor,
    compact_logits: torch.Tensor,
    candidates: torch.Tensor,
) -> None:
    """Map compact candidate offsets to request-local compressed-token IDs."""
    if (
        compact_indices.dtype != torch.int32
        or compact_indices.ndim != 2
        or compact_logits.dtype != torch.float32
        or compact_logits.ndim != 2
        or candidates.dtype != torch.int32
        or candidates.ndim != 2
    ):
        raise ValueError(
            "candidate mapping requires int32 indices/candidates and fp32 logits"
        )
    rows, topk = compact_indices.shape
    if compact_logits.shape[0] != rows or candidates.shape[0] != rows:
        raise ValueError("candidate mapping row counts must match")
    if compact_logits.shape[1] != candidates.shape[1] * CANDIDATE_BLOCK_SIZE:
        raise ValueError("compact logit width must equal candidate blocks times 8")
    if topk == 0 or rows == 0:
        return
    if (
        compact_indices.stride(1) != 1
        or compact_logits.stride(1) != 1
        or candidates.stride(1) != 1
    ):
        raise ValueError("top-k, logits, and candidate rows must be contiguous")
    _map_candidate_topk_kernel[(rows,)](
        compact_indices,
        compact_logits,
        candidates,
        compact_indices.stride(0),
        compact_logits.stride(0),
        candidates.stride(0),
        compact_logits.shape[1],
        candidates.shape[1],
        TOPK=topk,
        PADDED_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "dsv41_mxfp4_candidate_logits",
    "dsv41_mxfp4_dense_logits",
    "map_candidate_topk_",
]
