# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable SM89/SM120 MXFP4 kernels for the DeepSeek V4.1 indexer.

On SM120 the logits kernel feeds the packed e2m1 bytes and ue8m0 scales of q and K
straight into the block-scaled FP4 tensor-core MMA through `tl.dot_scaled`
(`mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`); nothing is decoded. Other
architectures keep the inline decode to BF16 and `tl.dot`. Products are exact in
both paths (e2m1 times a power of two is representable), so the two differ only in
fp32 accumulation order; the top-k sets they produce are identical on the shapes
covered by the tests."""

import functools
import os

import torch

from vllm.triton_utils import tl, triton

INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
PACKED_HEAD_DIM = 64
SCALE_GROUP_SIZE = 32
SCALES_PER_HEAD = 4
CANDIDATE_BLOCK_SIZE = 8
# Tile and pipeline configuration per code path. The FP4 tensor-core path has
# no per-element decode, so wide tiles amortize the address and page-table work
# (128 columns, one stage measured best on RTX PRO 6000). SM89 also benefits
# from that tile, especially for candidate scoring with a few query rows.
_LOGITS_BLOCK_N = 16
_LOGITS_NUM_STAGES = 2
_LOGITS_BLOCK_N_FP4 = 128
_LOGITS_NUM_STAGES_FP4 = 1
# Each program strides over a row's column tiles, so the grid does not depend
# on the row's context and CUDA graph replays stay valid. The number of
# programs per row targets a fixed total: few rows (decode) get many programs
# for parallelism, many rows (prefill) get few so q is loaded once per row and
# the K tiles are shared through L2 across the rows scoring the same tile.
_LOGITS_TARGET_PROGRAMS = 16_384
_LOGITS_MIN_TILE_PROGRAMS = 8
_LOGITS_NUM_WARPS = 4
_CUDA_GRID_Y_LIMIT = 65_535

_INDEX_HEADS = tl.constexpr(32)
_INDEX_HEAD_DIM = tl.constexpr(128)
_PACKED_HEAD_DIM = tl.constexpr(64)
_SCALE_GROUP_SIZE = tl.constexpr(32)
_SCALES_PER_HEAD = tl.constexpr(4)
_CANDIDATE_BLOCK_SIZE = tl.constexpr(8)
_FILL_BLOCK = tl.constexpr(256)


@triton.jit
def _decode_e2m1(nibble):
    nibble = nibble.to(tl.uint32)
    code = nibble & 7
    normalized = (((code >> 1) + 126) << 23) | ((code & 1) << 22)
    magnitude_bits = tl.where(code == 0, 0, tl.where(code == 1, 0x3F000000, normalized))
    bits = magnitude_bits | ((nibble & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _mxfp4_indexer_q_decode_kernel(
    q_values,
    q_scales,
    q_decoded,
    qv_stride_row,
    qv_stride_head,
    qs_stride_row,
    qs_stride_head,
):
    # One program per row: decode 32 heads x 128 dims of MXFP4 q to BF16 once,
    # with exactly the arithmetic the logits kernel uses inline.
    row = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, _INDEX_HEAD_DIM)
    heads = tl.arange(0, _INDEX_HEADS)
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
    tl.store(
        q_decoded
        + row * (_INDEX_HEADS * _INDEX_HEAD_DIM)
        + heads[:, None] * _INDEX_HEAD_DIM
        + dims[None, :],
        q.to(tl.bfloat16),
    )


@triton.jit
def _mxfp4_indexer_logits_kernel(
    q_values,
    q_scales,
    q_decoded,
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
    BLOCK_N: tl.constexpr,
    Q_PREDECODED: tl.constexpr,
    FP4_DOT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    first_tile = tl.program_id(1)
    tile_programs = tl.num_programs(1)
    dims = tl.arange(0, _INDEX_HEAD_DIM)
    heads = tl.arange(0, _INDEX_HEADS)

    # Per-row bounds. Dense rows score only [first_col, stop); the columns
    # outside get -inf from the fill loops below, so the output equals the
    # one-tile-per-program kernel bit for bit while the work follows the
    # row's context instead of the launch width.
    if PREFILL:
        start = tl.load(row_starts + row)
        end = tl.load(row_ends + row)
        has_context = end > start
        req = tl.load(token_to_seq + start, mask=has_context, other=0).to(tl.int64)
        if USE_CANDIDATES:
            first_col = 0
            stop = width
        else:
            first_col = tl.minimum(start, width)
            stop = tl.minimum(end, width)
    else:
        req = row // ROW_REPEAT
        end = tl.load(row_ends + row)
        first_col = 0
        stop = width if USE_CANDIDATES else tl.minimum(end, width)
    first_tile_of_row = first_col // BLOCK_N
    stop = tl.maximum(stop, first_tile_of_row * BLOCK_N)
    last_tile = (stop + BLOCK_N - 1) // BLOCK_N
    my_first_tile = first_tile_of_row + first_tile
    num_tiles = (last_tile - my_first_tile + tile_programs - 1) // tile_programs

    if FP4_DOT:
        # q stays packed: [32 heads, 64 bytes] e2m1 pairs (even dim in the low
        # nibble) and [32, 4] ue8m0 scales, the layouts tl.dot_scaled expects.
        packed = tl.arange(0, _PACKED_HEAD_DIM)
        grp = tl.arange(0, _SCALES_PER_HEAD)
        q_bytes = tl.load(
            q_values
            + row * qv_stride_row
            + heads[:, None] * qv_stride_head
            + packed[None, :]
        )
        q_scale_bytes = tl.load(
            q_scales
            + row * qs_stride_row
            + heads[:, None] * qs_stride_head
            + grp[None, :]
        )
    else:
        packed_dim = dims // 2
        if Q_PREDECODED:
            # BF16 q written by _mxfp4_indexer_q_decode_kernel; equal to the inline
            # decode below after its cast to BF16 for tl.dot.
            q = tl.load(
                q_decoded
                + row * (_INDEX_HEADS * _INDEX_HEAD_DIM)
                + heads[:, None] * _INDEX_HEAD_DIM
                + dims[None, :]
            )
        else:
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
    head_weights = tl.load(weights + row * weight_stride_row + heads)

    for i in range(0, num_tiles):
        col_tile = my_first_tile + i * tile_programs
        cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        if PREFILL:
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
                tile_req = req
            else:
                packed_col = cols
                valid = (
                    (packed_col >= start) & (packed_col < end) & (packed_col < width)
                )
                tile_req = tl.load(token_to_seq + packed_col, mask=valid, other=0).to(
                    tl.int64
                )
                req_start = tl.load(cu_seq_lens + tile_req, mask=valid, other=0)
                token = packed_col - req_start
        else:
            tile_req = req
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
            block_tables + tile_req * block_table_stride + logical_block,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (physical_block >= 0) & (physical_block < num_cache_blocks)

        if FP4_DOT:
            # K stays packed too: [64 bytes, BLOCK_N] values and [BLOCK_N, 4]
            # scales (dot_scaled wants rhs scales as [N, K // 32]).
            value_offsets = (
                physical_block[None, :] * cache_block_stride
                + block_offset[None, :] * _PACKED_HEAD_DIM
                + packed[:, None]
            )
            k_bytes = tl.load(k_cache + value_offsets, mask=valid[None, :], other=0)
            scale_offsets = (
                physical_block[:, None] * cache_block_stride
                + PAGE_SIZE * _PACKED_HEAD_DIM
                + block_offset[:, None] * _SCALES_PER_HEAD
                + grp[None, :]
            )
            k_scale_bytes = tl.load(
                k_cache + scale_offsets, mask=valid[:, None], other=127
            )
            head_scores = tl.dot_scaled(
                q_bytes, q_scale_bytes, "e2m1", k_bytes, k_scale_bytes, "e2m1"
            )
        else:
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
            k_scale_bytes = tl.load(
                k_cache + scale_offsets, mask=valid[None, :], other=0
            )
            k = _decode_e2m1(k_nibbles) * tl.exp2(k_scale_bytes.to(tl.float32) - 127.0)

            head_scores = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16))
        scores = tl.sum(tl.maximum(head_scores, 0.0) * head_weights[:, None], axis=0)
        tl.store(
            logits + row * logits_stride_row + cols,
            tl.where(valid, scores, -float("inf")),
            mask=cols < width,
        )

    if not USE_CANDIDATES:
        # Columns outside [first_tile_of_row * BLOCK_N, last_tile * BLOCK_N) hold
        # no keys for this row: write -inf without touching the cache.
        fill_cols = tl.arange(0, _FILL_BLOCK)
        neg_inf = tl.full([_FILL_BLOCK], -float("inf"), tl.float32)
        tail_start = last_tile * BLOCK_N + first_tile * _FILL_BLOCK
        fill_step = tile_programs * _FILL_BLOCK
        num_tail = (width - tail_start + fill_step - 1) // fill_step
        for i in range(0, num_tail):
            cols = tail_start + i * fill_step + fill_cols
            tl.store(
                logits + row * logits_stride_row + cols, neg_inf, mask=cols < width
            )
        if PREFILL:
            head_stop = first_tile_of_row * BLOCK_N
            head_start = first_tile * _FILL_BLOCK
            num_head = (head_stop - head_start + fill_step - 1) // fill_step
            for i in range(0, num_head):
                cols = head_start + i * fill_step + fill_cols
                tl.store(
                    logits + row * logits_stride_row + cols,
                    neg_inf,
                    mask=cols < head_stop,
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


@functools.cache
def _q_predecode_for_device(device_index: int) -> bool:
    major, _ = torch.cuda.get_device_capability(device_index)
    return major == 12


def _use_q_predecode(device: torch.device) -> bool:
    """Decode q once per row in a separate kernel instead of in every program.

    Only used when the FP4 tensor-core path is off (non-SM120 targets or
    VLLM_DSV41_INDEXER_FP4_DOT=0). On SM120 each logits program otherwise
    re-decodes the whole MXFP4 q (32 x 128 values) to score only BLOCK_N keys;
    predecoding is 1.6-1.8x faster on production shapes with bit-identical
    output. Other architectures keep the inline decode until measured.
    """
    if device.type != "cuda":
        return False
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    return _q_predecode_for_device(index)


def _predecode_q(q_values: torch.Tensor, q_scales: torch.Tensor) -> torch.Tensor:
    rows = q_values.shape[0]
    q_decoded = torch.empty(
        (rows, INDEX_HEADS, INDEX_HEAD_DIM),
        device=q_values.device,
        dtype=torch.bfloat16,
    )
    if rows:
        _mxfp4_indexer_q_decode_kernel[(rows,)](
            q_values,
            q_scales,
            q_decoded,
            q_values.stride(0),
            q_values.stride(1),
            q_scales.stride(0),
            q_scales.stride(1),
            num_warps=4,
        )
    return q_decoded


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


@functools.cache
def _fp4_dot_for_device(device_index: int) -> bool:
    major, _ = torch.cuda.get_device_capability(device_index)
    return major == 12 and hasattr(tl, "dot_scaled")


def _use_fp4_dot(device: torch.device) -> bool:
    """Score with the block-scaled FP4 tensor-core MMA instead of decoding.

    SM120 lowers tl.dot_scaled(e2m1, e2m1) to a native MMA; on other targets
    Triton would emulate it by upcasting, which is slower than the inline decode.
    VLLM_DSV41_INDEXER_FP4_DOT=0 forces the decode path for bisecting.
    """
    if (
        device.type != "cuda"
        or os.environ.get("VLLM_DSV41_INDEXER_FP4_DOT", "1") == "0"
    ):
        return False
    index = (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )
    return _fp4_dot_for_device(index)


def _logits_config(device: torch.device) -> tuple[bool, int, int]:
    """(fp4_dot, BLOCK_N, num_stages) for the logits kernel on this device."""
    if _use_fp4_dot(device):
        return True, _LOGITS_BLOCK_N_FP4, _LOGITS_NUM_STAGES_FP4
    if device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 9):
        return False, 128, 1
    return False, _LOGITS_BLOCK_N, _LOGITS_NUM_STAGES


def _logits_launch_grid(
    rows: int, width: int, block_n: int = _LOGITS_BLOCK_N
) -> tuple[int, int]:
    col_tiles = triton.cdiv(width, block_n)
    per_row = max(_LOGITS_MIN_TILE_PROGRAMS, _LOGITS_TARGET_PROGRAMS // max(rows, 1))
    tile_programs = max(1, min(col_tiles, per_row, _CUDA_GRID_Y_LIMIT))
    return rows, tile_programs


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
    fp4_dot, block_n, num_stages = _logits_config(q_values.device)
    grid = _logits_launch_grid(rows, width, block_n)
    q_predecoded = not fp4_dot and _use_q_predecode(q_values.device)
    q_decoded = _predecode_q(q_values, q_scales) if q_predecoded else q_values
    _mxfp4_indexer_logits_kernel[grid](
        q_values,
        q_scales,
        q_decoded,
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
        BLOCK_N=block_n,
        Q_PREDECODED=q_predecoded,
        FP4_DOT=fp4_dot,
        num_warps=_LOGITS_NUM_WARPS,
        num_stages=num_stages,
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
    fp4_dot, block_n, num_stages = _logits_config(q_values.device)
    grid = _logits_launch_grid(rows, width, block_n)
    q_predecoded = not fp4_dot and _use_q_predecode(q_values.device)
    q_decoded = _predecode_q(q_values, q_scales) if q_predecoded else q_values
    _mxfp4_indexer_logits_kernel[grid](
        q_values,
        q_scales,
        q_decoded,
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
        BLOCK_N=block_n,
        Q_PREDECODED=q_predecoded,
        FP4_DOT=fp4_dot,
        num_warps=_LOGITS_NUM_WARPS,
        num_stages=num_stages,
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
