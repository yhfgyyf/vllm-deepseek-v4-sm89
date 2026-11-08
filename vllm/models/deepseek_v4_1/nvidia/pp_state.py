# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact cache deltas for the two sharing cuts in balanced V4.1 PP4."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _copy_cache_rows(
    cache,
    slots,
    payload,
    num_slots,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    RESTORE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    slot = tl.load(slots + row, mask=row < num_slots, other=-1)
    valid = (slot >= 0) & (slot < CACHE_ROWS)
    cache_row = cache + slot // PAGE_ROWS * PAGE_STRIDE + slot % PAGE_ROWS * ROW_STRIDE
    packet_row = payload + row * WIDTH
    if RESTORE:
        value = tl.load(packet_row + offsets, mask=offsets < WIDTH, other=0)
        tl.store(cache_row + offsets, value, mask=valid & (offsets < WIDTH))
    else:
        value = tl.load(cache_row + offsets, mask=valid & (offsets < WIDTH), other=0)
        tl.store(packet_row + offsets, value, mask=offsets < WIDTH)


def copy_cache_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    payload: torch.Tensor,
    *,
    restore: bool,
) -> None:
    """Gather/scatter packed bytes using each stage's own physical slot mapping."""
    if payload.shape[0] == 0:
        return
    assert cache.dtype == payload.dtype == torch.uint8
    assert cache.ndim == 3 and payload.ndim == 2
    assert cache.shape[-1] == payload.shape[-1]
    assert cache.stride(-1) == 1 and payload.is_contiguous()
    _copy_cache_rows[(payload.shape[0],)](
        cache,
        slots,
        payload,
        slots.numel(),
        PAGE_STRIDE=cache.stride(0),
        ROW_STRIDE=cache.stride(1),
        PAGE_ROWS=cache.shape[1],
        CACHE_ROWS=cache.shape[0] * cache.shape[1],
        WIDTH=payload.shape[1],
        BLOCK=triton.next_power_of_2(payload.shape[1]),
        RESTORE=restore,
    )


def pp_payload_shapes(
    start_layer: int, index_topk: int, candidate_topk: int, index_cache_width: int
):
    """Incoming fields; all normal decode fields have one row per scheduled token."""
    if start_layer not in (10, 30, 31, 32):
        return {}
    fields: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "pp_kv": ((288,), torch.uint8),
    }
    if start_layer in (10, 30, 31):
        fields["pp_topk"] = ((index_topk,), torch.int32)
    if start_layer in (30, 31, 32):
        fields.update(
            pp_index_k=((index_cache_width,), torch.uint8),
            pp_candidates=((candidate_topk,), torch.int32),
            pp_token_ids=((), torch.int64),
        )
    return fields
