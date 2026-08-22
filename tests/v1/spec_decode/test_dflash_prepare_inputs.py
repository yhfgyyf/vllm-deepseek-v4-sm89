# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _run_prepare(*, target_positions: list[int], block_table_values: list[int]):
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_speculative_steps = 3
    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    query_slots = torch.full((max_num_tokens,), -2, dtype=torch.int64, device=device)
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slots = torch.full((max_num_tokens,), -2, dtype=torch.int64, device=device)
    sample_indices = torch.full(
        (max_num_reqs * num_speculative_steps,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    sample_pos = torch.full_like(sample_indices, -1)
    sample_idx_mapping = torch.full(
        sample_indices.shape, -1, dtype=torch.int32, device=device
    )
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.zeros_like(last_sampled)
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch,
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        block_table,
        4,
        123,
        num_speculative_steps,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=True,
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slots=query_slots.cpu(),
        context_positions=context_positions.cpu(),
        context_slots=context_slots.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
    )


def test_prepare_dflash_inputs_excludes_rejected_context_suffix():
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_positions[:4].tolist() == [10, 11, 0, 0]
    assert out.context_slots[:4].tolist() == [30, 31, PAD_SLOT_ID, PAD_SLOT_ID]
    assert out.input_buffers.input_ids[:3].cpu().tolist() == [99, 123, 123]
    assert out.input_buffers.positions[:3].cpu().tolist() == [12, 13, 14]
    assert out.query_slots[:3].tolist() == [32, 33, 34]
    assert out.sample_indices[:3].tolist() == [0, 1, 2]
    assert out.sample_pos[:3].tolist() == [13, 14, 15]
    assert out.sample_idx_mapping[:3].tolist() == [2, 2, 2]


def test_prepare_dflash_inputs_never_writes_the_null_block():
    out = _run_prepare(
        target_positions=[2, 3, 4, 5],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )

    assert out.context_slots[:4].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
    assert out.query_slots[:3].tolist() == [
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
