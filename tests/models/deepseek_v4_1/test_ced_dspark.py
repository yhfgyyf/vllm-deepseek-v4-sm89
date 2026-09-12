# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED context must preserve replay rows, request ownership and acceptance."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia.ced import (
    CED_METADATA_KEY,
    CEDDraftContext,
    CEDRequest,
    CEDStep,
    plan_ced_step,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def speculator(num_reqs=3, device="cpu"):
    spec = object.__new__(DSparkSpeculator)
    spec.device = torch.device(device)
    spec.num_query_per_req = spec.num_speculative_steps = 5
    spec.draft_kv_cache_group_ids = [1, 0]
    spec._layer_group_idx = [0, 1, 0]
    spec.block_tables = SimpleNamespace(
        slot_mappings=torch.full((2, 5 * num_reqs), 100, dtype=torch.int64),
        input_block_tables=[
            torch.arange(4, 4 + num_reqs * 256).view(num_reqs, 256),
            torch.arange(20, 20 + num_reqs * 128).view(num_reqs, 128),
        ],
        kernel_block_sizes=[4, 8],
        block_sizes=[16, 16],
    )
    spec.sample_idx_mapping = torch.arange(num_reqs).repeat_interleave(5).int()
    spec.input_buffers = SimpleNamespace(seq_lens=torch.full((num_reqs,), 999))
    spec.hidden_states = torch.full((256, 2), -99.0)
    spec.context_positions = torch.arange(256)
    spec._context_slot_mappings = torch.arange(512).view(2, 256)
    spec.model = SimpleNamespace(
        combine_hidden_states=Mock(side_effect=lambda values: values[:, :2] + 1),
        precompute_and_store_context_kv=Mock(),
    )
    spec.block_tables.slot_mappings = spec.block_tables.slot_mappings.to(device)
    spec.block_tables.input_block_tables = [
        table.to(device) for table in spec.block_tables.input_block_tables
    ]
    spec.sample_idx_mapping = spec.sample_idx_mapping.to(device)
    spec.input_buffers.seq_lens = spec.input_buffers.seq_lens.to(device)
    spec.hidden_states = spec.hidden_states.to(device)
    spec.context_positions = spec.context_positions.to(device)
    spec._context_slot_mappings = spec._context_slot_mappings.to(device)
    return spec


def make_step(context):
    plan = plan_ced_step((CEDRequest(0, 0, 1, 1, True),))
    return CEDStep(plan, {}, context.positions, draft_context=context)


def test_compact_context_preserves_128_replay_rows_and_masks_rejected_decode():
    spec = speculator()
    # Original request0: final chunk1, request1: decode verification3, request2:
    # unfinished encoder prefill. Compact order deliberately differs from batch.
    positions = torch.tensor([18, 19, 20, *range(385, 513)])
    reqs = torch.tensor([1, 1, 1, *([0] * 128)])
    aux = [positions[:, None].expand(-1, 2).float() + layer for layer in range(3)]
    context = CEDDraftContext(positions, reqs, (1, 0), aux)
    batch = SimpleNamespace(
        num_reqs=3, num_tokens=68, seq_lens=torch.tensor([513, 21, 600])
    )
    query_ptr = spec.block_tables.slot_mappings.data_ptr()
    sample_ptr = spec.sample_idx_mapping.data_ptr()
    spec._precompute_context_kv(
        batch,
        {CED_METADATA_KEY: make_step(context)},
        torch.zeros(68, 2),
        [],
        torch.tensor([0, 2, 0]),
        dummy_run=False,
    )
    projected, actual_positions, slots = (
        spec.model.precompute_and_store_context_kv.call_args.args
    )
    torch.testing.assert_close(projected, aux[0] + 1)
    torch.testing.assert_close(actual_positions, positions)
    assert projected.shape[0] == 131  # not the68 current target rows or final1
    for layer, gid in enumerate([1, 0, 1]):
        block_size = spec.block_tables.kernel_block_sizes[gid]
        expected = spec.block_tables.input_block_tables[gid][
            reqs, positions // block_size
        ]
        expected = expected * block_size + positions % block_size
        expected[1:3] = -1
        torch.testing.assert_close(slots[layer], expected)
    assert spec.block_tables.slot_mappings.data_ptr() == query_ptr
    assert spec.sample_idx_mapping.data_ptr() == sample_ptr
    assert (spec.block_tables.slot_mappings[:, :10] == 100).all()
    assert (spec.block_tables.slot_mappings[:, 10:] == -1).all()
    assert spec.sample_idx_mapping.tolist() == [0] * 5 + [1] * 5 + [-1] * 5
    assert spec.input_buffers.seq_lens.tolist() == [999, 999, 0]
    assert (spec.hidden_states == -99).all()  # no dense-zero context staging


def test_intermediate_prefill_publishes_no_context_and_has_inert_draft_queries():
    spec = speculator()
    context = CEDDraftContext(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), (), []
    )
    spec._precompute_context_kv(
        SimpleNamespace(num_reqs=3),
        {CED_METADATA_KEY: make_step(context)},
        torch.zeros(100, 2),
        [],
        torch.zeros(3, dtype=torch.int32),
        dummy_run=False,
    )
    spec.model.combine_hidden_states.assert_not_called()
    spec.model.precompute_and_store_context_kv.assert_not_called()
    assert (spec.block_tables.slot_mappings == -1).all()
    assert (spec.sample_idx_mapping == -1).all()


def test_split_null_pages_and_out_of_range_positions_never_become_context_slots():
    context = CEDDraftContext(
        positions=torch.tensor([-1, 0, 4, 8, 12, 16, 20, 24]),
        request_indices=torch.zeros(8, dtype=torch.int64),
        active_requests=(0,),
        aux_hidden_states=[],
    )
    slots = context.slot_mapping(
        torch.tensor([[0, 1, 2, 3, 7, 8]]),
        4,
        16,
        torch.tensor([30]),
        torch.tensor([0]),
    )
    assert slots.tolist() == [-1, -1, -1, -1, -1, 28, 32, -1]


@pytest.mark.parametrize("dummy_run", [False, True])
def test_non_ced_dspark_preserves_normal_dense_auxiliary_context(dummy_run):
    spec = speculator()
    aux = [torch.arange(8).view(4, 2).float()]
    spec._precompute_context_kv(
        SimpleNamespace(num_tokens=4),
        {},
        torch.zeros(4, 2),
        aux,
        torch.zeros(3, dtype=torch.int32),
        dummy_run=dummy_run,
    )
    hidden, positions, slots = spec.model.precompute_and_store_context_kv.call_args.args
    torch.testing.assert_close(hidden, aux[0] + 1)
    assert positions.tolist() == [0, 1, 2, 3]
    if dummy_run:
        assert slots is None
    else:
        assert [row.tolist() for row in slots] == [
            list(range(4)),
            list(range(256, 260)),
            list(range(4)),
        ]
