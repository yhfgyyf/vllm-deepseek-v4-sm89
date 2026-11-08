# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, post_update

DEVICE = current_platform.device_type


def test_deferred_pp_post_update_applies_fresh_drafts_only_to_live_slots():
    device = torch.device(DEVICE)
    computed = torch.tensor([10, 20, 30], dtype=torch.int32, device=device)
    sampled = torch.zeros(3, dtype=torch.int64, device=device)
    drafts = torch.full((3, 3), -9, dtype=torch.int64, device=device)
    all_ids = torch.zeros(3, 64, dtype=torch.int32, device=device)
    lengths = torch.tensor([10, 20, 30], dtype=torch.int32, device=device)
    post_update(
        torch.tensor([2, -1, 0], dtype=torch.int64, device=device),
        computed,
        sampled,
        None,
        torch.tensor(
            [[11, 12, 0, 0], [99, 0, 0, 0], [31, 0, 0, 0]],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor([2, 1, 1], dtype=torch.int32, device=device),
        torch.tensor([1, 0, 3], dtype=torch.int32, device=device),
        None,
        all_ids,
        lengths,
        torch.tensor(
            [[41, 42, 43], [99, 99, 99], [51, 52, 53]], dtype=torch.int64, device=device
        ),
        drafts,
    )
    assert computed.tolist() == [7, 20, 29]
    assert sampled.tolist() == [31, 0, 12]
    assert lengths.tolist() == [11, 20, 32]
    assert drafts.tolist() == [[51, 52, 53], [-9, -9, -9], [41, 42, 43]]
    assert all_ids[2, 30:32].tolist() == [11, 12]


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )
