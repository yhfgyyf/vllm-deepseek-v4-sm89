# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89/SM120-runnable unit test for the DSpark non-causal SWA index kernel.

The full ``test_dspark_noncausal_sparse_mla.py`` suite only exercises the
FlashMLA-sparse (SM90) and FlashInfer-TRTLLM (SM100) decode backends and thus
skips on Ada (SM89), where the portable Triton sparse-MLA path runs instead.

On the Triton path the only DSpark-specific new code is
``_compute_dspark_noncausal_swa_indices_kernel`` (it fills the per-token slot
index list that the Triton decode kernel then gathers). This test validates that
kernel directly: it needs only CUDA + Triton, so it runs on Ada. With an identity
block table the slot id of position ``p`` is exactly ``p``, so the expected
output is the contiguous block-anchored window ``[max(prefix - window, 0), seq)``
padded with ``-1`` -- the same semantics as the suite's
``_build_dspark_noncausal_indices`` reference.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        "DSpark non-causal SWA index kernel test only supports CUDA.",
        allow_module_level=True,
    )

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_dspark_noncausal_swa_indices_kernel,
)

DEVICE = current_platform.device_type


def _run_kernel(seq_lens, query_lens, window, block_size, device):
    num_reqs = len(seq_lens)
    num_decode_tokens = sum(query_lens)
    num_spec = max(query_lens)
    index_width = cdiv(window + num_spec, 128) * 128

    # token -> req mapping (decodes are laid out req-major).
    token_to_req = []
    for r, q in enumerate(query_lens):
        token_to_req.extend([r] * q)
    token_to_req_indices = torch.tensor(token_to_req, dtype=torch.int32, device=device)

    # query_start_loc over the decode-token array.
    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + q)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    is_valid_token = torch.ones(num_decode_tokens, dtype=torch.bool, device=device)

    # Identity block table: block b -> physical block b, so slot(p) == p.
    max_blocks = max(cdiv(s, block_size) for s in seq_lens)
    block_table = (
        torch.arange(max_blocks, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .repeat(num_reqs, 1)
        .contiguous()
    )

    swa_indices = torch.zeros(
        num_decode_tokens, 1, index_width, dtype=torch.int32, device=device
    )
    swa_lens = torch.zeros(num_decode_tokens, dtype=torch.int32, device=device)

    _compute_dspark_noncausal_swa_indices_kernel[(num_decode_tokens,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        window,
        index_width,
        query_start_loc,
        seq_lens_t,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table.stride(0),
        block_size,
        TRITON_BLOCK_SIZE=1024,
    )
    return swa_indices, swa_lens, index_width


@pytest.mark.parametrize(
    "seq_lens,query_lens,window,block_size",
    [
        ([20, 10], [4, 4], 8, 4),  # one req clamps at 0, the other doesn't
        ([130], [7], 128, 16),  # window + block > 128 -> width grows past 128
        ([64, 33, 9], [3, 3, 3], 16, 8),  # ragged batch
    ],
)
def test_noncausal_index_kernel_matches_reference(
    seq_lens, query_lens, window, block_size
):
    device = torch.device(DEVICE)
    swa_indices, swa_lens, index_width = _run_kernel(
        seq_lens, query_lens, window, block_size, device
    )

    # Expected: every query token in a block shares the contiguous range
    # [max(prefix - window, 0), seq_len); identity block table => slot == pos.
    expected = torch.full_like(swa_indices[:, 0, :], -1)
    expected_lens = torch.empty_like(swa_lens)
    t = 0
    for s_len, q_len in zip(seq_lens, query_lens):
        prefix = s_len - q_len
        start = max(prefix - window, 0)
        rng = torch.arange(start, s_len, dtype=torch.int32, device=device)
        for _ in range(q_len):
            expected[t, : rng.numel()] = rng
            expected_lens[t] = rng.numel()
            t += 1

    torch.testing.assert_close(swa_lens, expected_lens)
    torch.testing.assert_close(swa_indices[:, 0, :], expected)
    # Non-causality witness: an early block query's list contains a later block
    # position (future-pointing), which a causal window would never include.
    if query_lens[0] >= 2 and seq_lens[0] - 1 not in (-1,):
        last_pos = seq_lens[0] - 1
        first_token_list = swa_indices[0, 0, : swa_lens[0]]
        assert (first_token_list == last_pos).any(), (
            "first block query must attend to the last (future) block position"
        )


def test_noncausal_index_kernel_marks_invalid_tokens():
    device = torch.device(DEVICE)
    # A padded/invalid token must get swa_len == 0 (the kernel early-returns).
    seq_lens, query_lens, window, block_size = [16], [4], 8, 4
    num_decode_tokens = sum(query_lens)
    num_spec = max(query_lens)
    index_width = cdiv(window + num_spec, 128) * 128

    token_to_req_indices = torch.zeros(
        num_decode_tokens, dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor([0, 4], dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    is_valid_token = torch.tensor(
        [True, True, False, True], dtype=torch.bool, device=device
    )
    max_blocks = cdiv(seq_lens[0], block_size)
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )

    swa_indices = torch.zeros(
        num_decode_tokens, 1, index_width, dtype=torch.int32, device=device
    )
    swa_lens = torch.full((num_decode_tokens,), -1, dtype=torch.int32, device=device)

    _compute_dspark_noncausal_swa_indices_kernel[(num_decode_tokens,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        window,
        index_width,
        query_start_loc,
        seq_lens_t,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_table.stride(0),
        block_size,
        TRITON_BLOCK_SIZE=1024,
    )
    assert int(swa_lens[2]) == 0
    assert int(swa_lens[0]) > 0 and int(swa_lens[3]) > 0
