# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded replay must retain chunk tails, request identity, and sampling rows."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia.ced import (
    CEDRequest,
    CEDTailState,
    clamp_replay_swa,
    plan_ced_step,
    validate_ced_config,
)


def boundary(start, count, offset=0):
    ids = torch.arange(start, start + count) + offset
    hidden = ids[:, None, None].expand(-1, 4, 8).to(torch.bfloat16)
    pre_mix = ids[:, None].expand(-1, 4).float() / 1024
    return hidden, pre_mix, ids


@pytest.mark.parametrize("length", [1, 127, 128, 129, 257])
@pytest.mark.parametrize("chunk", [1, 31, 128, 512])
def test_chunked_prefill_replays_exactly_the_last_encoder_window(length, chunk):
    state = CEDTailState(2, 8)
    for start in range(0, length, chunk):
        count = min(chunk, length - start)
        plan = plan_ced_step((CEDRequest(1, start, count, length, True),))
        actual = state.pack(plan, *boundary(start, count))
        if start + count != length:
            assert plan.num_decoder_tokens == 0
            assert actual[0].shape == (0, 4, 8)
            continue
        first = max(0, length - 128)
        expected = boundary(first, length - first)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        assert plan.output_rows == (count - 1,)
        assert plan.decoder_output_rows == (length - first - 1,)


def test_mixed_batch_reorders_replay_without_changing_sampling_rows():
    state = CEDTailState(4, 8)
    state.pack(plan_ced_step((CEDRequest(3, 0, 128, 129, True),)), *boundary(0, 128))
    requests = (
        CEDRequest(3, 128, 1, 129, True),
        CEDRequest(0, 20, 1, 20, False),
        CEDRequest(2, 0, 3, 100, True),
        CEDRequest(1, 0, 1, 1, True),
    )
    rows = [boundary(r.start, r.query_len) for r in requests]
    inputs = tuple(torch.cat([r[i] for r in rows]) for i in range(3))
    plan = plan_ced_step(requests)
    actual = state.pack(plan, *inputs)
    assert plan.decoder_requests == (1, 3, 0)
    assert plan.query_start_loc == (0, 1, 2, 130)
    assert plan.num_decoder_tokens > plan.num_input_tokens
    assert actual[2].tolist() == [20, 0] + list(range(1, 129))
    assert plan.output_rows == (1, 5, 0)
    assert plan.decoder_output_rows == (0, 1, 129)


def test_reused_request_slot_cannot_consume_an_old_tail():
    state = CEDTailState(1, 8)
    state.pack(plan_ced_step((CEDRequest(0, 0, 128, 129, True),)), *boundary(0, 128))
    state.reset(0)
    with pytest.raises(ValueError, match="tail missing"):
        state.pack(
            plan_ced_step((CEDRequest(0, 128, 1, 129, True),)), *boundary(128, 1)
        )
    plan = plan_ced_step((CEDRequest(0, 0, 3, 3, True),))
    result = state.pack(plan, *boundary(0, 3, offset=200))
    assert result[2].tolist() == [200, 201, 202]


def test_prefix_hit_seeds_progress_without_claiming_cached_encoder_rows():
    state = CEDTailState(1, 8)
    state.reset(0, 128)
    state.pack(
        plan_ced_step((CEDRequest(0, 128, 128, 257, True),)), *boundary(128, 128)
    )
    plan = plan_ced_step((CEDRequest(0, 256, 1, 257, True),))
    actual = state.pack(plan, *boundary(256, 1))
    for value, expected in zip(actual, boundary(129, 128)):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_prefix_hit_must_leave_a_full_replay_window_even_with_stale_ring_values():
    state = CEDTailState(1, 8)
    state.pack(plan_ced_step((CEDRequest(0, 0, 128, 129, True),)), *boundary(0, 128))
    state.reset(0, 128)
    with pytest.raises(ValueError, match="insufficient encoder replay"):
        state.pack(
            plan_ced_step((CEDRequest(0, 128, 1, 129, True),)), *boundary(128, 1)
        )


def test_generated_history_recovery_does_not_move_the_original_replay_boundary():
    state = CEDTailState(1, 8)
    plan = plan_ced_step((CEDRequest(0, 0, 127, 127, True),))
    state.pack(plan, *boundary(0, 127))
    plan = plan_ced_step((CEDRequest(0, 127, 10, 127, False),))
    actual = state.pack(plan, *boundary(127, 10))
    assert actual[2].tolist() == list(range(127, 137))
    assert plan.output_rows == tuple(range(10))
    assert state.ends[0] == 127


def test_compact_queries_are_ordered_for_both_speculative_backend_thresholds():
    plan = plan_ced_step(
        (
            CEDRequest(0, 128, 1, 129, True),
            CEDRequest(1, 20, 8, 20, False),
            CEDRequest(2, 0, 10, 10, True),
            CEDRequest(3, 0, 3, 3, True),
        )
    )
    assert plan.decoder_requests == (3, 1, 2, 0)
    assert plan.query_start_loc == (0, 3, 11, 21, 149)


def test_replay_swa_clamp_uses_logical_not_physical_cache_positions():
    # Noncontiguous physical slots deliberately have no ordering relation.
    decode_slots = torch.tensor([[900, 5, 20, -1]], dtype=torch.int32)
    prefill_slots = torch.tensor([[[800, 30, 9, 55]], [[99, 600, 8, 2]]])
    metadata = SimpleNamespace(
        num_decode_tokens=1,
        decode_swa_indices=decode_slots,
        decode_swa_lens=torch.tensor([3]),
        prefill_swa_indices=prefill_slots,
        prefill_swa_lens=torch.tensor([4, 4]),
    )
    clamp_replay_swa(metadata, torch.tensor([2, 8, 9]), torch.tensor([0, 8, 8]), 4)
    assert decode_slots.tolist() == [[900, 5, 20, -1]]
    assert prefill_slots.tolist() == [[[-1, -1, -1, 55]], [[-1, -1, 8, 2]]]
    assert metadata.prefill_swa_lens.tolist() == [1, 2]


@pytest.mark.parametrize(
    "requests",
    [
        (CEDRequest(0, 0, 1, 1, True), CEDRequest(0, 0, 1, 1, True)),
        (CEDRequest(-1, 0, 1, 1, True),),
        (CEDRequest(0, 1, 1, 1, True),),
        (CEDRequest(0, 0, 1, 2, False),),
    ],
)
def test_invalid_progress_is_rejected_before_cache_mutation(requests):
    with pytest.raises(ValueError):
        plan_ced_step(requests)


def valid_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=40,
                sliding_window=128,
                hc_mult=4,
                kv_source_layer_ids=[2, 8, 14, 20],
                engram_layer_ids=[1, 14],
            )
        ),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=20),
    )


@pytest.mark.parametrize(
    "field",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "ubatch_size",
    ],
)
def test_ced_rejects_unvalidated_parallel_schedules(field):
    config = valid_config()
    validate_ced_config(config, False)
    setattr(config.parallel_config, field, 2)
    with pytest.raises(NotImplementedError):
        validate_ced_config(config, False)


def test_ced_reserves_replay_capacity_not_just_scheduled_input_rows():
    config = valid_config()
    config.scheduler_config.max_num_batched_tokens = 2048
    with pytest.raises(ValueError, match="128"):
        validate_ced_config(config, False)
    config.scheduler_config.max_num_batched_tokens = 2560
    validate_ced_config(config, False)


@pytest.mark.parametrize(
    "field", ["speculative_config", "kv_transfer_config", "lora_config"]
)
def test_ced_rejects_unvalidated_state_restoration_or_draft_contracts(field):
    config = valid_config()
    setattr(config, field, object())
    with pytest.raises(NotImplementedError):
        validate_ced_config(config, False)


def test_ced_accepts_prefix_cache_and_native_dspark_together():
    config = valid_config()
    config.cache_config.enable_prefix_caching = True
    config.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        num_speculative_tokens=5,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                dspark_target_layer_ids=[37, 38, 39],
                sliding_window=128,
                compress_ratios=[1] * 40 + [0] * 3,
            )
        ),
    )
    validate_ced_config(config, False)
    config.speculative_config.draft_model_config.hf_config.dspark_target_layer_ids = [
        19
    ]
    with pytest.raises(NotImplementedError, match="decoder-only"):
        validate_ced_config(config, False)
