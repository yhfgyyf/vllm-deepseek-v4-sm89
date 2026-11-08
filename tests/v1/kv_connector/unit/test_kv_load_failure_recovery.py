# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import SupportsHMA
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.request import Request, RequestStatus

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


def _make_get_num_new_matched_tokens(
    req_num_new_matched_tokens: dict[str, int],
    async_load,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        value = req_num_new_matched_tokens.get(request.request_id, 0)
        return value, async_load

    return get_num_new_matched_tokens


@pytest.fixture
def scheduler():
    vllm_config = create_vllm_config(kv_load_failure_policy="recompute")
    return create_scheduler(vllm_config)


def _create_ced_scheduler(*, kv_load_failure_policy: str = "recompute") -> Scheduler:
    block_size = 32
    vllm_config = create_vllm_config(
        max_num_seqs=1,
        max_num_batched_tokens=256,
        block_size=block_size,
        max_model_len=2048,
        kv_load_failure_policy=kv_load_failure_policy,
    )
    global_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=288,
        model_version="deepseek_v4_1",
    )
    encoder_swa_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=528,
        sliding_window=128,
        model_version="deepseek_v4_1",
        allow_prefix_caching=True,
    )
    decoder_scratch_spec = SlidingWindowMLASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=528,
        sliding_window=128,
        model_version="deepseek_v4_1",
        allow_prefix_caching=False,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=256,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["global"], global_spec),
            KVCacheGroupSpec(["encoder_swa"], encoder_swa_spec),
            KVCacheGroupSpec(["decoder_scratch"], decoder_scratch_spec),
        ],
    )
    scheduler = create_scheduler(
        vllm_config,
        num_blocks=kv_cache_config.num_blocks,
        kv_cache_config=kv_cache_config,
    )
    scheduler.ced_prefill = True
    scheduler.kv_cache_manager.prefix_replay_window = 128
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        manager.prefix_replay_window = 128
    assert scheduler.ced_prefill
    return scheduler


class _MockHMAConnector(SupportsHMA):
    supports_divergent_local_hybrid_hits = False
    requires_kv_delivery = False

    def __init__(self, matched_tokens: int, async_load: bool):
        self.get_num_new_matched_tokens = Mock(
            return_value=(matched_tokens, async_load)
        )
        self.on_new_request = Mock()
        self.update_state_after_alloc = Mock()
        self.build_connector_meta = Mock(return_value=None)
        self.update_connector_output = Mock()
        self.get_kv_connector_stats = Mock(return_value=None)
        self.take_events = Mock(return_value=())

    def request_finished_all_groups(self, request, block_ids):
        return False, None


def _mock_connector(
    scheduler: Scheduler, matched_tokens: int, *, async_load: bool
) -> _MockHMAConnector:
    connector = _MockHMAConnector(matched_tokens, async_load)
    scheduler.connector = connector
    return connector


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_async_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    (req2_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request2.request_id)
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        reqs=[],
        finished_recving={request1.request_id, request3.request_id},
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    min_invalid_block_idx = min(invalid_block_idxs)

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 3
    for request in scheduler.skipped_waiting:
        if request.request_id == request2.request_id:
            assert request.num_computed_tokens == (
                min_invalid_block_idx * scheduler.block_size
            )
        else:
            assert request.num_computed_tokens == num_external_computed_tokens
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.failed_recving_kv_req_ids == {request2.request_id}
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 98}),
        (100, 99, {50, 98}),
        (100, 99, {98}),
    ],
)
def test_sync_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request1 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request1)
    request2 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request2)
    request3 = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request3)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: num_external_computed_tokens,
        request3.request_id: num_external_computed_tokens,
    }

    assert len(scheduler.running) == 3
    assert len(scheduler_output.scheduled_new_reqs) == 3
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3

    # Simulate a failure in loading some of request2 blocks.
    req2_block_ids = scheduler_output.scheduled_new_reqs[1].block_ids[0]
    invalid_block_ids = {req2_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2, request3],
        invalid_block_ids=invalid_block_ids,
        use_eos=True,
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    assert len(scheduler.running) == 1
    assert scheduler.running[0].request_id == request2.request_id
    assert scheduler.running[0].num_computed_tokens == (
        min(invalid_block_idxs) * scheduler.block_size
    )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 3
    assert scheduler.connector.request_finished.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,"
    "num_external_computed_blocks,"
    "num_common_prefix_blocks,"
    "invalid_block_idxs",
    [
        (100, 99, 50, {0, 49}),
        (100, 99, 50, {25, 49}),
        (100, 99, 50, {49}),
    ],
)
def test_sync_load_failure_with_shared_blocks(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    num_common_prefix_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks >= num_common_prefix_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size
    common_prefix_len = num_common_prefix_blocks * scheduler.block_size

    request1 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request1)
    request2 = create_request(
        num_tokens=num_prompt_tokens, common_prefix_len=common_prefix_len
    )
    scheduler.add_request(request=request2)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request1.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=False)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    # req_id -> num_computed_tokens
    expected_computed_tokens = {
        request1.request_id: num_external_computed_tokens,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    assert len(scheduler_output.scheduled_new_reqs) == 2
    for request in scheduler_output.scheduled_new_reqs:
        assert request.num_computed_tokens == expected_computed_tokens[request.req_id]
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2

    # Simulate a failure in loading some of the shared blocks.
    req1_block_ids = scheduler_output.scheduled_new_reqs[0].block_ids[0]
    invalid_block_ids = {req1_block_ids[i] for i in invalid_block_idxs}
    model_runner_output = create_model_runner_output(
        [request1, request2], invalid_block_ids=invalid_block_ids, use_eos=True
    )

    scheduler.update_from_output(scheduler_output, model_runner_output)

    # req_id -> num_computed_tokens
    # all the common prefix blocks will be computed by request1
    expected_computed_tokens = {
        request1.request_id: min(invalid_block_idxs) * scheduler.block_size,
        request2.request_id: common_prefix_len,
    }

    assert len(scheduler.running) == 2
    for request in scheduler.running:
        assert (
            request.num_computed_tokens == expected_computed_tokens[request.request_id]
        )
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 2


@pytest.mark.parametrize(
    "num_prompt_blocks,num_external_computed_blocks,invalid_block_idxs",
    [
        (100, 99, {0, 50, 98}),
        (100, 99, {98, 50, 0}),
    ],
)
def test_async_progressive_load_failure(
    scheduler: Scheduler,
    num_prompt_blocks: int,
    num_external_computed_blocks: int,
    invalid_block_idxs: set[int],
):
    assert num_prompt_blocks >= num_external_computed_blocks

    num_prompt_tokens = num_prompt_blocks * scheduler.block_size
    num_external_computed_tokens = num_external_computed_blocks * scheduler.block_size

    request = create_request(num_tokens=num_prompt_tokens)
    scheduler.add_request(request=request)

    # Mock KV connector method.
    # req_id -> num_external_computed_tokens
    req_num_new_matched_tokens = {
        request.request_id: num_external_computed_tokens,
    }

    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(req_num_new_matched_tokens, async_load=True)
    )
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()

    assert len(scheduler.waiting) == 0
    assert len(scheduler.skipped_waiting) == 1
    assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
    assert request.num_computed_tokens == num_external_computed_tokens
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.connector.get_num_new_matched_tokens.call_count == 1

    min_invalid_block_idx = max(invalid_block_idxs) + 1
    # Simulate failures when progressively loading request blocks.
    for invalid_block_idx in invalid_block_idxs:
        (req_block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
        invalid_block_ids = {req_block_ids[invalid_block_idx]}
        model_runner_output = create_model_runner_output(
            reqs=[],
            finished_recving=set(),
            invalid_block_ids=invalid_block_ids,
            use_eos=True,
        )

        scheduler.update_from_output(scheduler_output, model_runner_output)

        min_invalid_block_idx = min(min_invalid_block_idx, invalid_block_idx)

        assert len(scheduler.waiting) == 0
        assert len(scheduler.skipped_waiting) == 1
        assert scheduler.skipped_waiting.peek_request().request_id == request.request_id
        assert request.num_computed_tokens == (
            min_invalid_block_idx * scheduler.block_size
        )
        assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert scheduler.failed_recving_kv_req_ids == {request.request_id}
        assert scheduler.connector.get_num_new_matched_tokens.call_count == 1


def test_ced_async_failure_in_encoder_swa_restarts_from_zero():
    scheduler = _create_ced_scheduler()
    connector = _mock_connector(scheduler, 192, async_load=True)
    connector.get_num_new_matched_tokens.side_effect = [(192, True), (0, False)]
    request = create_request(num_tokens=321, block_size=scheduler.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_prompt_tokens - request.num_computed_tokens >= 128
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    assert len(block_ids) == 3
    invalid_block_id = block_ids[1][2]
    initially_allocated = {
        block_id for group_block_ids in block_ids for block_id in group_block_ids
    }

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[], invalid_block_ids={invalid_block_id}, use_eos=False
        ),
    )

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 0
    assert request.request_id in scheduler.failed_recving_kv_req_ids
    assert initially_allocated <= {
        block.block_id
        for blocks in scheduler.kv_cache_manager.get_blocks(request.request_id).blocks
        for block in blocks
    }

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[], finished_recving={request.request_id}, use_eos=False
        ),
    )
    scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 256
    assert request.request_id not in scheduler.failed_recving_kv_req_ids
    assert request.request_id not in scheduler.finished_recving_kv_req_ids
    assert request in scheduler.running
    assert connector.get_num_new_matched_tokens.call_count == 2
    assert all(
        block.ref_cnt >= 0
        for block in scheduler.kv_cache_manager.block_pool.blocks
        if not block.is_null
    )


def test_ced_async_success_waits_before_replay_admission():
    scheduler = _create_ced_scheduler()
    connector = _mock_connector(scheduler, 192, async_load=True)
    request = create_request(num_tokens=321, block_size=scheduler.block_size)
    scheduler.add_request(request)

    load_output = scheduler.schedule()

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 192
    assert not load_output.scheduled_new_reqs

    scheduler.update_from_output(
        load_output,
        create_model_runner_output(
            reqs=[], finished_recving={request.request_id}, use_eos=False
        ),
    )

    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    replay_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert len(replay_output.scheduled_new_reqs) == 1
    admitted = replay_output.scheduled_new_reqs[0]
    assert admitted.req_id == request.request_id
    assert admitted.num_computed_tokens == 192
    assert request.num_prompt_tokens - admitted.num_computed_tokens >= 128
    assert connector.get_num_new_matched_tokens.call_count == 1


def test_ced_sync_failure_preempts_and_drops_stale_output():
    scheduler = _create_ced_scheduler()
    _mock_connector(scheduler, 192, async_load=False)
    request = create_request(num_tokens=321, block_size=scheduler.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.num_in_flight_tokens > 0
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    cached_prefix_ids = {
        block_id
        for group_block_ids in block_ids[:2]
        for block_id in group_block_ids
        if scheduler.kv_cache_manager.block_pool.blocks[block_id].block_hash is not None
    }
    assert cached_prefix_ids

    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[request], invalid_block_ids={block_ids[1][2]}, use_eos=False
        ),
    )

    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.num_in_flight_tokens == 0
    assert request.num_stale_output_tokens == 0
    assert request.drop_stale_output
    assert request.num_output_tokens == 0
    assert request in scheduler.waiting
    assert request not in scheduler.running
    assert request.request_id in scheduler.reset_preempted_req_ids
    assert all(
        scheduler.kv_cache_manager.block_pool.blocks[block_id].block_hash is None
        for block_id in cached_prefix_ids
    )


def test_ced_async_failure_policy_fail_finishes_without_leaking_blocks():
    scheduler = _create_ced_scheduler(kv_load_failure_policy="fail")
    _mock_connector(scheduler, 192, async_load=True)
    request = create_request(num_tokens=321, block_size=scheduler.block_size)
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()
    block_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)

    outputs = scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            reqs=[],
            finished_recving={request.request_id},
            invalid_block_ids={block_ids[1][2]},
            use_eos=False,
        ),
    )

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert request not in scheduler._inflight_prefills
    assert not any(scheduler.kv_cache_manager.get_block_ids(request.request_id))
    assert all(
        block.ref_cnt >= 0
        for block in scheduler.kv_cache_manager.block_pool.blocks
        if not block.is_null
    )
    assert outputs[request.client_index].outputs[0].finish_reason == FinishReason.ERROR
