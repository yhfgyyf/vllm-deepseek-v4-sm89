# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4_1.attention import DeepseekV41SWACache
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_request_block_hasher,
    group_and_unify_kv_cache_specs,
    init_none_hash,
)
from vllm.v1.core.sched.scheduler import _cap_ced_replay_chunk
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlotMappingPolicy,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=288,
        model_version="deepseek_v4_1",
    )


def _swa_spec(*, cacheable: bool) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        state_content_bytes=528,
        sliding_window=64,
        model_version="deepseek_v4_1",
        allow_prefix_caching=cacheable,
    )


def _request(request_id: str, token_ids: list[int]) -> Request:
    sampling_params = SamplingParams(max_tokens=16)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(32, sha256),
    )


def _manager(prefix_replay_window: int) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["global"], _mla_spec()),
            KVCacheGroupSpec(["encoder_swa"], _swa_spec(cacheable=True)),
            KVCacheGroupSpec(["decoder_swa"], _swa_spec(cacheable=False)),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=32,
        hash_block_size=32,
        enable_caching=True,
        prefix_replay_window=prefix_replay_window,
    )


@pytest.mark.parametrize(("replay_window", "expected_hit"), [(0, 320), (128, 192)])
def test_ced_prefix_hit_reuses_only_cacheable_groups(replay_window, expected_hit):
    init_none_hash(sha256)
    manager = _manager(replay_window)
    token_ids = list(range(321))

    producer = _request("producer", token_ids)
    blocks, hit, _ = manager.get_computed_blocks(producer)
    assert hit == 0
    assert manager.allocate_slots(producer, producer.num_tokens, 0, blocks) is not None

    producer_blocks = manager.get_blocks(producer.request_id).blocks
    assert all(block.block_hash is None for block in producer_blocks[2])

    consumer = _request("consumer", token_ids)
    computed_blocks, hit, _ = manager.get_computed_blocks(consumer)
    assert hit == expected_hit
    assert computed_blocks.blocks[0]
    assert computed_blocks.blocks[1]
    assert computed_blocks.blocks[2] == []

    suffix = consumer.num_tokens - hit
    assert (
        manager.allocate_slots(
            consumer,
            suffix,
            num_new_computed_tokens=hit,
            new_computed_blocks=computed_blocks,
        )
        is not None
    )

    consumer_blocks = manager.get_blocks(consumer.request_id).blocks
    private_decoder_blocks = [
        block for block in consumer_blocks[2] if not block.is_null
    ]
    assert private_decoder_blocks
    assert all(block.block_hash is None for block in private_decoder_blocks)
    assert {block.block_id for block in private_decoder_blocks}.isdisjoint(
        block.block_id for block in producer_blocks[2] if not block.is_null
    )


def test_ced_swa_grouping_keeps_cacheability_distinct():
    grouped = group_and_unify_kv_cache_specs(
        {
            "global": _mla_spec(),
            "encoder_swa": _swa_spec(cacheable=True),
            "decoder_swa": _swa_spec(cacheable=False),
        }
    )

    assert grouped is not None
    assert len(grouped) == 3
    assert [group.prefix_cacheable for group in grouped] == [True, True, False]

    worker_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(list(group.kv_cache_specs), group) for group in grouped
        ],
    )
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    assert [
        group.kv_cache_spec.prefix_cacheable
        for group in scheduler_config.kv_cache_groups
    ] == [True, True, False]


def test_v41_swa_cache_publishes_constructor_cacheability_tag():
    cache = DeepseekV41SWACache.__new__(DeepseekV41SWACache)
    cache.block_size = 32
    cache.head_dim = 192
    cache.window_size = 128
    cache.cache_config = SimpleNamespace(cache_dtype="fp8_ds_mla")
    cache.allow_prefix_caching = False

    spec = cache.get_kv_cache_spec(SimpleNamespace())

    assert not spec.prefix_cacheable
    assert spec.slot_mapping_policy == SlotMappingPolicy.PAGED


def test_ced_replay_chunks_preserve_prompt_boundary_and_bound_history():
    request = SimpleNamespace(num_prompt_tokens=256, num_tokens=600)

    assert _cap_ced_replay_chunk(request, 128, 256) == 128
    assert _cap_ced_replay_chunk(request, 256, 300) == 128
    assert _cap_ced_replay_chunk(request, 600, 7) == 7
