# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v4_1.nvidia.ced import CED_METADATA_KEY, CEDRequest
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlotMappingPolicy,
)
from vllm.v1.worker.gpu.model_runner import _requires_ced_eager
from vllm.v1.worker.utils import AttentionGroup


class _FakeMetadataBuilder:
    requires_block_table_width = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.buffer = object()

    def build(self, common_prefix_len, common_attn_metadata):
        assert common_prefix_len == 0
        return SimpleNamespace(
            common=common_attn_metadata,
            builder_buffer=self.buffer,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)


class _FakeBackend:
    @staticmethod
    def get_builder_cls():
        return _FakeMetadataBuilder


def _request(**overrides):
    fields: dict[str, object] = dict(
        req_id="req",
        sampling_params=SimpleNamespace(prompt_logprobs=None),
        prompt_embeds=None,
        mm_features=[],
        num_computed_tokens=0,
        prompt_len=128,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _request_state() -> DeepseekV41ModelState:
    state = object.__new__(DeepseekV41ModelState)
    state.ced_enabled = True
    state.ced_tail = Mock()
    state._ced_request_slots = {}
    state._ced_prompt_lens = {}
    state.rope_state = None
    state.prompt_embeds_state = None
    state._mm_prefix_prompt_token_ids = {}
    state.model_config = SimpleNamespace(hf_config=SimpleNamespace())
    return state


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"sampling_params": SimpleNamespace(prompt_logprobs=0)},
            "prompt log probabilities",
        ),
        (
            {"sampling_params": SimpleNamespace(prompt_logprobs=1)},
            "prompt log probabilities",
        ),
        ({"prompt_embeds": torch.zeros(1, 2)}, "prompt embeddings"),
        ({"mm_features": [object()]}, "multimodal inputs"),
        ({"num_computed_tokens": 1}, "insufficient encoder replay"),
    ],
)
def test_ced_rejects_request_features_before_reset(overrides, message):
    state = _request_state()

    with pytest.raises(ValueError, match=message):
        state.add_request(0, _request(**overrides))

    state.ced_tail.reset.assert_not_called()
    assert state._ced_request_slots == {}


def test_ced_resets_tail_when_slot_zero_is_added_and_removed():
    state = _request_state()

    state.add_request(0, _request())
    state.remove_request("req")

    assert state.ced_tail.reset.call_args_list == [((0, 0),), ((0,),)]
    assert state._ced_request_slots == {}
    assert state._ced_prompt_lens == {}


def test_ced_prefix_hit_retains_original_prompt_length_for_resumed_history():
    state = _request_state()
    state.add_request(
        3,
        _request(
            prompt_len=257, num_computed_tokens=128, prefill_token_ids=list(range(400))
        ),
    )
    state.ced_tail.reset.assert_called_once_with(3, 128)
    assert state._ced_prompt_lens == {3: 257}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ced_exact_metadata_transfer_is_allowed_with_gpu_sync_check(monkeypatch):
    import vllm.utils.gpu_sync_debug as gsd

    state = _request_state()
    state._ced_prompt_lens = {3: 257}
    state.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256)
    )
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([3]),
        query_start_loc=torch.tensor([0, 128], device="cuda"),
        seq_lens=torch.tensor([128], device="cuda"),
        positions=torch.arange(128, device="cuda"),
    )
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    step = gsd.with_gpu_sync_check(state._build_ced_step)(batch, (), [], None)
    assert step.plan.requests == (CEDRequest(3, 0, 128, 257, True),)


@pytest.mark.parametrize(
    ("enabled", "has_prefill", "dummy_run", "expected"),
    [
        (True, True, False, True),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, True, False),
    ],
)
def test_ced_eager_guard_only_applies_to_real_prefill_batches(
    enabled, has_prefill, dummy_run, expected
):
    hf_config = SimpleNamespace(model_type="deepseek_v41", ced_prefill=enabled)
    batch = SimpleNamespace(has_prefill=has_prefill)

    assert _requires_ced_eager(hf_config, batch, dummy_run) is expected


def test_ced_eager_flag_does_not_change_other_model_families():
    hf_config = SimpleNamespace(model_type="glm_moe_dsa", ced_prefill=True)
    assert not _requires_ced_eager(hf_config, SimpleNamespace(has_prefill=True), False)


def test_ced_slot_mappings_preserve_paged_ring_and_none_policies():
    positions = torch.tensor([0, 5], dtype=torch.int64)
    mappings = DeepseekV41ModelState._ced_slot_mappings(
        (
            torch.tensor([[2, 3]], dtype=torch.int32),
            torch.tensor([[7]], dtype=torch.int32),
            torch.tensor([[9]], dtype=torch.int32),
        ),
        positions,
        (0, 2),
        (
            SlotMappingPolicy.PAGED,
            SlotMappingPolicy.SINGLE_BLOCK_RING,
            SlotMappingPolicy.NONE,
        ),
        (4, 4, 4),
    )

    assert torch.equal(
        mappings,
        torch.tensor([[8, 13], [28, 29], [-1, -1]], dtype=torch.int64),
    )


@pytest.mark.parametrize("optimistic_cpu_counts", [False, True])
def test_ced_prepare_attn_uses_independent_compact_metadata_builders(
    optimistic_cpu_counts,
):
    state = object.__new__(DeepseekV41ModelState)
    state.ced_enabled = True
    state._ced_attn_groups = None
    state._ced_prompt_lens = {9: 128, 3: 10}
    state.supports_mm_inputs = False
    state.max_model_len = 256
    state.device = torch.device("cpu")
    state.model_config = SimpleNamespace(is_mm_prefix_lm=False)
    state.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=256),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
    )

    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
    )
    group = AttentionGroup(_FakeBackend, ["layer"], spec, 0)  # type: ignore[arg-type]
    group.create_metadata_builders(state.vllm_config, state.device, 4)
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    input_batch = SimpleNamespace(
        req_ids=["prefill", "decode"],
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping_np=np.array([9, 3], dtype=np.intp),
        num_computed_tokens_np=np.array([72, 10], dtype=np.int32),
        num_scheduled_tokens=np.array([56, 1], dtype=np.int32),
        prefill_len_np=np.array([128, 10], dtype=np.int32),
        is_prefilling_np=np.array([True, False]),
        has_prefill=True,
        query_start_loc_np=np.array([0, 56, 57], dtype=np.int32),
        query_start_loc=torch.tensor([0, 56, 57], dtype=torch.int32),
        num_tokens=57,
        num_tokens_after_padding=57,
        positions=torch.arange(57, dtype=torch.int64),
        seq_lens=torch.tensor([128, 11], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([128, 11], dtype=torch.int32),
        dcp_local_seq_lens=None,
        max_query_len=56,
        prompt_lens=None,
    )
    block_tables = (
        torch.stack(
            (
                torch.arange(64, dtype=torch.int32),
                torch.arange(100, 164, dtype=torch.int32),
            )
        ),
    )
    source_slot_mappings = torch.zeros((1, 57), dtype=torch.int64)
    if optimistic_cpu_counts:
        input_batch.num_computed_tokens_np[1] = 20
        input_batch.num_scheduled_tokens[1] = 8
        input_batch.query_start_loc_np[-1] = 64

    metadata = state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables,
        source_slot_mappings,
        [[group]],
        kv_cache_config,
    )

    source = metadata["layer"]
    step = metadata[CED_METADATA_KEY]
    decoder = step.decoder_metadata["layer"]
    assert source.builder_buffer is group.get_metadata_builder(0).buffer
    assert decoder.builder_buffer is not source.builder_buffer
    assert source.common.query_start_loc_cpu.tolist() == [
        0,
        56,
        64 if optimistic_cpu_counts else 57,
    ]
    assert step.plan.decoder_requests == (1, 0)
    assert decoder.common.query_start_loc_cpu.tolist() == [0, 1, 129]
    assert decoder.common.seq_lens.tolist() == [11, 128]
    assert decoder.common.is_prefilling.tolist() == [False, True]
    assert step.positions.tolist() == [10, *range(128)]
    assert step.decoder_slot_mapping["layer"][0].item() == 410

    capture_metadata = state.prepare_attn(
        input_batch,
        CUDAGraphMode.FULL,
        block_tables,
        source_slot_mappings,
        [[group]],
        kv_cache_config,
        for_capture=True,
    )
    assert CED_METADATA_KEY not in capture_metadata
