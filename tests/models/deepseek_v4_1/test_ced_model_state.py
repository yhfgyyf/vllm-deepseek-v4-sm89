# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import VLLMValidationError
from vllm.inputs.engine import embeds_input, mm_input, tokens_input
from vllm.models.deepseek_v4_1.nvidia.ced import CED_METADATA_KEY, CEDRequest
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor
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
    fields.setdefault("prefill_token_ids", list(range(cast(int, fields["prompt_len"]))))
    return SimpleNamespace(**fields)


def _mm_feature(
    modality: str = "image", offset: int = 1, length: int = 1
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=f"cached-{modality}",
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _request_state() -> DeepseekV41ModelState:
    state = object.__new__(DeepseekV41ModelState)
    state.ced_enabled = True
    state.ced_tail = Mock()
    state.ced_tail.window = 384
    state._ced_request_slots = {}
    state._ced_prompt_lens = {}
    state._ced_replay_starts = {}
    state._ced_mm_ranges = {}
    state.rope_state = None
    state.prompt_embeds_state = None
    state._mm_prefix_prompt_token_ids = {}
    state.model_config = SimpleNamespace(hf_config=SimpleNamespace())
    return state


def _input_processor(
    ced_enabled=True, model_type="deepseek_v41", max_num_batched_tokens=512
):
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type, ced_prefill=ced_enabled),
        max_model_len=512,
        runner_type="generate",
        max_logprobs=20,
        get_vocab_size=lambda: 130000,
        logits_processors=None,
        is_diffusion=False,
        return_sampling_mask=False,
    )
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
        ),
        reasoning_config=None,
    )
    processor.renderer = SimpleNamespace(tokenizer=None, get_eos_token_id=lambda: None)
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.lora_config = None
    processor.generation_config_fields = {}
    processor.supports_mm_inputs = True
    processor.mm_encoder_cache_size = 512
    processor.skip_prompt_length_check = False
    processor.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_scheduled_tokens=None,
        long_prefill_token_threshold=0,
    )
    return processor


def _image_input(modality="image", *, prompt_len=3, offset=1, length=1):
    token_ids = [1] * prompt_len
    token_ids[-1] = 2
    token_ids[offset : offset + length] = [129264] * length
    return mm_input(
        token_ids,
        {modality: [None]},  # Encoder-cache hit still represents multimodal input.
        {modality: ["cached-feature"]},
        {modality: [PlaceholderRange(offset=offset, length=length)]},
    )


@pytest.mark.parametrize(
    "prompt, prompt_logprobs, feature",
    [
        (_image_input("video"), None, "multimodal inputs"),
        (embeds_input(torch.zeros(3, 8)), None, "prompt embeddings"),
        (tokens_input([1, 2, 3]), 0, "prompt log probabilities"),
        (tokens_input([1, 2, 3]), 1, "prompt log probabilities"),
    ],
)
def test_ced_unsupported_input_is_rejected_before_engine_request(
    prompt, prompt_logprobs, feature
):
    processor = _input_processor()
    params = SamplingParams(max_tokens=1, prompt_logprobs=prompt_logprobs)

    with pytest.raises(VLLMValidationError, match=feature) as exc_info:
        processor.process_inputs("unsupported", prompt, params, ("generate",))

    error = create_error_response(exc_info.value).error
    assert error.code == 400
    assert error.type == "BadRequestError"
    assert error.param == "ced_prefill"

    request = processor.process_inputs(
        "text", tokens_input([1, 2, 3]), SamplingParams(max_tokens=1), ("generate",)
    )
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.mm_features is None


def test_ced_accepts_image_before_engine_request():
    processor = _input_processor()

    request = processor.process_inputs(
        "image", _image_input(), SamplingParams(max_tokens=1), ("generate",)
    )

    assert request.mm_features is not None
    assert len(request.mm_features) == 1
    assert request.mm_features[0].modality == "image"
    assert request.mm_features[0].identifier == "cached-feature"
    assert request.mm_features[0].mm_position == PlaceholderRange(offset=1, length=1)
    assert request.prompt_token_ids == [1, 129264, 2]


def test_ced_rejects_image_replay_larger_than_the_worker_capacity():
    processor = _input_processor(max_num_batched_tokens=256)
    prompt = _image_input(prompt_len=400, offset=96, length=220)

    with pytest.raises(
        VLLMValidationError, match="image replay exceeds max_num_batched_tokens"
    ) as exc_info:
        processor.process_inputs(
            "oversized-image", prompt, SamplingParams(max_tokens=1), ("generate",)
        )

    error = create_error_response(exc_info.value).error
    assert error.code == 400
    assert error.type == "BadRequestError"
    assert error.param == "ced_prefill"


@pytest.mark.parametrize("image_length", [253, 254, 256])
def test_ced_image_admission_reserves_dspark_draft_slots(image_length):
    processor = _input_processor(max_num_batched_tokens=256)
    processor.speculative_config = SimpleNamespace(max_num_new_slots_for_drafting=3)
    prompt = _image_input(prompt_len=512, offset=0, length=image_length)
    params = SamplingParams(max_tokens=1)
    if image_length == 253:
        processor._validate_ced_inputs(prompt, params)
    else:
        with pytest.raises(
            VLLMValidationError, match="fit in one prefill chunk"
        ) as exc:
            processor._validate_ced_inputs(prompt, params)
        assert create_error_response(exc.value).error.code == 400


@pytest.mark.parametrize(
    "ced_enabled, model_type",
    [(False, "deepseek_v41"), (True, "other_model")],
)
def test_ced_input_guard_preserves_other_execution_paths(ced_enabled, model_type):
    processor = _input_processor(ced_enabled, model_type)
    request = processor.process_inputs(
        "image", _image_input(), SamplingParams(max_tokens=1), ("generate",)
    )
    assert request.mm_features is not None
    assert request.mm_features[0].modality == "image"
    assert request.prompt_token_ids == [1, 129264, 2]


@pytest.mark.parametrize("placeholders", [{}, {"image": []}])
def test_ced_accepts_text_rendered_by_multimodal_processor(placeholders):
    processor = _input_processor()
    request = processor.process_inputs(
        "text",
        mm_input([1, 2, 3], {}, {}, placeholders),
        SamplingParams(max_tokens=1),
        ("generate",),
    )
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.mm_features == []


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
        ({"mm_features": [_mm_feature("video")]}, "multimodal inputs"),
        ({"num_computed_tokens": 1}, "insufficient encoder replay"),
    ],
)
def test_ced_rejects_request_features_before_reset(overrides, message):
    state = _request_state()

    with pytest.raises(ValueError, match=message):
        state.add_request(0, _request(**overrides))

    state.ced_tail.reset.assert_not_called()
    assert state._ced_request_slots == {}


@pytest.mark.parametrize("draft_slots", [0, 3])
@pytest.mark.parametrize(
    ("prompt_len", "image_spans", "expected_ranges", "expected_replay_start"),
    [
        pytest.param(
            480,
            ((16, 220), (236, 220)),
            [(16, 235), (236, 455)],
            236,
            id="adjacent-second-crosses-tail",
        ),
        pytest.param(
            336,
            ((16, 220), (260, 48)),
            [(16, 235), (260, 307)],
            16,
            id="first-crosses-tail",
        ),
        pytest.param(
            160,
            ((40, 48), (96, 48)),
            [(40, 87), (96, 143)],
            32,
            id="both-inside-tail",
        ),
    ],
)
def test_ced_multi_image_request_retains_full_replay_ranges(
    draft_slots, prompt_len, image_spans, expected_ranges, expected_replay_start
):
    processor = _input_processor(max_num_batched_tokens=2048)
    if draft_slots:
        processor.speculative_config = SimpleNamespace(
            max_num_new_slots_for_drafting=draft_slots
        )
    token_ids = [7] * prompt_len
    placeholders = [
        PlaceholderRange(offset=offset, length=length) for offset, length in image_spans
    ]
    for offset, length in image_spans:
        token_ids[offset : offset + length] = [129264] * length
    prompt = mm_input(
        token_ids,
        {"image": [None, None]},
        {"image": ["cached-image-0", "cached-image-1"]},
        {"image": placeholders},
    )

    request = processor.process_inputs(
        "multi-image", prompt, SamplingParams(max_tokens=1), ("generate",)
    )
    assert request.mm_features is not None
    assert [
        (feature.modality, feature.identifier, feature.mm_position)
        for feature in request.mm_features
    ] == [
        ("image", "cached-image-0", placeholders[0]),
        ("image", "cached-image-1", placeholders[1]),
    ]
    state = _request_state()
    state.add_request(
        2, _request(prompt_len=prompt_len, mm_features=request.mm_features)
    )

    state.ced_tail.reset.assert_called_once_with(2, 0)
    assert state._ced_replay_starts == {2: expected_replay_start}
    assert state._ced_mm_ranges == {2: expected_ranges}


def test_ced_resets_tail_when_slot_zero_is_added_and_removed():
    state = _request_state()

    state.add_request(0, _request())
    state.remove_request("req")

    assert state.ced_tail.reset.call_args_list == [((0, 0),), ((0,),)]
    assert state._ced_request_slots == {}
    assert state._ced_prompt_lens == {}
    assert state._ced_replay_starts == {}
    assert state._ced_mm_ranges == {}


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
    state._ced_replay_starts = {9: 0, 3: 0}
    state._ced_mm_ranges = {9: [(4, 123)], 3: []}
    state.ced_tail = SimpleNamespace(window=384)
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
    assert step.plan.cache_window == 384
    assert decoder.common.query_start_loc_cpu.tolist() == [0, 1, 129]
    assert decoder.common.seq_lens.tolist() == [11, 128]
    assert decoder.common.is_prefilling.tolist() == [False, True]
    assert decoder.common.mm_req_doc_ranges == {0: [], 1: [(4, 123)]}
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
