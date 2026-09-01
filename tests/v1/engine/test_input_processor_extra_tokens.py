# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.multimodal.inputs import PlaceholderRange
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.v1.engine.input_processor import InputProcessor


def _processor(hf_config):
    processor = InputProcessor.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        hf_config=hf_config,
        max_model_len=16,
        runner_type="generate",
        get_vocab_size=lambda: 100,
    )
    processor.renderer = SimpleNamespace(tokenizer=SimpleNamespace(max_token_id=99))
    processor.mm_encoder_cache_size = 16
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = False
    return processor


def test_declared_extra_input_token_id_is_allowed():
    processor = _processor(SimpleNamespace(extra_input_token_ids=[1005]))

    processor._validate_model_input(
        {
            "type": "multimodal",
            "prompt_token_ids": [1, 1005],
            "mm_placeholders": {"image": [PlaceholderRange(offset=1, length=1)]},
        },
        "decoder",
    )


def test_declared_extra_input_token_range_is_allowed():
    processor = _processor(
        SimpleNamespace(extra_input_token_ranges=[{"start": 1000, "end": 1010}])
    )

    processor._validate_model_input(
        {
            "type": "multimodal",
            "prompt_token_ids": [1, 1009],
            "mm_placeholders": {"image": [PlaceholderRange(offset=1, length=1)]},
        },
        "decoder",
    )


def test_undeclared_extra_input_token_still_rejected():
    processor = _processor(SimpleNamespace(extra_input_token_ranges=[(1000, 1010)]))

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [1010]},
            "decoder",
        )


def test_negative_input_token_still_rejected():
    processor = _processor(SimpleNamespace(extra_input_token_ranges=[(-10, 0)]))

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [-1]},
            "decoder",
        )


def test_deepseek_v4_vision_config_declares_sentinel_token_range():
    config = DeepseekV4Config(vocab_size=1000, vision_n_layers=1)
    processor = _processor(config)

    processor._validate_model_input(
        {
            "type": "multimodal",
            "prompt_token_ids": [1000, 1001, 1002, 1003, 1004],
            "mm_placeholders": {"image": [PlaceholderRange(offset=0, length=5)]},
        },
        "decoder",
    )

    assert config.vllm_mm_prefix_start_token_id == 1000
    assert config.vllm_mm_prefix_end_token_id == 1004
    assert config.vision_n_layers == 1
    assert config.is_mm_prefix_lm


def test_declared_extra_input_token_requires_multimodal_placeholder():
    processor = _processor(SimpleNamespace(extra_input_token_ids=[1005]))

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [1, 1005]},
            "decoder",
        )

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(
            {
                "type": "multimodal",
                "prompt_token_ids": [1, 1005],
                "mm_placeholders": {"image": [PlaceholderRange(offset=0, length=1)]},
            },
            "decoder",
        )


def test_deepseek_v4_text_config_does_not_allow_sentinel_tokens():
    config = DeepseekV4Config(vocab_size=1000, vision_n_layers=0)
    processor = _processor(config)

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [1000]},
            "decoder",
        )
