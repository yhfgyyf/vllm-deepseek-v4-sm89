# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.config.engram import EngramConfig
from vllm.config.utils import hash_config
from vllm.config.vllm import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


def _model_config(tmp_path, architecture: str, layer_ids: tuple[int, ...] = (20, 24)):
    config = DeepseekV41Config(
        text_config={
            "engram_layer_ids": list(layer_ids),
            "max_position_embeddings": 128,
        },
    )
    config.architectures = [architecture]
    config.save_pretrained(tmp_path)
    return ModelConfig(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        max_model_len=128,
    )


def test_engram_config_hash_tracks_storage_mode():
    assert hash_config(EngramConfig(cpu_offload=True)) != hash_config(
        EngramConfig(cpu_offload=False)
    )


def test_engram_config_accepts_only_v41_engram_models(tmp_path):
    EngramConfig().verify_model_config(
        _model_config(tmp_path, "DeepseekV41ForCausalLM")
    )

    with pytest.raises(ValueError, match="supported Engram embeddings"):
        EngramConfig().verify_model_config(
            _model_config(tmp_path / "v4", "DeepseekV4ForCausalLM")
        )
    with pytest.raises(ValueError, match="non-empty n-gram layer ids"):
        EngramConfig().verify_model_config(
            _model_config(tmp_path / "empty", "DeepseekV41ForCausalLM", layer_ids=())
        )


def test_engine_args_converts_engram_mapping(tmp_path):
    _model_config(tmp_path, "DeepseekV41ForCausalLM")
    args = EngineArgs(model=str(tmp_path), engram_config={"cpu_offload": False})

    assert args.engram_config == EngramConfig(cpu_offload=False)


def test_vllm_config_resolves_engram_storage_and_hashes_it(tmp_path):
    config = VllmConfig(
        engram_config=EngramConfig(cpu_offload=True),
        model_config=_model_config(tmp_path, "DeepseekV41ForCausalLM"),
    )
    assert config.engram_config is not None
    assert config.engram_config.cpu_offload
    assert config.engram_config.dp_shared_memory is False
    assert hash_config(config.engram_config) == hash_config(
        EngramConfig(cpu_offload=True, dp_shared_memory=False)
    )
