# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.engram import EngramConfig
from vllm.config.vllm import VllmConfig
from vllm.engine.arg_utils import EngineArgs


def _model_config(architecture: str, layer_ids: tuple[int, ...] = (20, 24)):
    return SimpleNamespace(
        architecture=architecture,
        hf_text_config=SimpleNamespace(engram_layer_ids=layer_ids),
    )


def test_engram_config_hash_tracks_storage_mode():
    assert (
        EngramConfig(cpu_offload=True).compute_hash()
        != EngramConfig(cpu_offload=False).compute_hash()
    )


def test_engram_config_accepts_only_v41_engram_models(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)

    EngramConfig().verify_model_config(_model_config("DeepseekV41ForCausalLM"))

    with pytest.raises(ValueError, match="supported Engram embeddings"):
        EngramConfig().verify_model_config(_model_config("DeepseekV4ForCausalLM"))
    with pytest.raises(ValueError, match="non-empty engram_layer_ids"):
        EngramConfig().verify_model_config(
            _model_config("DeepseekV41ForCausalLM", layer_ids=())
        )


def test_engine_args_converts_engram_mapping():
    args = EngineArgs(model="dummy", engram_config={"cpu_offload": False})

    assert args.engram_config == EngramConfig(cpu_offload=False)


def test_verify_engram_config_logs_hashable_value(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_cuda", lambda: True)
    config = SimpleNamespace(
        engram_config=EngramConfig(cpu_offload=True),
        model_config=_model_config("DeepseekV41ForCausalLM"),
        speculative_config=None,
    )

    VllmConfig._verify_engram_config(config)
