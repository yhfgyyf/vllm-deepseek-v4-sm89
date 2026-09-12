# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only DFlash behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.model_executor.models.qwen3_dflash import (
    _dflash_layer_causal,
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
    _resolve_dspark_attention_backend,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu_model_runner import (
    _build_lookback_token_ids,
    _v41_mm_prefix_span,
)


def _config(num_hidden_layers, layer_types=None, causal_override=None, is_causal=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
        is_causal=is_causal,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # DFlash2 stores the explicit attention semantics at the top level.
        (
            _config(
                2,
                layer_types=["sliding_attention"] * 2,
                is_causal=False,
            ),
            True,
        ),
        (
            _config(2, layer_types=["full_attention"] * 2, is_causal=True),
            False,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def test_dflash_layer_causal_honors_top_level_override():
    config = _config(
        2,
        layer_types=["sliding_attention", "full_attention"],
        is_causal=False,
    )
    assert _dflash_layer_causal(config, 0) is False
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )


@pytest.mark.parametrize(
    "model_type,expected",
    [("deepseek_v4", (38, 39, 40)), ("deepseek_v41", (37, 38, 39))],
)
def test_dspark_eagle_aux_layer_index_semantics(model_type, expected):
    config = _vllm_config(model_type=model_type, dspark_target_layer_ids=[37, 38, 39])
    assert get_eagle3_aux_layers_from_config(config.speculative_config) == expected


def test_dspark_deepseek_v41_reuses_target_attention_backend():
    draft = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
    target = AttentionBackendEnum.FLASHINFER_MLA
    assert _resolve_dspark_attention_backend(draft, None, target) is target


@pytest.mark.parametrize(
    "computed,prompt,expected",
    [
        (0, 0, [-1, -1, -1]),
        (1, 3, [10, -1, -1]),
        (3, 3, [12, 11, 10]),
        (5, 3, [-1, -1, 12]),
    ],
)
def test_lookback_token_ids_masks_non_prompt_positions(computed, prompt, expected):
    tokens = np.array([[10, 11, 12, -7, -7, -7]], dtype=np.int32)
    result = _build_lookback_token_ids(
        tokens,
        np.array([computed], dtype=np.int32),
        np.array([prompt], dtype=np.int32),
        3,
    )
    np.testing.assert_array_equal(result[0], expected)


def test_lookback_token_ids_empty_warmup_batch():
    result = _build_lookback_token_ids(
        np.empty((0, 4), dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        3,
    )
    assert result.shape == (0, 3)


@pytest.mark.parametrize(
    "offset,length,expected",
    [(0, 8, (7, 7)), (1, 8, (7, 8)), (2, 8, (7, 9)), (8, 16, (15, 23))],
)
def test_v41_mm_prefix_span_leading_pad(offset, length, expected):
    assert _v41_mm_prefix_span(offset, length, 8) == expected
