# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.spec_decode import speculator as speculator_module
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module


def test_draft_metadata_uses_cpu_upper_bound_plus_step(monkeypatch) -> None:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        speculator_module, "build_attn_metadata", fake_build_attn_metadata
    )

    speculator = SimpleNamespace(
        arange=torch.arange(5, dtype=torch.int32),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.zeros((4, 2), dtype=torch.int32)],
            slot_mappings=torch.zeros((1, 12), dtype=torch.int64),
        ),
        input_buffers=SimpleNamespace(
            query_start_loc=torch.zeros(5, dtype=torch.int32),
            seq_lens=torch.zeros(4, dtype=torch.int32),
        ),
        attn_groups=[[]],
        kv_cache_config=object(),
        draft_max_seq_len=128,
        max_model_len=100,
    )

    out = speculator_module.DraftModelSpeculator._build_draft_attn_metadata(
        speculator,
        num_reqs=2,
        num_reqs_padded=4,
        num_tokens_padded=12,
        seq_lens_cpu_upper_bound=torch.tensor([10, 99], dtype=torch.int32),
        step=7,
        num_query_per_req=3,
        causal=False,
    )

    assert out == {"ok": True}
    assert captured["query_start_loc_cpu"].tolist() == [0, 3, 6, 6, 6]
    assert captured["max_query_len"] == 3
    assert captured["seq_lens_cpu_upper_bound"].tolist() == [17, 100, 0, 0]
    assert captured["causal"] is False


@pytest.mark.parametrize(
    ("support", "expected_mode"),
    [
        (AttentionCGSupport.UNIFORM_BATCH, CUDAGraphMode.FULL_DECODE_ONLY),
        (AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE, CUDAGraphMode.NONE),
    ],
)
def test_dflash_draft_graph_is_gated_by_attention_support(
    monkeypatch, support, expected_mode
) -> None:
    created = {}

    class FakeDFlashCudaGraphManager:
        def __init__(self, vllm_config, device, cudagraph_mode, **kwargs) -> None:
            created["mode"] = cudagraph_mode
            created["kwargs"] = kwargs

    monkeypatch.setattr(
        dflash_module, "DFlashCudaGraphManager", FakeDFlashCudaGraphManager
    )

    speculator = dflash_module.DFlashSpeculator.__new__(dflash_module.DFlashSpeculator)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_query_per_req = 7
    speculator.dflash_causal = False
    speculator.method = "dflash"
    speculator._speculator_name = "DFlash"
    speculator.attn_cg_support = SimpleNamespace(
        min_cg_support=support,
        min_cg_attn_backend="fake",
    )

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)

    assert created["mode"] == expected_mode
    assert created["kwargs"]["decode_query_len"] == 7
    assert "causal" not in created["kwargs"]


def test_dflash_capture_passes_group_causal() -> None:
    captured = {}

    class FakeDFlashCudaGraphManager:
        def capture(self, *args, **kwargs) -> None:
            captured.update(kwargs)

    speculator = dflash_module.DFlashSpeculator.__new__(dflash_module.DFlashSpeculator)
    speculator._speculator_name = "DFlash"
    speculator.sample_indices = torch.ones(1, dtype=torch.int64)
    speculator.sample_pos = torch.ones(1, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.zeros(1, dtype=torch.int32)
    speculator.query_cudagraph_manager = FakeDFlashCudaGraphManager()
    speculator._generate_draft = object()
    speculator.input_buffers = object()
    speculator.block_tables = object()
    speculator.attn_groups = []
    speculator.kv_cache_config = object()
    speculator.max_model_len = 128
    speculator._group_causal = {0: False, 1: True}

    speculator.capture()

    assert captured["causal"] == {0: False, 1: True}
    assert speculator.sample_indices.tolist() == [0]
    assert speculator.sample_pos.tolist() == [0]
    assert speculator.sample_idx_mapping.tolist() == [-1]


def test_dspark_draft_graph_is_enabled_with_uniform_batch_support(
    monkeypatch,
) -> None:
    created = {}

    class FakeDFlashCudaGraphManager:
        def __init__(self, vllm_config, device, cudagraph_mode, **kwargs) -> None:
            created["mode"] = cudagraph_mode

    monkeypatch.setattr(
        dflash_module, "DFlashCudaGraphManager", FakeDFlashCudaGraphManager
    )
    speculator = dflash_module.DFlashSpeculator.__new__(dflash_module.DFlashSpeculator)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_query_per_req = 7
    speculator.dflash_causal = False
    speculator.method = "dspark"
    speculator._speculator_name = "DSpark"
    speculator.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH,
        min_cg_attn_backend="fake",
    )

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)

    assert created["mode"] == CUDAGraphMode.FULL_DECODE_ONLY


def test_model_runner_sets_speculator_attention_before_graph_manager() -> None:
    source = model_runner_module.GPUModelRunner.initialize_kv_cache.__code__.co_names
    assert source.index("set_attn") < source.index("init_cudagraph_manager")
