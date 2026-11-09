# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-local dependencies and materialized CED boundary ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v4_1.common.pipeline import (
    get_sharing_dependencies,
    validate_pipeline_partition,
)
from vllm.models.deepseek_v4_1.nvidia import model as model_module
from vllm.models.deepseek_v4_1.nvidia.ced import CED_METADATA_KEY
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV41NativeMixedAttention,
)
from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV4Model
from vllm.models.deepseek_v4_1.nvidia.model_state import DeepseekV41ModelState
from vllm.models.deepseek_v4_1.nvidia.pp_state import copy_cache_rows
from vllm.sequence import IntermediateTensors


def config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        engram_layer_ids=[1, 14],
    )


@pytest.mark.parametrize("tp", [2, 4])
def test_pp2_keeps_persistent_and_per_forward_sources_stage_local(tp):
    hf = config()
    hf.tensor_parallel_size = tp
    ranges = [(0, 20), (20, 40)]
    validate_pipeline_partition(hf, ranges)
    dependencies = get_sharing_dependencies(hf, ranges)
    assert {d.kind for d in dependencies} == {"kv", "index", "index_k", "candidate"}
    assert all(d.source_stage == d.consumer_stage for d in dependencies)
    assert ("candidate", 20, 36) in {
        (d.kind, d.source_layer, d.consumer_layer) for d in dependencies
    }


def test_pp1_sharing_is_protected():
    validate_pipeline_partition(config(), [(0, 40)])


@pytest.mark.parametrize("last_start", [30, 31, 32])
def test_balanced_pp4_preserves_producer_ids_with_explicit_sharing_cuts(last_start):
    ranges = [(0, 10), (10, 20), (20, last_start), (last_start, 40)]
    validate_pipeline_partition(config(), ranges)
    cross = {
        (d.kind, d.source_layer, d.source_stage, d.consumer_stage)
        for d in get_sharing_dependencies(config(), ranges)
        if d.source_stage != d.consumer_stage
    }
    expected = {
        ("kv", 8, 0, 1),
        ("index", 8, 0, 1),
        ("kv", 20, 2, 3),
        ("index_k", 20, 2, 3),
        ("candidate", 20, 2, 3),
    }
    if last_start < 32:
        expected.add(("index", 28, 2, 3))
    assert cross == expected


@pytest.mark.parametrize("width", [68, 132, 288])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pp_cache_delta_replay_preserves_bytes_with_different_physical_pages(width):
    """Graph replay and rejected-token overwrites must not touch history rows."""
    source = torch.empty_strided(
        (4, 128, width),
        (128 * width + 128, width, 1),
        dtype=torch.uint8,
        device="cuda",
    )
    target = torch.empty_strided(
        (5, 128, width),
        (128 * width + 256, width, 1),
        dtype=torch.uint8,
        device="cuda",
    )
    source.random_(0, 256)
    target.fill_(37)
    source_slots = torch.tensor([131, -1, 0, 511], device="cuda")
    target_slots = torch.tensor([260, -1, 129, 639], device="cuda")
    packet = torch.empty((4, width), dtype=torch.uint8, device="cuda")

    def transfer():
        copy_cache_rows(source, source_slots, packet, restore=False)
        copy_cache_rows(target, target_slots, packet, restore=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        transfer()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transfer()
    for iteration in range(2):
        # A rejected speculative row is overwritten on the next replay.
        source.random_(0, 256)
        source_slots[0] = 131 + iteration
        expected = target.clone()
        expected[2, 4] = source[1, 3 + iteration]
        expected[1, 1] = source[0, 0]
        expected[4, 127] = source[3, 127]
        graph.replay()
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        assert not packet[1].any()


@pytest.mark.parametrize("include_topk", [False, True])
@pytest.mark.parametrize("scheduled,replay", [(8192, 0), (8192, 128), (1, 128)])
def test_pp4_ced_copies_compact_queries_and_full_cache_deltas(
    scheduled, replay, include_topk
):
    state = DeepseekV41ModelState.__new__(DeepseekV41ModelState)
    state.pp4_last_stage = state.ced_enabled = True
    batch = SimpleNamespace(has_prefill=True, num_tokens_after_padding=scheduled)
    query_keys = [
        "hidden_states",
        "pre_mix",
        "pp_candidates",
        "pp_token_ids",
    ]
    if include_topk:
        query_keys.append("pp_topk")
    cache_keys = ["pp_kv", "pp_index_k"]
    received = {
        key: torch.arange(replay if key in query_keys else scheduled)
        for key in query_keys + cache_keys
    }
    capacity = max(scheduled, replay)
    destination = IntermediateTensors(
        {key: torch.full((capacity,), -1) for key in received}
    )
    actual = state.prepare_pp_intermediate_tensors(destination, received, batch, False)
    for key, expected in received.items():
        torch.testing.assert_close(actual[key], expected)
        assert (destination[key][len(expected) :] == -1).all()


@pytest.mark.parametrize("tp", [2, 4])
def test_pp_hc_payload_keeps_full_hidden_width_for_each_tp_lane(tp):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=5120)
    model.hc_mult = 4
    model.pp4 = False
    payload = model.make_empty_intermediate_tensors(
        3, torch.bfloat16, torch.device("cpu")
    )
    assert set(payload.tensors) == {"hidden_states", "pre_mix"}
    assert payload["hidden_states"].shape == (3, 4, 5120)
    assert payload["hidden_states"].dtype == torch.bfloat16
    assert payload["pre_mix"].shape == (3, 4)
    assert payload["pre_mix"].dtype == torch.float32
    local_heads = 32 // tp
    assert (
        DeepseekV41NativeMixedAttention.get_padded_num_q_heads(local_heads)
        == local_heads
    )


def test_illegal_cut_reports_producer_consumer_before_construction():
    with pytest.raises(NotImplementedError, match="kv source layer 8.*layer 10"):
        validate_pipeline_partition(config(), [(0, 10), (10, 40)])


def test_closed_but_unvalidated_partition_is_not_silently_supported():
    with pytest.raises(NotImplementedError, match="20\\|20"):
        validate_pipeline_partition(config(), [(0, 8), (8, 14), (14, 20), (20, 40)])


@pytest.mark.parametrize("values", [[8, 2], [2, 2], [0, 2], [-1, 2], [2, 40]])
def test_invalid_sources_are_rejected(values):
    hf = config()
    hf.kv_source_layer_ids = values
    with pytest.raises(ValueError, match="kv_source_layer_ids"):
        validate_pipeline_partition(hf, [(0, 40)])


def test_candidate_producer_must_publish_indices():
    hf = config()
    hf.candidate_source_layer_id = 21
    with pytest.raises(ValueError, match="candidate source"):
        validate_pipeline_partition(hf, [(0, 20), (20, 40)])


def test_candidate_only_cross_stage_dependency_is_not_hidden_by_local_kv():
    hf = config()
    hf.candidate_source_layer_id = 14
    with pytest.raises(
        NotImplementedError, match="candidate source layer 14.*layer 20"
    ):
        validate_pipeline_partition(hf, [(0, 20), (20, 40)])


def test_engram_must_remain_on_first_stage():
    hf = config()
    hf.engram_layer_ids = [1, 20]
    with pytest.raises(NotImplementedError, match="Engram"):
        validate_pipeline_partition(hf, [(0, 20), (20, 40)])


def test_ced_receiving_stage_uses_materialized_boundary_once(monkeypatch):
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.use_mega_moe = model.use_sequence_parallel = False
    model.engram_hash = None
    model.start_layer, model.end_layer = 20, 40
    model.ced_enabled = True
    model.pp4 = False
    model.layers = nn.ModuleList([nn.Identity() for _ in range(40)])
    step = object()
    decoder = Mock(return_value=torch.ones(3, 8))
    monkeypatch.setattr(model, "_forward_ced_decoder", decoder)
    monkeypatch.setattr(
        model_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False)
    )
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        model_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={CED_METADATA_KEY: step}),
    )
    post = Mock(
        side_effect=AssertionError("materialized HC must not be reconstructed twice")
    )
    monkeypatch.setattr(model_module, "mhc_post_tilelang", post)
    boundary = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    pre_mix = torch.randn(3, 4, dtype=torch.float32)
    received = IntermediateTensors(
        {"hidden_states": boundary.clone(), "pre_mix": pre_mix.clone()}
    )
    ids, positions = torch.tensor([129264, 3, 5]), torch.arange(3)

    actual = model(ids, positions, received)

    assert actual is decoder.return_value
    post.assert_not_called()
    assert decoder.call_args.args[0] is step
    assert decoder.call_args.args[1].data_ptr() != boundary.data_ptr()
    torch.testing.assert_close(decoder.call_args.args[1], boundary)
    torch.testing.assert_close(decoder.call_args.args[2], pre_mix)
    assert decoder.call_args.args[3] is ids


def test_pp_aux_rejects_nonlocal_taps_but_keeps_capture_numbering():
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    model._set_aux_hidden_state_layers((37, 38, 39))
    assert model.aux_hidden_state_layers == (37, 38, 39)
    with pytest.raises(NotImplementedError, match="decoder-only"):
        model._set_aux_hidden_state_layers((19, 38, 39))


@pytest.mark.parametrize("dspark", [False, True])
@pytest.mark.parametrize(("tp", "draft_tp"), [(2, 2), (4, 4), (2, 4)])
def test_last_stage_dspark_embedding_is_real_and_loads_checkpoint_values(
    monkeypatch, dspark, tp, draft_tp
):
    hf = config()
    hf.vocab_size, hf.hidden_size = 16, 8
    hf.hc_mult, hf.hc_eps, hf.rms_norm_eps = 4, 1e-6, 1e-6
    hf.num_attention_heads, hf.index_topk = 32, 2
    parallel = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=tp,
        enable_expert_parallel=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, dtype=torch.bfloat16),
        quant_config=None,
        engram_config=None,
        parallel_config=parallel,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128),
        speculative_config=(
            SimpleNamespace(
                use_dspark=lambda: True, draft_tensor_parallel_size=draft_tp
            )
            if dspark
            else None
        ),
    )

    def embedding(vocab, hidden, **kwargs):
        return nn.Embedding(vocab, hidden)

    monkeypatch.setattr(model_module, "VocabParallelEmbedding", embedding)
    monkeypatch.setattr(model_module, "RMSNorm", nn.LayerNorm)
    monkeypatch.setattr(model_module.torch.cuda, "Stream", lambda: None)
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    monkeypatch.setattr(
        model_module,
        "make_layers",
        lambda *args, **kwargs: (
            20,
            40,
            nn.ModuleList([nn.Identity() for _ in range(40)]),
        ),
    )
    monkeypatch.setattr(model_module.EngramLayout, "from_config", lambda hf: None)
    monkeypatch.setattr(
        model_module, "get_tensor_model_parallel_world_size", lambda: tp
    )
    model = DeepseekV4Model(vllm_config=vllm_config)
    if not dspark or draft_tp != tp:
        assert isinstance(model.embed_tokens, PPMissingLayer)
        return
    model.get_expert_mapping = lambda: []
    weight = torch.arange(128).reshape(16, 8).float()
    loaded = model.load_weights([("embed_tokens.weight", weight)])
    assert loaded == {"embed_tokens.weight"}
    torch.testing.assert_close(
        model.embed_input_ids(torch.tensor([0, 3, 15])), weight[[0, 3, 15]]
    )
