# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def _handler(monkeypatch, last=False, num_spec=3):
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.is_last_rank = last
    handler.last_rank = 1
    handler.max_sample_len = num_spec + 1
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()
    handler.broadcast_stream = Mock()
    handler.broadcast_group = Mock()
    handler.queue = deque([None] * 2)
    handler.req_idx_gen_np = np.zeros(8, dtype=np.int32)
    handler.disabled = False
    handler.enable_adaptive_verification = False
    handler.aux_hidden_state_relay_keys = ()
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(
        pp_utils, "async_copy_to_gpu", lambda a, **kwargs: torch.from_numpy(a)
    )
    return handler


def _decode_batch():
    batch = _batch([10, 0, 10], [8, 100, 8], [4, 4, 4])
    batch.idx_mapping_np = np.array([2, 3, 4], dtype=np.int64)
    batch.idx_mapping = torch.from_numpy(batch.idx_mapping_np.copy())
    return batch


def test_sampled_counts_then_fresh_drafts_have_fixed_wire_shapes(monkeypatch):
    receiver = _handler(monkeypatch)
    batch = _decode_batch()
    payloads = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kwargs: payloads.append(tensor)
    )
    assert not receiver.receive(batch)
    assert [tuple(t.shape) for t in payloads] == [(3, 4), (2, 4), (3, 3)]
    sender = _handler(monkeypatch, last=True)
    payloads.clear()
    sender.broadcast(
        torch.tensor([[11], [22], [33]]),
        torch.ones(3, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
        batch,
    )
    table = torch.arange(24).reshape(8, 3)
    sender.broadcast_drafts(table, batch)
    assert [tuple(t.shape) for t in payloads] == [(3, 4), (2, 4), (3, 3)]
    assert payloads[0].tolist() == [[11, 0, 0, 0], [22, 0, 0, 0], [33, 0, 0, 0]]
    expected = table[batch.idx_mapping].clone()
    table.zero_()
    torch.testing.assert_close(payloads[2], expected)


def test_deferred_fifo_filters_cancelled_reused_and_prefill_slots(monkeypatch):
    handler = _handler(monkeypatch)
    batch = _decode_batch()
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    handler.receive(batch)
    slot = handler.queue[-1]
    assert slot is not None
    # Reuse/mutate the original batch arrays; the pending slot must be isolated.
    batch.idx_mapping_np[:] = 7
    handler.on_req_idx_freed(2)
    assert handler.get_prev_sampled_outputs() is None
    outputs = handler.get_prev_sampled_outputs()
    assert outputs is not None
    assert outputs["idx_mapping"].tolist() == [-1, -1, 4]
    assert outputs["broadcast_drafts"] is slot.draft_tokens
    handler.main_stream.wait_event.assert_called_once_with(slot.event)


def test_adaptive_confidences_follow_drafts_and_exclude_reused_slots(monkeypatch):
    receiver = _handler(monkeypatch)
    receiver.enable_adaptive_verification = True
    batch = _decode_batch()
    payloads = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda tensor, **kwargs: payloads.append(tensor)
    )
    receiver.receive(batch)
    assert [tuple(t.shape) for t in payloads] == [(3, 4), (2, 4), (3, 3), (3, 3)]
    slot = receiver.queue[-1]
    slot.draft_confidences.copy_(torch.arange(9).reshape(3, 3))
    receiver.on_req_idx_freed(2)
    assert receiver.get_prev_sampled_outputs() is None
    outputs = receiver.get_prev_sampled_outputs()
    assert outputs["confidence_indices"].tolist() == [4]
    assert outputs["broadcast_confidences"].tolist() == [[6.0, 7.0, 8.0]]

    sender = _handler(monkeypatch, last=True)
    sender.enable_adaptive_verification = True
    confidences = torch.rand(3, 3)
    expected = confidences.clone()
    payloads.clear()
    sender.broadcast_drafts(torch.zeros(8, 3, dtype=torch.int64), batch, confidences)
    confidences.zero_()
    assert len(payloads) == 2
    torch.testing.assert_close(payloads[1], expected)


@pytest.mark.parametrize("last", [False, True])
def test_disabled_warmup_enqueues_no_collectives(monkeypatch, last):
    handler = _handler(monkeypatch, last=last)
    handler.set_disabled(True)
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    batch = _decode_batch()
    if last:
        handler.broadcast(torch.zeros(3, 1, dtype=torch.int64), None, None, batch)
        handler.broadcast_drafts(torch.zeros(8, 3, dtype=torch.int64), batch)
    else:
        assert not handler.receive(batch)
    broadcast.assert_not_called()
    assert list(handler.queue) == [None, None]


@pytest.mark.parametrize("num_reqs", [1, 3, 4, 5])
def test_count_views_remain_aligned_for_odd_batches(num_reqs):
    counts = pp_utils._alloc_combined(num_reqs, torch.device("cpu"))
    assert counts.shape[1] >= num_reqs
    assert counts[0].data_ptr() % 16 == counts[1].data_ptr() % 16 == 0


def test_middle_stage_relays_aux_without_replacing_local_outputs(monkeypatch):
    handler = _handler(monkeypatch)
    handler.aux_hidden_state_relay_keys = ("aux_hidden_states_0",)
    upstream = torch.ones(2, 8)
    local = torch.full((2, 8), 2.0)
    result = handler.relay_aux_hidden_states(
        IntermediateTensors({"aux_hidden_states_0": upstream}),
        IntermediateTensors({"hidden_states": local, "aux_hidden_states_1": local}),
    )
    assert result["aux_hidden_states_0"] is upstream
    assert result["aux_hidden_states_1"] is local


@pytest.mark.parametrize("own_embed", [False, True])
def test_last_stage_draft_never_shares_missing_target_embedding(monkeypatch, own_embed):
    from types import SimpleNamespace

    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    draft_inner = torch.nn.Module()
    draft_inner.embed_tokens = torch.nn.Embedding(8, 4)
    own = draft_inner.embed_tokens
    target = SimpleNamespace(embed_tokens=PPMissingLayer())
    draft = SimpleNamespace(has_own_embed_tokens=own_embed)
    if own_embed:
        utils.maybe_share_target_embed(draft, draft_inner, target)
        assert draft_inner.embed_tokens is own
    else:
        with pytest.raises(RuntimeError, match="needs the target input embedding"):
            utils.maybe_share_target_embed(draft, draft_inner, target)


def test_last_stage_dspark_shares_loaded_target_weights(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=2))
    target = SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4))
    draft_inner = torch.nn.Module()
    draft_inner.embed_tokens = torch.nn.Embedding(8, 4)
    utils.maybe_share_target_embed(
        SimpleNamespace(has_own_embed_tokens=False), draft_inner, target
    )
    assert draft_inner.embed_tokens is target.embed_tokens


@pytest.mark.parametrize(
    "pp_size,load_format", [(1, "fastsafetensors"), (2, "auto"), (2, "fastsafetensors")]
)
def test_draft_only_loader_avoids_world_collectives(monkeypatch, pp_size, load_format):
    from types import SimpleNamespace

    from vllm.config import LoadConfig
    from vllm.v1.worker.gpu.spec_decode import utils

    monkeypatch.setattr(
        utils, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size)
    )
    original = LoadConfig(load_format=load_format)
    warning = Mock()
    monkeypatch.setattr(utils.logger, "warning_once", warning)
    draft = utils.get_pp_safe_draft_load_config(original)
    if pp_size == 2 and load_format == "fastsafetensors":
        assert draft.load_format == "auto"
        assert original.load_format == "fastsafetensors"
        warning.assert_called_once()
        assert "WORLD collectives" in warning.call_args.args[0]
        assert "target loader is unchanged" in warning.call_args.args[0]
    else:
        assert draft is original
        warning.assert_not_called()


@pytest.mark.parametrize("pp_size", [1, 2])
def test_draft_parallel_config_preserves_existing_dp_ep_settings(pp_size):
    from types import SimpleNamespace

    from vllm.config import ParallelConfig
    from vllm.v1.worker.gpu.spec_decode.utils import get_pp_draft_parallel_config

    target = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=pp_size,
        enable_expert_parallel=True,
        disable_custom_all_reduce=True,
        distributed_executor_backend="external_launcher",
    )
    config = SimpleNamespace(
        parallel_config=target,
        speculative_config=SimpleNamespace(
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=2)
        ),
    )
    draft = get_pp_draft_parallel_config(config)
    assert draft.pipeline_parallel_size == 1
    assert draft.tensor_parallel_size == 2
    assert draft.enable_expert_parallel
    assert draft.disable_custom_all_reduce
    assert draft.data_parallel_size == target.data_parallel_size
    assert target.pipeline_parallel_size == pp_size
    if pp_size == 1:
        assert draft is target


def test_cross_stage_dspark_parallel_config_keeps_target_and_draft_sizes_separate():
    from types import SimpleNamespace

    from vllm.config import ParallelConfig, SpeculativeConfig
    from vllm.v1.worker.gpu.spec_decode.utils import get_pp_draft_parallel_config

    target = ParallelConfig(
        tensor_parallel_size=2,
        pipeline_parallel_size=4,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
    )
    with pytest.raises(ValueError, match="target stage's TP"):
        SpeculativeConfig.create_draft_parallel_config(target, 4)
    draft = SpeculativeConfig.create_draft_parallel_config(
        target, 4, allow_pp_dspark_tp4=True
    )
    config = SimpleNamespace(
        parallel_config=target,
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_parallel_config=draft,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41")
            ),
        ),
    )
    execution = get_pp_draft_parallel_config(config)
    assert (execution.tensor_parallel_size, execution.pipeline_parallel_size) == (4, 1)
    assert execution.enable_expert_parallel
    assert (target.tensor_parallel_size, target.pipeline_parallel_size) == (2, 4)
    config.speculative_config.method = "eagle3"
    with pytest.raises(ValueError, match="target stage's TP"):
        get_pp_draft_parallel_config(config)


def test_draft_parallel_groups_restore_target_after_exception(monkeypatch):
    from vllm.distributed import parallel_state as state

    target_tp, target_ep, draft_tp, draft_ep = [object() for _ in range(4)]
    monkeypatch.setattr(state, "_TP", target_tp)
    monkeypatch.setattr(state, "_EP", target_ep)
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        state.use_draft_parallel_groups(draft_tp, draft_ep),
    ):
        assert state.get_tp_group() is draft_tp
        assert state.get_ep_group() is draft_ep
        with state.use_draft_parallel_groups(target_tp):
            assert state.get_tp_group() is target_tp
            assert state.get_ep_group() is draft_ep
        assert state.get_tp_group() is draft_tp
        raise RuntimeError("interrupted")
    assert state.get_tp_group() is target_tp
    assert state.get_ep_group() is target_ep


@pytest.mark.parametrize("ubatch_size", [0, 1, 2])
def test_cross_stage_draft_accepts_disabled_microbatch_defaults(
    monkeypatch, ubatch_size
):
    from types import SimpleNamespace

    from vllm.config import ParallelConfig
    from vllm.platforms import current_platform
    from vllm.v1.worker.gpu.spec_decode import pp_draft

    monkeypatch.setattr(current_platform, "device_count", lambda: 8)
    config = SimpleNamespace(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2, pipeline_parallel_size=4, ubatch_size=ubatch_size
        ),
        speculative_config=SimpleNamespace(
            method="dspark", draft_tensor_parallel_size=4
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41", ced_prefill=True)
        ),
        use_v2_model_runner=True,
    )
    created = object()
    monkeypatch.setattr(pp_draft, "PPDraftGroup", lambda *_: created)
    if ubatch_size < 2:
        assert (
            pp_draft.maybe_create_pp_draft_group(config, torch.device("cpu")) is created
        )
    else:
        with pytest.raises(NotImplementedError, match="microbatching"):
            pp_draft.maybe_create_pp_draft_group(config, torch.device("cpu"))


def test_cross_tp_dspark_keeps_loaded_weights_when_target_stage_has_no_head(
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm.model_executor import model_loader
    from vllm.model_executor.models import qwen3_dflash
    from vllm.model_executor.models import utils as model_utils
    from vllm.v1.worker.gpu.spec_decode.dspark import utils
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    head = torch.nn.Linear(4, 8, bias=False)
    draft = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4)),
        lm_head=head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=model_utils.PPMissingLayer()),
        lm_head=model_utils.PPMissingLayer(),
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_tensor_parallel_size=4,
            target_parallel_config=SimpleNamespace(tensor_parallel_size=2),
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), get_vocab_size=lambda: 8
            ),
            attention_backend=None,
            kv_cache_dtype=None,
        ),
        attention_config=SimpleNamespace(backend=None),
        load_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        model_config=SimpleNamespace(get_vocab_size=lambda: 8),
    )
    monkeypatch.setattr(utils, "replace", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        utils, "get_pp_draft_parallel_config", lambda _: SimpleNamespace()
    )
    monkeypatch.setattr(utils, "get_pp_safe_draft_load_config", lambda value: value)
    monkeypatch.setattr(utils, "_resolve_dspark_attention_backend", lambda *args: None)
    monkeypatch.setattr(model_utils, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(qwen3_dflash, "dflash_has_any_non_causal", lambda _: False)
    monkeypatch.setattr(model_loader, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(
        eagle_utils, "get_pp_group", lambda: SimpleNamespace(world_size=4)
    )
    assert utils.load_dspark_model(target, config) is draft
    assert draft.lm_head is head


def test_aux_intermediate_factory_reserves_persistent_remote_buffers(monkeypatch):
    from types import SimpleNamespace

    from vllm.distributed import parallel_state
    from vllm.v1.worker.gpu.spec_decode.eagle import eagle3_utils

    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=2, is_first_rank=False),
    )
    inner = SimpleNamespace(
        supports_aux_hidden_states_over_pp=True,
        _aux_slot_base_cached=2,
        AUX_HIDDEN_STATE_KEY="aux_hidden_states_",
        config=SimpleNamespace(hidden_size=8),
    )
    model = SimpleNamespace(
        get_language_model=lambda: inner,
        make_empty_intermediate_tensors=lambda n, dtype, device: IntermediateTensors(
            {"hidden_states": torch.zeros(n, 4, 8, dtype=dtype, device=device)}
        ),
    )
    eagle3_utils.verify_supports_aux_hidden_states_over_pp(model, "dspark")
    eagle3_utils.reserve_aux_intermediate_tensor_slots(model)
    tensors = model.make_empty_intermediate_tensors(5, torch.bfloat16, "cpu")
    assert set(tensors.tensors) == {
        "hidden_states",
        "aux_hidden_states_0",
        "aux_hidden_states_1",
    }
    assert tensors["aux_hidden_states_1"].shape == (5, 8)
    assert tensors["aux_hidden_states_1"].dtype == torch.bfloat16
