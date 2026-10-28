# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-backed decoder replay, not checkpoint quality or serving evaluation.

Use real delayed mHC, packed-cache codecs and native FlashInfer attention in
twenty small decoder layers. Dense FFNs stand in for independently tested MoE.
The encoder boundary is supplied explicitly to isolate the changed schedule.
"""

from itertools import islice
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.deepseek_v4_1.test_ced_dspark import speculator
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.kernels.mhc.deepseek_v41 import (
    hc_collapse_triton,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.deepseek_v4_1.common.ops import fused_q_rope_swa_insert
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v4_1.nvidia.ced import (
    CED_METADATA_KEY,
    CEDRequest,
    CEDStep,
    CEDTailState,
    plan_ced_step,
)
from vllm.models.deepseek_v4_1.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
)
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4Model,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
CAPACITY = 512
HIDDEN = 256
IMAGE_TOKEN_ID = 129264


def logical_swa_slots(positions, replay_start, image_span=None):
    """Build logical ``(causal SWA) OR (same image span)`` cache slots."""
    first = (positions - 127).clamp_min(0)
    end = positions + 1
    width = 128
    if image_span is not None:
        image_start, image_end = image_span
        in_image = (positions >= image_start) & (positions < image_end)
        first = torch.where(in_image, first.clamp_max(image_start), first)
        end = torch.where(in_image, end.clamp_min(image_end), end)
        width = max(width, image_end - min(image_start, replay_start))
    first = first.clamp_min(replay_start)
    slots = first[:, None] + torch.arange(width, device=positions.device)
    return torch.where(
        (slots < end[:, None]) & (slots >= replay_start), slots, -1
    ).int()


class PackedAttention(nn.Module):
    def __init__(self, global_cache, source=False):
        super().__init__()
        self.global_cache = global_cache
        self.source = source
        self.ced_global_kv_prebuilt = False
        self.cache = torch.zeros(
            CAPACITY // 32, 32, 528, device="cuda", dtype=torch.uint8
        )
        self.cs = torch.cat(
            (
                torch.ones(CAPACITY, 32, device="cuda"),
                torch.zeros(CAPACITY, 32, device="cuda"),
            ),
            -1,
        )
        self.rows_seen = []
        self.global_writes = []

    def build_ced_global_kv(self, hidden, positions):
        self.global_writes.append(positions.clone())
        rope_quant_insert(
            hidden.repeat(1, 2), positions, self.cs, self.global_cache, positions, 1
        )

    def forward(self, positions, hidden, _):
        from flashinfer.mla.deepseek_v41 import deepseek_v41_mixed_sparse_attention

        if self.source and not self.ced_global_kv_prebuilt:
            self.build_ced_global_kv(hidden, positions)
        self.rows_seen.append(positions.clone())
        kv = hidden.repeat(1, 2).contiguous()
        q = kv[:, None].expand(-1, 8, -1).contiguous()
        q = fused_q_rope_swa_insert(q, kv, positions, self.cs, self.cache, positions)
        metadata = get_forward_context().attn_metadata
        swa = logical_swa_slots(
            positions, metadata["replay_start"], metadata.get("image_span")
        )
        global_slots = torch.arange(CAPACITY, device="cuda")[None].expand(
            len(positions), -1
        )
        global_slots = torch.where(
            global_slots <= positions[:, None], global_slots, -1
        ).int()
        out = deepseek_v41_mixed_sparse_attention(
            q,
            self.cache,
            self.global_cache,
            swa.contiguous(),
            global_slots.contiguous(),
            512**-0.5,
        )
        return (out[:, 0, :HIDDEN] * 0.03).contiguous()


class DenseFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.002
        )
        self.input_ids_seen = []

    def forward(self, hidden, input_ids):
        self.input_ids_seen.append(input_ids.clone())
        return (hidden @ self.weight).contiguous()


def make_model(seed, tail_capacity=128):
    torch.manual_seed(seed)
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=HIDDEN)
    model.ced_tail = CEDTailState(1, HIDDEN, window=tail_capacity).cuda()
    model.norm = RMSNorm(HIDDEN, 1e-6).cuda().bfloat16()
    global_cache = torch.zeros(
        CAPACITY // 128, 128, 288, device="cuda", dtype=torch.uint8
    )
    layers = [nn.Identity() for _ in range(20)]
    for index in range(20):
        layer = DeepseekV4DecoderLayer.__new__(DeepseekV4DecoderLayer)
        nn.Module.__init__(layer)
        layer.hc_mult = 4
        layer.hc_eps = layer.rms_norm_eps = 1e-6
        layer.hc_post_alpha = 2.0
        layer.hc_sinkhorn_iters = 20
        layer.use_sequence_parallel = False
        layer.engram = None
        layer.attn_norm = RMSNorm(HIDDEN, 1e-6).cuda().bfloat16()
        layer.ffn_norm = RMSNorm(HIDDEN, 1e-6).cuda().bfloat16()
        for sublayer in ("attn", "ffn"):
            setattr(
                layer,
                f"hc_{sublayer}_fn",
                nn.Parameter(torch.randn(24, HIDDEN * 4, device="cuda") * 0.001),
            )
            setattr(
                layer,
                f"hc_{sublayer}_base",
                nn.Parameter(torch.zeros(24, device="cuda")),
            )
            setattr(
                layer,
                f"hc_{sublayer}_scale",
                nn.Parameter(torch.full((3,), 0.1, device="cuda")),
            )
        layer.hc_attn_fn_broadcast = None
        layer.attn = PackedAttention(global_cache, source=index == 0)
        layer.ffn = DenseFFN()
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


def context(start=0, image_span=None):
    attn_metadata = {"replay_start": start}
    if image_span is not None:
        attn_metadata["image_span"] = image_span
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=attn_metadata,
        slot_mapping={},
        skip_compiled=True,
    )


def explicit_replay_reference(
    model, boundary, pre_mix, input_ids, positions, replay_start, image_span
):
    """Independent schedule: build all global rows, slice once, run twenty layers."""
    layer = model.layers[20]
    _, _, normalized, _ = mhc_pre_delayed_tilelang(
        boundary,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        layer.rms_norm_eps,
        layer.hc_eps,
        layer.hc_eps,
        layer.hc_post_alpha,
        layer.hc_sinkhorn_iters,
        pre_mix=pre_mix,
        norm_weight=layer.attn_norm.weight,
        norm_eps=layer.attn_norm.variance_epsilon,
    )
    layer.attn.build_ced_global_kv(normalized, positions)
    layer.attn.ced_global_kv_prebuilt = True
    hidden = boundary[replay_start:].contiguous()
    carried = pre_mix[replay_start:].contiguous()
    token_ids = input_ids[replay_start:].contiguous()
    query_positions = positions[replay_start:]
    residual = post = res = None
    auxiliary = []
    with override_forward_context(context(replay_start, image_span)):
        for idx, layer in enumerate(islice(model.layers, 20, 40), start=20):
            hidden, residual, post, res, carried = layer(
                hidden, query_positions, token_ids, carried, post, res, residual
            )
            if idx + 1 in (37, 38, 39):
                auxiliary.append(mhc_post_tilelang(hidden, residual, post, res).mean(1))
        hidden = mhc_post_tilelang(hidden, residual, post, res)
        return model.norm(hc_collapse_triton(hidden, carried)), auxiliary


def transfer_prefix_block(
    worker: CPUOffloadingWorker, job_id: int, gpu_to_cpu: bool
) -> int:
    gpu_spec = GPULoadStoreSpec([0], group_sizes=(1,), block_indices=(0,))
    cpu_spec = CPULoadStoreSpec([0])
    if gpu_to_cpu:
        assert worker.submit_store(job_id, gpu_spec, cpu_spec)
    else:
        assert worker.submit_load(job_id, cpu_spec, gpu_spec)
    worker.wait({job_id})
    results = worker.get_finished()
    assert len(results) == 1
    assert results[0].job_id == job_id
    assert results[0].success
    assert results[0].transfer_size is not None
    return results[0].transfer_size


def make_ced_cpu_connector(model, monkeypatch):
    source = model.layers[20].attn
    scratch_before = model.layers[20].attn.cache
    scratch_after = model.layers[39].attn.cache
    scratch_spec = CircularBufferSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=264,
        dtype=torch.uint8,
    )
    global_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=288,
        dtype=torch.uint8,
    )
    scratch_page = scratch_spec.page_size_bytes
    global_page = global_spec.page_size_bytes
    kv_cache_config = KVCacheConfig(
        num_blocks=CAPACITY // 128,
        kv_cache_tensors=[
            KVCacheTensor(scratch_before.numel(), ["scratch_before"], 0, scratch_page),
            KVCacheTensor(global_page * (CAPACITY // 128), ["global"], 0, global_page),
            KVCacheTensor(scratch_after.numel(), ["scratch_after"], 0, scratch_page),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["scratch_before"], scratch_spec),
            KVCacheGroupSpec(["global"], global_spec),
            KVCacheGroupSpec(["scratch_after"], scratch_spec),
        ],
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
        world_size=1,
        rank=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel,
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 1),
    )
    config = OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=128,
                layer_names=("global",),
                group_id=1,
            ),
        ),
        worker_kv_bytes_per_block=global_page,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": global_page},
        engine_id="ced-kernel-offload",
        model=OffloadingModelConfig(name="deepseek-v4.1-test", dtype="uint8"),
        cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )
    monkeypatch.setattr(CPUOffloadingSpec, "_uses_shared_region", lambda self: False)
    spec = CPUOffloadingSpec(config)
    connector = OffloadingConnectorWorker(spec, vllm_config, kv_cache_config)
    connector.register_kv_caches(
        {
            "scratch_before": scratch_before,
            "global": source.global_cache,
            "scratch_after": scratch_after,
        }
    )
    assert spec.group_ids == (1,)
    assert isinstance(connector.worker, CPUOffloadingWorker)
    assert len(connector.worker._load_handler.layer_refs_per_group) == 1
    gpu_tensors = connector.worker._store_handler.src_tensors
    assert len(gpu_tensors) == 1
    assert gpu_tensors[0].data_ptr() == source.global_cache.data_ptr()
    return connector


@torch.inference_mode()
def test_logical_image_slots_expand_only_queries_inside_the_image():
    image_span = (64, 320)
    positions = torch.tensor([64, 191, 319, 320, 399], device="cuda")

    slots = logical_swa_slots(positions, replay_start=64, image_span=image_span)

    image_slots = torch.arange(*image_span, device="cuda", dtype=torch.int32)
    for row in range(3):
        torch.testing.assert_close(slots[row], image_slots)
    for row, position in ((3, 320), (4, 399)):
        expected = torch.arange(
            position - 127, position + 1, device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(slots[row, :128], expected)
        assert (slots[row, 128:] == -1).all()


@torch.inference_mode()
@pytest.mark.parametrize(
    "length,chunk,prefix_hit,image_span,tail_capacity",
    [
        (127, 127, 0, None, 128),
        (129, 128, 0, None, 128),
        (257, 128, 0, None, 128),
        (257, 128, 128, None, 128),
        (400, 128, 0, (64, 320), 384),
        (400, 128, 64, (64, 320), 384),
    ],
)
@pytest.mark.parametrize("dspark", [False, True])
def test_ced_chunked_decoder_matches_independent_bounded_replay(
    length,
    chunk,
    prefix_hit,
    image_span,
    tail_capacity,
    dspark,
    default_vllm_config,
):
    torch.manual_seed(83)
    boundary = torch.randn(length, 4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(length, 4, device="cuda"), -1)
    positions = torch.arange(length, device="cuda")
    input_ids = positions.clone()
    if image_span is not None:
        image_start, image_end = image_span
        input_ids[image_start:image_end] = IMAGE_TOKEN_ID
    replay_start = image_span[0] if image_span is not None else max(0, length - 128)
    reference = make_model(14, tail_capacity)
    expected, expected_aux = explicit_replay_reference(
        reference,
        boundary,
        pre_mix,
        input_ids,
        positions,
        replay_start,
        image_span,
    )
    actual_model = make_model(14, tail_capacity)
    actual_model.aux_hidden_state_layers = (37, 38, 39) if dspark else ()
    if prefix_hit:
        # Only prefix-independent global KV is shared. No decoder SWA or
        # encoder hidden-tail values are copied from the previous request.
        actual_model.layers[20].attn.global_cache.flatten(0, 1)[:prefix_hit].copy_(
            reference.layers[20].attn.global_cache.flatten(0, 1)[:prefix_hit]
        )
        actual_model.ced_tail.reset(0, prefix_hit)
    for first in range(prefix_hit, length, chunk):
        end = min(first + chunk, length)
        plan = plan_ced_step(
            (
                CEDRequest(
                    0,
                    first,
                    end - first,
                    length,
                    True,
                    replay_start=replay_start,
                ),
            ),
            cache_window=tail_capacity,
        )
        decoder_pos = torch.tensor(plan.positions, device="cuda", dtype=torch.int64)
        decoder_metadata = {"replay_start": replay_start}
        if image_span is not None:
            decoder_metadata["image_span"] = image_span
        step = CEDStep(plan, decoder_metadata, decoder_pos)
        ctx = context()
        old_metadata, old_slots = ctx.attn_metadata, ctx.slot_mapping
        with override_forward_context(ctx):
            actual = actual_model._forward_ced_decoder(
                step,
                boundary[first:end],
                pre_mix[first:end],
                input_ids[first:end],
                positions[first:end],
            )
        assert ctx.attn_metadata is old_metadata and ctx.slot_mapping is old_slots
        assert not actual_model.layers[20].attn.ced_global_kv_prebuilt
        if dspark:
            actual, dense_aux = actual
            assert dense_aux == []
            assert step.draft_context is not None
            torch.testing.assert_close(step.draft_context.positions, decoder_pos)
            assert (step.draft_context.request_indices == 0).all()
        if end != length:
            assert torch.count_nonzero(actual) == 0
            assert not actual_model.layers[20].attn.rows_seen
            if dspark:
                assert step.draft_context.aux_hidden_states == []
        else:
            torch.testing.assert_close(actual[-1], expected[-1], rtol=0.005, atol=0.005)
            if dspark:
                for actual_aux, ref_aux in zip(
                    step.draft_context.aux_hidden_states, expected_aux
                ):
                    torch.testing.assert_close(
                        actual_aux, ref_aux, rtol=0.005, atol=0.005
                    )
                check_draft_context(step)
    source = actual_model.layers[20].attn
    torch.testing.assert_close(torch.cat(source.global_writes), positions[prefix_hit:])
    replay_positions = positions[replay_start:]
    replay_input_ids = input_ids[replay_start:]
    for layer in islice(actual_model.layers, 20, 40):
        assert len(layer.attn.rows_seen) == 1
        torch.testing.assert_close(layer.attn.rows_seen[0], replay_positions)
        assert len(layer.ffn.input_ids_seen) == 1
        torch.testing.assert_close(layer.ffn.input_ids_seen[0], replay_input_ids)
    assert actual_model.ced_tail.ends == [length]
    assert actual_model.ced_tail.valid_starts == [replay_start]
    tail_slots = replay_positions.remainder(tail_capacity)
    torch.testing.assert_close(
        actual_model.ced_tail.input_ids[tail_slots], replay_input_ids
    )
    torch.testing.assert_close(
        source.global_cache, reference.layers[20].attn.global_cache, rtol=0, atol=0
    )

    # A new decode token must consume the replay-built SWA/global history.
    next_boundary = torch.randn(1, 4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    next_mix = torch.softmax(torch.randn(1, 4, device="cuda"), -1)
    next_position = torch.tensor([length], device="cuda")
    hidden, carried = next_boundary, next_mix
    residual = post = res = None
    reference.layers[20].attn.ced_global_kv_prebuilt = False
    with override_forward_context(context(image_span=image_span)):
        for layer in islice(reference.layers, 20, 40):
            hidden, residual, post, res, carried = layer(
                hidden, next_position, next_position, carried, post, res, residual
            )
        hidden = mhc_post_tilelang(hidden, residual, post, res)
        expected_next = reference.norm(hc_collapse_triton(hidden, carried))
    decode_plan = plan_ced_step(
        (CEDRequest(0, length, 1, length, False),), cache_window=tail_capacity
    )
    decode_step = CEDStep(decode_plan, {"replay_start": 0}, next_position)
    with override_forward_context(context()):
        actual_next = actual_model._forward_ced_decoder(
            decode_step, next_boundary, next_mix, next_position, next_position
        )
    if dspark:
        actual_next, _ = actual_next
        assert len(decode_step.draft_context.aux_hidden_states) == 3
    torch.testing.assert_close(actual_next, expected_next, rtol=0.005, atol=0.005)
    for layer in islice(actual_model.layers, 20, 40):
        torch.testing.assert_close(layer.ffn.input_ids_seen[-1], next_position)


@torch.inference_mode()
def test_ced_cpu_offload_restores_prefix_and_rebuilds_private_state(
    default_vllm_config,
    monkeypatch,
):
    length = 257
    prefix_hit = 128
    replay_start = length - 128
    torch.manual_seed(97)
    boundary = torch.randn(length, 4, HIDDEN, device="cuda", dtype=torch.bfloat16)
    pre_mix = torch.softmax(torch.randn(length, 4, device="cuda"), -1)
    positions = torch.arange(length, device="cuda")
    input_ids = positions.clone()

    model = make_model(29)
    model.aux_hidden_state_layers = (37, 38, 39)
    reference_plan = plan_ced_step(
        (
            CEDRequest(
                0,
                0,
                length,
                length,
                True,
                replay_start=replay_start,
            ),
        ),
        cache_window=128,
    )
    reference_positions = torch.tensor(
        reference_plan.positions, device="cuda", dtype=torch.int64
    )
    reference_step = CEDStep(
        reference_plan,
        {"replay_start": replay_start},
        reference_positions,
    )
    with override_forward_context(context()):
        reference_output, dense_aux = model._forward_ced_decoder(
            reference_step, boundary, pre_mix, input_ids, positions
        )
    assert dense_aux == []
    assert reference_step.draft_context is not None
    reference_aux = [
        hidden.clone() for hidden in reference_step.draft_context.aux_hidden_states
    ]

    source = model.layers[20].attn
    global_cache = source.global_cache
    full_global_cache = global_cache.clone()
    page_size_bytes = global_cache[0].numel() * global_cache.element_size()
    connector = make_ced_cpu_connector(model, monkeypatch)
    offloader = connector.worker
    assert isinstance(offloader, CPUOffloadingWorker)
    try:
        stored_bytes = transfer_prefix_block(offloader, 1, gpu_to_cpu=True)

        global_cache.zero_()
        for layer in islice(model.layers, 20, 40):
            layer.attn.cache.zero_()
            layer.attn.rows_seen.clear()
            layer.attn.global_writes.clear()
            layer.ffn.input_ids_seen.clear()
        model.ced_tail.hidden.fill_(torch.nan)
        model.ced_tail.pre_mix.fill_(torch.nan)
        model.ced_tail.input_ids.fill_(-1)
        model.ced_tail.reset(0, prefix_hit)

        loaded_bytes = transfer_prefix_block(offloader, 2, gpu_to_cpu=False)
        torch.testing.assert_close(global_cache[0], full_global_cache[0])
        assert torch.count_nonzero(global_cache[1:]) == 0
        for layer in islice(model.layers, 20, 40):
            assert torch.count_nonzero(layer.attn.cache) == 0

        recovery_plan = plan_ced_step(
            (
                CEDRequest(
                    0,
                    prefix_hit,
                    length - prefix_hit,
                    length,
                    True,
                    replay_start=replay_start,
                ),
            ),
            cache_window=128,
        )
        recovery_positions = torch.tensor(
            recovery_plan.positions, device="cuda", dtype=torch.int64
        )
        recovery_step = CEDStep(
            recovery_plan,
            {"replay_start": replay_start},
            recovery_positions,
        )
        with override_forward_context(context()):
            recovered_output, dense_aux = model._forward_ced_decoder(
                recovery_step,
                boundary[prefix_hit:],
                pre_mix[prefix_hit:],
                input_ids[prefix_hit:],
                positions[prefix_hit:],
            )
    finally:
        connector.shutdown()

    assert stored_bytes == loaded_bytes == page_size_bytes
    torch.testing.assert_close(
        recovered_output[-1], reference_output[-1], rtol=0.005, atol=0.005
    )
    assert dense_aux == []
    assert recovery_step.draft_context is not None
    for recovered_aux, expected_aux in zip(
        recovery_step.draft_context.aux_hidden_states,
        reference_aux,
        strict=True,
    ):
        torch.testing.assert_close(recovered_aux, expected_aux, rtol=0.005, atol=0.005)
    check_draft_context(recovery_step)

    torch.testing.assert_close(global_cache, full_global_cache, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(source.global_writes), positions[prefix_hit:], rtol=0, atol=0
    )
    assert model.ced_tail.ends == [length]
    assert model.ced_tail.valid_starts == [replay_start]
    tail_slots = positions[replay_start:].remainder(128)
    assert not torch.isnan(model.ced_tail.hidden[tail_slots]).any()
    assert not torch.isnan(model.ced_tail.pre_mix[tail_slots]).any()
    torch.testing.assert_close(
        model.ced_tail.input_ids[tail_slots], input_ids[replay_start:]
    )
    for layer in islice(model.layers, 20, 40):
        cache = layer.attn.cache.flatten(0, 1)
        assert torch.count_nonzero(cache[:replay_start]) == 0
        assert torch.count_nonzero(cache[replay_start:length]) > 0


class ContextProjection(nn.Linear):
    def forward(self, hidden):
        return super().forward(hidden), None


def check_draft_context(step):
    """Real V4.1 draft projection/codec, then native C0 noncausal graph replay.

    Small random weights stand in for checkpoint weights; the production DSpark
    context method and new speculator hook both execute, without a model server.
    """
    from flashinfer.mla.deepseek_v41 import (
        deepseek_v41_mixed_sparse_attention,
        deepseek_v41_pack_swa_cache,
    )

    draft = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    nn.Module.__init__(draft)
    draft.main_proj = nn.Linear(
        HIDDEN * 3, HIDDEN, bias=False, device="cuda", dtype=torch.bfloat16
    )
    draft.main_norm = RMSNorm(HIDDEN, 1e-6).cuda().bfloat16()
    draft.layers = nn.ModuleList()
    for _ in range(3):
        layer = nn.Module()
        attn = nn.Module()
        attn.fused_wqa_wkv = ContextProjection(
            HIDDEN, 16 + 512, bias=False, device="cuda", dtype=torch.bfloat16
        )
        attn.q_lora_rank = 16
        attn.n_local_heads, attn.head_dim = 8, 512
        attn.kv_norm = RMSNorm(512, 1e-6).cuda().bfloat16()
        attn.swa_cache_layer = SimpleNamespace(
            kv_cache=torch.zeros(36, 32, 528, device="cuda", dtype=torch.uint8)
        )
        attn.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.cat(
                (
                    torch.ones(CAPACITY, 32, device="cuda"),
                    torch.zeros(CAPACITY, 32, device="cuda"),
                ),
                -1,
            )
        )
        layer.attn = attn
        draft.layers.append(layer)
    wrapper = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = draft
    runner = speculator(1, "cuda")
    runner.model = wrapper
    runner.block_tables.kernel_block_sizes = [32, 32]
    runner.block_tables.block_sizes = [32, 32]
    runner.block_tables.input_block_tables = [
        torch.arange(2, 18, device="cuda")[None],
        torch.arange(20, 36, device="cuda")[None],
    ]
    payload = step.draft_context
    positions = payload.positions
    end = int(positions[-1]) + 1
    batch = SimpleNamespace(
        num_reqs=1, num_tokens=1, seq_lens=torch.tensor([end], device="cuda")
    )
    runner._precompute_context_kv(
        batch,
        {CED_METADATA_KEY: step},
        torch.zeros(1, HIDDEN, device="cuda"),
        [],
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        dummy_run=False,
    )
    main = draft.combine_hidden_states(torch.cat(payload.aux_hidden_states, -1))
    for layer, base in zip(draft.layers, (20, 2, 20)):
        projected, _ = layer.attn.fused_wqa_wkv(main)
        kv = layer.attn.kv_norm(projected[:, 16:])
        cache = layer.attn.swa_cache_layer.kv_cache
        expected_cache = torch.zeros_like(cache)
        deepseek_v41_pack_swa_cache(kv, positions + base * 32, expected_cache)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        assert not torch.count_nonzero(cache[0])  # null block untouched

    # Five noncausal queries consume all128 replay context rows plus each other.
    attn = draft.layers[0].attn
    cache = attn.swa_cache_layer.kv_cache
    query_positions = torch.arange(end, end + 5, device="cuda")
    query_kv = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
    query = query_kv[:, None].expand(-1, 8, -1).contiguous()
    query = fused_q_rope_swa_insert(
        query,
        query_kv,
        query_positions,
        attn.rotary_emb.cos_sin_cache,
        cache,
        query_positions + 20 * 32,
    )
    selected = torch.arange(max(0, end - 128), end + 5, device="cuda") + 20 * 32
    slots = torch.full((5, 256), -1, device="cuda", dtype=torch.int32)
    slots[:, : len(selected)] = selected.int()
    rows = cache.flatten(0, 1)[selected]
    values = rows[:, :512].view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    scales = rows[:, 512:].view(torch.float8_e8m0fnu).float()
    decoded = (values * scales[:, :, None]).flatten(1)
    probabilities = (query.float() @ decoded.T * (512**-0.5)).softmax(-1).bfloat16()
    expected = (probabilities.float() @ decoded).bfloat16()

    def run():
        return deepseek_v41_mixed_sparse_attention(
            query, cache, None, slots, None, 512**-0.5
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(5):
        graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
