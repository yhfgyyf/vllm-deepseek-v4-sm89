# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, cast

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal.utils import get_mm_safe_replay_start
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, SlotMappingPolicy
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from .ced import (
    CED_METADATA_KEY,
    CEDRequest,
    CEDStep,
    clamp_replay_swa,
    plan_ced_step,
)


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


class DeepseekV41ModelState(DefaultModelState):
    """DeepSeek V4.1 runner state for engram lookback and opt-in CED replay.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        inner_model = getattr(model, "language_model", model).model
        self.ced_enabled = bool(getattr(inner_model, "ced_enabled", False))
        self.ced_tail = getattr(inner_model, "ced_tail", None)
        if self.ced_enabled and self.ced_tail is None:
            raise RuntimeError("CED is enabled without its reserved tail state")
        self._ced_request_slots: dict[str, int] = {}
        self._ced_prompt_lens: dict[int, int] = {}
        self._ced_replay_starts: dict[int, int] = {}
        self._ced_mm_ranges: dict[int, list[tuple[int, int]]] = {}
        self._ced_attn_groups: list[list[AttentionGroup]] | None = None

        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.ced_enabled:
            sampling_params = new_req_data.sampling_params
            if (
                sampling_params is not None
                and sampling_params.prompt_logprobs is not None
            ):
                raise ValueError("CED does not support prompt log probabilities")
            if new_req_data.prompt_embeds is not None:
                raise ValueError("CED does not support prompt embeddings")
            if any(feature.modality != "image" for feature in new_req_data.mm_features):
                raise ValueError(
                    "CED does not support multimodal inputs other than images"
                )
            mm_ranges = [
                span
                for feature in new_req_data.mm_features
                for span in feature.mm_position.extract_embeds_range()
            ]
            replay_start = get_mm_safe_replay_start(
                new_req_data.prompt_len, 128, mm_ranges
            )
            assert self.ced_tail is not None
            if new_req_data.prompt_len - replay_start > self.ced_tail.window:
                raise ValueError("CED image replay exceeds the reserved tail capacity")
            if new_req_data.num_computed_tokens > replay_start:
                raise ValueError("CED prefix hit leaves insufficient encoder replay")

        super().add_request(req_index, new_req_data)
        if self.ced_enabled:
            assert self.ced_tail is not None
            self.ced_tail.reset(req_index, new_req_data.num_computed_tokens)
            self._ced_request_slots[new_req_data.req_id] = req_index
            self._ced_prompt_lens[req_index] = new_req_data.prompt_len
            self._ced_replay_starts[req_index] = replay_start
            self._ced_mm_ranges[req_index] = mm_ranges

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        slot = self._ced_request_slots.pop(req_id, None)
        if self.ced_enabled and slot is not None:
            assert self.ced_tail is not None
            self.ced_tail.reset(slot)
            self._ced_prompt_lens.pop(slot, None)
            self._ced_replay_starts.pop(slot, None)
            self._ced_mm_ranges.pop(slot, None)

    def _get_ced_attn_groups(
        self, attn_groups: list[list[AttentionGroup]]
    ) -> list[list[AttentionGroup]]:
        """Clone builders so compact metadata cannot overwrite encoder buffers."""
        if self._ced_attn_groups is not None:
            return self._ced_attn_groups

        cloned_groups: list[list[AttentionGroup]] = []
        for groups in attn_groups:
            cloned_cache_group = []
            for group in groups:
                source_builder = group.get_metadata_builder(0)
                kernel_block_size = getattr(source_builder, "kernel_block_size", None)
                if kernel_block_size is None:
                    raise RuntimeError(
                        "CED requires an explicit attention kernel block size"
                    )
                clone = AttentionGroup(
                    group.backend,
                    list(group.layer_names),
                    group.kv_cache_spec,
                    group.kv_cache_group_id,
                )
                clone.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                cloned_cache_group.append(clone)
            cloned_groups.append(cloned_cache_group)
        self._ced_attn_groups = cloned_groups
        return cloned_groups

    @staticmethod
    def _ced_slot_mappings(
        block_tables: tuple[torch.Tensor, ...],
        positions: torch.Tensor,
        query_start_loc: tuple[int, ...],
        policies: tuple[SlotMappingPolicy, ...],
        kernel_block_sizes: tuple[int, ...],
        manager_block_sizes: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Map compact decoder rows using their original absolute positions."""
        num_tokens = positions.numel()
        query_lens = torch.tensor(
            [end - start for start, end in zip(query_start_loc, query_start_loc[1:])],
            dtype=torch.int64,
            device=positions.device,
        )
        token_to_req = torch.repeat_interleave(
            torch.arange(len(query_lens), device=positions.device),
            query_lens,
            output_size=num_tokens,
        )
        mappings = []
        for block_table, policy, block_size, manager_size in zip(
            block_tables,
            policies,
            kernel_block_sizes,
            manager_block_sizes or kernel_block_sizes,
        ):
            if policy == SlotMappingPolicy.NONE:
                mappings.append(positions.new_full((num_tokens,), -1))
                continue
            block_indices = (
                torch.zeros_like(positions)
                if policy == SlotMappingPolicy.SINGLE_BLOCK_RING
                else positions // block_size
            )
            valid = (positions >= 0) & (block_indices < block_table.shape[1])
            block_numbers = block_table[
                token_to_req, block_indices.clamp(0, block_table.shape[1] - 1)
            ]
            valid &= block_numbers >= manager_size // block_size
            mappings.append(
                torch.where(
                    valid, block_numbers * block_size + positions % block_size, -1
                )
            )
        return torch.stack(mappings)

    def _build_ced_step(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> CEDStep:
        # Async rejection and adaptive verification update GPU progress before
        # its CPU upper bounds. CED prefill is already eager: read one small
        # metadata packet rather than constructing replay from stale positions.
        with gpu_sync_allowed():
            starts, ends, seq_ends = (
                torch.stack(
                    (
                        input_batch.query_start_loc[: input_batch.num_reqs],
                        input_batch.query_start_loc[1 : input_batch.num_reqs + 1],
                        input_batch.seq_lens[: input_batch.num_reqs],
                    )
                )
                .cpu()
                .tolist()
            )
        requests = tuple(
            CEDRequest(
                slot=int(input_batch.idx_mapping_np[i]),
                start=seq_ends[i] - (ends[i] - starts[i]),
                query_len=ends[i] - starts[i],
                prefill_len=self._ced_prompt_lens[int(input_batch.idx_mapping_np[i])],
                is_prefilling=(seq_ends[i] - (ends[i] - starts[i]))
                < self._ced_prompt_lens[int(input_batch.idx_mapping_np[i])],
                replay_start=self._ced_replay_starts.get(
                    int(input_batch.idx_mapping_np[i])
                ),
            )
            for i in range(input_batch.num_reqs)
        )
        assert self.ced_tail is not None
        plan = plan_ced_step(requests, cache_window=self.ced_tail.window)
        if (
            plan.num_decoder_tokens
            > self.vllm_config.scheduler_config.max_num_batched_tokens
        ):
            raise ValueError("CED compact replay exceeds the reserved token capacity")
        positions = torch.tensor(
            plan.positions, dtype=torch.int64, device=input_batch.positions.device
        )
        if not plan.decoder_requests:
            return CEDStep(plan, {}, positions, {})

        selected = torch.tensor(
            plan.decoder_requests, dtype=torch.int64, device=positions.device
        )
        decoder_block_tables = tuple(
            block_table.index_select(0, selected.to(block_table.device))
            for block_table in block_tables
        )
        ced_groups = self._get_ced_attn_groups(attn_groups)
        kernel_block_sizes = tuple(
            int(cast(Any, groups[0].get_metadata_builder(0)).kernel_block_size)
            for groups in ced_groups
        )
        policies = tuple(
            group.kv_cache_spec.slot_mapping_policy
            for group in kv_cache_config.kv_cache_groups
        )
        decoder_slot_mappings = self._ced_slot_mappings(
            decoder_block_tables,
            positions,
            plan.query_start_loc,
            policies,
            kernel_block_sizes,
            tuple(
                group.kv_cache_spec.block_size
                for group in kv_cache_config.kv_cache_groups
            ),
        )
        query_start_loc_cpu = torch.tensor(plan.query_start_loc, dtype=torch.int32)
        query_start_loc_gpu = query_start_loc_cpu.to(device=positions.device)
        selected_for_seq_lens = selected.to(input_batch.seq_lens.device)
        seq_lens = input_batch.seq_lens.index_select(0, selected_for_seq_lens)
        seq_lens_cpu = torch.tensor(
            [requests[i].start + requests[i].query_len for i in plan.decoder_requests],
            dtype=torch.int32,
        )
        is_prefilling = torch.tensor(
            [requests[i].is_prefilling for i in plan.decoder_requests],
            dtype=torch.bool,
        )
        dcp_local_seq_lens = input_batch.dcp_local_seq_lens
        if dcp_local_seq_lens is not None:
            dcp_local_seq_lens = dcp_local_seq_lens.index_select(
                0, selected.to(dcp_local_seq_lens.device)
            )
        prompt_lens = input_batch.prompt_lens
        if prompt_lens is not None:
            prompt_lens = prompt_lens.index_select(0, selected.to(prompt_lens.device))

        decoder_metadata = build_attn_metadata(
            attn_groups=ced_groups,
            num_reqs=len(plan.decoder_requests),
            num_tokens=plan.num_decoder_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max(
                end - start
                for start, end in zip(plan.query_start_loc, plan.query_start_loc[1:])
            ),
            seq_lens=seq_lens,
            max_seq_len=max(seq_lens_cpu.tolist()),
            block_tables=decoder_block_tables,
            slot_mappings=decoder_slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            dcp_local_seq_lens=dcp_local_seq_lens,
            positions=positions,
            is_prefilling=is_prefilling,
            mm_req_doc_ranges={
                index: self._ced_mm_ranges.get(requests[original].slot, [])
                for index, original in enumerate(plan.decoder_requests)
            },
            rswa_prefix_lens=prompt_lens,
        )
        replay_starts = torch.tensor(
            plan.replay_starts, dtype=torch.int64, device=positions.device
        )
        seen_metadata: set[int] = set()
        for metadata in decoder_metadata.values():
            if isinstance(metadata, DeepseekSparseSWAMetadata) and id(metadata) not in (
                seen_metadata
            ):
                clamp_replay_swa(metadata, positions, replay_starts, window=128)
                seen_metadata.add(id(metadata))

        return CEDStep(
            plan,
            decoder_metadata,
            positions,
            build_slot_mappings_by_layer(decoder_slot_mappings, kv_cache_config),
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
        )
        if self.ced_enabled and input_batch.has_prefill and not for_capture:
            attn_metadata[CED_METADATA_KEY] = self._build_ced_step(
                input_batch, block_tables, attn_groups, kv_cache_config
            )
        return attn_metadata

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs
