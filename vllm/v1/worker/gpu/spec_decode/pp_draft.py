# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A TP4 DSpark group spanning the last two TP2 target pipeline stages."""

import copy
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_world_group,
    init_model_parallel_group,
    use_draft_parallel_groups,
)
from vllm.models.deepseek_v4_1.nvidia.ced import (
    CED_METADATA_KEY,
    CEDDraftContext,
    CEDStep,
    get_ced_replay_capacity,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_draft_parallel_config


def maybe_create_pp_draft_group(vllm_config: VllmConfig, device: torch.device):
    spec = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    if (
        spec is None
        or parallel.pipeline_parallel_size == 1
        or spec.draft_tensor_parallel_size == parallel.tensor_parallel_size
    ):
        return None
    if not (
        spec.method == "dspark"
        and spec.draft_tensor_parallel_size == 4
        and parallel.tensor_parallel_size == 2
        and parallel.pipeline_parallel_size == 4
        and parallel.data_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not parallel.use_ubatching
        and vllm_config.model_config.hf_config.model_type == "deepseek_v41"
        and vllm_config.model_config.hf_config.ced_prefill
        and vllm_config.use_v2_model_runner
    ):
        raise NotImplementedError(
            "Cross-stage drafting requires V4.1 CED DSpark TP4 with a TP2/PP4 "
            "target, DP1 and no context parallelism or microbatching"
        )
    return PPDraftGroup(vllm_config, device)


class PPDraftGroup:
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        world = get_world_group()
        if world.world_size != 8:
            raise ValueError("The cross-stage DSpark group requires eight ranks")
        self.participates = world.rank >= 4
        self.is_source = world.rank == 6
        groups = [[0], [1], [2], [3], [4, 5, 6, 7]]
        self.tp = init_model_parallel_group(
            groups, world.local_rank, "nccl", group_name="draft_tp"
        )
        with use_draft_parallel_groups(self.tp):
            self.ep = init_model_parallel_group(
                groups,
                world.local_rank,
                "nccl",
                group_name="ep",
                use_all2all=vllm_config.parallel_config.enable_expert_parallel,
            )
        self.execution_config = copy.copy(vllm_config)
        self.execution_config.parallel_config = get_pp_draft_parallel_config(
            vllm_config
        )
        if not self.participates:
            return
        hf = vllm_config.model_config.hf_config
        spec = vllm_config.speculative_config
        assert spec is not None
        self.num_aux = len(spec.draft_model_config.hf_config.dspark_target_layer_ids)
        self.hidden_size = hf.hidden_size
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_rows = self.max_num_reqs * get_ced_replay_capacity(hf)
        self.aux_buffer = torch.empty(
            max_rows,
            self.num_aux * self.hidden_size,
            dtype=vllm_config.model_config.dtype,
            device=device,
        )
        self.values = torch.zeros(
            5 * self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.temperatures = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.last_sampled = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.next_prefill = torch.zeros_like(self.last_sampled).unsqueeze(0)
        self.seeds = torch.zeros_like(self.last_sampled)
        self.temperature = torch.zeros_like(self.temperatures)
        # Initialize the relay's NCCL communicator before cache sizing/warmup.
        self.tp.broadcast(self.values[:1], src=2)

    def context(self):
        return use_draft_parallel_groups(self.tp, self.ep)

    def propose(
        self,
        speculator,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
        num_sampled: torch.Tensor | None = None,
        num_rejected: torch.Tensor | None = None,
        last_sampled: torch.Tensor | None = None,
        next_prefill_tokens: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        *,
        dummy_run: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert self.participates
        num_reqs = input_batch.num_reqs
        if dummy_run:
            # Memory/graph warmup is collective, even on a helper target stage.
            hidden = torch.zeros_like(
                speculator.hidden_states[: input_batch.num_tokens]
            )
            aux_hidden_states = [hidden] * self.num_aux
            return speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                hidden,
                aux_hidden_states,
                torch.ones(num_reqs, dtype=torch.int32, device=hidden.device),
                torch.zeros(num_reqs, dtype=torch.int32, device=hidden.device),
                self.last_sampled,
                self.next_prefill,
                self.temperature,
                self.seeds,
                dummy_run=True,
                **kwargs,
            )

        step = attn_metadata.get(CED_METADATA_KEY)
        if isinstance(step, CEDStep):
            rows = step.plan.num_decoder_tokens
            if not rows:
                # Nonfinal encoder chunks publish neither samples nor draft KV.
                return speculator.draft_tokens[:num_reqs]
        else:
            if input_batch.has_prefill:
                raise RuntimeError("Cross-stage DSpark prefill requires CED metadata")
            rows = input_batch.num_tokens
        if rows > self.aux_buffer.shape[0]:
            raise ValueError("Cross-stage DSpark auxiliary context exceeds capacity")
        values = self.values[: 5 * num_reqs].view(5, num_reqs)
        temperatures = self.temperatures[:num_reqs]
        auxiliaries = self.aux_buffer[:rows]
        idx = input_batch.idx_mapping
        if self.is_source:
            assert num_sampled is not None and num_rejected is not None
            assert last_sampled is not None and next_prefill_tokens is not None
            assert temperature is not None and seeds is not None
            values[0].copy_(num_sampled)
            values[1].copy_(num_rejected)
            values[2].copy_(last_sampled.reshape(-1)[idx])
            values[3].copy_(next_prefill_tokens[0, idx])
            values[4].copy_(seeds[idx])
            temperatures.copy_(temperature[idx])
            if isinstance(step, CEDStep):
                assert step.draft_context is not None
                aux_hidden_states = step.draft_context.aux_hidden_states
            assert aux_hidden_states is not None
            assert len(aux_hidden_states) == self.num_aux
            torch.cat(
                [value[:rows] for value in aux_hidden_states], dim=-1, out=auxiliaries
            )
        self.tp.broadcast(values, src=2)
        self.tp.broadcast(temperatures, src=2)
        self.tp.broadcast(auxiliaries, src=2)
        self.last_sampled[idx] = values[2]
        self.next_prefill[0, idx] = values[3]
        self.seeds[idx] = values[4]
        self.temperature[idx] = temperatures
        aux_hidden_states = list(auxiliaries.split(self.hidden_size, dim=-1))
        if isinstance(step, CEDStep):
            # Use this stage's request ordering and KV block tables, never the
            # producer's physical cache addresses or persistent request slots.
            step.draft_context = CEDDraftContext(
                positions=step.positions,
                request_indices=torch.tensor(
                    [
                        request
                        for request, start, end in zip(
                            step.plan.decoder_requests,
                            step.plan.query_start_loc,
                            step.plan.query_start_loc[1:],
                        )
                        for _ in range(end - start)
                    ],
                    dtype=torch.int64,
                    device=auxiliaries.device,
                ),
                active_requests=step.plan.decoder_requests,
                aux_hidden_states=aux_hidden_states,
            )
        return speculator.propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            speculator.hidden_states[: input_batch.num_tokens],
            aux_hidden_states,
            values[0].to(torch.int32),
            values[1].to(torch.int32),
            self.last_sampled,
            self.next_prefill,
            self.temperature,
            self.seeds,
            **kwargs,
        )
