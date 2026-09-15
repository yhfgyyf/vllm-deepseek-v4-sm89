# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned state for the experimental bounded-replay CED schedule.

This implements the report's approximate decoder replay, not an equivalence
transformation of full decoder prefill. The ordinary model path is unchanged.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

CED_METADATA_KEY = "__deepseek_v41_ced__"


def get_ced_replay_capacity(hf_config: Any) -> int:
    """Return the physical replay capacity required by the model config."""
    vision_tokens = (
        hf_config.vision_max_n_token
        if getattr(hf_config, "vision_n_layers", 0) > 0
        else 0
    )
    return 128 + vision_tokens


def validate_ced_config(vllm_config: Any, use_sequence_parallel: bool) -> None:
    """Fail closed for combinations not covered by the bounded-replay prototype."""
    hf = vllm_config.model_config.hf_config
    parallel = vllm_config.parallel_config
    unsupported = []
    for condition, name in (
        (not vllm_config.use_v2_model_runner, "model runner v1"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (parallel.data_parallel_size != 1, "data parallelism"),
        (parallel.decode_context_parallel_size != 1, "decode context parallelism"),
        (parallel.prefill_context_parallel_size != 1, "prefill context parallelism"),
        (use_sequence_parallel, "sequence parallelism"),
        (parallel.enable_dbo or parallel.ubatch_size > 1, "microbatching/DBO"),
        (vllm_config.kv_transfer_config is not None, "KV transfer/restoration"),
        (getattr(vllm_config, "lora_config", None) is not None, "LoRA"),
    ):
        if condition:
            unsupported.append(name)
    if unsupported:
        raise NotImplementedError(
            "Experimental CED does not support " + ", ".join(unsupported)
        )
    spec = vllm_config.speculative_config
    if spec is not None:
        if not getattr(spec, "use_dspark", lambda: False)():
            raise NotImplementedError(
                "CED supports DSpark, not other speculative decoding"
            )
        draft = spec.draft_model_config.hf_config
        aux_layers = tuple(getattr(draft, "dspark_target_layer_ids", ()))
        if (
            getattr(draft, "model_type", None) != "deepseek_v41"
            or not aux_layers
            or tuple(sorted(set(aux_layers))) != aux_layers
            or any(layer <= 20 or layer > 40 for layer in aux_layers)
            or draft.sliding_window > 128
            or any(draft.compress_ratios[40:])
        ):
            raise NotImplementedError(
                "CED DSpark requires V4.1 decoder-only auxiliaries and SWA-only drafts"
            )
        if not 1 <= spec.num_speculative_tokens <= 127:
            raise ValueError(
                "CED verification must fit within the 128-row replay budget"
            )
    if hf.num_hidden_layers != 40 or hf.sliding_window != 128 or hf.hc_mult != 4:
        raise ValueError("CED requires the V4.1 20+20 layer, SWA128, HC4 architecture")
    if [i for i in hf.kv_source_layer_ids if i >= 20] != [20]:
        raise ValueError("CED requires decoder global KV to be owned only by layer 20")
    if any(i >= 20 for i in getattr(hf, "engram_layer_ids", ())):
        raise ValueError("CED decoder Engram replay is not implemented")
    scheduler = vllm_config.scheduler_config
    if scheduler.max_num_batched_tokens < 128 * scheduler.max_num_seqs:
        raise ValueError("CED requires max_num_batched_tokens >= 128 * max_num_seqs")


@dataclass(frozen=True)
class CEDRequest:
    slot: int
    start: int
    query_len: int
    prefill_len: int
    is_prefilling: bool
    replay_start: int | None = None


@dataclass(frozen=True)
class CEDPlan:
    requests: tuple[CEDRequest, ...]
    semantic_window: int
    cache_window: int
    request_replay_starts: tuple[int, ...]
    decoder_requests: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    positions: tuple[int, ...]
    replay_starts: tuple[int, ...]
    store_rows: tuple[int, ...]
    store_slots: tuple[int, ...]
    cache_rows: tuple[int, ...]
    current_rows: tuple[int, ...]
    use_cache: tuple[bool, ...]
    output_rows: tuple[int, ...]
    decoder_output_rows: tuple[int, ...]
    num_input_tokens: int

    @property
    def num_decoder_tokens(self) -> int:
        return len(self.positions)


def plan_ced_step(
    requests: tuple[CEDRequest, ...],
    window: int = 128,
    cache_window: int | None = None,
) -> CEDPlan:
    """Plan on CPU request metadata, including a one-token final prefill."""
    if window < 1:
        raise ValueError("CED replay window must be positive")
    if cache_window is None:
        cache_window = window
    if cache_window < 1:
        raise ValueError("CED replay cache window must be positive")
    if len({r.slot for r in requests}) != len(requests):
        raise ValueError("CED request slots must be unique within a batch")
    active, query_start = [], [0]
    positions, replay_starts = [], []
    store_rows: list[int] = []
    store_slots: list[int] = []
    cache_rows, current_rows, use_cache = [], [], []
    output_rows, decoder_output_rows = [], []
    request_replay_starts, replay_rows = [], []
    input_offset = 0
    offsets = []
    for batch_index, request in enumerate(requests):
        offsets.append(input_offset)
        slot, start, count = request.slot, request.start, request.query_len
        if slot < 0 or start < 0 or count < 1:
            raise ValueError("CED requires nonnegative slots/positions and real rows")
        end = start + count
        default_replay_start = max(0, request.prefill_len - window)
        replay_start = request.replay_start
        if replay_start is None:
            replay_start = default_replay_start
        elif not 0 <= replay_start <= default_replay_start:
            raise ValueError("Invalid CED replay start for the full prompt")
        request_replay_starts.append(replay_start)
        request_replay_rows = request.prefill_len - replay_start
        if request_replay_rows > cache_window:
            raise ValueError("CED replay exceeds the physical cache window")
        replay_rows.append(request_replay_rows)
        if request.is_prefilling:
            if start >= request.prefill_len or end > request.prefill_len:
                raise ValueError("Invalid CED prefill progress")
            first_saved = max(start, end - request_replay_rows)
            store_rows.extend(input_offset + p - start for p in range(first_saved, end))
            store_slots.extend(
                slot * cache_window + p % cache_window for p in range(first_saved, end)
            )
            if end == request.prefill_len:
                active.append(batch_index)
        else:
            if start < request.prefill_len:
                raise ValueError("Decode cannot precede the completed prefill")
            active.append(batch_index)
        input_offset += count

    # Builders partition short queries before prefills. A one-token final
    # prefill remains a prefill semantically even though its replay is short.
    active.sort(
        key=lambda i: (
            replay_rows[i] if requests[i].is_prefilling else requests[i].query_len
        )
    )
    for batch_index in active:
        request = requests[batch_index]
        slot, start, count = request.slot, request.start, request.query_len
        end = start + count
        offset = offsets[batch_index]
        if request.is_prefilling:
            replay_start = request_replay_starts[batch_index]
            for pos in range(replay_start, end):
                positions.append(pos)
                replay_starts.append(replay_start)
                cache_rows.append(slot * cache_window + pos % cache_window)
                current_rows.append(0)
                use_cache.append(True)
            output_rows.append(offset + count - 1)
            decoder_output_rows.append(len(positions) - 1)
        else:
            for pos in range(start, end):
                positions.append(pos)
                replay_starts.append(0)
                cache_rows.append(0)
                current_rows.append(offset + pos - start)
                use_cache.append(False)
                output_rows.append(offset + pos - start)
                decoder_output_rows.append(len(positions) - 1)
        query_start.append(len(positions))
    return CEDPlan(
        requests=requests,
        semantic_window=window,
        cache_window=cache_window,
        request_replay_starts=tuple(request_replay_starts),
        decoder_requests=tuple(active),
        query_start_loc=tuple(query_start),
        positions=tuple(positions),
        replay_starts=tuple(replay_starts),
        store_rows=tuple(store_rows),
        store_slots=tuple(store_slots),
        cache_rows=tuple(cache_rows),
        current_rows=tuple(current_rows),
        use_cache=tuple(use_cache),
        output_rows=tuple(output_rows),
        decoder_output_rows=tuple(decoder_output_rows),
        num_input_tokens=input_offset,
    )


class CEDTailState(nn.Module):
    """A fixed-size request-slot ring, including delayed-mHC coefficients."""

    def __init__(self, max_requests: int, hidden_size: int, window: int = 128):
        super().__init__()
        if min(max_requests, hidden_size, window) < 1:
            raise ValueError("CED state dimensions must be positive")
        self.window = window
        self.max_requests = max_requests
        self.ends = [0] * max_requests
        self.valid_starts = [0] * max_requests
        self.register_buffer(
            "hidden",
            torch.zeros(max_requests * window, 4, hidden_size, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "pre_mix",
            torch.zeros(max_requests * window, 4, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "input_ids",
            torch.zeros(max_requests * window, dtype=torch.int64),
            persistent=False,
        )

    def reset(self, slot: int, start: int = 0) -> None:
        # Position validation prevents stale rows being consumed; no large clear.
        if not 0 <= slot < self.max_requests or start < 0:
            raise ValueError("Invalid CED tail slot or restored progress")
        self.ends[slot] = start
        self.valid_starts[slot] = start

    def pack(
        self,
        plan: CEDPlan,
        hidden: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if plan.cache_window != self.window:
            raise ValueError("CED plan and tail cache windows do not match")
        if hidden.shape != (plan.num_input_tokens, 4, self.hidden.shape[-1]):
            raise ValueError("CED boundary residual shape does not match its plan")
        if pre_mix.shape != (plan.num_input_tokens, 4):
            raise ValueError("CED requires carried per-stream pre-mix coefficients")
        if hidden.dtype != torch.bfloat16 or pre_mix.dtype != torch.float32:
            raise ValueError("CED preserves BF16 residuals and FP32 pre-mix")
        if input_ids.shape != (plan.num_input_tokens,):
            raise ValueError("CED token IDs must match its boundary rows")
        if (
            not hidden.device
            == pre_mix.device
            == input_ids.device
            == self.hidden.device
        ):
            raise ValueError(
                "CED state and boundary tensors must be on the same device"
            )
        for request, replay_start in zip(
            plan.requests, plan.request_replay_starts, strict=True
        ):
            if request.slot >= self.max_requests:
                raise ValueError("CED request slot exceeds the reserved capacity")
            if request.is_prefilling:
                if request.start == 0:
                    self.reset(request.slot)
                if self.ends[request.slot] != request.start:
                    raise ValueError("CED tail missing: prefix restore requires replay")
                end = request.start + request.query_len
                if (
                    end == request.prefill_len
                    and self.valid_starts[request.slot] > replay_start
                ):
                    raise ValueError(
                        "CED prefix hit leaves insufficient encoder replay"
                    )
        device = hidden.device

        def indices(values):
            return torch.tensor(values, dtype=torch.int64, device=device)

        if plan.store_rows:
            source, target = indices(plan.store_rows), indices(plan.store_slots)
            self.hidden.index_copy_(0, target, hidden.index_select(0, source))
            self.pre_mix.index_copy_(0, target, pre_mix.index_select(0, source))
            self.input_ids.index_copy_(0, target, input_ids[source].to(torch.int64))
        for request, replay_start in zip(
            plan.requests, plan.request_replay_starts, strict=True
        ):
            if request.is_prefilling:
                self.ends[request.slot] = request.start + request.query_len
                self.valid_starts[request.slot] = max(
                    self.valid_starts[request.slot],
                    self.ends[request.slot] - (request.prefill_len - replay_start),
                )
        cached, current = indices(plan.cache_rows), indices(plan.current_rows)
        mask = torch.tensor(plan.use_cache, dtype=torch.bool, device=device)
        return (
            torch.where(mask[:, None, None], self.hidden[cached], hidden[current]),
            torch.where(mask[:, None], self.pre_mix[cached], pre_mix[current]),
            torch.where(mask, self.input_ids[cached], input_ids[current].long()),
        )


@dataclass
class CEDDraftContext:
    """Real decoder auxiliaries in compact order, outside draft graph capture."""

    positions: torch.Tensor
    request_indices: torch.Tensor
    active_requests: tuple[int, ...]
    aux_hidden_states: list[torch.Tensor]

    def slot_mapping(
        self,
        block_table: torch.Tensor,
        kernel_block_size: int,
        manager_block_size: int,
        seq_lens: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """Reject speculative suffixes and every kernel page of the null block."""
        if manager_block_size % kernel_block_size:
            raise ValueError("CED cache block sizes must be divisible")
        reqs = self.request_indices
        positions = self.positions
        if positions.shape != reqs.shape:
            raise ValueError("CED draft positions and request ownership must align")
        valid = (reqs >= 0) & (reqs < block_table.shape[0]) & (positions >= 0)
        safe_reqs = reqs.clamp(0, block_table.shape[0] - 1)
        columns = positions // kernel_block_size
        valid &= columns < block_table.shape[1]
        blocks = block_table[safe_reqs, columns.clamp(0, block_table.shape[1] - 1)]
        valid &= blocks >= manager_block_size // kernel_block_size
        valid &= positions < seq_lens[safe_reqs] - num_rejected[safe_reqs]
        return torch.where(
            valid,
            blocks.to(torch.int64) * kernel_block_size + positions % kernel_block_size,
            -1,
        )


@dataclass
class CEDStep:
    plan: CEDPlan
    decoder_metadata: dict[str, Any]
    positions: torch.Tensor
    decoder_slot_mapping: dict[str, torch.Tensor] | None = None
    draft_context: CEDDraftContext | None = None


def clamp_replay_swa(
    metadata: Any, positions: torch.Tensor, replay_starts: torch.Tensor, window: int
) -> None:
    """Exclude unbuilt SWA keys to the left of a decoder replay window."""
    decode = metadata.num_decode_tokens
    mm_prefix_query_ranges = getattr(metadata, "mm_prefix_query_ranges", None)
    for slots, lengths, rows in (
        (metadata.decode_swa_indices, metadata.decode_swa_lens, slice(0, decode)),
        (
            metadata.prefill_swa_indices,
            metadata.prefill_swa_lens,
            slice(decode, positions.numel()),
        ),
    ):
        if slots is None:
            continue
        view = slots[:, 0] if slots.ndim == 3 else slots
        pos = positions[rows]
        first = (pos - window + 1).clamp_min(0)
        if mm_prefix_query_ranges is not None:
            spans = mm_prefix_query_ranges[rows].to(device=pos.device, dtype=pos.dtype)
            span_start, span_end = spans.unbind(dim=1)
            in_span = (span_start >= 0) & (span_start <= pos) & (pos <= span_end)
            first = torch.where(in_span, torch.minimum(first, span_start), first)
        logical_keys = first[:, None] + torch.arange(view.shape[1], device=pos.device)
        view.masked_fill_(logical_keys < replay_starts[rows, None], -1)
        if lengths is not None:
            lengths.copy_((view >= 0).sum(-1).to(lengths.dtype))
