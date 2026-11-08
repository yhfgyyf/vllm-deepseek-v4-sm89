# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None
    draft_confidences: torch.Tensor | None = None


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


def _alloc_combined(num_reqs: int, device: torch.device) -> torch.Tensor:
    """Keep both count views 16-byte aligned, including odd batch sizes."""
    return torch.zeros(2, -(-num_reqs // 4) * 4, dtype=torch.int32, device=device)


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
        enable_adaptive_verification: bool = False,
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.enable_adaptive_verification = enable_adaptive_verification
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        if current_platform.is_cuda() and get_pp_group().world_size > 2:
            # Account for the larger PP communicator before sizing the KV cache.
            # Warmup disables sampling, so its first real broadcast is too late.
            ready = torch.zeros(1, dtype=torch.int32, device=device)
            torch.distributed.broadcast(
                ready, src=self.last_rank, group=self.broadcast_group
            )
            self.main_stream.synchronize()
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()
        self.disabled = False

    def set_disabled(self, disabled: bool) -> None:
        self.disabled = disabled

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        inputs: IntermediateTensors | None,
        outputs: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return outputs
        assert inputs is not None
        return IntermediateTensors(
            outputs.tensors
            | {key: inputs[key] for key in self.aux_hidden_state_relay_keys}
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        outputs = dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            broadcast_drafts=slot.draft_tokens,
        )
        if slot.draft_confidences is not None:
            rows = async_copy_to_gpu(np.flatnonzero(~exclude_mask), device=self.device)
            outputs["broadcast_confidences"] = slot.draft_confidences[rows]
            outputs["confidence_indices"] = idx_mapping[rows]
        return outputs

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        if self.disabled:
            return False
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        idx_mapping_np = input_batch.idx_mapping_np.copy()
        gen_at_receive_np = self.req_idx_gen_np[idx_mapping_np].copy()

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = _alloc_combined(num_reqs, self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            draft_confidences = None
            if self.max_sample_len > 1:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.max_sample_len - 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
                if self.enable_adaptive_verification:
                    draft_confidences = torch.empty_like(
                        draft_tokens, dtype=torch.float32
                    )
                    torch.distributed.broadcast(
                        draft_confidences,
                        src=self.last_rank,
                        group=self.broadcast_group,
                    )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined[:, :num_reqs].unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
            if draft_confidences is not None:
                draft_confidences.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            idx_mapping_np,
            need_sampled_mask.copy(),
            gen_at_receive_np,
            draft_tokens,
            draft_confidences,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if self.disabled or compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64
        assert num_sampled.dtype == torch.int32
        assert num_rejected.dtype == torch.int32
        assert sampled_token_ids.shape[1] <= self.max_sample_len

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids, (0, self.max_sample_len - sampled_token_ids.shape[1])
            ).contiguous()
            torch.distributed.broadcast(
                send_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = _alloc_combined(input_batch.num_reqs, self.device)
            combined[0, : input_batch.num_reqs] = num_sampled
            combined[1, : input_batch.num_reqs] = num_rejected
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)

    def broadcast_drafts(
        self,
        draft_token_table: torch.Tensor,
        input_batch: InputBatch,
        confidence_probs: torch.Tensor | None = None,
    ) -> None:
        assert self.is_last_rank
        if self.disabled or self.max_sample_len == 1:
            return
        if compute_need_sampled_mask(input_batch) is None:
            return
        assert draft_token_table.dtype == torch.int64
        assert draft_token_table.shape[1] == self.max_sample_len - 1
        # Snapshot on the producer stream before the next propose overwrites it.
        drafts = draft_token_table[input_batch.idx_mapping]
        confidences = None
        if self.enable_adaptive_verification:
            assert confidence_probs is not None
            confidences = confidence_probs[: input_batch.num_reqs].clone()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                drafts, src=self.last_rank, group=self.broadcast_group
            )
            drafts.record_stream(self.broadcast_stream)
            if confidences is not None:
                torch.distributed.broadcast(
                    confidences, src=self.last_rank, group=self.broadcast_group
                )
                confidences.record_stream(self.broadcast_stream)
