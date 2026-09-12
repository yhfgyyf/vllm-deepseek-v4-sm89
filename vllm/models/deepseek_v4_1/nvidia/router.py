# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

_SUPPORTED_ROUTING_SHAPES = frozenset({(128, 3), (384, 6)})


def can_use_deepseek_v41_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    correction_bias_vl: torch.Tensor | None,
    topk: int,
    indices_dtype: torch.dtype,
    input_tokens: torch.Tensor | None,
) -> bool:
    if not (
        gating_output.is_cuda
        and gating_output.dtype == torch.float32
        and gating_output.ndim == 2
        and (gating_output.shape[1], topk) in _SUPPORTED_ROUTING_SHAPES
        and gating_output.is_contiguous()
        and correction_bias.is_cuda
        and correction_bias.dtype == torch.float32
        and correction_bias.shape == (gating_output.shape[1],)
        and correction_bias.is_contiguous()
        and correction_bias.device == gating_output.device
        and indices_dtype in (torch.int32, torch.uint32, torch.int64)
    ):
        return False

    if correction_bias_vl is None:
        return True

    return (
        correction_bias_vl.is_cuda
        and correction_bias_vl.dtype == torch.float32
        and correction_bias_vl.shape == (gating_output.shape[1],)
        and correction_bias_vl.is_contiguous()
        and correction_bias_vl.device == gating_output.device
        and input_tokens is not None
        and input_tokens.is_cuda
        and input_tokens.device == gating_output.device
        and input_tokens.dtype in (torch.int32, torch.int64)
        and input_tokens.ndim == 1
        and input_tokens.shape[0] == gating_output.shape[0]
        and input_tokens.is_contiguous()
    )


if current_platform.is_cuda():

    @triton.jit
    def _deepseek_v41_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        correction_bias_vl_ptr,
        input_tokens_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        image_sentinel_lo: tl.constexpr,
        NUM_EXPERTS: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_VL_BIAS: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS

        bias = tl.load(
            correction_bias_ptr + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_VL_BIAS:
            token = tl.load(input_tokens_ptr + row)
            is_image = (token >= image_sentinel_lo) & (token < image_sentinel_lo + 5)
            bias_vl = tl.load(
                correction_bias_vl_ptr + expert_offsets,
                mask=expert_mask,
                other=0.0,
            ).to(tl.float32)
            bias = tl.where(is_image, bias_vl, bias)

        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        logits = tl.load(
            gating_output_ptr + row * NUM_EXPERTS + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        softplus = tl.maximum(logits, 0.0) + tldevice.log1p(
            tldevice.exp(-tl.abs(logits))
        )
        scores = tl.sqrt(softplus)
        candidates = tl.where(expert_mask, scores + bias, -float("inf"))
        candidates = tl.where(candidates == candidates, candidates, -1e30)

        topk_offsets = tl.arange(0, 8)
        selected_weights = tl.zeros([8], dtype=tl.float32)
        selected_ids = tl.zeros([8], dtype=tl.int32)
        for slot in tl.static_range(0, TOP_K):
            max_value = tl.max(candidates, axis=0)
            tied_ids = tl.where(candidates == max_value, expert_offsets, NUM_EXPERTS)
            expert_id = tl.min(tied_ids, axis=0).to(tl.int32)
            selected_weight = tl.sum(
                tl.where(expert_offsets == expert_id, scores, 0.0), axis=0
            )
            is_slot = topk_offsets == slot
            selected_weights = tl.where(is_slot, selected_weight, selected_weights)
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            candidates = tl.where(
                expert_offsets == expert_id, -float("inf"), candidates
            )

        weight_sum = tl.sum(selected_weights, axis=0)
        selected_weights *= routed_scaling_factor / tl.where(
            weight_sum > 0.0, weight_sum, 1.0
        )
        output_mask = topk_offsets < TOP_K
        output_offsets = row * TOP_K + topk_offsets

        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()

        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def deepseek_v41_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
    *,
    topk: int,
    correction_bias_vl: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_deepseek_v41_topk(
        gating_output,
        correction_bias,
        correction_bias_vl,
        topk,
        indices_dtype,
        input_tokens,
    ):
        raise ValueError(
            "DeepSeek V4.1 native routing requires contiguous CUDA FP32 logits "
            "and bias with (experts, topk) equal to (384, 6) or (128, 3)."
        )

    num_tokens, num_experts = gating_output.shape
    shape = (num_tokens, topk)
    topk_weights = gating_output.new_empty(shape, dtype=torch.float32)
    topk_ids = gating_output.new_empty(shape, dtype=indices_dtype)
    if num_tokens == 0:
        return topk_weights, topk_ids

    correction_bias_vl_arg = (
        correction_bias if correction_bias_vl is None else correction_bias_vl
    )
    input_tokens_arg = input_tokens if input_tokens is not None else topk_ids[:, 0]
    with torch.cuda.device(gating_output.device):
        _deepseek_v41_topk_kernel[(num_tokens,)](
            gating_output,
            correction_bias,
            correction_bias_vl_arg,
            input_tokens_arg,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            IMAGE_SENTINEL_BASE_ID,
            NUM_EXPERTS=num_experts,
            TOP_K=topk,
            BLOCK_N=triton.next_power_of_2(num_experts),
            HAS_VL_BIAS=correction_bias_vl is not None,
            num_warps=1,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    return topk_weights, topk_ids


class DeepseekV41TopKBiasRouter(FusedTopKBiasRouter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (self.global_num_experts, self.top_k) not in _SUPPORTED_ROUTING_SHAPES:
            raise ValueError(
                "DeepSeek V4.1 routing supports (experts, topk) equal to "
                "(384, 6) or (128, 3)."
            )
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("DeepSeek V4.1 routing requires sqrtsoftplus scoring.")
        if not self.renormalize:
            raise ValueError("DeepSeek V4.1 routing requires normalized weights.")
        if self._hash_indices_table is not None:
            raise ValueError("DeepSeek V4.1 does not support hash MoE routing.")

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        if self.e_score_correction_bias is None:
            raise ValueError("DeepSeek V4.1 routing requires a correction bias.")
        if self.e_score_correction_bias_vl is not None and input_ids is None:
            raise ValueError(
                "DeepSeek V4.1 mixed text/image routing requires input_ids."
            )

        output_indices_dtype = torch.int32 if indices_type is None else indices_type
        topk_weights, topk_ids = deepseek_v41_topk(
            router_logits,
            self.e_score_correction_bias.data,
            output_indices_dtype,
            self.routed_scaling_factor,
            topk=self.top_k,
            correction_bias_vl=(
                self.e_score_correction_bias_vl.data
                if self.e_score_correction_bias_vl is not None
                else None
            ),
            input_tokens=input_ids,
        )

        if self.num_fused_shared_experts > 0:
            num_tokens = topk_ids.shape[0]
            num_shared = self.num_fused_shared_experts
            shared_ids = torch.arange(
                self.global_num_experts,
                self.global_num_experts + num_shared,
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            ).expand(num_tokens, num_shared)
            shared_weights = torch.full(
                (num_tokens, num_shared),
                self.shared_expert_weight,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat((topk_ids, shared_ids), dim=-1)
            topk_weights = torch.cat((topk_weights, shared_weights), dim=-1)

        return topk_weights, topk_ids
