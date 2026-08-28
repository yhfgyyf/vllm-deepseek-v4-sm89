# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
    FlashInferMLASparseMetadata,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(
    model_type: str | None,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
) -> str:
    if model_type in ("glm5_next", "glm5_next_text"):
        if qk_nope_head_dim == 256 and qk_rope_head_dim == 0 and kv_lora_rank == 512:
            return "arbitrary_fp32_nope"
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(FlashInferMLASparseImpl):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(
            model_type,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.kv_lora_rank,
        )

        from vllm.utils.flashinfer import (
            has_flashinfer_sparse_mla_sm120,
            has_flashinfer_sparse_mla_sm120_glm_nope,
        )

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        if (
            self.kv_scale_format == "arbitrary_fp32_nope"
            and not has_flashinfer_sparse_mla_sm120_glm_nope()
        ):
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 GLM NoPE requires FlashInfer's "
                "kv_scale_format API and GLM NoPE cache pack/gather helpers."
            )
        assert self.topk_indices_buffer is not None

        # The native SM120 operator accepts a BF16 query while dequantizing its
        # packed FP8 KV tiles in-kernel.
        self.supports_quant_query_input = False

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if self.kv_scale_format != "arbitrary_fp32_nope":
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
            )
        if kv_cache.numel() == 0:
            return
        if k_pe.shape[-1] != 0:
            raise ValueError(
                "GLM NoPE cache update expects qk_rope_head_dim=0, "
                f"got k_pe.shape={tuple(k_pe.shape)}"
            )

        from flashinfer.mla._sparse_mla_sm120_cache import (
            glm_nope_quantize_and_cache,
        )

        glm_nope_quantize_and_cache(
            kv_c_normed,
            kv_cache.view(torch.uint8),
            slot_mapping.flatten(),
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # fp8_ds_mla stores per-128-element scales inside each cache entry, so
        # no separate per-tensor query/KV scale is part of this kernel contract.
        self.bmm1_scale = self.scale
        self.bmm2_scale = 1.0
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
