# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for `FLASHINFER_MLA_SPARSE_SM120`."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
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


class FlashInferMLASparseSM120Impl(SparseMLACommonImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def get_fp8_ds_mla_row_bytes(self) -> int:
        if self.kv_lora_rank == 512 and self.qk_rope_head_dim == 0:
            return 528
        return super().get_fp8_ds_mla_row_bytes()

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
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        if kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        topk_indices_buffer = mla_args.pop("topk_indices_buffer", None)
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
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
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
            has_flashinfer_sparse_mla_sm89,
            has_flashinfer_sparse_mla_sm89_glm_nope,
            has_flashinfer_sparse_mla_sm120,
            has_flashinfer_sparse_mla_sm120_glm_nope,
        )

        capability = current_platform.get_device_capability()
        is_sm89 = capability == DeviceCapability(8, 9)
        if is_sm89:
            has_sparse_mla = (
                has_flashinfer_sparse_mla_sm89_glm_nope()
                if self.kv_scale_format == "arbitrary_fp32_nope"
                else has_flashinfer_sparse_mla_sm89()
            )
        else:
            has_sparse_mla = has_flashinfer_sparse_mla_sm120()
        if not has_sparse_mla:
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires a FlashInfer sparse MLA "
                "build compatible with the current GPU."
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
        self._workspace_buffer: torch.Tensor | None = None

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        skip_scale: bool = False,
    ) -> None:
        if self.kv_scale_format != "arbitrary_fp32_nope":
            return super().do_kv_cache_update(
                kv_c_normed,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                skip_scale=skip_scale,
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
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        index_group = self.index_group
        if isinstance(index_group, HiSparseMLAIndexGroup):
            num_decode_tokens = attn_metadata.num_decode_tokens
            physical_kv_cache = index_group.physical_kv_cache(self.index_group_index)
            if (
                num_decode_tokens == num_actual_toks
                or index_group.cache(self.index_group_index).all_context_pages_resident
            ):
                topk_indices_physical = cast(
                    torch.Tensor,
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices,
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=False,
                    ),
                )
                return (
                    self._run_mqa_kernel(q, physical_kv_cache, topk_indices_physical),
                    None,
                )

            outputs = []
            if num_decode_tokens:
                topk_indices_physical = cast(
                    torch.Tensor,
                    index_group.convert_logical_to_physical_topk(
                        self.index_group_index,
                        topk_indices[:num_decode_tokens],
                        attn_metadata,
                        block_stride_rows=None,
                        return_valid_counts=False,
                    ),
                )
                outputs.append(
                    self._run_mqa_kernel(
                        q[:num_decode_tokens],
                        physical_kv_cache,
                        topk_indices_physical,
                    )
                )
            prefill_cache, block_table, req_ids = index_group.stage_prefill_rows(
                self.index_group_index,
                kv_c_and_k_pe_cache,
                attn_metadata,
            )
            topk_indices_physical = cast(
                torch.Tensor,
                triton_convert_req_index_to_global_index(
                    req_ids,
                    block_table,
                    topk_indices[num_decode_tokens:],
                    BLOCK_SIZE=attn_metadata.block_size,
                    NUM_TOPK_TOKENS=topk_indices.shape[1],
                ),
            )
            outputs.append(
                self._run_mqa_kernel(
                    q[num_decode_tokens:],
                    prefill_cache,
                    topk_indices_physical,
                )
            )
            output = torch.cat(outputs) if len(outputs) > 1 else outputs[0]
            return output, None

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                BLOCK_STRIDE_ROWS=block_stride_rows,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )
        return (
            self._run_mqa_kernel(
                q,
                # The SM120 kernels are built for 64-row pages only.
                kv_rows.view(-1, 64, kv_rows.shape[-1]),
                topk_indices_physical,
            ),
            None,
        )

    def _run_mqa_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices_physical: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_toks = q.shape[0]

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        sparse_capacity = topk_indices_physical.shape[1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=sparse_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_capacity,
            kv_scale_format=self.kv_scale_format,
        )
        return out.squeeze(1)
