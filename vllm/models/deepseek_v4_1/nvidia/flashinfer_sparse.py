# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native DeepSeek V4.1 mixed FP8/FP4 sparse attention."""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV41SparseMLABackend,
    DeepseekV41SparseMLAMetadata,
    DeepseekV41SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _native_mixed_attention(*args, **kwargs) -> torch.Tensor | None:
    try:
        from flashinfer.mla.deepseek_v41 import (
            _deepseek_v41_mixed_sparse_attention_with_inv_rope,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek V4.1 requires FlashInfer's native "
            "mixed sparse attention with inverse-RoPE API; no FP8 or FlashMLA "
            "fallback is permitted."
        ) from exc
    return _deepseek_v41_mixed_sparse_attention_with_inv_rope(*args, **kwargs)


def _native_workspace_size(
    num_tokens: int,
    num_heads: int,
    num_swa_slots: int,
    num_global_slots: int,
) -> int:
    try:
        from flashinfer.mla import deepseek_v41_mixed_sparse_workspace_size
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek V4.1 requires FlashInfer's native mixed sparse "
            "attention workspace-size API."
        ) from exc
    return int(
        deepseek_v41_mixed_sparse_workspace_size(
            num_tokens,
            num_heads,
            num_swa_slots,
            num_global_slots,
        )
    )


def _get_native_workspace(
    q: torch.Tensor,
    swa_slots: torch.Tensor,
    global_slots: torch.Tensor | None,
) -> torch.Tensor:
    workspace_bytes = _native_workspace_size(
        q.shape[0],
        q.shape[1],
        swa_slots.shape[1],
        global_slots.shape[1] if global_slots is not None else 0,
    )
    (workspace,) = current_workspace_manager().get_simultaneous(
        ((workspace_bytes,), torch.uint8),
    )
    return workspace


def _max_native_workspace_size(
    num_heads: int,
    max_num_tokens: int,
    num_swa_slots: int,
    num_global_slots: int,
) -> int:
    token_counts = list(range(1, min(32, max_num_tokens) + 1))
    if max_num_tokens > 32:
        token_counts.append(max_num_tokens)
    return max(
        _native_workspace_size(
            num_tokens,
            num_heads,
            num_swa_slots,
            num_global_slots,
        )
        for num_tokens in token_counts
    )


def _native_workspace_size_upper_bound(
    max_num_tokens: int,
    num_heads: int,
    min_swa_slots: int,
    max_swa_slots: int,
    num_global_slots: int,
) -> int:
    try:
        from flashinfer.mla import (
            deepseek_v41_mixed_sparse_workspace_size_upper_bound,
        )
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek V4.1 requires FlashInfer's native mixed sparse "
            "attention workspace upper-bound API."
        ) from exc
    return int(
        deepseek_v41_mixed_sparse_workspace_size_upper_bound(
            max_num_tokens,
            num_heads,
            min_swa_slots,
            max_swa_slots,
            num_global_slots,
        )
    )


class DeepseekV41NativeSparseBackend(DeepseekV41SparseMLABackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_DSV41"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512]

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 or (
            capability.major == 8 and capability.minor == 9
        )

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if not cls.supports_compute_capability(device_capability):
            return "DeepSeek V4.1 native attention requires SM89 or SM12x"
        if kv_cache_dtype not in (None, "auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            return "DeepSeek V4.1 requires its native mixed FP8/FP4 cache"
        return None

    @staticmethod
    def get_builder_cls() -> type["DeepseekV41NativeMetadataBuilder"]:
        return DeepseekV41NativeMetadataBuilder


class DeepseekV41NativeMetadataBuilder(DeepseekV41SparseMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekV41NativeSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS


class DeepseekV41NativeSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [32]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 32

    @staticmethod
    def get_builder_cls() -> type[DeepseekV41NativeSWAMetadataBuilder]:
        return DeepseekV41NativeSWAMetadataBuilder


class DeepseekV41NativeMixedAttention(DeepseekV4Attention):
    """BF16-Q attention over 528-byte SWA and 288-byte global cache rows."""

    backend_cls = DeepseekV41NativeSparseBackend
    swa_backend_cls = DeepseekV41NativeSWABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_workspace_manager_initialized():
            return
        max_global_slots = self.index_topk if self.compress_ratio else 0
        workspace_bytes = _native_workspace_size_upper_bound(
            self.max_num_batched_tokens,
            self.n_local_heads,
            self.window_size,
            max(self.window_size, self.max_model_len),
            max_global_slots,
        )
        current_workspace_manager().get_simultaneous(
            ((workspace_bytes,), torch.uint8),
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads not in (8, 16, 32):
            raise ValueError(
                "DeepSeek V4.1 native attention supports 8, 16, or 32 "
                f"local query heads, got {num_heads}."
            )
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        # The native final store applies inverse RoPE and writes [G, T, D].
        group_dim = o.shape[1] * o.shape[2] // self.n_local_groups
        grouped_o = o.view(self.n_local_groups, o.shape[0], group_dim).transpose(0, 1)
        weight = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        projected = torch.einsum("bhd,hrd->bhr", grouped_o, weight)
        return self.wo_b(projected.flatten(1))

    def _global_slots(
        self,
        metadata: DeepseekV41SparseMLAMetadata,
        swa_metadata: "DeepseekSparseSWAMetadata",
        token_slice: slice,
    ) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("V4.1 compressed attention requires indexer top-k.")
        if swa_metadata.token_to_req_indices is None:
            raise RuntimeError("V4.1 attention request mapping is missing.")
        if swa_metadata.is_valid_token is None:
            raise RuntimeError("V4.1 attention validity metadata is missing.")
        slots, _ = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[token_slice],
            swa_metadata.token_to_req_indices[token_slice],
            metadata.block_table,
            metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[token_slice],
        )
        return slots

    def _run_native(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        global_cache: torch.Tensor | None,
        swa_slots: torch.Tensor,
        global_slots: torch.Tensor | None,
        output: torch.Tensor,
        positions: torch.Tensor,
        grouped_output: torch.Tensor,
        token_offset: int,
    ) -> None:
        if q.dtype != torch.bfloat16 or q.shape[-1] != 512:
            raise ValueError("V4.1 native attention requires BF16 Q[..., 512].")
        if swa_slots.ndim == 3:
            if swa_slots.shape[1] != 1:
                raise ValueError("V4.1 SWA slots must contain one KV head.")
            swa_slots = swa_slots[:, 0]
        elif swa_slots.ndim != 2:
            raise ValueError("V4.1 SWA slots must have shape [T, K] or [T, 1, K].")
        if swa_cache.ndim != 3 or swa_cache.shape[-1] != 528:
            raise ValueError("V4.1 SWA cache must have 528-byte interleaved rows.")
        if global_cache is not None and (
            global_cache.ndim != 3 or global_cache.shape[-1] != 288
        ):
            raise ValueError("V4.1 global cache must have 288-byte interleaved rows.")
        _native_mixed_attention(
            q,
            swa_cache,
            global_cache,
            swa_slots,
            global_slots,
            self.scale,
            sinks=self.attn_sink,
            swa_page_stride_bytes=swa_cache.stride(0),
            swa_row_stride_bytes=swa_cache.stride(1),
            global_page_stride_bytes=(
                global_cache.stride(0) if global_cache is not None else None
            ),
            global_row_stride_bytes=(
                global_cache.stride(1) if global_cache is not None else None
            ),
            out=output,
            workspace=_get_native_workspace(q, swa_slots, global_slots),
            positions=positions,
            cos_sin=self.rotary_emb.cos_sin_cache,
            grouped_out=grouped_output,
            token_offset=token_offset,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv
        if output.shape != q.shape or output.dtype != q.dtype:
            raise ValueError("V4.1 native attention output must match BF16 Q.")
        group_dim = q.shape[1] * q.shape[2] // self.n_local_groups
        grouped_output = output.view(self.n_local_groups, q.shape[0], group_dim)
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            output.zero_()
            return
        if not isinstance(attn_metadata, dict):
            raise RuntimeError("V4.1 attention requires per-layer metadata.")
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        if swa_metadata is None:
            raise RuntimeError("V4.1 SWA metadata is missing.")

        swa_only = self.compress_ratio == 0
        global_metadata = cast(
            "DeepseekV41SparseMLAMetadata | None",
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        if not swa_only and global_metadata is None:
            raise RuntimeError("V4.1 global-cache metadata is missing.")
        global_cache = None if swa_only else self._compressed_kv_cache()
        swa_cache = self.swa_cache_layer.kv_cache

        decode_tokens = swa_metadata.num_decode_tokens
        if decode_tokens:
            if swa_metadata.decode_swa_indices is None:
                raise RuntimeError("V4.1 decode SWA slots are missing.")
            decode_slice = slice(0, decode_tokens)
            global_slots = (
                None
                if swa_only
                else self._global_slots(global_metadata, swa_metadata, decode_slice)
            )
            self._run_native(
                q[decode_slice],
                swa_cache,
                global_cache,
                swa_metadata.decode_swa_indices,
                global_slots,
                output[decode_slice],
                positions[decode_slice],
                grouped_output,
                0,
            )

        prefill_tokens = swa_metadata.num_prefill_tokens
        if prefill_tokens:
            if swa_metadata.prefill_swa_indices is None:
                raise RuntimeError("V4.1 prefill SWA slots are missing.")
            prefill_slice = slice(decode_tokens, decode_tokens + prefill_tokens)
            global_slots = (
                None
                if swa_only
                else self._global_slots(global_metadata, swa_metadata, prefill_slice)
            )
            self._run_native(
                q[prefill_slice],
                swa_cache,
                global_cache,
                swa_metadata.prefill_swa_indices,
                global_slots,
                output[prefill_slice],
                positions[prefill_slice],
                grouped_output,
                decode_tokens,
            )
