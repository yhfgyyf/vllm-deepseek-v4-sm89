# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    select_candidate_blocks as _select_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    CANDIDATE_BLOCK_SIZE as DSV41_CANDIDATE_BLOCK_SIZE,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    dsv41_mxfp4_candidate_logits,
    dsv41_mxfp4_dense_logits,
    map_candidate_topk_,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
DSV41_CANDIDATE_BLOCKS = 2048
DSV41_CANDIDATE_WIDTH = DSV41_CANDIDATE_BLOCKS * DSV41_CANDIDATE_BLOCK_SIZE

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _has_cuda_indexer_mqa_backend(use_fp4_cache: bool) -> bool:
    return has_deep_gemm() or (
        HAS_TRITON
        and not use_fp4_cache
        and current_platform.is_device_capability((8, 9))
    )


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _dsv41_topk_from_candidates(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    topk_indices: torch.Tensor,
    workspace: torch.Tensor,
    lengths_out: torch.Tensor,
) -> None:
    """Select compact candidate logits and map them to request-local tokens."""
    rows, width = logits.shape
    compact_lens = lengths_out[:rows]
    compact_lens.fill_(width)
    torch.ops._C.persistent_topk(
        logits,
        compact_lens,
        topk_indices,
        workspace,
        topk_indices.shape[1],
        width,
    )
    map_candidate_topk_(topk_indices, logits, candidates)


def _dsv41_expand_decode_seq_lens(
    seq_lens: torch.Tensor,
    batch_size: int,
    next_n: int,
) -> torch.Tensor:
    if seq_lens.ndim == 2:
        if seq_lens.shape[0] != batch_size or seq_lens.shape[1] not in (1, next_n):
            raise ValueError("V4.1 decode lengths must be [B, 1] or [B, next_n]")
        seq_lens = seq_lens.reshape(-1)
    elif seq_lens.ndim != 1:
        raise ValueError("V4.1 decode lengths must be one- or two-dimensional")

    if seq_lens.numel() == batch_size * next_n:
        return seq_lens
    if seq_lens.numel() != batch_size:
        raise ValueError(
            "V4.1 decode lengths must have one entry per request or query row"
        )
    return seq_lens.reshape(batch_size, 1).expand(batch_size, next_n).reshape(-1)


def _dsv41_workspace_specs(
    max_model_len: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    max_logits_elements = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4
    if max_logits_elements < max(max_model_len, DSV41_CANDIDATE_WIDTH):
        raise ValueError(
            "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB must fit at least one V4.1 "
            "dense and candidate row"
        )
    block_score_elements = max_logits_elements // DSV41_CANDIDATE_BLOCK_SIZE
    row_capacity = max_logits_elements // DSV41_CANDIDATE_WIDTH
    return (
        ((max_logits_elements,), torch.float32),
        ((block_score_elements,), torch.float32),
        ((row_capacity,), torch.int32),
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )


def _dsv41_rows_per_chunk(
    width: int,
    max_logits_elements: int,
    *,
    needs_block_scores: bool,
    group_size: int = 1,
) -> int:
    if width <= 0 or group_size <= 0:
        raise ValueError("V4.1 indexer width and decode group size must be positive")
    row_capacity = max_logits_elements // DSV41_CANDIDATE_WIDTH
    rows = min(row_capacity, max_logits_elements // width)
    if needs_block_scores:
        score_width = max(
            DSV41_CANDIDATE_BLOCKS,
            (width + DSV41_CANDIDATE_BLOCK_SIZE - 1) // DSV41_CANDIDATE_BLOCK_SIZE,
        )
        rows = min(
            rows,
            (max_logits_elements // DSV41_CANDIDATE_BLOCK_SIZE) // score_width,
        )
    rows = rows // group_size * group_size
    if rows < group_size:
        raise ValueError(
            "V4.1 indexer workspace cannot fit one complete decode group; "
            "increase VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"
        )
    return rows


def _dsv41_native_indexer(
    attn_metadata: DeepseekV32IndexerMetadata,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
) -> torch.Tensor:
    """Run the V4.1 MXFP4 indexer without DeepGEMM or dense K expansion."""
    if candidate_write and candidate_blocks is None:
        raise ValueError("V4.1 candidate source requires a candidate output buffer")
    if candidate_blocks is not None:
        if candidate_block_size != DSV41_CANDIDATE_BLOCK_SIZE:
            raise ValueError(
                "DeepSeek V4.1 native indexer requires candidate_block_size=8, "
                f"got {candidate_block_size}"
            )
        if (
            candidate_blocks.dtype != torch.int32
            or candidate_blocks.ndim != 2
            or candidate_blocks.shape[1] != DSV41_CANDIDATE_BLOCKS
        ):
            raise ValueError(
                "V4.1 candidate buffer must be int32 with 2048 blocks per row"
            )
    workspace_manager = current_workspace_manager()
    logits_scratch, block_scores_scratch, lengths_scratch, topk_workspace = (
        workspace_manager.get_simultaneous(*_dsv41_workspace_specs(max_model_len))
    )

    if attn_metadata.num_prefills > 0:
        assert attn_metadata.prefill is not None
        for chunk in attn_metadata.prefill.chunks:
            reads_candidates = candidate_blocks is not None and not candidate_write
            width = (
                candidate_blocks.shape[1] * candidate_block_size
                if reads_candidates
                else chunk.local_total_seq_lens
            )
            rows_per_chunk = _dsv41_rows_per_chunk(
                width,
                logits_scratch.numel(),
                needs_block_scores=candidate_blocks is not None and candidate_write,
            )
            num_rows = chunk.token_end - chunk.token_start
            for row_offset in range(0, num_rows, rows_per_chunk):
                token_start = chunk.token_start + row_offset
                token_end = min(token_start + rows_per_chunk, chunk.token_end)
                rows = token_end - token_start
                local_rows = slice(row_offset, row_offset + rows)
                q_values = q_quant[token_start:token_end]
                q_scales = q_scale[token_start:token_end]
                chunk_weights = weights[token_start:token_end]
                topk_indices = topk_indices_buffer[token_start:token_end, :topk_tokens]
                chunk_candidates = (
                    candidate_blocks[token_start:token_end]
                    if candidate_blocks is not None
                    else None
                )
                logits_out = logits_scratch[: rows * width].view(rows, width)
                if reads_candidates:
                    assert chunk_candidates is not None
                    logits = dsv41_mxfp4_candidate_logits(
                        q_values,
                        q_scales,
                        kv_cache,
                        chunk_weights,
                        chunk.block_table,
                        chunk.cu_seqlen_ke[local_rows],
                        chunk_candidates,
                        row_starts=chunk.cu_seqlen_ks[local_rows],
                        cu_seq_lens=chunk.cu_seq_lens,
                        token_to_seq=chunk.token_to_seq,
                        out=logits_out,
                    )
                    _dsv41_topk_from_candidates(
                        logits,
                        chunk_candidates,
                        topk_indices,
                        topk_workspace,
                        lengths_scratch,
                    )
                    continue

                logits = dsv41_mxfp4_dense_logits(
                    q_values,
                    q_scales,
                    kv_cache,
                    chunk_weights,
                    chunk.block_table,
                    chunk.cu_seqlen_ke[local_rows],
                    width,
                    row_starts=chunk.cu_seqlen_ks[local_rows],
                    cu_seq_lens=chunk.cu_seq_lens,
                    token_to_seq=chunk.token_to_seq,
                    out=logits_out,
                )
                if chunk_candidates is not None:
                    num_blocks = (
                        width + candidate_block_size - 1
                    ) // candidate_block_size
                    score_width = max(chunk_candidates.shape[1], num_blocks)
                    scores_out = block_scores_scratch[: rows * score_width].view(
                        rows, score_width
                    )
                    _select_candidate_blocks(
                        logits,
                        chunk.cu_seqlen_ks[local_rows],
                        chunk.cu_seqlen_ke[local_rows],
                        chunk_candidates.shape[1],
                        candidate_block_size,
                        chunk_candidates,
                        topk_workspace,
                        scores_out=scores_out,
                        block_lens_out=lengths_scratch[:rows],
                    )
                ops.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks[local_rows],
                    chunk.cu_seqlen_ke[local_rows],
                    topk_indices,
                    rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if attn_metadata.num_decodes > 0:
        assert attn_metadata.decode is not None
        decode_metadata = attn_metadata.decode
        decode_lens = decode_metadata.decode_lens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if num_decode_tokens == 0:
            padded_q_values = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scales = q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
            padded_weights = weights[:1]
            padded_candidates = (
                candidate_blocks[:1] if candidate_blocks is not None else None
            )
        elif decode_metadata.requires_padding:
            padded_q_values = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens, pad_value=0
            )
            padded_q_scales = pack_seq_triton(
                q_scale[:num_decode_tokens], decode_lens, pad_value=0
            )
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).flatten(0, 1)
            padded_candidates = (
                pack_seq_triton(
                    candidate_blocks[:num_decode_tokens], decode_lens, pad_value=-1
                ).flatten(0, 1)
                if candidate_blocks is not None
                else None
            )
        else:
            padded_q_values = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            padded_q_scales = q_scale[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_scale.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens]
            padded_candidates = (
                candidate_blocks[:num_decode_tokens]
                if candidate_blocks is not None
                else None
            )

        batch_size, next_n = padded_q_values.shape[:2]
        num_padded_tokens = batch_size * next_n
        seq_lens = _dsv41_expand_decode_seq_lens(
            decode_metadata.seq_lens, batch_size, next_n
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        reads_candidates = padded_candidates is not None and not candidate_write
        if reads_candidates:
            width = padded_candidates.shape[1] * candidate_block_size
        else:
            width = decode_metadata.max_indexer_kv_len
            if width is None:
                raise ValueError(
                    "V4.1 dense decode requires an active indexer KV length"
                )
            if width > max_model_len:
                raise ValueError(
                    "V4.1 dense decode context exceeds the configured maximum: "
                    f"{width} > {max_model_len}"
                )
        rows_per_chunk = _dsv41_rows_per_chunk(
            width,
            logits_scratch.numel(),
            needs_block_scores=padded_candidates is not None and candidate_write,
            group_size=next_n,
        )
        requests_per_chunk = rows_per_chunk // next_n
        for request_start in range(0, batch_size, requests_per_chunk):
            request_end = min(request_start + requests_per_chunk, batch_size)
            row_start = request_start * next_n
            row_end = request_end * next_n
            rows = row_end - row_start
            row_slice = slice(row_start, row_end)
            request_slice = slice(request_start, request_end)
            q_values = padded_q_values[request_slice]
            q_scales = padded_q_scales[request_slice]
            chunk_weights = padded_weights[row_slice]
            chunk_candidates = (
                padded_candidates[row_slice] if padded_candidates is not None else None
            )
            chunk_topk_indices = topk_indices[row_slice]
            chunk_seq_lens = seq_lens[row_slice]
            logits_out = logits_scratch[: rows * width].view(rows, width)
            if reads_candidates:
                assert chunk_candidates is not None
                logits = dsv41_mxfp4_candidate_logits(
                    q_values,
                    q_scales,
                    kv_cache,
                    chunk_weights,
                    decode_metadata.block_table[request_slice],
                    chunk_seq_lens,
                    chunk_candidates,
                    row_repeat=next_n,
                    out=logits_out,
                )
                _dsv41_topk_from_candidates(
                    logits,
                    chunk_candidates,
                    chunk_topk_indices,
                    topk_workspace,
                    lengths_scratch,
                )
                continue

            logits = dsv41_mxfp4_dense_logits(
                q_values,
                q_scales,
                kv_cache,
                chunk_weights,
                decode_metadata.block_table[request_slice],
                chunk_seq_lens,
                width,
                row_repeat=next_n,
                out=logits_out,
            )
            if chunk_candidates is not None:
                num_blocks = (width + candidate_block_size - 1) // candidate_block_size
                score_width = max(chunk_candidates.shape[1], num_blocks)
                scores_out = block_scores_scratch[: rows * score_width].view(
                    rows, score_width
                )
                _select_candidate_blocks(
                    logits,
                    None,
                    chunk_seq_lens,
                    chunk_candidates.shape[1],
                    candidate_block_size,
                    chunk_candidates,
                    topk_workspace,
                    scores_out=scores_out,
                    block_lens_out=lengths_scratch[:rows],
                )
            torch.ops._C.persistent_topk(
                logits,
                chunk_seq_lens,
                chunk_topk_indices,
                topk_workspace,
                topk_tokens,
                width,
            )

        if decode_metadata.requires_padding:
            unpacked = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, topk_tokens), decode_lens
            )
            topk_indices_buffer[: unpacked.shape[0], :topk_tokens] = unpacked
            if candidate_write:
                assert candidate_blocks is not None and padded_candidates is not None
                unpacked_candidates = unpack_seq_triton(
                    padded_candidates.reshape(batch_size, next_n, -1), decode_lens
                )
                candidate_blocks[: unpacked_candidates.shape[0]] = unpacked_candidates

    return topk_indices_buffer


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    use_v41_native: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if use_v41_native:
        if not use_fp4_cache or q_scale is None:
            raise ValueError("V4.1 native indexer requires packed MXFP4 Q and K")
        if dcp_world_size != 1 or use_pcp:
            raise NotImplementedError(
                "V4.1 native indexer currently requires DCP=PCP=1"
            )

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        if use_v41_native:
            current_workspace_manager().get_simultaneous(
                *_dsv41_workspace_specs(max_model_len)
            )
        else:
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            current_workspace_manager().get_simultaneous(
                values_spec,
                scales_spec,
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )

        if not use_v41_native:
            # Dummy allocation to simulate peak logits memory in the old path.
            # FP8 elements so elements == bytes.
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
            use_v41_native=use_v41_native,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if use_pcp:
        num_tokens //= get_pcp_group().world_size
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if use_v41_native:
        assert q_scale is not None
        return _dsv41_native_indexer(
            attn_metadata_narrowed,
            kv_cache,
            q_quant,
            q_scale,
            weights,
            topk_indices_buffer,
            topk_tokens,
            max_model_len,
            candidate_blocks,
            candidate_block_size,
            candidate_write,
        )
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if chunk.local_total_seq_lens == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                row_starts=chunk.cu_seqlen_ks,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 64
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
        if use_cooperative_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.cooperative_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        elif use_persistent_topk:
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    use_v41_native: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "candidate_blocks"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
        use_v41_native: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.use_v41_native = use_v41_native
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        parallel_config = get_current_vllm_config().parallel_config
        self._parallel_config = parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self._cp_kv_cache_interleave_size: int | None = None
        if (
            current_platform.is_cuda()
            and not use_v41_native
            and not _has_cuda_indexer_mqa_backend(use_fp4_cache)
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM, or an "
                "active Triton backend on SM89 with FP8 query cache."
            )

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """With PD+DCP, the real value isn't known until block_size is finalized,
        which happens after this layer is built. Safe to cache after the first access,
        as long as the adjustment always runs before any forward pass
        (it's set up in Worker.initialize_from_config, ahead of warmup/serving).
        """
        if self._cp_kv_cache_interleave_size is None:
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            use_v41_native=self.use_v41_native,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        from vllm.platforms.rocm import on_gfx11

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            or on_gfx11()
        ):
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
                compress_ratio=self.compress_ratio,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
