# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import math

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm import _cpu_ops as cpu_ops
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask as _apply_candidate_mask,
)
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    select_candidate_blocks as _select_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_candidate_blocks import (
    select_candidate_blocks as _select_dsv41_candidate_blocks,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    CANDIDATE_BLOCK_SIZE as DSV41_CANDIDATE_BLOCK_SIZE,
)
from vllm.model_executor.kernels.attention.dsa.dsv41_indexer import (
    dsv41_mxfp4_candidate_logits,
    dsv41_mxfp4_dense_logits,
    map_candidate_topk_,
)
from vllm.model_executor.kernels.attention.dsa.fused_mqa_topk import (
    fused_mqa_topk_chunk,
    fused_mqa_topk_key_scales,
    fused_mqa_topk_prefill,
    get_fused_mqa_topk_workspaces,
    use_fused_mqa_topk,
)
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    deep_select_topk,
    get_indexer_topk,
    is_deep_select_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    mqa_logits_mx_inputs,
    mqa_logits_use_mx_scales,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    dense_logits_rows,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

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
    logits_dtype: torch.dtype = torch.float32,
    num_positions: int | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    `topk_indices` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most `topk_tokens - 1` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships `dcp_world_size * topk_tokens` candidates instead of the whole
    score row. Overwrites `topk_indices` with global token ids (`-1` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    The selector keys only the score bits that `logits_dtype` (the dtype the
    MQA logits were computed in, before the FP32 upcast) carries and the id
    bits that positions below `num_positions` use.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        dcp_topk_id_bits,
        dcp_topk_score_bits,
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
    # Gather along dim 0 and let the selector read the rank-major
    # [world_size, rows, topk, 2] result: a dim-1 gather would reorder it into
    # [rows, world_size * topk, 2] with a full extra copy.
    gathered = get_dcp_group().all_gather(packed, dim=0)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered.view(dcp_world_size, *packed.shape),
        topk_tokens,
        out=topk_indices,
        score_bits=dcp_topk_score_bits(logits_dtype),
        id_bits=32 if num_positions is None else dcp_topk_id_bits(num_positions),
    )


def dcp_gather_kv_rows(
    local_padded: torch.Tensor,
    deinterleave_idx: torch.Tensor,
    gathered_buf: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """All-gather this rank's KV shard and return it in global token order."""
    gathered = gathered_buf[: get_dcp_group().world_size * local_padded.shape[0]]
    dist.all_gather_into_tensor(
        gathered, local_padded.contiguous(), group=get_dcp_group().device_group
    )
    return torch.index_select(
        gathered, 0, deinterleave_idx, out=out_buf[: deinterleave_idx.shape[0]]
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


@triton.jit
def _upcast_logits_kernel(src_ptr, dst_ptr, row_ends_ptr, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    end = tl.load(row_ends_ptr + row)
    src_ptr += row.to(tl.int64) * stride
    dst_ptr += row.to(tl.int64) * stride
    for start in range(tl.program_id(1) * BLOCK, end, tl.num_programs(1) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < end
        vals = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, vals.to(tl.float32), mask=mask)


def as_fp32_logits(logits: torch.Tensor, row_ends: torch.Tensor) -> torch.Tensor:
    """Upcast BF16 MQA logits for the FP32 top-k kernels.

    Only columns below `row_ends` (int32, one per row) are converted, the
    span the consumers read; DeepGEMM's aligned row stride is kept.
    """
    if logits.dtype == torch.float32:
        return logits
    out = torch.empty_strided(
        logits.shape, logits.stride(), dtype=torch.float32, device=logits.device
    )
    num_rows, width = logits.shape
    if num_rows == 0 or width == 0:
        return out
    assert row_ends.numel() == num_rows and row_ends.dtype == torch.int32
    block = 1024
    splits = min(triton.cdiv(width, block), max(1, 1024 // num_rows))
    _upcast_logits_kernel[(num_rows, splits)](
        logits, out, row_ends.contiguous(), logits.stride(0), BLOCK=block
    )
    return out


def _decode_chunk_ranges(
    rows: int, width: int, budget: int, itemsize: int
) -> list[tuple[int, int]]:
    """Partition independent Q rows under a payload budget of logits whose
    elements take `itemsize` bytes."""
    if budget < 0:
        raise ValueError("Decode logits budget must be nonnegative")
    if not rows or not budget or rows * width * itemsize <= budget:
        return []
    limit = budget // (itemsize * width) // 16 * 16
    if limit < 16:
        raise ValueError("Decode logits budget must fit at least 16 logits rows")
    units = (rows + 15) // 16
    count = (units + limit // 16 - 1) // (limit // 16)
    base, extra = divmod(units, count)
    ranges = []
    start = 0
    # Allocate the largest scratch first so later graph nodes can reuse it.
    for i in range(count):
        end = min(rows, start + 16 * (base + (i < extra)))
        ranges.append((start, end))
        start = end
    return ranges


def _varlen_decode_rows(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    decode: DeepSeekV32IndexerDecodeMetadata,
    max_model_len: int,
    max_seq_len: int,
    topk_backend: str,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    candidate_blocks: torch.Tensor | None,
    candidate_block_size: int,
    candidate_write: bool,
    logits_dtype: torch.dtype,
    start: int,
    end: int,
) -> None:
    """The varlen (SM100) decode logits, candidate selection and top-k of rows
    `[start, end)`, with the slice's own DeepGEMM schedule."""
    assert decode.indices is not None
    seq_lens = decode.seq_lens[start:end]
    indices = decode.indices[start:end]
    schedule = decode.schedule_metadata
    if end - start < decode.seq_lens.shape[0]:
        schedule = get_paged_mqa_logits_metadata(
            seq_lens, kv_cache.shape[1], schedule.shape[0] - 1, indices=indices
        )
    logits = fp8_fp4_paged_mqa_logits(
        (q[start:end], q_scale[start:end] if q_scale is not None else None),
        kv_cache,
        weights[start:end],
        seq_lens,
        decode.block_table[start:end],
        schedule,
        max_model_len=max_model_len,
        logits_dtype=logits_dtype,
        indices=indices,
    )
    row_ends = seq_lens.view(-1)
    selector = get_indexer_topk(topk_backend)
    if not selector.takes_bf16_logits(logits, topk_tokens, end - start):
        logits = as_fp32_logits(logits, row_ends)
    if candidate_blocks is not None:
        candidates = candidate_blocks[start:end]
        if candidate_write:
            _select_candidate_blocks(
                logits,
                None,
                row_ends,
                candidates.shape[1],
                candidate_block_size,
                candidates,
                1,
                score_dtype=(
                    torch.float32 if topk_backend in ("auto", "persistent") else None
                ),
            )
        else:
            _apply_candidate_mask(
                logits, None, row_ends, candidates, candidate_block_size, 1
            )
    selector(
        logits,
        seq_lens,
        1,
        topk_indices_buffer[start:end, :topk_tokens],
        topk_tokens,
        max_seq_len,
    )


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
    """4D `[num_blocks, block_size, 1, head_width]` view expected by
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
            if reads_candidates:
                assert candidate_blocks is not None
                width = candidate_blocks.shape[1] * candidate_block_size
            else:
                width = chunk.local_total_seq_lens
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
                    _select_dsv41_candidate_blocks(
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
            assert padded_candidates is not None
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
                _select_dsv41_candidate_blocks(
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
    pcp_shard_decode_requests: bool,
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
    topk_backend: str = "auto",
    logits_dtype: torch.dtype = torch.float32,
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
    if candidate_blocks is not None:
        # Candidate blocks are request-local; the DCP-sharded logits layout
        # would need per-rank translation that is not implemented.
        assert dcp_world_size == 1, (
            "v4.1 two-level candidate filtering is not supported with DCP."
        )
        assert candidate_block_size > 0

    fused_mqa_topk = not use_v41_native and use_fused_mqa_topk(
        query_shape=q_quant.shape[1:],
        topk=topk_tokens,
        fp4_cache=use_fp4_cache,
        full_context=use_pcp or dcp_world_size == 1,
        candidate_selection=candidate_blocks is not None,
    )

    # DeepSelect ranks BF16 decode logits directly; other top-k backends and
    # the DCP merge read an FP32 copy.
    fp32_decode_logits = topk_backend != "deep_select" or dcp_world_size != 1

    decode_budget = envs.VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB * 1024 * 1024
    chunk_decode = (
        decode_budget > 0
        and current_platform.is_device_capability_family(100)
        and not use_pcp
        and dcp_world_size == 1
        and topk_backend in ("auto", "persistent")
        and topk_tokens in (512, 1024, 2048)
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
            profile_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
                values_spec,
                scales_spec,
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            ]
            if use_pcp and dcp_world_size > 1:
                # The PCP+DCP path takes an all-gather destination and a
                # de-interleaved result.
                gather_spec = _gather_workspace_shapes(
                    total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
                )
                profile_specs.extend(gather_spec * 2)
            current_workspace_manager().get_simultaneous(*profile_specs)
            if fused_mqa_topk:
                # Allocate the largest fused_mqa_topk workspaces so they are
                # accounted for (freed again, they overlap the dense logits below).
                get_fused_mqa_topk_workspaces(q_quant.device, q_quant.shape[0])

            # Decode rows per step: each decode is padded to 1 + speculative tokens.
            # The profiling run sets the vLLM config; other dummy runs may not.
            max_decode_tokens = 0
            if (cfg := get_current_vllm_config_or_none()) is not None:
                max_decode_tokens = min(
                    cfg.scheduler_config.max_num_seqs,
                    cfg.scheduler_config.max_num_batched_tokens,
                ) * (1 + cfg.num_speculative_tokens)

            # Needed with the fused kernel too: chunks fused_mqa_topk_chunk keeps
            # dense take this path (row-split to stay within the budget).
            # Dummy allocation for the peak logits memory during inference: a
            # prefill chunk's, or the decode logits, which eager and piecewise
            # steps allocate outside the CUDA graph pools. DeepGEMM emits them in
            # `logits_dtype`, each row holding whole KV splits at a 1024-byte
            # stride; an FP32 copy of BF16 logits keeps that stride. Prefill always
            # reads FP32 logits, and the chunk splitter caps them at the budget.
            split_kv = 384 if current_platform.is_device_capability_family(100) else 256
            row_align = math.lcm(split_kv, 1024 // logits_dtype.itemsize)
            copy_bytes = 0 if logits_dtype == torch.float32 else torch.float32.itemsize
            decode_copy_bytes = copy_bytes if fp32_decode_logits else 0
            if chunk_decode:
                # A decode chunk holds at most the budget, read in place by top-k.
                max_decode_tokens = min(
                    max_decode_tokens,
                    decode_budget // (logits_dtype.itemsize * max_model_len),
                )
                decode_copy_bytes = 0
            prefill_budget = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            max_logits_bytes = max(
                prefill_budget
                // torch.float32.itemsize
                * (logits_dtype.itemsize + copy_bytes),
                max_decode_tokens
                * round_up(max_model_len, row_align)
                * (logits_dtype.itemsize + decode_copy_bytes),
            )
            _ = torch.empty(
                max_logits_bytes, dtype=torch.uint8, device=hidden_states.device
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
            pcp_shard_decode_requests,
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
    prefill_metadata = attn_metadata_narrowed.prefill
    # If all prefills are shorter than the compression ratio, chunks are empty:
    # there are no compressed KV entries to score or select with top-k.
    # Skip prefill workspace setup, which calls max() on the chunks and would
    # raise ValueError for an empty list.
    has_prefill = prefill_metadata is not None and bool(prefill_metadata.chunks)
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
    if k is not None:
        # MTP draft decodes are replicated and use an unexpanded slot mapping.
        if use_pcp and num_tokens > k.shape[0]:
            num_tokens //= get_pcp_group().world_size
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
            pcp_shard_decode_requests=pcp_shard_decode_requests,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            "mxfp8" if mqa_logits_use_mx_scales() else scale_fmt,
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

        # Get the shared workspace buffers once, sized for this batch's largest
        # chunk rather than the profiled bound (total_seq_lens), so the scratch
        # stays at the workspace head and leaves retained tail spans intact.
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            max(c.max_local_total_seq_lens for c in prefill_metadata.chunks),
            head_dim,
            fp8_dtype,
            use_fp4_cache,
        )
        # PCP + DCP needs two more pairs: the rank-major all-gather destination
        # and the de-interleaved result.
        pcp_chunks = [
            (c, c.pcp_deinterleave_idx)
            for c in prefill_metadata.chunks
            if c.pcp_deinterleave_idx is not None
        ]
        gather_specs: list[tuple[tuple[int, int], torch.dtype]] = []
        if pcp_chunks:
            gathered_rows = max(
                dcp_world_size * c.local_total_seq_lens for c, _ in pcp_chunks
            )
            deinterleaved_rows = max(idx.shape[0] for _, idx in pcp_chunks)
            for rows in (gathered_rows, deinterleaved_rows):
                gather_specs.extend(
                    _gather_workspace_shapes(rows, head_dim, fp8_dtype, use_fp4_cache)
                )
        k_quant_full, k_scale_full, *gather_bufs = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
            *gather_specs,
        )
        # Under ShardQ the chunks index the shardq_gather_indices tokens.
        prefill_q, prefill_q_scale, prefill_weights = q_quant, q_scale, weights
        if (indices := prefill_metadata.shardq_gather_indices) is not None:
            assert candidate_blocks is None
            prefill_q, prefill_weights = q_quant[indices], weights[indices]
            if q_scale is not None:
                prefill_q_scale = q_scale[indices]
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

            row_start, row_end = chunk.token_start, chunk.token_end

            # PCP + DCP KV all-gather.
            deinterleave_idx = chunk.pcp_deinterleave_idx
            if deinterleave_idx is not None:
                # local_total_seq_lens is the PCP-padded extent here, so every
                # rank contributes an identically shaped shard.
                gathered_values, gathered_scales, out_values, out_scales = gather_bufs
                shard = k_quant[: chunk.local_total_seq_lens]
                k_quant = dcp_gather_kv_rows(
                    shard, deinterleave_idx, gathered_values, out_values
                )
                k_scale = dcp_gather_kv_rows(
                    k_scale[: chunk.local_total_seq_lens],
                    deinterleave_idx,
                    gathered_scales,
                    out_scales,
                )

            q_slice = prefill_q[row_start:row_end]
            q_scale_slice = (
                prefill_q_scale[row_start:row_end]
                if prefill_q_scale is not None
                else None
            )
            topk_indices = topk_indices_buffer[row_start:row_end, :topk_tokens]

            # DeepGEMM returns compressed logits: row i starts at K position
            # cu_seqlen_ks[i]. None means every row starts at column zero.
            row_starts: torch.Tensor | None = None
            # Rows per dense logits call under the logits budget.
            dense_rows = dense_logits_rows(chunk.max_seqlen_k)
            use_deep_select = False
            if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            elif fused_mqa_topk and fused_mqa_topk_chunk(
                chunk.max_seqlen_k,
                k_quant.shape[0],
                topk_tokens,
                q_slice.shape[0],
                dense_rows,
            ):
                fused_mqa_topk_prefill(
                    q_slice,
                    k_quant,
                    fused_mqa_topk_key_scales(k_scale),
                    prefill_weights[row_start:row_end],
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                )
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
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
                    k_scale_cast = k_scale.view(
                        torch.int32 if mqa_logits_use_mx_scales() else torch.float32
                    ).squeeze(-1)
                # Chunks planned for fused_mqa_topk are not bounded by the logits
                # budget: split their rows (budget-planned chunks, the only ones
                # under DCP, are one window, so the merge below sees all the
                # rows).
                for sub in range(0, q_slice.shape[0], dense_rows):
                    rows = slice(sub, min(sub + dense_rows, q_slice.shape[0]))
                    tokens = slice(row_start + rows.start, row_start + rows.stop)
                    if current_platform.is_xpu():
                        if q_scale_slice is not None:
                            raise RuntimeError(
                                "XPU fp8_mqa_logits does not support FP4 Q"
                            )
                        logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                            q_slice_cast[rows],
                            k_quant_cast,
                            k_scale_cast,
                            prefill_weights[tokens],
                            cu_seqlen_ks[rows],
                            cu_seqlen_ke[rows],
                        )
                        row_starts = cu_seqlen_ks[rows]
                        row_ends = cu_seqlen_ke[rows]
                    else:
                        q_scale_rows, chunk_weights = mqa_logits_mx_inputs(
                            q_slice_cast[rows],
                            q_scale_slice[rows] if q_scale_slice is not None else None,
                            prefill_weights[tokens],
                            logits_dtype,
                        )
                        row_ends = cu_seqlen_ke[rows] - cu_seqlen_ks[rows]
                        logits = fp8_fp4_mqa_logits(
                            (q_slice_cast[rows], q_scale_rows),
                            (k_quant_cast, k_scale_cast),
                            chunk_weights,
                            cu_seqlen_ks[rows],
                            cu_seqlen_ke[rows],
                            chunk.max_seqlen_k,
                            logits_dtype=logits_dtype,
                        )
                        # DeepSelect ranks the bf16 logits in place of their exact
                        # fp32 copy; fp32 logits keep the per-row top-k.
                        use_deep_select = (
                            topk_backend == "deep_select"
                            and logits.dtype == torch.bfloat16
                            and dcp_world_size == 1
                            and is_deep_select_supported(logits, topk_tokens)
                        )
                        if not use_deep_select:
                            logits = as_fp32_logits(logits, row_ends)
                    num_rows = logits.shape[0]
                    if candidate_blocks is not None:
                        # Two-level selection (v4.1): the candidate source
                        # publishes its top blocks; later indexers mask their
                        # scores to them. Both before the row top-k.
                        chunk_candidates = candidate_blocks[tokens]
                        if candidate_write:
                            _select_candidate_blocks(
                                logits,
                                row_starts,
                                row_ends,
                                chunk_candidates.shape[1],
                                candidate_block_size,
                                chunk_candidates,
                            )
                        else:
                            _apply_candidate_mask(
                                logits,
                                row_starts,
                                row_ends,
                                chunk_candidates,
                                candidate_block_size,
                            )
                    if use_deep_select:
                        deep_select_topk(
                            logits,
                            topk_tokens,
                            end=row_ends,
                            output_idx=topk_indices[rows],
                        )
                    else:
                        ops.top_k_per_row_prefill(
                            logits,
                            torch.zeros_like(row_ends)
                            if row_starts is None
                            else row_starts,
                            row_ends,
                            topk_indices[rows],
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            topk_tokens,
                        )

            if deinterleave_idx is None:
                # Under the PCP path the top-k already ran over the whole
                # context.
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=row_starts,
                    logits_dtype=logits_dtype,
                    num_positions=max_model_len,
                )
            # Free this chunk's logits before the next chunk allocates its own.
            del logits

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        # requires_padding can be computed False for ragged warmup/mixed
        # batches (seen on SM120 TP=2: 8 tokens over 6 seqs -> uniform
        # reshape below would crash). The token-count/batch-size divisibility
        # check is a necessary-but-not-sufficient proxy for "uniform batch"
        # (e.g. decode_lens=[1, 2, 3] divides evenly by 3 but isn't uniform);
        # it catches the specific case seen in practice without requiring a
        # device sync to inspect decode_lens directly, which would break
        # CUDA-graph-capture safety on this path. Shared by both the pack
        # branch below and the matching unpack branch further down — they
        # must agree on which path was taken.
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif needs_padded_path:
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
        if chunk_decode:
            assert decode_metadata.indices is not None
            assert decode_metadata.global_seq_lens is None
            assert not needs_padded_path
            ranges = _decode_chunk_ranges(
                num_padded_tokens,
                max_model_len,
                decode_budget,
                logits_dtype.itemsize,
            )
            if ranges:
                padded_q_scale, decode_weights = mqa_logits_mx_inputs(
                    padded_q_quant_cast,
                    padded_q_scale,
                    weights[:num_padded_tokens],
                    logits_dtype,
                )
                for start, end in ranges:
                    _varlen_decode_rows(
                        padded_q_quant_cast,
                        padded_q_scale,
                        decode_weights,
                        kv_cache,
                        decode_metadata,
                        max_model_len,
                        attn_metadata_narrowed.max_seq_len,
                        topk_backend,
                        topk_tokens,
                        topk_indices_buffer,
                        candidate_blocks,
                        candidate_block_size,
                        candidate_write,
                        logits_dtype,
                        start,
                        end,
                    )
                return topk_indices_buffer
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
            padded_q_scale, decode_weights = mqa_logits_mx_inputs(
                padded_q_quant_cast,
                padded_q_scale,
                weights[:num_padded_tokens],
                logits_dtype,
            )
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                decode_weights,
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                logits_dtype=logits_dtype,
                indices=decode_metadata.indices,
            )
            # Keeping bf16 halves top-k/candidate reads. SM100 decode is varlen:
            # one context length per row.
            if dcp_world_size != 1 or not get_indexer_topk(
                topk_backend
            ).takes_bf16_logits(logits, topk_tokens, logits.shape[0]):
                logits = as_fp32_logits(logits, seq_lens.view(-1))
        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            # Two-level selection (v4.1) on the decode logits; columns are
            # request-local compressed positions. seq_lens is (B, next_n)
            # for native spec decode (per-row effective lens) and (B, 1)
            # otherwise.
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                    score_dtype=(
                        torch.float32
                        if topk_backend in ("auto", "cooperative", "persistent")
                        else None
                    ),
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
                logits_dtype=logits_dtype,
                num_positions=max_model_len,
            )

        if needs_padded_path:
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
    pcp_shard_decode_requests: bool,
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
    topk_backend: str = "auto",
    logits_dtype: torch.dtype = torch.float32,
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
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        self.logits_dtype = STR_DTYPE_TO_TORCH_DTYPE[
            vllm_config.attention_config.resolve_indexer_logits_dtype("float32")
        ]
        self._parallel_config = parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.pcp_shard_decode_requests = parallel_config.pcp_shard_decode_requests
        if (
            current_platform.is_cuda()
            and not use_v41_native
            and not _has_cuda_indexer_mqa_backend(use_fp4_cache)
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM, or an "
                "active Triton backend on SM89 with FP8 query cache."
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            from vllm.v1.attention.ops.common import (
                _PACK_SEQ_TRITON_KERNEL,
                _UNPACK_SEQ_TRITON_KERNEL,
            )

            pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()
            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=pack_dtype,
                pad_value=0 if use_fp4_cache else -float("inf"),
            )
            _UNPACK_SEQ_TRITON_KERNEL.register_warmup()

            if self.dcp_world_size > 1 and current_platform.is_cuda() and has_cutedsl():
                from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                    _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
                )

                _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()

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
            if self.use_pcp and self.dcp_world_size > 1:
                raise NotImplementedError(
                    "The ROCm sparse-indexer path does not support PCP+DCP."
                )
            return self.forward_hip(hidden_states, q_quant, k, weights)
        elif current_platform.is_cpu():
            return self.forward_cpu(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm, XPU and CPU platforms."
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
            self.pcp_shard_decode_requests,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self._parallel_config.cp_kv_cache_interleave_size,
            False,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            topk_backend=self.topk_backend,
            logits_dtype=self.logits_dtype,
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
        from vllm.platforms.rocm import on_gfx950

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            # The so-called AITER sparse indexer op has a native gfx950 path:
            # its cache insert, MQA logits, and top-k fallbacks are implemented
            # by local Triton/C++ kernels and do not require the aiter package.
            or on_gfx950()
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
                candidate_blocks=self.candidate_blocks,
                candidate_block_size=self.candidate_block_size,
                candidate_write=self.candidate_write,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path requires AITER or a supported "
            "native architecture (gfx950)."
        )

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        """CPU sparse attention indexer: cache write stays eager Python glue
        (own K-cache layout, not shared with the main attention cache
        write). PREFILL and DECODE both call the ported
        `fp8_paged_mqa_logits_cpu`/`topk_transform_512_cpu` kernels,
        which read the paged K-cache directly via `page_table` -- no
        eager gather step, no per-request Python loop.

        `prefill_metadata.chunks` always has exactly one entry here:
        `DeepseekV4CPUIndexerMetadataBuilder` overrides the base chunk
        split to always return the whole step's prefill batch as one
        chunk, since the base chunking only bounds CUDA/XPU's dense M*N
        logits tensor and flat K-gather workspace, neither of which this
        paged kernel allocates.
        """
        assert not self.use_fp4_cache, (
            "CPU sparse indexer doesn't support fp4 cache yet"
        )
        assert isinstance(q_quant, torch.Tensor), (
            "CPU sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        assert self.dcp_world_size <= 1 and not self.use_pcp, (
            "CPU sparse indexer doesn't support decode/prefill context parallelism yet."
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        attn_metadata_narrowed: DeepseekV32IndexerMetadata | None = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata[self.k_cache.prefix]
            assert isinstance(metadata, DeepseekV32IndexerMetadata)
            attn_metadata_narrowed = metadata
        if attn_metadata_narrowed is None:
            # Profiling/dummy run: no real metadata to act on.
            return self.topk_indices_buffer

        kv_cache = self.k_cache.kv_cache
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer
        slot_mapping = attn_metadata_narrowed.slot_mapping
        has_decode = attn_metadata_narrowed.num_decodes > 0
        has_prefill = attn_metadata_narrowed.num_prefills > 0
        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

        num_tokens = slot_mapping.shape[0]
        if k is not None:
            k = k[:num_tokens]

        if not self.skip_k_cache_insert:
            # Only reachable via DeepseekV32Attention with
            # prefill_context_parallel_size > 1 on CPU -- set_k_cpu/set_s_cpu
            # (the kernels this used to call) have been removed as unused/
            # untested (csrc/cpu/sgl-kernels/store_cache.cpp).
            raise NotImplementedError(
                "SparseAttnIndexer.forward_cpu: skip_k_cache_insert=False "
                "(prefill context parallel on CPU) is not supported."
            )

        topk_indices_buffer[: hidden_states.shape[0]] = -1

        if has_prefill:
            assert topk_tokens == 512, (
                "topk_transform_512_cpu only supports index_topk == 512."
            )
            prefill_metadata = attn_metadata_narrowed.prefill
            assert prefill_metadata is not None
            assert len(prefill_metadata.chunks) == 1, (
                "forward_cpu expects the prefill metadata builder to always "
                "produce a single chunk -- see "
                "DeepseekV4CPUIndexerMetadataBuilder._split_indexer_prefill_chunks."
            )
            chunk = prefill_metadata.chunks[0]
            # kv_cache is a per-layer view into vLLM's shared multi-layer
            # cache allocation, so its block stride generally exceeds
            # block_size * page_width; fp8_paged_mqa_logits_cpu reads
            # kv_view.stride(0) explicitly, so no copy is needed here.
            kv_view = kv_cache.view(kv_cache.shape[0], -1)
            block_size = kv_cache.shape[1]
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.local_total_seq_lens == 0:
                topk_indices.fill_(-1)
            else:
                assert chunk.local_cu_seq_lens is not None
                # Each token's own (per-request, DCP-local) causal length.
                local_seq_lens = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                # Recover each token's owning request from its row-start
                # tag. Ties (from zero-length requests) are harmless: those
                # tokens have local length 0 and never dereference
                # page_table.
                req_idx = (
                    torch.searchsorted(
                        chunk.local_cu_seq_lens, chunk.cu_seqlen_ks, right=True
                    )
                    - 1
                )
                page_table = chunk.block_table[req_idx]
                # The true max over this chunk's own rows, NOT
                # chunk.max_local_total_seq_lens (that field sums every
                # request's length in the chunk, bounding the old
                # flat-gather buffer this paged path no longer allocates).
                max_seq_len = int(local_seq_lens.max().item())

                logits = cpu_ops.fp8_paged_mqa_logits_cpu(
                    q_slice,
                    kv_view,
                    weights[chunk.token_start : chunk.token_end],
                    local_seq_lens,
                    page_table,
                    block_size,
                    max_seq_len,
                )
                out_page_scratch = torch.empty(
                    (q_slice.shape[0], topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                cpu_ops.topk_transform_512_cpu(
                    logits,
                    local_seq_lens,
                    page_table,
                    out_page_scratch,
                    block_size,
                    topk_indices,
                )

        if has_decode:
            decode_metadata = attn_metadata_narrowed.decode
            assert decode_metadata is not None
            assert not decode_metadata.requires_padding, (
                "CPU sparse indexer decode path does not support speculative "
                "decoding (native MTP) yet."
            )
            batch_size = decode_metadata.decode_lens.shape[0]
            if batch_size > 0:
                # No native MTP on CPU (asserted above) => exactly one
                # query token per decode request, so the flat slice below
                # is already the batch-major layout
                # fp8_paged_mqa_logits_cpu wants.
                assert num_decode_tokens == batch_size, (
                    "CPU sparse indexer decode path expects exactly one query "
                    "token per decode request."
                )
                assert topk_tokens == 512, (
                    "topk_transform_512_cpu only supports index_topk == 512."
                )
                seq_lens = decode_metadata.seq_lens
                seq_lens = (
                    seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
                )
                block_table = decode_metadata.block_table[:batch_size]
                block_size = kv_cache.shape[1]
                kv_view = kv_cache.view(kv_cache.shape[0], -1)

                logits = cpu_ops.fp8_paged_mqa_logits_cpu(
                    q_quant[:num_decode_tokens],
                    kv_view,
                    weights[:num_decode_tokens],
                    seq_lens,
                    block_table,
                    block_size,
                    attn_metadata_narrowed.max_seq_len,
                )

                # out_page_indices is a required kernel output but unused:
                # the indexer's topk output must stay local/compressed-
                # context positions (resolved later by
                # DeepseekV4CPUAttention.forward_mqa via
                # map_local_to_global_slots_cpu).
                out_page_scratch = torch.empty(
                    (batch_size, topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                cpu_ops.topk_transform_512_cpu(
                    logits,
                    seq_lens,
                    block_table,
                    out_page_scratch,
                    block_size,
                    topk_indices_buffer[:num_decode_tokens, :topk_tokens],
                )

        return topk_indices_buffer
