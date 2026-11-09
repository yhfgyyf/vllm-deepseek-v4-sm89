# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import get_pp_indices
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.deepseek_v41 import (
    hc_collapse_triton,
    mhc_post_pre_delayed,
    mhc_pre_delayed_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    drop_checkpoint_cache,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v4.nvidia.model import (
    make_deepseek_v4_expert_params_mapping,
)
from vllm.models.deepseek_v4_1.attention import (
    DeepseekV4Attention,
    _indexer_k_cache_head_dim,
    _replace_layer_index,
    dsv41_indexer_uses_fp4,
)
from vllm.models.deepseek_v4_1.nvidia.ced import (
    CED_METADATA_KEY,
    CEDDraftContext,
    CEDStep,
    CEDTailState,
    get_ced_replay_capacity,
    validate_ced_config,
)
from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
    DeepseekV41NativeMixedAttention,
)
from vllm.models.deepseek_v4_1.nvidia.pp_state import copy_cache_rows, pp_payload_shapes
from vllm.models.deepseek_v4_1.nvidia.router import DeepseekV41TopKBiasRouter
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    EngramBatch,
    EngramLayout,
    NgramHashState,
    can_share_engram_tables,
    gather_engram_hashes,
)
from vllm.models.deepseek_v41.nvidia.ops.mhc import hc_expand
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.mm_preprocess import image_sentinel_mask
from ..common.pipeline import validate_pipeline_partition

if typing.TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        if vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS:
            raise NotImplementedError(
                "DeepSeek V4.1 native routing currently supports modular MoE "
                "backends only."
            )

        super().__init__(
            vllm_config,
            prefix=prefix,
            use_sequence_parallel=use_sequence_parallel,
            num_hash_layers=0,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
        )
        if self.experts.is_monolithic:
            raise NotImplementedError(
                "DeepSeek V4.1 native routing requires a modular MoE backend."
            )
        previous_router = self.experts.router
        self.experts.router = DeepseekV41TopKBiasRouter(
            top_k=self.n_activated_experts,
            global_num_experts=self.n_routed_experts,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            bias_vl=self.gate.bias_vl,
            image_sentinel_lo=self.image_sentinel_lo,
            renormalize=self.renormalize,
            routed_scaling_factor=self.routed_scaling_factor,
            eplb_state=previous_router.eplb_state,
            scoring_func=self.scoring_func,
            num_fused_shared_experts=getattr(
                previous_router, "num_fused_shared_experts", 0
            ),
            shared_expert_weight=getattr(previous_router, "shared_expert_weight", 1.0),
        )


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Select only the native V4.1 mixed-cache attention implementation."""
    if current_platform.is_rocm():
        raise NotImplementedError(
            "DeepSeek V4.1 native mixed-cache attention supports NVIDIA "
            "SM89 and SM12x only."
        )
    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend is not None and backend not in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41_NATIVE,
    ):
        raise ValueError(
            f"{backend.name} is not a native DeepSeek V4.1 attention backend. "
            "V4.1 does not fall back to FlashMLA or the DeepSeek V4 FP8 cache."
        )
    if device_capability is not None and not (
        device_capability.major == 12
        or (device_capability.major == 8 and device_capability.minor == 9)
    ):
        raise NotImplementedError(
            "DeepSeek V4.1 native mixed-cache attention supports SM89 and SM12x."
        )
    return DeepseekV41NativeMixedAttention


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    parallel_config = vllm_config.parallel_config
    use_mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
    return (
        parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (use_mega_moe or parallel_config.data_parallel_size > 1)
    )


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
        engram_prefetch_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                    prefetch_stream=engram_prefetch_stream,
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.use_sequence_parallel:
            self.attn.wo_b.reduce_results = False
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        if residual is None:
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                residual = hc_expand(x, self.hc_mult)
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    x=x,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
            else:
                residual = x
                if self.engram is not None and engram_hashes is not None:
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        engram_mask,
                    )
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                    norm_weight=self.attn_norm.weight,
                    norm_eps=self.attn_norm.variance_epsilon,
                )
        elif self.engram is None or engram_hashes is None:
            residual, post_mix, res_mix, x, attn_pre = mhc_post_pre_delayed(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )
        else:
            residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if self.engram is not None and engram_hashes is not None:
                # Engram injection happens between the previous sublayer's
                # post and this block's pre, on the full hc stream, so the
                # mix coefficients see the injected stream.
                residual = self.engram(
                    residual,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )
            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=self.attn_norm.weight,
                norm_eps=self.attn_norm.variance_epsilon,
            )

        if self.use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)

        residual, post_mix, res_mix, x, ffn_pre = mhc_post_pre_delayed(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre


class DeepseekV4Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            validate_pipeline_partition(
                config,
                [
                    get_pp_indices(config.num_hidden_layers, rank, pp_size)
                    for rank in range(pp_size)
                ],
            )
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.pp4 = pp_size == 4
        self.pp_index_width = (
            _indexer_k_cache_head_dim(
                config.index_head_dim, dsv41_indexer_uses_fp4(vllm_config)
            )
            if self.pp4
            else 0
        )
        self.use_mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        self.ced_enabled = bool(getattr(config, "ced_prefill", False))
        self.ced_tail: CEDTailState | None = None
        if self.ced_enabled:
            validate_ced_config(vllm_config, self.use_sequence_parallel)
        self.ced_decoder_owner = self.ced_enabled and (
            get_pp_group().rank_in_group == 2
            if self.pp4
            else get_pp_group().is_last_rank
        )
        self.ced_decoder_stage = self.ced_enabled and (
            get_pp_group().rank_in_group >= 2
            if self.pp4
            else get_pp_group().is_last_rank
        )
        if self.ced_decoder_owner:
            self.ced_tail = CEDTailState(
                vllm_config.scheduler_config.max_num_seqs,
                config.hidden_size,
                window=get_ced_replay_capacity(config),
            )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if (
            candidate_source_layer >= 0
            and candidate_topk_blocks > 0
            and (
                get_pp_group().is_last_rank
                or (self.pp4 and get_pp_group().rank_in_group == 2)
            )
        ):
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        spec_config = vllm_config.speculative_config
        if get_pp_group().is_first_rank or (
            get_pp_group().is_last_rank
            and spec_config is not None
            and spec_config.use_dspark()
            and spec_config.draft_tensor_parallel_size
            == vllm_config.parallel_config.tensor_parallel_size
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)
        engram_config = vllm_config.engram_config
        self.vllm_config = vllm_config
        engram_prefetch_stream = (
            torch.cuda.Stream()
            if self.engram_layout is not None
            and engram_config is not None
            and engram_config.cpu_offload
            else None
        )
        if (
            self.engram_layout is not None
            and engram_config is not None
            and engram_config.dp_shared_memory
        ):
            engram_config.dp_shared_memory = can_share_engram_tables(self.engram_layout)

        if self.engram_layout is not None and engram_config and engram_config.use_thp:
            # Release old checkpoint cache before allocating the Engram host tables.
            model_config = vllm_config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=vllm_config.load_config.download_dir,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
                engram_prefetch_stream=engram_prefetch_stream,
            ),
            prefix=f"{prefix}.layers",
        )

        self.engram_batch = EngramBatch.create(
            [
                layer.engram
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if getattr(layer, "engram", None) is not None
            ],
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_dp_shared_memory = bool(
            vllm_config.engram_config and vllm_config.engram_config.dp_shared_memory
        )
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            engram = next(
                (
                    layer.engram
                    for layer in islice(self.layers, self.start_layer, self.end_layer)
                    if isinstance(layer, DeepseekV4DecoderLayer)
                    and layer.engram is not None
                ),
                None,
            )
            if engram is not None:
                # Shared host tables and GPU peer reads look up only local tokens.
                self.engram_dp_shared_memory = engram.embed_tokens.dp_size == 1
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        needs_mtp_hidden_states = (
            spec_config is not None
            and not spec_config.use_dspark()
            and (spec_config.use_eagle() or spec_config.uses_draft_model())
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        pp_size = self.parallel_config.pipeline_parallel_size
        last_stage_start = get_pp_indices(40, pp_size - 1, pp_size)[0]
        if self.parallel_config.pipeline_parallel_size > 1 and any(
            not last_stage_start < layer <= 40 for layer in layers
        ):
            raise NotImplementedError(
                "V4.1 PP requires decoder-only auxiliary captures on the last stage"
            )
        super()._set_aux_hidden_state_layers(layers)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.hc_mult, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
        }
        if self.pp4:
            for key, (shape, field_dtype) in pp_payload_shapes(
                self.start_layer,
                self.config.index_topk,
                self.config.candidate_topk_blocks,
                self.pp_index_width,
            ).items():
                tensors[key] = torch.zeros(
                    (batch_size, *shape), dtype=field_dtype, device=device
                )
        return IntermediateTensors(tensors)

    def _pp_shared_cache(self, source: int, *, index: bool = False):
        attention = self.layers[self.start_layer].attn
        prefix = _replace_layer_index(attention.prefix, source)
        if index:
            prefix += ".indexer.k_cache"
        return attention._static_forward_context[prefix]

    def _pp_cache_deltas(self, num_tokens: int) -> dict[str, torch.Tensor]:
        if not self.pp4 or self.end_layer not in (10, 30, 31, 32):
            return {}
        source = 8 if self.end_layer == 10 else 20
        metadata = get_forward_context().attn_metadata
        deltas = {}
        for key, index, width in (
            ("pp_kv", False, 288),
            *(([("pp_index_k", True, self.pp_index_width)]) if source == 20 else []),
        ):
            owner = self._pp_shared_cache(source, index=index)
            payload = torch.zeros(
                (num_tokens, width),
                dtype=torch.uint8,
                device=self.topk_indices_buffer.device,
            )
            if isinstance(metadata, dict) and owner.kv_cache.numel():
                copy_cache_rows(
                    owner.kv_cache,
                    typing.cast(typing.Any, metadata[owner.prefix]).slot_mapping,
                    payload,
                    restore=False,
                )
            deltas[key] = payload
        return deltas

    def _pp_intermediate(
        self,
        hidden: torch.Tensor,
        pre_mix: torch.Tensor,
        token_ids: torch.Tensor,
        deltas: dict[str, torch.Tensor] | None = None,
    ) -> IntermediateTensors:
        tensors = {"hidden_states": hidden, "pre_mix": pre_mix}
        if self.pp4 and self.end_layer in (10, 30, 31, 32):
            rows = hidden.shape[0]
            tensors.update(self._pp_cache_deltas(rows) if deltas is None else deltas)
            if self.end_layer in (10, 30, 31):
                tensors["pp_topk"] = self.topk_indices_buffer[:rows].clone()
            if self.end_layer in (30, 31, 32):
                assert self.candidate_block_buffer is not None
                tensors["pp_candidates"] = self.candidate_block_buffer[:rows].clone()
                tensors["pp_token_ids"] = token_ids.to(torch.int64)
        return IntermediateTensors(tensors)

    def _restore_pp_state(self, tensors: IntermediateTensors) -> None:
        if not self.pp4 or self.start_layer not in (10, 30, 31, 32):
            return
        source = 8 if self.start_layer == 10 else 20
        metadata = get_forward_context().attn_metadata
        for key, index in (
            ("pp_kv", False),
            *(([("pp_index_k", True)]) if source == 20 else []),
        ):
            owner = self._pp_shared_cache(source, index=index)
            if isinstance(metadata, dict) and owner.kv_cache.numel():
                copy_cache_rows(
                    owner.kv_cache,
                    typing.cast(typing.Any, metadata[owner.prefix]).slot_mapping,
                    tensors[key],
                    restore=True,
                )
        if self.start_layer in (10, 30, 31):
            topk = tensors["pp_topk"]
            self.topk_indices_buffer[: topk.shape[0]].copy_(topk)
        if source == 20:
            candidates = tensors["pp_candidates"]
            assert self.candidate_block_buffer is not None
            self.candidate_block_buffer[: candidates.shape[0]].copy_(candidates)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            self._restore_pp_state(intermediate_tensors)
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Skipped on
        # profile runs (KV cache unbound).
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    raise NotImplementedError(
                        "engram needs `lookback_token_ids` from the model runner"
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                )
            else:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them; the memory profile also has to
                # see the Engram activations.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                if self.engram_batch is not None:
                    self.engram_batch.prepare_embeddings(engram_hashes)
                else:
                    gathered_hashes = gather_engram_hashes(
                        engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                    )
                    for layer in islice(self.layers, self.start_layer, self.end_layer):
                        engram = getattr(layer, "engram", None)
                        if engram is not None:
                            engram.prepare_embeddings(
                                gathered_hashes[:, engram.layer_hash_index]
                            )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call
        ced_step = None
        if self.ced_enabled and is_forward_context_available():
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                ced_step = typing.cast(CEDStep | None, metadata.get(CED_METADATA_KEY))
        if ced_step is not None and self.start_layer in (30, 31, 32):
            assert intermediate_tensors is not None
            return self._forward_ced_decoder(
                ced_step,
                hidden_states,
                pre_mix,
                input_ids,
                positions,
                replay_token_ids=intermediate_tensors["pp_token_ids"],
            )
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx == 20 and ced_step is not None:
                boundary = (
                    hidden_states
                    if self.start_layer == 20
                    else mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
                )
                assert pre_mix is not None
                return self._forward_ced_decoder(
                    ced_step, boundary, pre_mix, input_ids, positions
                )
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_state = aux_recon.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            else:
                hidden_states = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return self._pp_intermediate(hidden_states, pre_mix, input_ids)

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = hc_collapse_triton(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _forward_ced_decoder(
        self,
        step: CEDStep,
        boundary: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        replay_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        """Full-context KV-only preparation followed by request-owned SWA replay."""
        source = None
        deltas = None
        if self.start_layer <= 20:
            assert self.ced_tail is not None
            source = self.layers[20]
            _, _, normalized, _ = mhc_pre_delayed_tilelang(
                boundary,
                source.hc_attn_fn,
                source.hc_attn_scale,
                source.hc_attn_base,
                source.rms_norm_eps,
                source.hc_eps,
                source.hc_eps,
                source.hc_post_alpha,
                source.hc_sinkhorn_iters,
                pre_mix=pre_mix,
                norm_weight=source.attn_norm.weight,
                norm_eps=source.attn_norm.variance_epsilon,
            )
            source.attn.build_ced_global_kv(normalized, positions)
            deltas = self._pp_cache_deltas(positions.shape[0])
            rows = step.plan.num_input_tokens
            hidden, carried, token_ids = self.ced_tail.pack(
                step.plan, boundary[:rows], pre_mix[:rows], input_ids[:rows]
            )
        else:
            assert self.start_layer in (30, 31, 32) and replay_token_ids is not None
            hidden, carried, token_ids = boundary, pre_mix, replay_token_ids
            assert hidden.shape[0] == token_ids.shape[0] == step.plan.num_decoder_tokens
        result = boundary.new_zeros((positions.shape[0], self.config.hidden_size))
        aux_layers = tuple(self.aux_hidden_state_layers)
        aux_hidden_states: list[torch.Tensor] = []
        if aux_layers:
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
                    device=result.device,
                ),
                active_requests=step.plan.decoder_requests,
                aux_hidden_states=aux_hidden_states,
            )
        if step.plan.num_decoder_tokens == 0:
            if not get_pp_group().is_last_rank:
                return self._pp_intermediate(hidden, carried, token_ids, deltas)
            return (result, []) if aux_layers else result

        ctx = get_forward_context()
        old_metadata, old_slots = ctx.attn_metadata, ctx.slot_mapping
        old_padding, old_descriptor = ctx.is_padding, ctx.batch_descriptor
        old_prebuilt = (
            source.attn.ced_global_kv_prebuilt if source is not None else False
        )
        try:
            ctx.attn_metadata = step.decoder_metadata
            if step.decoder_slot_mapping is not None:
                ctx.slot_mapping = step.decoder_slot_mapping
            ctx.is_padding = None
            ctx.batch_descriptor = None
            if source is not None:
                source.attn.ced_global_kv_prebuilt = True
            residual = post = res = None
            start = max(20, self.start_layer)
            for idx, layer in enumerate(
                islice(self.layers, start, self.end_layer), start=start
            ):
                hidden, residual, post, res, carried = layer(
                    hidden, step.positions, token_ids, carried, post, res, residual
                )
                if idx + 1 in aux_layers:
                    aux_hidden_states.append(
                        mhc_post_tilelang(hidden, residual, post, res).mean(dim=1)
                    )
            hidden = mhc_post_tilelang(hidden, residual, post, res)
            if not get_pp_group().is_last_rank:
                return self._pp_intermediate(hidden, carried, token_ids, deltas)
            decoded = self.norm(hc_collapse_triton(hidden, carried))
            target = torch.tensor(step.plan.output_rows, device=result.device)
            source_rows = torch.tensor(
                step.plan.decoder_output_rows, device=result.device
            )
            result.index_copy_(0, target, decoded.index_select(0, source_rows))
        finally:
            if source is not None:
                source.attn.ced_global_kv_prebuilt = old_prebuilt
            ctx.attn_metadata, ctx.slot_mapping = old_metadata, old_slots
            ctx.is_padding, ctx.batch_descriptor = old_padding, old_descriptor
        return (result, []) if aux_layers else result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.use_sequence_parallel
        )

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_native_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=first_layer.ffn.n_redundant_experts,
        )

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers `weight_scale`;
    block-FP8 linear layers register `weight_scale_inv`.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".ffn.gate.bias_vl": ".ffn.gate.bias_vl",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV4MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV4MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model
    requires_raw_input_tokens = True

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mhc_broadcast_weights()
        config = self.model.vllm_config
        if config.engram_config and config.engram_config.use_thp:
            # Loading weights refills the file cache; release it before MADV_COLLAPSE.
            model_config = config.model_config
            drop_checkpoint_cache(
                model_config.model_weights or model_config.model,
                revision=model_config.revision,
                cache_dir=config.load_config.download_dir,
            )
            for layer in islice(
                self.model.layers, self.model.start_layer, self.model.end_layer
            ):
                if layer.engram is not None:
                    layer.engram.embed_tokens.collapse_huge_pages()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
