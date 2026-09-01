# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.decorators import (
    should_torch_compile_mm_encoder,
    support_torch_compile,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tp_rank,
)
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.models.vision import is_vit_use_data_parallel

IMAGE_START = 0
IMAGE_PAD = 1
IMAGE = 2
IMAGE_NEWLINE = 3
IMAGE_END = 4


@lru_cache(maxsize=64)
def _cached_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _get_vision_cos_sin(
    n_h: int,
    n_w: int,
    dim: int,
    theta: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _cached_vision_cos_sin(n_h, n_w, dim, theta)
    return cos.to(device=device, non_blocking=True), sin.to(
        device=device, non_blocking=True
    )


def _apply_vision_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return x.to(dtype=dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        patch_size = config.vision_patch_size
        self.proj = ReplicatedLinear(
            3 * patch_size**2,
            config.vision_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim > 2:
            patches = patches.flatten(1)
        hidden_states, _ = self.proj(patches)
        return hidden_states


class DeepseekV4VisionAttention(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = config.vision_n_heads
        self.hidden_size = config.vision_dim
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                "vision_dim must be divisible by vision_n_heads "
                f"(got {self.hidden_size=} and {self.num_heads=})."
            )
        tp_size = get_tp_size()
        use_data_parallel = is_vit_use_data_parallel()
        disable_tp = use_data_parallel or self.num_heads % tp_size != 0
        self.tp_size = 1 if disable_tp else tp_size
        self.tp_rank = 0 if disable_tp else get_tp_rank()
        self.num_heads_per_partition = self.num_heads // self.tp_size
        self.rope_dim = self.head_dim // 2
        self.rope_theta = config.vision_rope_theta
        self.wqkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=disable_tp,
        )
        self.wo = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=disable_tp,
        )
        self.attn = MMEncoderAttention(
            self.num_heads_per_partition,
            self.head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        n_tokens = x.shape[0]
        qkv, _ = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(n_tokens, self.num_heads_per_partition, self.head_dim)
        k = k.view(n_tokens, self.num_heads_per_partition, self.head_dim)
        v = v.view(n_tokens, self.num_heads_per_partition * self.head_dim)

        cos, sin = _get_vision_cos_sin(
            n_vit_h,
            n_vit_w,
            self.rope_dim,
            self.rope_theta,
            device=x.device,
        )
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)

        q = q.reshape(1, n_tokens, -1)
        k = k.reshape(1, n_tokens, -1)
        v = v.reshape(1, n_tokens, -1)
        out = self.attn(q, k, v).reshape(n_tokens, -1)
        out, _ = self.wo(out)
        return out


class DeepseekV4VisionMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.w1 = MergedColumnParallelLinear(
            config.vision_dim,
            [config.vision_inter_dim, config.vision_inter_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
        )
        self.w2 = RowParallelLinear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.w1(x)
        x = self.act(x)
        x, _ = self.w2(x)
        return x


@support_torch_compile(
    dynamic_arg_dims={"x": 0},
    enable_if=should_torch_compile_mm_encoder,
    is_encoder=True,
)
class DeepseekV4VisionBlock(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.attn = DeepseekV4VisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)
        self.mlp = DeepseekV4VisionMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), n_vit_h, n_vit_w)
        x = x + self.mlp(self.norm2(x))
        return x


class DeepseekV4VisionTransformer(nn.Module):
    packed_modules_mapping = {
        "wqkv": ["wqkv"],
    }

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = DeepseekV4PatchEmbed(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.patch_embed",
        )
        self.blocks = nn.ModuleList(
            DeepseekV4VisionBlock(
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{idx}",
            )
            for idx in range(config.vision_n_layers)
        )
        self.norm = RMSNorm(config.vision_dim, eps=1e-6, dtype=torch.float32)

    def forward(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, n_vit_h, n_vit_w)
        return self.norm(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class DeepseekV4Aligner(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = ReplicatedLinear(
            in_dim,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
        )
        self.w2 = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        x, _ = self.w1(x)
        x = F.gelu(x)
        x, _ = self.w2(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def build_image_sentinel_embeddings(
    *,
    image_features: torch.Tensor,
    token_types: torch.Tensor,
    image_indices: torch.Tensor,
    image_start: torch.Tensor,
    image_end: torch.Tensor,
    image_newline: torch.Tensor,
    image_pad: torch.Tensor,
) -> torch.Tensor:
    params = torch.stack([image_start, image_pad, image_pad, image_newline, image_end])
    token_types = token_types.to(device=params.device, dtype=torch.long)
    embeddings = params[token_types]
    image_mask = token_types == IMAGE
    if image_mask.any():
        image_indices = image_indices.to(
            device=image_features.device,
            dtype=torch.long,
        )
        num_image_tokens = int(image_mask.sum().item())
        if image_indices.numel() != num_image_tokens:
            raise ValueError(
                "image_indices must contain one entry per IMAGE token "
                f"({image_indices.numel()} != {num_image_tokens})"
            )
        if image_indices.min() < 0 or image_indices.max() >= image_features.shape[0]:
            raise ValueError("image_indices contains an out-of-range feature index")
        embeddings[image_mask] = image_features[image_indices].to(embeddings)
    return embeddings
