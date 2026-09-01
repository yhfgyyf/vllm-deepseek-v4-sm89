# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .multimodal_processor import (
    DeepseekV4VisionDummyInputsBuilder,
    DeepseekV4VisionMultiModalProcessor,
    DeepseekV4VisionProcessingInfo,
)
from .vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEWLINE,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4Aligner,
    DeepseekV4VisionTransformer,
    build_image_sentinel_embeddings,
)

if current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM
else:
    from .nvidia.model import DeepseekV4ForCausalLM

_IMAGE_SENTINEL_NAMES = {
    "image_start",
    "image_end",
    "image_newline",
    "image_pad",
}
_IMAGE_SENTINEL_TYPES = (IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEWLINE, IMAGE_END)
_IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
_VIDEO_PLACEHOLDER = "<|place_holder_mm_span_0436|>"


class DeepseekV4ImageInputs(NamedTuple):
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    token_types: torch.Tensor
    image_indices: torch.Tensor


class DeepseekV4VideoFrameInputs(NamedTuple):
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int


class DeepseekV4VideoInputs(NamedTuple):
    frames: tuple[DeepseekV4VideoFrameInputs, ...]
    token_types: torch.Tensor
    image_indices: torch.Tensor


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VisionMultiModalProcessor,
    info=DeepseekV4VisionProcessingInfo,
    dummy_inputs=DeepseekV4VisionDummyInputsBuilder,
)
class DeepseekV4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    packed_modules_mapping = {
        **DeepseekV4ForCausalLM.packed_modules_mapping,
        "wqkv": ["wqkv"],
    }
    hf_to_vllm_mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    lora_skip_prefixes = getattr(DeepseekV4ForCausalLM, "lora_skip_prefixes", [])
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision = DeepseekV4VisionTransformer(
                config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "vision"),
            )
            self.aligner = DeepseekV4Aligner(
                config,
                quant_config=None,
                prefix=maybe_prefix(prefix, "aligner"),
            )
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
            self.image_pad = nn.Parameter(torch.empty(config.hidden_size))

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV4ForCausalLM(
                vllm_config=vllm_config,
                prefix=prefix,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.configure_mm_token_handling(
            config.vocab_size,
            [config.vocab_size + token_type for token_type in _IMAGE_SENTINEL_TYPES],
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("image"):
            return _IMAGE_PLACEHOLDER
        if modality.startswith("video"):
            return _VIDEO_PLACEHOLDER
        raise ValueError("Only image and video modalities are supported")

    def _parse_image_inputs(
        self,
        **kwargs: object,
    ) -> tuple[DeepseekV4ImageInputs, ...]:
        image_embeds = kwargs.get("image_embeds")
        if image_embeds is not None:
            raise NotImplementedError(
                "DeepSeek-V4 Vision does not support precomputed image "
                "embeddings without the image layout metadata"
            )

        patches = kwargs.get("pixel_values", kwargs.get("patches"))
        if patches is None:
            return ()

        if not isinstance(patches, torch.Tensor):
            raise TypeError("DeepSeek V4 vision input 'pixel_values' must be a tensor")

        grid_thw = kwargs.get("image_grid_thw")
        if isinstance(grid_thw, torch.Tensor):
            grids = [
                (int(grid[-2].item()), int(grid[-1].item()))
                for grid in grid_thw.reshape(-1, grid_thw.shape[-1])
            ]
        else:
            n_vit_h = self._as_int_list(kwargs.get("n_vit_h"), "n_vit_h")
            n_vit_w = self._as_int_list(kwargs.get("n_vit_w"), "n_vit_w")
            if len(n_vit_h) != len(n_vit_w):
                raise ValueError("n_vit_h and n_vit_w must have the same length")
            grids = list(zip(n_vit_h, n_vit_w))
        if not grids:
            raise ValueError("DeepSeek V4 vision input contains no image grids")

        patch_counts_obj = kwargs.get("num_image_patches")
        if patch_counts_obj is not None:
            patch_counts = self._as_int_list(
                patch_counts_obj,
                "num_image_patches",
            )
        else:
            patch_counts = [n_vit_h * n_vit_w for n_vit_h, n_vit_w in grids]
        if len(patch_counts) != len(grids):
            raise ValueError("num_image_patches must contain one value per image grid")
        if sum(patch_counts) != patches.shape[0]:
            raise ValueError(
                "num_image_patches does not match the flattened pixel_values"
            )

        token_types = kwargs.get("image_token_types", kwargs.get("token_types"))
        replacement_ids = kwargs.get("image_replacement_ids")
        if token_types is None and isinstance(replacement_ids, torch.Tensor):
            token_types = replacement_ids.to(dtype=torch.long) - self.config.vocab_size
        if token_types is None:
            raise TypeError(
                "DeepSeek V4 vision input 'image_token_types' or "
                "'image_replacement_ids' is required"
            )
        token_types = torch.as_tensor(token_types, dtype=torch.long).flatten()

        end_offsets = token_types.eq(IMAGE_END).nonzero().flatten().tolist()
        if len(end_offsets) != len(grids):
            raise ValueError("image_token_types must contain one IMAGE_END per image")
        token_slices: list[torch.Tensor] = []
        token_start = 0
        for token_end in end_offsets:
            token_slices.append(token_types[token_start : token_end + 1])
            token_start = token_end + 1
        if token_start != token_types.numel():
            raise ValueError("Unexpected token types after the final IMAGE_END")

        image_indices_obj = kwargs.get("image_indices", kwargs.get("perm"))
        if image_indices_obj is None:
            raise TypeError("DeepSeek V4 vision input 'image_indices' is required")
        image_indices = torch.as_tensor(
            image_indices_obj,
            dtype=torch.long,
        ).flatten()
        image_inputs: list[DeepseekV4ImageInputs] = []
        patch_offset = 0
        image_index_offset = 0
        for (grid_n_vit_h, grid_n_vit_w), patch_count, token_slice in zip(
            grids,
            patch_counts,
            token_slices,
        ):
            patch_slice = patches[patch_offset : patch_offset + patch_count]
            patch_offset += patch_count
            image_count = int(token_slice.eq(IMAGE).sum().item())
            indices = image_indices[
                image_index_offset : image_index_offset + image_count
            ]
            image_index_offset += image_count
            image_inputs.append(
                DeepseekV4ImageInputs(
                    patch_slice,
                    grid_n_vit_h,
                    grid_n_vit_w,
                    token_slice,
                    indices,
                )
            )

        if image_index_offset != image_indices.numel():
            raise ValueError("image_indices must contain one entry per IMAGE token")

        return tuple(image_inputs)

    def _parse_video_inputs(
        self,
        **kwargs: object,
    ) -> tuple[DeepseekV4VideoInputs, ...]:
        video_embeds = kwargs.get("video_embeds")
        if video_embeds is not None:
            raise NotImplementedError(
                "DeepSeek-V4 Vision does not support precomputed video "
                "embeddings without the frame layout metadata"
            )

        patches = kwargs.get("video_pixel_values")
        if patches is None:
            return ()

        if not isinstance(patches, torch.Tensor):
            raise TypeError(
                "DeepSeek V4 vision input 'video_pixel_values' must be a tensor"
            )

        n_vit_h = self._as_int_list(kwargs.get("video_n_vit_h"), "video_n_vit_h")
        n_vit_w = self._as_int_list(kwargs.get("video_n_vit_w"), "video_n_vit_w")
        frame_patch_counts = self._as_int_list(
            kwargs.get("num_video_frame_patches"),
            "num_video_frame_patches",
        )
        video_num_frames = self._as_int_list(
            kwargs.get("video_num_frames"),
            "video_num_frames",
        )
        if not (len(n_vit_h) == len(n_vit_w) == len(frame_patch_counts)):
            raise ValueError("Video frame geometry must contain one value per frame")
        if sum(frame_patch_counts) != patches.shape[0]:
            raise ValueError(
                "num_video_frame_patches does not match video_pixel_values"
            )
        if sum(video_num_frames) != len(frame_patch_counts):
            raise ValueError(
                "video_num_frames does not match the number of processed frames"
            )

        token_types_obj = kwargs.get("video_token_types")
        if token_types_obj is None:
            raise TypeError("DeepSeek V4 vision input 'video_token_types' is required")
        token_types = torch.as_tensor(token_types_obj, dtype=torch.long).flatten()
        end_offsets = token_types.eq(IMAGE_END).nonzero().flatten().tolist()
        if len(end_offsets) != sum(video_num_frames):
            raise ValueError(
                "video_token_types must contain one IMAGE_END per sampled frame"
            )

        video_token_slices: list[torch.Tensor] = []
        token_start = 0
        end_offset_start = 0
        for num_frames in video_num_frames:
            frame_end_offsets = end_offsets[
                end_offset_start : end_offset_start + num_frames
            ]
            if not frame_end_offsets:
                raise ValueError("Each video must contain at least one frame")
            token_end = frame_end_offsets[-1]
            video_token_slices.append(token_types[token_start : token_end + 1])
            token_start = token_end + 1
            end_offset_start += num_frames
        if token_start != token_types.numel():
            raise ValueError("Unexpected token types after the final video frame")

        image_indices_obj = kwargs.get("video_image_indices")
        if image_indices_obj is None:
            raise TypeError(
                "DeepSeek V4 vision input 'video_image_indices' is required"
            )
        image_indices = torch.as_tensor(
            image_indices_obj,
            dtype=torch.long,
        ).flatten()

        video_inputs: list[DeepseekV4VideoInputs] = []
        patch_offset = 0
        frame_offset = 0
        image_index_offset = 0
        for num_frames, token_slice in zip(
            video_num_frames,
            video_token_slices,
            strict=True,
        ):
            frames: list[DeepseekV4VideoFrameInputs] = []
            for frame_idx in range(frame_offset, frame_offset + num_frames):
                patch_count = frame_patch_counts[frame_idx]
                frame_patches = patches[patch_offset : patch_offset + patch_count]
                patch_offset += patch_count
                frames.append(
                    DeepseekV4VideoFrameInputs(
                        frame_patches,
                        n_vit_h[frame_idx],
                        n_vit_w[frame_idx],
                    )
                )
            frame_offset += num_frames
            image_count = int(token_slice.eq(IMAGE).sum().item())
            indices = image_indices[
                image_index_offset : image_index_offset + image_count
            ]
            image_index_offset += image_count
            video_inputs.append(
                DeepseekV4VideoInputs(
                    tuple(frames),
                    token_slice,
                    indices,
                )
            )

        if image_index_offset != image_indices.numel():
            raise ValueError(
                "video_image_indices must contain one entry per IMAGE token"
            )

        return tuple(video_inputs)

    @staticmethod
    def _as_int_list(value: object, name: str) -> list[int]:
        if isinstance(value, int):
            return [value]
        if isinstance(value, torch.Tensor):
            return [int(item) for item in value.flatten().tolist()]
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [int(item) for item in value]
        raise TypeError(
            f"DeepSeek V4 vision inputs 'image_grid_thw' or '{name}' are required"
        )

    def encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        hidden_states = self.vision(patches, n_vit_h, n_vit_w)
        return self.aligner(hidden_states, n_vit_h, n_vit_w)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        images = self._parse_image_inputs(**kwargs)
        videos = self._parse_video_inputs(**kwargs)
        if not images and not videos:
            return []

        embeddings: list[torch.Tensor] = []
        for image in images:
            if image.n_vit_h == 0 and image.n_vit_w == 0:
                image_features = image.patches
            else:
                image_features = self.encode_image(
                    image.patches,
                    image.n_vit_h,
                    image.n_vit_w,
                )
            embeddings.append(
                build_image_sentinel_embeddings(
                    image_features=image_features,
                    token_types=image.token_types,
                    image_indices=image.image_indices,
                    image_start=self.image_start,
                    image_end=self.image_end,
                    image_newline=self.image_newline,
                    image_pad=self.image_pad,
                )
            )
        for video in videos:
            frame_features: list[torch.Tensor] = []
            for frame in video.frames:
                if frame.n_vit_h == 0 and frame.n_vit_w == 0:
                    frame_features.append(frame.patches)
                else:
                    frame_features.append(
                        self.encode_image(
                            frame.patches,
                            frame.n_vit_h,
                            frame.n_vit_w,
                        )
                    )
            embeddings.append(
                build_image_sentinel_embeddings(
                    image_features=torch.cat(frame_features, dim=0),
                    token_types=video.token_types,
                    image_indices=video.image_indices,
                    image_start=self.image_start,
                    image_end=self.image_end,
                    image_newline=self.image_newline,
                    image_pad=self.image_pad,
                )
            )
        return tuple(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        vision_weights: list[tuple[str, torch.Tensor]] = []
        aligner_weights: list[tuple[str, torch.Tensor]] = []
        language_weights: list[tuple[str, torch.Tensor]] = []
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.startswith("vision."):
                vision_weights.append((name.removeprefix("vision."), loaded_weight))
            elif name.startswith("aligner."):
                aligner_weights.append((name.removeprefix("aligner."), loaded_weight))
            elif name in _IMAGE_SENTINEL_NAMES:
                param = getattr(self, name)
                assert param.shape == loaded_weight.shape, (
                    f"{name}: expected {tuple(param.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
                default_weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                language_weights.append((name, loaded_weight))

        loaded_params.update(
            f"vision.{name}" for name in self.vision.load_weights(vision_weights)
        )
        loaded_params.update(
            f"aligner.{name}" for name in self.aligner.load_weights(aligner_weights)
        )
        loaded_params.update(
            f"language_model.{name}"
            for name in self.language_model.load_weights(language_weights)
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()


DeepseekV4VisionForCausalLM = DeepseekV4ForConditionalGeneration
