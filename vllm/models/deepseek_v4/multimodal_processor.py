# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 Vision multimodal input processing."""

import math
from collections.abc import Mapping, MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from PIL import Image, ImageOps

from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.inputs import MultiModalDataDict, MultiModalHashes
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalKwargsOptionalItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    VideoEmbeddingItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
    iter_token_matches,
)

from .vision import IMAGE, IMAGE_END, IMAGE_NEWLINE, IMAGE_PAD, IMAGE_START

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import BatchFeature


_IMAGE_TOKEN = "<｜deepseek_image｜>"
_VIDEO_TOKEN = "<|place_holder_mm_span_0436|>"
_MAX_VIDEO_FRAMES = 8
_COMPRESS_PAD_TO = 4
_IMAGE_BLOCK_END_OFFSET_MOD = 1


def _get_vision_attr(config: object, name: str, default: Any) -> Any:
    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and hasattr(vision_config, name):
        return getattr(vision_config, name)
    return getattr(config, name, default)


def _get_patch_size(config: object) -> int:
    return int(_get_vision_attr(config, "vision_patch_size", 14))


def _get_downsample_ratio(config: object) -> int:
    return int(_get_vision_attr(config, "vision_downsample_ratio", 3))


def _get_max_image_tokens(config: object) -> int:
    return int(_get_vision_attr(config, "vision_max_n_token", 384))


def _get_min_pixels(config: object) -> int:
    return int(_get_vision_attr(config, "vision_min_pixels", 147456))


def _get_max_wh_ratio(config: object) -> float | None:
    value = _get_vision_attr(config, "vision_max_wh_ratio", 8)
    return None if value is None else float(value)


def _get_vocab_size(config: object) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("DeepSeek-V4 Vision processing requires config.vocab_size")
    return int(vocab_size)


def _image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if array.ndim != 3:
        raise ValueError(f"Expected 3-D image data, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if (
        np.issubdtype(array.dtype, np.floating)
        and np.nanmin(array) >= 0.0
        and np.nanmax(array) <= 1.0
    ):
        array = array * 255.0
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def _iter_video_frames(video: Any) -> list[Any]:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    if isinstance(video, np.ndarray):
        return [video[idx] for idx in range(video.shape[0])]
    if isinstance(video, Sequence) and not isinstance(video, str | bytes):
        return list(video)
    raise TypeError(f"Unsupported video type: {type(video)!r}")


def _sample_video_frames(video: Any) -> list[Image.Image]:
    frames = _iter_video_frames(video)
    if not frames:
        raise ValueError("DeepSeek-V4 Vision video inputs must contain frames")
    if len(frames) > _MAX_VIDEO_FRAMES:
        frame_indices = np.linspace(
            0,
            len(frames) - 1,
            _MAX_VIDEO_FRAMES,
            dtype=int,
        )
        frames = [frames[int(idx)] for idx in frame_indices]
    return [_image_to_pil(frame) for frame in frames]


def _grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += ((n_llm_h + 1) // 2 * (n_llm_w + 1)) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def _solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("DeepSeek-V4 Vision image token budget is too small")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def _safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    max_n_token -= _COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = _solve_resize_ratio(
            height,
            width,
            patch_size,
            downsample_ratio,
            budget,
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def _select_resized_geometry(
    image: Image.Image,
    config: object,
) -> tuple[int, int, int, int, int, int]:
    patch_size = _get_patch_size(config)
    downsample_ratio = _get_downsample_ratio(config)
    max_n_token = _get_max_image_tokens(config)
    min_pixels = _get_min_pixels(config)
    max_wh_ratio = _get_max_wh_ratio(config)

    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = int(height * max_wh_ratio)
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = _safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample_ratio,
        max_n_token,
    )
    return (
        best_height // patch_size,
        best_width // patch_size,
        n_llm_h,
        n_llm_w,
        best_height,
        best_width,
    )


def _extract_patches(
    image: Image.Image,
    config: object,
) -> tuple[torch.Tensor, int, int, int, int]:
    patch_size = _get_patch_size(config)
    n_vit_h, n_vit_w, n_llm_h, n_llm_w, best_height, best_width = (
        _select_resized_geometry(image, config)
    )
    max_wh_ratio = _get_max_wh_ratio(config)
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(
            image,
            (best_width, best_height),
            color=(127, 127, 127),
        )
    chw = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
    chw = ((chw / 255.0 - 0.5) / 0.5).to(torch.bfloat16).contiguous()
    patches = (
        chw.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
        .contiguous()
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def _build_image_token_types(
    n_llm_h: int,
    n_llm_w: int,
    output_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    compress_pad = _COMPRESS_PAD_TO - 1 - output_offset % _COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEWLINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.long,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.long)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.long),
            torch.tensor([IMAGE_START], dtype=torch.long),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.long),
            torch.tensor([IMAGE_END], dtype=torch.long),
        ]
    )
    return types, perm


def _build_image_block(
    *,
    vocab_size: int,
    n_llm_h: int,
    n_llm_w: int,
    output_offset: int,
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    token_type_tensor, image_indices = _build_image_token_types(
        n_llm_h,
        n_llm_w,
        output_offset,
    )
    return (
        [vocab_size + int(token_type) for token_type in token_type_tensor.tolist()],
        token_type_tensor,
        image_indices,
    )


class DeepseekV4VisionProcessingInfo(BaseProcessingInfo):
    """Processing info for DeepSeek-V4 Vision Exp."""

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_image_token(self) -> str:
        return _IMAGE_TOKEN

    def get_video_token(self) -> str:
        return _VIDEO_TOKEN

    def get_image_size_with_most_features(self) -> ImageSize:
        config = self.get_hf_config()
        patch_size = _get_patch_size(config)
        downsample_ratio = _get_downsample_ratio(config)
        max_tokens = _get_max_image_tokens(config)
        max_wh_ratio = _get_max_wh_ratio(config)
        layout_budget = max_tokens - (_COMPRESS_PAD_TO - 1)

        best: tuple[int, int, int] | None = None
        for n_llm_h in range(1, layout_budget + 1):
            for n_llm_w in range(1, layout_budget + 1):
                if max_wh_ratio is not None and n_llm_w > max_wh_ratio * n_llm_h:
                    continue
                _, _, num_tokens = _grid_tokens(
                    n_llm_h * patch_size * downsample_ratio,
                    n_llm_w * patch_size * downsample_ratio,
                    patch_size,
                    downsample_ratio,
                )
                if num_tokens > layout_budget:
                    continue
                candidate = n_llm_h * n_llm_w, n_llm_h, n_llm_w
                if best is None or candidate > best:
                    best = candidate

        if best is None:
            raise ValueError("DeepSeek-V4 Vision image token budget is too small")
        _, n_llm_h, n_llm_w = best
        grid_unit = patch_size * downsample_ratio
        return ImageSize(
            width=n_llm_w * grid_unit,
            height=n_llm_h * grid_unit,
        )

    def get_max_image_tokens(self) -> int:
        return _get_max_image_tokens(self.get_hf_config())

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_image_tokens() * _MAX_VIDEO_FRAMES,
        }


class DeepseekV4VisionDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.get_image_token() * mm_counts.get(
            "image", 0
        ) + self.info.get_video_token() * mm_counts.get("video", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        image_size = self.info.get_image_size_with_most_features()
        video_options = cast(VideoDummyOptions | None, mm_options.get("video"))
        return {
            "image": self._get_dummy_images(
                width=image_size.width,
                height=image_size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=image_size.width,
                height=image_size.height,
                num_frames=_MAX_VIDEO_FRAMES,
                num_videos=mm_counts.get("video", 0),
                overrides=video_options,
            ),
        }


class DeepseekV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionProcessingInfo]
):
    """Native processor for DeepSeek-V4 Vision image placeholders."""

    def _get_mm_output_offset_mods(
        self,
        prompt: list[int],
    ) -> dict[str, list[int]]:
        tokenizer = self.info.get_tokenizer()
        target_by_modality = {
            "image": cached_encode(
                tokenizer,
                self.info.get_image_token(),
                add_special_tokens=False,
            ),
            "video": cached_encode(
                tokenizer,
                self.info.get_video_token(),
                add_special_tokens=False,
            ),
        }
        matches: list[tuple[int, int, str]] = []
        for modality, target in target_by_modality.items():
            matches.extend(
                (match.start_idx, match.end_idx, modality)
                for match in iter_token_matches(prompt, target)
            )
        matches.sort(key=lambda match: match[0])

        offsets = {"image": list[int](), "video": list[int]()}
        output_offset_mod = 0
        previous_end = 0
        for start_idx, end_idx, modality in matches:
            output_offset_mod = (
                output_offset_mod + start_idx - previous_end
            ) % _COMPRESS_PAD_TO
            offsets[modality].append(output_offset_mod)
            output_offset_mod = _IMAGE_BLOCK_END_OFFSET_MOD
            previous_end = end_idx
        return offsets

    def _get_mm_hashes(self, inputs: ProcessorInputs) -> MultiModalHashes:
        mm_hashes = super()._get_mm_hashes(inputs)
        if not mm_hashes.get("image") and not mm_hashes.get("video"):
            return mm_hashes

        output_offset_mods = self._get_mm_output_offset_mods(inputs.prompt)
        hash_algorithm = self.info.ctx.get_mm_config().mm_hasher_algorithm
        out_hashes = dict(mm_hashes)
        for modality in ("image", "video"):
            hashes = mm_hashes.get(modality, [])
            if not hashes:
                continue
            offsets = output_offset_mods[modality]
            if len(offsets) != len(hashes):
                raise ValueError(
                    "DeepSeek-V4 Vision requires one "
                    f"{modality} placeholder per {modality} "
                    f"({len(offsets)} != {len(hashes)})"
                )
            out_hashes[modality] = [
                MultiModalHasher.hash_kwargs(
                    hash_algorithm,
                    model_id=self.info.model_id,
                    mm_hash=mm_hash,
                    modality=modality,
                    output_offset_mod4=output_offset_mod,
                )
                for mm_hash, output_offset_mod in zip(
                    hashes,
                    offsets,
                    strict=True,
                )
            ]
        return out_hashes

    def _get_hf_processor_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields: dict[str, MultiModalFieldConfig] = {}
        if "pixel_values" in hf_inputs:
            fields["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "image",
                hf_inputs["num_image_patches"],
            )
        if "image_embeds" in hf_inputs:
            fields["image_embeds"] = MultiModalFieldConfig.batched("image")
        if "video_pixel_values" in hf_inputs:
            fields["video_pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                "video",
                hf_inputs["num_video_patches"],
            )
        for key in (
            "n_vit_h",
            "n_vit_w",
            "n_llm_h",
            "n_llm_w",
            "num_image_patches",
        ):
            if key in hf_inputs:
                fields[key] = MultiModalFieldConfig.batched(
                    "image",
                    keep_on_cpu=True,
                )
        for key in (
            "video_n_vit_h",
            "video_n_vit_w",
            "video_n_llm_h",
            "video_n_llm_w",
            "num_video_frame_patches",
        ):
            if key in hf_inputs:
                fields[key] = MultiModalFieldConfig.flat_from_sizes(
                    "video",
                    hf_inputs["video_num_frames"],
                    keep_on_cpu=True,
                )
        for key in (
            "num_video_patches",
            "video_num_frames",
        ):
            if key in hf_inputs:
                fields[key] = MultiModalFieldConfig.batched(
                    "video",
                    keep_on_cpu=True,
                )
        return fields

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        del hf_processor_mm_kwargs

        from transformers.feature_extraction_utils import BatchFeature

        if "image" not in mm_items and "video" not in mm_items:
            return BatchFeature({})

        config = self.info.get_hf_config()
        patches_by_image = list[torch.Tensor]()
        n_vit_h = list[int]()
        n_vit_w = list[int]()
        n_llm_h = list[int]()
        n_llm_w = list[int]()
        num_image_patches = list[int]()
        if "image" in mm_items:
            image_items = mm_items["image"]
            if isinstance(image_items, ImageEmbeddingItems):
                raise NotImplementedError(
                    "DeepSeek-V4 Vision requires raw images because its prompt "
                    "layout depends on the resized image geometry; precomputed "
                    "image embeddings are not supported"
                )
            if not isinstance(image_items, ImageProcessorItems):
                raise TypeError(f"Unsupported image items: {type(image_items)!r}")
            for item_idx in range(image_items.get_count()):
                image = _image_to_pil(image_items.get(item_idx))
                patches, item_n_vit_h, item_n_vit_w, item_n_llm_h, item_n_llm_w = (
                    _extract_patches(
                        image,
                        config,
                    )
                )
                patches_by_image.append(patches)
                n_vit_h.append(item_n_vit_h)
                n_vit_w.append(item_n_vit_w)
                n_llm_h.append(item_n_llm_h)
                n_llm_w.append(item_n_llm_w)
                num_image_patches.append(patches.shape[0])

        video_patches = list[torch.Tensor]()
        video_n_vit_h = list[int]()
        video_n_vit_w = list[int]()
        video_n_llm_h = list[int]()
        video_n_llm_w = list[int]()
        num_video_frame_patches = list[int]()
        num_video_patches = list[int]()
        video_num_frames = list[int]()
        if "video" in mm_items:
            video_items = mm_items["video"]
            if isinstance(video_items, VideoEmbeddingItems):
                raise NotImplementedError(
                    "DeepSeek-V4 Vision requires raw videos because its prompt "
                    "layout depends on the resized frame geometry; precomputed "
                    "video embeddings are not supported"
                )
            if not isinstance(video_items, VideoProcessorItems):
                raise TypeError(f"Unsupported video items: {type(video_items)!r}")
            for item_idx in range(video_items.get_count()):
                frames = _sample_video_frames(video_items.get(item_idx))
                frame_patches = list[torch.Tensor]()
                frame_n_vit_h_values = list[int]()
                frame_n_vit_w_values = list[int]()
                frame_n_llm_h_values = list[int]()
                frame_n_llm_w_values = list[int]()
                frame_patch_counts = list[int]()
                for frame in frames:
                    (
                        patches,
                        frame_n_vit_h,
                        frame_n_vit_w,
                        frame_n_llm_h,
                        frame_n_llm_w,
                    ) = _extract_patches(frame, config)
                    frame_patches.append(patches)
                    frame_n_vit_h_values.append(frame_n_vit_h)
                    frame_n_vit_w_values.append(frame_n_vit_w)
                    frame_n_llm_h_values.append(frame_n_llm_h)
                    frame_n_llm_w_values.append(frame_n_llm_w)
                    frame_patch_counts.append(patches.shape[0])
                video_patches.extend(frame_patches)
                video_n_vit_h.extend(frame_n_vit_h_values)
                video_n_vit_w.extend(frame_n_vit_w_values)
                video_n_llm_h.extend(frame_n_llm_h_values)
                video_n_llm_w.extend(frame_n_llm_w_values)
                num_video_frame_patches.extend(frame_patch_counts)
                num_video_patches.append(sum(frame_patch_counts))
                video_num_frames.append(len(frames))

        output: dict[str, torch.Tensor] = {}
        if patches_by_image:
            output.update(
                {
                    "pixel_values": torch.cat(patches_by_image, dim=0),
                    "n_vit_h": torch.tensor(n_vit_h, dtype=torch.long),
                    "n_vit_w": torch.tensor(n_vit_w, dtype=torch.long),
                    "n_llm_h": torch.tensor(n_llm_h, dtype=torch.long),
                    "n_llm_w": torch.tensor(n_llm_w, dtype=torch.long),
                    "num_image_patches": torch.tensor(
                        num_image_patches,
                        dtype=torch.long,
                    ),
                }
            )
        if video_patches:
            output.update(
                {
                    "video_pixel_values": torch.cat(video_patches, dim=0),
                    "video_n_vit_h": torch.tensor(video_n_vit_h, dtype=torch.long),
                    "video_n_vit_w": torch.tensor(video_n_vit_w, dtype=torch.long),
                    "video_n_llm_h": torch.tensor(video_n_llm_h, dtype=torch.long),
                    "video_n_llm_w": torch.tensor(video_n_llm_w, dtype=torch.long),
                    "num_video_frame_patches": torch.tensor(
                        num_video_frame_patches,
                        dtype=torch.long,
                    ),
                    "num_video_patches": torch.tensor(
                        num_video_patches,
                        dtype=torch.long,
                    ),
                    "video_num_frames": torch.tensor(
                        video_num_frames,
                        dtype=torch.long,
                    ),
                }
            )

        return BatchFeature(output)

    @staticmethod
    def _append_image_block(
        *,
        vocab_size: int,
        n_llm_h: int,
        n_llm_w: int,
        output_offset: int,
        token_ids: list[int],
        token_types: list[int],
        image_indices: list[int],
    ) -> None:
        frame_token_ids, frame_token_types, frame_image_indices = _build_image_block(
            vocab_size=vocab_size,
            n_llm_h=n_llm_h,
            n_llm_w=n_llm_w,
            output_offset=output_offset + len(token_ids),
        )
        token_ids.extend(frame_token_ids)
        token_types.extend(int(token_type) for token_type in frame_token_types)
        image_index_offset = len(image_indices)
        image_indices.extend(
            image_index_offset + int(image_idx) for image_idx in frame_image_indices
        )

    def _image_geometry(
        self,
        out_mm_kwargs: MultiModalKwargsItems,
        item_idx: int,
    ) -> tuple[int, int]:
        item = out_mm_kwargs["image"][item_idx]
        if item is None:
            raise RuntimeError(f"Missing DeepSeek-V4 image kwargs for item {item_idx}")
        data = item.get_data()
        return int(cast(Any, data["n_llm_h"])), int(cast(Any, data["n_llm_w"]))

    def _video_geometry(
        self,
        out_mm_kwargs: MultiModalKwargsItems,
        item_idx: int,
    ) -> list[tuple[int, int]]:
        item = out_mm_kwargs["video"][item_idx]
        if item is None:
            raise RuntimeError(f"Missing DeepSeek-V4 video kwargs for item {item_idx}")
        data = item.get_data()
        n_llm_h = cast(torch.Tensor, data["video_n_llm_h"]).flatten().tolist()
        n_llm_w = cast(torch.Tensor, data["video_n_llm_w"]).flatten().tolist()
        return [(int(h), int(w)) for h, w in zip(n_llm_h, n_llm_w, strict=True)]

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del hf_processor_mm_kwargs
        if (
            mm_items.get_count("image", strict=False) == 0
            and mm_items.get_count("video", strict=False) == 0
        ):
            return []

        vocab_size = _get_vocab_size(self.info.get_hf_config())

        def image_replacement(
            item_idx: int,
            output_offset: int,
        ) -> PromptUpdateDetails:
            n_llm_h, n_llm_w = self._image_geometry(out_mm_kwargs, item_idx)
            token_ids, _, _ = _build_image_block(
                vocab_size=vocab_size,
                n_llm_h=n_llm_h,
                n_llm_w=n_llm_w,
                output_offset=output_offset,
            )
            return PromptUpdateDetails.from_seq(token_ids)

        def video_replacement(
            item_idx: int,
            output_offset: int,
        ) -> PromptUpdateDetails:
            token_ids = list[int]()
            for n_llm_h, n_llm_w in self._video_geometry(out_mm_kwargs, item_idx):
                frame_token_ids, _, _ = _build_image_block(
                    vocab_size=vocab_size,
                    n_llm_h=n_llm_h,
                    n_llm_w=n_llm_w,
                    output_offset=output_offset + len(token_ids),
                )
                token_ids.extend(frame_token_ids)
            return PromptUpdateDetails.from_seq(token_ids)

        updates: list[PromptUpdate] = [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    self.info.get_tokenizer(),
                    self.info.get_image_token(),
                    add_special_tokens=False,
                ),
                replacement=image_replacement,
            )
        ]
        updates.append(
            PromptReplacement(
                modality="video",
                target=cached_encode(
                    self.info.get_tokenizer(),
                    self.info.get_video_token(),
                    add_special_tokens=False,
                ),
                replacement=video_replacement,
            )
        )
        return updates

    def _inject_token_metadata(
        self,
        mm_kwargs: MultiModalKwargsOptionalItems,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
    ) -> None:
        vocab_size = _get_vocab_size(self.info.get_hf_config())
        field = MultiModalFlatField(slices=[slice(None)], keep_on_cpu=True)

        image_items = cast(
            MutableSequence[MultiModalKwargsItem | None],
            mm_kwargs.get("image", []),
        )
        for placeholder in placeholders.get("image", []):
            item = image_items[placeholder.item_idx]
            if item is None:
                continue
            item = MultiModalKwargsItem(dict(item))
            image_items[placeholder.item_idx] = item

            data = item.get_data()
            _, token_types, image_indices = _build_image_block(
                vocab_size=vocab_size,
                n_llm_h=int(cast(Any, data["n_llm_h"])),
                n_llm_w=int(cast(Any, data["n_llm_w"])),
                output_offset=placeholder.start_idx,
            )
            if token_types.numel() != len(placeholder.tokens):
                raise RuntimeError(
                    "DeepSeek-V4 image placeholder metadata is inconsistent "
                    "with the prompt replacement span."
                )

            item["image_token_types"] = MultiModalFieldElem(
                data=token_types,
                field=field,
            )
            item["image_indices"] = MultiModalFieldElem(
                data=image_indices,
                field=field,
            )

        video_items = cast(
            MutableSequence[MultiModalKwargsItem | None],
            mm_kwargs.get("video", []),
        )
        for placeholder in placeholders.get("video", []):
            item = video_items[placeholder.item_idx]
            if item is None:
                continue
            item = MultiModalKwargsItem(dict(item))
            video_items[placeholder.item_idx] = item

            data = item.get_data()
            n_llm_h = cast(torch.Tensor, data["video_n_llm_h"]).flatten().tolist()
            n_llm_w = cast(torch.Tensor, data["video_n_llm_w"]).flatten().tolist()
            video_token_ids: list[int] = []
            video_token_types: list[int] = []
            video_image_indices: list[int] = []
            for frame_n_llm_h, frame_n_llm_w in zip(
                n_llm_h,
                n_llm_w,
                strict=True,
            ):
                self._append_image_block(
                    vocab_size=vocab_size,
                    n_llm_h=int(frame_n_llm_h),
                    n_llm_w=int(frame_n_llm_w),
                    output_offset=placeholder.start_idx,
                    token_ids=video_token_ids,
                    token_types=video_token_types,
                    image_indices=video_image_indices,
                )
            if len(video_token_types) != len(placeholder.tokens):
                raise RuntimeError(
                    "DeepSeek-V4 video placeholder metadata is inconsistent "
                    "with the prompt replacement span."
                )

            item["video_token_types"] = MultiModalFieldElem(
                data=torch.tensor(video_token_types, dtype=torch.long),
                field=field,
            )
            item["video_image_indices"] = MultiModalFieldElem(
                data=torch.tensor(video_image_indices, dtype=torch.long),
                field=field,
            )

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        prompt_ids, placeholders = super()._maybe_apply_prompt_updates(
            mm_items,
            prompt_ids,
            mm_kwargs,
            mm_prompt_updates,
        )
        self._inject_token_metadata(mm_kwargs, placeholders)
        return prompt_ids, placeholders
