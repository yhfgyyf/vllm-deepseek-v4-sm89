# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

import vllm.models.deepseek_v4.multimodal_processor as processor_module
from vllm.models.deepseek_v4.multimodal_processor import (
    DeepseekV4VisionDummyInputsBuilder,
    DeepseekV4VisionMultiModalProcessor,
    DeepseekV4VisionProcessingInfo,
)
from vllm.models.deepseek_v4.vision_model import DeepseekV4ForConditionalGeneration
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing.processor import (
    PromptReplacement,
    PromptUpdateDetails,
    apply_token_matches,
)
from vllm.multimodal.utils import group_and_batch_mm_kwargs

VOCAB_SIZE = 1000
IMAGE_TOKEN = "<｜deepseek_image｜>"
VIDEO_TOKEN = "<|place_holder_mm_span_0436|>"
LOCAL_MODEL_ENV = "VLLM_TEST_DEEPSEEK_V4_VISION_MODEL"


def _get_local_model_path() -> Path:
    value = os.environ.get(LOCAL_MODEL_ENV)
    if value is None:
        pytest.skip(f"set {LOCAL_MODEL_ENV} to run reference-model tests")
    assert value is not None
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"reference model directory does not exist: {path}")
    return path


def _load_reference_processor():
    reference_processor = _get_local_model_path() / "inference" / "image_processor.py"
    if not reference_processor.is_file():
        pytest.skip(f"reference processor does not exist: {reference_processor}")
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_vision_reference_processor",
        reference_processor,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    vocab = {IMAGE_TOKEN: 500, VIDEO_TOKEN: 501}
    max_token_id = 255

    def encode(self, text, **kwargs):
        del kwargs
        ids = []
        i = 0
        while i < len(text):
            for token, token_id in self.vocab.items():
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def decode(self, token_ids, **kwargs):
        del kwargs
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse.get(token_id, chr(token_id)) for token_id in token_ids)


class _HFProcessor:
    image_token = IMAGE_TOKEN
    image_processor = SimpleNamespace(size={"width": 14, "height": 14})


class _MMConfig:
    enable_mm_embeds = False
    allow_missing_mm_embeddings = False
    mm_processor_cache_gb = 1
    mm_hasher_algorithm = "sha256"

    def merge_mm_processor_kwargs(self, kwargs):
        return dict(kwargs)

    def get_limit_per_prompt(self, modality):
        del modality
        return 4


class _Context:
    def __init__(self):
        self.processor = _HFProcessor()
        self.tokenizer = _Tokenizer()
        mm_config = _MMConfig()
        self.model_config = SimpleNamespace(
            model="deepseek-v4-vision-test",
            encoder_config={},
            max_model_len=4096,
            multimodal_config=mm_config,
            get_multimodal_config=lambda: mm_config,
            get_inputs_embeds_size=lambda: None,
        )

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_processor(self, **kwargs):
        del kwargs
        return self.processor

    def get_hf_config(self):
        return SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_max_n_token=384,
            vision_min_pixels=147456,
            vision_max_wh_ratio=8,
        )

    def get_mm_config(self):
        return self.model_config.multimodal_config

    def get_merged_mm_kwargs(self, kwargs):
        return self.get_mm_config().merge_mm_processor_kwargs(kwargs)

    def call_hf_processor(self, hf_processor, data, kwargs):
        raise AssertionError("DeepSeek-V4 Vision processor must not call HF offsets")


def _make_processor(ctx=None, cache=None):
    info = DeepseekV4VisionProcessingInfo(ctx or _Context())
    dummy = DeepseekV4VisionDummyInputsBuilder(info)
    return DeepseekV4VisionMultiModalProcessor(info, dummy, cache=cache)


def _reference_block(n_vit_h, n_vit_w, output_offset):
    ref = _load_reference_processor()
    token_types, perm = ref.build_image_block(n_vit_h, n_vit_w, output_offset)
    types = token_types.tolist()
    return [VOCAB_SIZE + token_type for token_type in types], types, perm.tolist()


@pytest.mark.parametrize("n_llm_h,n_llm_w", [(1, 1), (10, 10), (10, 19), (14, 21)])
@pytest.mark.parametrize("output_offset", range(4))
def test_deepseek_v4_image_block_ends_at_fixed_alignment(
    n_llm_h, n_llm_w, output_offset
):
    token_ids, _, _ = _reference_block(n_llm_h, n_llm_w, output_offset)

    assert (output_offset + len(token_ids)) % 4 == 1


def test_deepseek_v4_local_tokenizer_multimodal_token_contract():
    tokenizer = AutoTokenizer.from_pretrained(
        _get_local_model_path(), local_files_only=True
    )

    assert tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) == 129264
    assert tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False) == [129264]
    assert tokenizer.convert_tokens_to_ids(VIDEO_TOKEN) == 129265
    assert tokenizer.encode(VIDEO_TOKEN, add_special_tokens=False) == [129265]
    assert tokenizer.encode(VIDEO_TOKEN + "\n", add_special_tokens=False)[:1] == [
        129265
    ]


def test_deepseek_v4_image_token_does_not_require_hf_processor(monkeypatch):
    ctx = _Context()
    monkeypatch.setattr(
        ctx,
        "get_hf_processor",
        lambda **kwargs: pytest.fail(
            "native image processing must not require a HuggingFace processor"
        ),
    )

    assert DeepseekV4VisionProcessingInfo(ctx).get_image_token() == IMAGE_TOKEN


def test_deepseek_v4_dummy_image_profiles_near_max_features():
    processor = _make_processor()
    image_size = processor.info.get_image_size_with_most_features()
    image = Image.new("RGB", (image_size.width, image_size.height))
    hf_inputs = processor._apply_hf_processor_main(
        processor.info.parse_mm_data({"image": image}),
        {},
    )
    _, token_types, _ = _reference_block(
        int(hf_inputs["n_llm_h"][0]),
        int(hf_inputs["n_llm_w"][0]),
        output_offset=0,
    )

    assert len(token_types) >= processor.info.get_max_image_tokens() - 3


def test_deepseek_v4_video_limits_and_dummy_cover_eight_frames():
    processor = _make_processor()

    assert processor.info.get_supported_mm_limits() == {"image": None, "video": None}
    assert processor.info.get_mm_max_tokens_per_item(4096, {}) == {
        "image": processor.info.get_max_image_tokens(),
        "video": processor.info.get_max_image_tokens() * 8,
    }
    assert processor.dummy_inputs.get_dummy_text({"image": 1, "video": 1}) == (
        IMAGE_TOKEN + VIDEO_TOKEN
    )

    dummy = processor.dummy_inputs.get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"video": 2},
        mm_options={},
    )

    assert len(dummy["video"]) == 2
    assert dummy["video"][0].shape[0] == 8


def test_deepseek_v4_processor_rejects_precomputed_image_embeddings():
    processor = _make_processor()
    mm_items = MultiModalDataItems({"image": ImageEmbeddingItems(torch.zeros(1, 4, 8))})

    with pytest.raises(NotImplementedError, match="raw images"):
        processor._apply_hf_processor_main(mm_items, {})


def test_deepseek_v4_video_processor_samples_at_most_eight_frames(monkeypatch):
    processor = _make_processor()
    seen_colours: list[int] = []

    def fake_extract_patches(image, config):
        del config
        seen_colours.append(image.getpixel((0, 0))[0])
        return torch.full((1, 3, 1, 1), image.getpixel((0, 0))[0]), 1, 1, 1, 1

    monkeypatch.setattr(processor_module, "_extract_patches", fake_extract_patches)
    frames = np.stack(
        [np.full((2, 2, 3), frame_idx, dtype=np.uint8) for frame_idx in range(12)]
    )

    hf_inputs = processor._apply_hf_processor_main(
        processor.info.parse_mm_data({"video": frames}),
        {},
    )

    assert seen_colours == [0, 1, 3, 4, 6, 7, 9, 11]
    assert hf_inputs["video_num_frames"].tolist() == [8]
    assert hf_inputs["video_pixel_values"].shape == (8, 3, 1, 1)
    assert hf_inputs["num_video_frame_patches"].tolist() == [1] * 8
    assert hf_inputs["num_video_patches"].tolist() == [8]


def _image_record(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"data": buffer.getvalue()}


@pytest.mark.parametrize(
    "size",
    [
        (14, 14),
        (28, 14),
        (14, 28),
        (384, 384),
        (3000, 100),
    ],
)
def test_deepseek_v4_vision_processor_matches_official_resize_and_patches(size):
    processor = _make_processor()
    ref = _load_reference_processor()
    width, height = size
    image = Image.new("RGB", (width, height), color=(1, 128, 255))

    result = processor(
        prompt=IMAGE_TOKEN,
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    data = result["mm_kwargs"]["image"][0].get_data()
    ref_args = SimpleNamespace(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )
    ref_patches, ref_h, ref_w, ref_llm_h, ref_llm_w = ref.load_image(
        _image_record(image),
        ref_args,
    )

    assert data["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(data["pixel_values"], ref_patches)
    assert data["n_vit_h"].item() == ref_h
    assert data["n_vit_w"].item() == ref_w
    assert data["n_llm_h"].item() == ref_llm_h
    assert data["n_llm_w"].item() == ref_llm_w


def test_deepseek_v4_vision_processor_accepts_unit_float_images():
    processor = _make_processor()
    image_array = np.zeros((14, 14, 3), dtype=np.uint8)
    image_array[..., 0] = 1
    image_array[..., 1] = 128
    image_array[..., 2] = 255
    pil_image = Image.fromarray(image_array, mode="RGB")
    float_array = image_array.astype(np.float32) / 255.0
    float_tensor = torch.from_numpy(float_array).permute(2, 0, 1)

    def process(image):
        result = processor(
            prompt=IMAGE_TOKEN,
            mm_items=MultiModalDataItems({"image": ImageProcessorItems([image])}),
            hf_processor_mm_kwargs={},
        )
        return result["mm_kwargs"]["image"][0].get_data()

    expected = process(pil_image)

    for data in (process(float_array), process(float_tensor)):
        assert torch.equal(data["pixel_values"], expected["pixel_values"])
        assert data["n_vit_h"].item() == expected["n_vit_h"].item()
        assert data["n_vit_w"].item() == expected["n_vit_w"].item()
        assert data["n_llm_h"].item() == expected["n_llm_h"].item()
        assert data["n_llm_w"].item() == expected["n_llm_w"].item()


@pytest.mark.parametrize("prefix_len", [0, 1, 2, 3])
def test_offset_aware_replacement_uses_true_output_offset(prefix_len):
    def replacement(item_idx, output_offset):
        del item_idx
        token_ids, _, _ = _reference_block(1, 1, output_offset)
        return PromptUpdateDetails.from_seq(token_ids)

    update = PromptReplacement(
        modality="image",
        target=[500],
        replacement=replacement,
    )
    prompt = [7] * prefix_len + [500]

    new_prompt, result = apply_token_matches(prompt, {"image": [[update.resolve(0)]]})
    expected, _, _ = _reference_block(1, 1, prefix_len)

    assert result == {"image": [0]}
    assert new_prompt == [7] * prefix_len + expected


def test_deepseek_v4_vision_processor_handles_multi_image_n_layout():
    processor = _make_processor()
    first = Image.new("RGB", (384, 384), color=(255, 0, 0))
    second = Image.new("RGB", (768, 384), color=(0, 255, 0))

    result = processor(
        prompt=f"x{IMAGE_TOKEN}yy{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": [first, second]}),
        hf_processor_mm_kwargs={},
    )

    first_tokens, first_types, first_perm = _reference_block(10, 10, output_offset=1)
    second_offset = 1 + len(first_tokens) + 2
    second_tokens, second_types, second_perm = _reference_block(
        10,
        19,
        output_offset=second_offset,
    )
    assert result["prompt_token_ids"] == [
        ord("x"),
        *first_tokens,
        ord("y"),
        ord("y"),
        *second_tokens,
    ]

    placeholders = result["mm_placeholders"]["image"]
    assert [(p.offset, p.length, p.get_num_embeds()) for p in placeholders] == [
        (1, len(first_tokens), len(first_tokens)),
        (second_offset, len(second_tokens), len(second_tokens)),
    ]
    assert all(p.is_embed is None for p in placeholders)

    first_item, second_item = result["mm_kwargs"]["image"]
    first_data = first_item.get_data()
    second_data = second_item.get_data()
    assert first_data["pixel_values"].shape == (784, 3, 14, 14)
    assert second_data["pixel_values"].shape == (1540, 3, 14, 14)
    assert first_data["pixel_values"].dtype == torch.bfloat16
    assert first_data["n_vit_h"].item() == 28
    assert first_data["n_vit_w"].item() == 28
    assert first_data["n_llm_h"].item() == 10
    assert first_data["n_llm_w"].item() == 10
    assert second_data["n_vit_h"].item() == 28
    assert second_data["n_vit_w"].item() == 55
    assert second_data["n_llm_h"].item() == 10
    assert second_data["n_llm_w"].item() == 19
    assert first_data["image_token_types"].tolist() == first_types
    assert second_data["image_token_types"].tolist() == second_types
    assert first_data["image_indices"].tolist() == first_perm
    assert second_data["image_indices"].tolist() == second_perm

    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    image_inputs = model._parse_image_inputs(**result["mm_kwargs"].get_data())
    assert [(item.n_vit_h, item.n_vit_w) for item in image_inputs] == [
        (28, 28),
        (28, 55),
    ]
    assert image_inputs[0].token_types.tolist() == first_types
    assert image_inputs[1].token_types.tolist() == second_types
    assert image_inputs[0].image_indices.tolist() == first_perm
    assert image_inputs[1].image_indices.tolist() == second_perm


def test_deepseek_v4_vision_processor_handles_video_placeholder_metadata(
    monkeypatch,
):
    processor = _make_processor()

    def fake_extract_patches(image, config):
        del image, config
        return torch.zeros(1, 3, 1, 1), 1, 1, 1, 1

    monkeypatch.setattr(processor_module, "_extract_patches", fake_extract_patches)
    video = np.zeros((2, 2, 2, 3), dtype=np.uint8)

    result = processor(
        prompt=f"x{VIDEO_TOKEN}",
        mm_items=processor.info.parse_mm_data({"video": video}),
        hf_processor_mm_kwargs={},
    )

    first_tokens, first_types, first_perm = _reference_block(1, 1, output_offset=1)
    second_tokens, second_types, second_perm = _reference_block(
        1,
        1,
        output_offset=1 + len(first_tokens),
    )
    assert result["prompt_token_ids"] == [
        ord("x"),
        *first_tokens,
        *second_tokens,
    ]
    placeholder = result["mm_placeholders"]["video"][0]
    assert (placeholder.offset, placeholder.length, placeholder.get_num_embeds()) == (
        1,
        len(first_tokens) + len(second_tokens),
        len(first_tokens) + len(second_tokens),
    )

    data = result["mm_kwargs"]["video"][0].get_data()
    assert data["video_pixel_values"].shape == (2, 3, 1, 1)
    assert data["video_num_frames"].item() == 2
    assert data["video_n_vit_h"].tolist() == [1, 1]
    assert data["num_video_frame_patches"].tolist() == [1, 1]
    assert data["video_token_types"].tolist() == first_types + second_types
    assert data["video_image_indices"].tolist() == first_perm + [
        idx + len(first_perm) for idx in second_perm
    ]


def test_deepseek_v4_vision_batches_videos_with_different_frame_counts(
    monkeypatch,
):
    processor = _make_processor()

    def fake_extract_patches(image, config):
        del config
        value = image.getpixel((0, 0))[0]
        return torch.full((1, 3, 1, 1), value), 1, 1, 1, 1

    monkeypatch.setattr(processor_module, "_extract_patches", fake_extract_patches)
    first_video = np.full((1, 2, 2, 3), 10, dtype=np.uint8)
    second_video = np.stack(
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in (20, 30, 40)]
    )

    result = processor(
        prompt=f"{VIDEO_TOKEN}x{VIDEO_TOKEN}",
        mm_items=MultiModalDataItems(
            {"video": VideoProcessorItems([first_video, second_video])}
        ),
        hf_processor_mm_kwargs={},
    )

    video_items = result["mm_kwargs"]["video"]
    groups = list(
        group_and_batch_mm_kwargs(
            [("video", item) for item in video_items],
        )
    )
    assert len(groups) == 1
    modality, num_items, batched_kwargs = groups[0]
    assert modality == "video"
    assert num_items == 2
    assert batched_kwargs["video_num_frames"].tolist() == [1, 3]
    assert batched_kwargs["num_video_frame_patches"].tolist() == [1, 1, 1, 1]
    assert batched_kwargs["video_pixel_values"][:, 0, 0, 0].tolist() == [
        10,
        20,
        30,
        40,
    ]

    model = DeepseekV4ForConditionalGeneration.__new__(
        DeepseekV4ForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    video_inputs = model._parse_video_inputs(**batched_kwargs)
    assert [len(video.frames) for video in video_inputs] == [1, 3]
    assert [
        [int(frame.patches[0, 0, 0, 0]) for frame in video.frames]
        for video in video_inputs
    ] == [[10], [20, 30, 40]]


def test_deepseek_v4_vision_processor_cache_hit_matches_miss():
    ctx = _Context()
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = _make_processor(ctx, cache=cache)
    image = Image.new("RGB", (14, 14), color=(1, 2, 3))

    first = processor(
        prompt=f"a{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    second = processor(
        prompt=f"a{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )

    assert cache.make_stats().hits > 0
    assert first["prompt_token_ids"] == second["prompt_token_ids"]
    assert first["mm_placeholders"] == second["mm_placeholders"]
    assert first["mm_hashes"] == second["mm_hashes"]
    assert first["mm_kwargs"] == second["mm_kwargs"]


def test_deepseek_v4_vision_processor_cache_key_tracks_prefix_pad_layout():
    ctx = _Context()
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = _make_processor(ctx, cache=cache)
    image = Image.new("RGB", (14, 14), color=(1, 2, 3))

    first = processor(
        prompt=f"{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    second = processor(
        prompt=f"z{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    third = processor(
        prompt=f"q{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )

    first_expected, first_types, _ = _reference_block(10, 10, output_offset=0)
    second_expected, second_types, _ = _reference_block(10, 10, output_offset=1)
    assert cache.make_stats().hits > 0
    assert first["mm_hashes"]["image"] != second["mm_hashes"]["image"]
    assert second["mm_hashes"]["image"] == third["mm_hashes"]["image"]
    assert first["prompt_token_ids"] == first_expected
    assert second["prompt_token_ids"] == [ord("z"), *second_expected]
    assert third["prompt_token_ids"] == [ord("q"), *second_expected]
    assert first["mm_kwargs"]["image"][0].get_data()["image_token_types"].tolist() == (
        first_types
    )
    assert second["mm_kwargs"]["image"][0].get_data()["image_token_types"].tolist() == (
        second_types
    )


def test_deepseek_v4_image_video_hashes_use_prompt_order_offsets(monkeypatch):
    ctx = _Context()
    cache = MultiModalProcessorOnlyCache(ctx.model_config)
    processor = _make_processor(ctx, cache=cache)

    def fake_extract_patches(image, config):
        del image, config
        return torch.zeros(1, 3, 1, 1), 1, 1, 1, 1

    monkeypatch.setattr(processor_module, "_extract_patches", fake_extract_patches)
    image = Image.new("RGB", (2, 2), color=(1, 2, 3))
    video = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    first = processor(
        prompt=f"{VIDEO_TOKEN}yy{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image, "video": video}),
        hf_processor_mm_kwargs={},
    )
    second = processor(
        prompt=f"z{VIDEO_TOKEN}y{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image, "video": video}),
        hf_processor_mm_kwargs={},
    )
    third = processor(
        prompt=f"{IMAGE_TOKEN}yy{VIDEO_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image, "video": video}),
        hf_processor_mm_kwargs={},
    )
    same_image_offset = processor(
        prompt=f"abc{IMAGE_TOKEN}",
        mm_items=processor.info.parse_mm_data({"image": image}),
        hf_processor_mm_kwargs={},
    )
    same_video_offset = processor(
        prompt=f"abc{VIDEO_TOKEN}",
        mm_items=processor.info.parse_mm_data({"video": video}),
        hf_processor_mm_kwargs={},
    )

    assert first["mm_hashes"]["video"] != second["mm_hashes"]["video"]
    assert first["mm_hashes"]["image"] != second["mm_hashes"]["image"]
    assert first["mm_hashes"]["image"] == same_image_offset["mm_hashes"]["image"]
    assert third["mm_hashes"]["video"] == same_video_offset["mm_hashes"]["video"]


def test_deepseek_v4_vision_processor_hashes_all_offset_alignments():
    processor = _make_processor()
    image = Image.new("RGB", (14, 14), color=(1, 2, 3))

    results = [
        processor(
            prompt=f"{'z' * prefix_len}{IMAGE_TOKEN}",
            mm_items=processor.info.parse_mm_data({"image": image}),
            hf_processor_mm_kwargs={},
        )
        for prefix_len in range(5)
    ]
    image_hashes = [result["mm_hashes"]["image"][0] for result in results]

    assert len(set(image_hashes[:4])) == 4
    assert image_hashes[0] == image_hashes[4]
