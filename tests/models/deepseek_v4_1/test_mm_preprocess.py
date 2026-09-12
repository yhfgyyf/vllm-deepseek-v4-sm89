# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from PIL import Image

from vllm.models.deepseek_v4.vision import (
    DeepseekV4Aligner as SharedDeepseekV4Aligner,
)
from vllm.models.deepseek_v4.vision import (
    DeepseekV4VisionTransformer as SharedDeepseekV4VisionTransformer,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD_ID,
    IMAGE_SENTINEL_BASE_ID,
    IMAGE_START,
    DeepseekV4VLProcessor,
    image_sentinel_mask,
    image_token_types,
)
from vllm.models.deepseek_v4_1.nvidia.vl_model import (
    DeepseekV4Aligner,
    DeepseekV4VisionTransformer,
    DeepseekV41ForCausalLM,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config


def _config() -> DeepseekV41Config:
    return DeepseekV41Config(
        text_config={"hidden_size": 8},
        vision_config={
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "patch_size": 2,
            "downsample_ratio": 2,
            "max_image_tokens": 32,
            "min_pixels": 1,
        },
    )


def test_v41_image_processor_preserves_patch_and_span_layout():
    image = Image.new("RGB", (6, 4), color=(255, 127, 0))

    outputs = DeepseekV4VLProcessor(_config())(images=[image])

    assert outputs["patches"].shape == (6, 3, 2, 2)
    assert outputs["patches"].dtype == torch.bfloat16
    assert torch.equal(outputs["vit_grid"], torch.tensor([[2, 3]]))
    assert torch.equal(outputs["llm_grid"], torch.tensor([[1, 2]]))
    assert torch.equal(
        outputs["types"],
        torch.tensor([IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END]),
    )


def test_v41_image_sentinel_mask_includes_span_and_alignment_pad():
    token_ids = torch.tensor(
        [7, IMAGE_SENTINEL_BASE_ID, IMAGE_PAD_ID, IMAGE_SENTINEL_BASE_ID + 2]
    )

    assert torch.equal(
        image_sentinel_mask(token_ids),
        torch.tensor([False, True, True, False]),
    )


def test_v41_image_span_places_shared_vision_rows_and_delimiters():
    assert DeepseekV4VisionTransformer is SharedDeepseekV4VisionTransformer
    assert DeepseekV4Aligner is SharedDeepseekV4Aligner

    model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    torch.nn.Module.__init__(model)
    model.image_start = torch.nn.Parameter(torch.tensor([10.0, 11.0]))
    model.image_newline = torch.nn.Parameter(torch.tensor([20.0, 21.0]))
    model.image_end = torch.nn.Parameter(torch.tensor([30.0, 31.0]))
    image_rows = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    span = model._build_image_span(image_rows, image_token_types(1, 2))

    torch.testing.assert_close(
        span,
        torch.tensor(
            [
                [10.0, 11.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        ),
    )
