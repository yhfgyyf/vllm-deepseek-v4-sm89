# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        vision_n_layers: int = 0,
        vision_patch_size: int = 14,
        vision_downsample_ratio: int = 3,
        vision_max_n_token: int = 384,
        vision_min_pixels: int = 147456,
        vision_max_wh_ratio: int | None = 8,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        self.vision_n_layers = vision_n_layers
        self.vision_patch_size = vision_patch_size
        self.vision_downsample_ratio = vision_downsample_ratio
        self.vision_max_n_token = vision_max_n_token
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_wh_ratio = vision_max_wh_ratio
        self.is_mm_prefix_lm = vision_n_layers > 0
        super().__init__(**kwargs)
        vocab_size = getattr(self, "vocab_size", None)
        if vision_n_layers > 0 and vocab_size is not None:
            self.vllm_extra_input_token_ranges = [(vocab_size, vocab_size + 5)]
            self.vllm_mm_prefix_start_token_id = vocab_size
            self.vllm_mm_prefix_end_token_id = vocab_size + 4
