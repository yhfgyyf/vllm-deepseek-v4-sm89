# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from .nvidia.dspark import DSparkDeepseekV4ForCausalLM
from .nvidia.vl_model import DeepseekV41ForCausalLM
from .quant_config import DeepseekV4FP8Config

__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
