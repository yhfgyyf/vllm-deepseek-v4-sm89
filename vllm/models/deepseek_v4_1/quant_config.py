# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization contract for DeepSeek-V4.1-Flash."""

from __future__ import annotations

from typing import cast

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.quantization import QuantSpec
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.modelopt import (
    CkptCtx,
    KMxfp8Static,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.mxfp4 import (
    DeepseekV41Mxfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.models.deepseek_v4.quant_config import (
    DeepseekV4FP8Config as DeepseekV4BaseFP8Config,
)


@register_weight_loader_v2_supported_method
class DeepseekV41WoALinearMethod(LinearMethodBase):
    """Load checkpoint MXFP8 ``wo_a`` once into its official BF16 form."""

    def __init__(self) -> None:
        self.unquantized_method = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % MXFP8_BLOCK_SIZE:
            raise ValueError("DeepSeek-V4.1 wo_a input size must be divisible by 32")
        if any(size % 32 for size in output_partition_sizes):
            raise ValueError("DeepSeek-V4.1 wo_a output shards must be divisible by 32")
        if params_dtype != torch.bfloat16:
            raise ValueError("DeepSeek-V4.1 wo_a runtime weight must use bfloat16")

        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=MXFP8_VALUE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)

        scale_loader = KMxfp8Static.get_scale_weight_loader(
            weight_loader, CkptCtx(scale_block_size=(32, 32))
        )
        weight_scale = ModelWeightParameter(
            data=torch.full(
                (
                    output_size_per_partition,
                    input_size_per_partition // MXFP8_BLOCK_SIZE,
                ),
                fill_value=255,
                dtype=MXFP8_SCALE_DTYPE,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=scale_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype == torch.bfloat16:
            return
        if layer.weight.dtype != MXFP8_VALUE_DTYPE:
            raise TypeError(
                "DeepSeek-V4.1 wo_a checkpoint weight must be float8_e4m3fn, "
                f"got {layer.weight.dtype}."
            )
        expected_scale_shape = (
            layer.weight.shape[0],
            layer.weight.shape[1] // MXFP8_BLOCK_SIZE,
        )
        if (
            layer.weight_scale.dtype != MXFP8_SCALE_DTYPE
            or tuple(layer.weight_scale.shape) != expected_scale_shape
        ):
            raise ValueError(
                "DeepSeek-V4.1 wo_a requires expanded row-major uint8 scales "
                f"with shape {expected_scale_shape}."
            )
        if torch.any(layer.weight_scale == 255):
            config = get_current_vllm_config_or_none()
            load_format = getattr(
                getattr(config, "load_config", None), "load_format", None
            )
            if load_format != "dummy":
                raise ValueError(
                    "DeepSeek-V4.1 wo_a checkpoint scale was not fully loaded"
                )
            layer.weight_scale.data.fill_(127)
        replace_parameter(
            layer,
            "weight",
            dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.unquantized_method.apply(layer, x, bias)


class DeepseekV4FP8Config(DeepseekV4BaseFP8Config):
    """V4.1-only 32x32 MXFP8 linear and MXFP4/MXFP8 MoE config."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v41_fp8"

    @property
    def expert_dtype(self) -> str:
        return "fp4"

    @property
    def is_scale_e8m0(self) -> bool:
        return True

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg,
        user_quant,
        hf_config=None,
    ) -> QuantizationMethods | None:
        if not isinstance(hf_quant_cfg, dict):
            return None
        if hf_quant_cfg.get("quant_method") not in (
            "fp8",
            "deepseek_v41_fp8",
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type in ("deepseek_v41", "deepseek_v41_text"):
            return "deepseek_v41_fp8"
        if user_quant == "deepseek_v41_fp8":
            return "deepseek_v41_fp8"
        return None

    @classmethod
    def from_config(cls, config: dict) -> DeepseekV4FP8Config:
        if config.get("weight_block_size") != [32, 32]:
            raise ValueError(
                "DeepSeek-V4.1 requires weight_block_size=[32, 32], got "
                f"{config.get('weight_block_size')!r}."
            )
        if config.get("scale_fmt") != "ue8m0":
            raise ValueError(
                "DeepSeek-V4.1 requires scale_fmt='ue8m0', got "
                f"{config.get('scale_fmt')!r}."
            )
        if config.get("expert_dtype", "fp4") != "fp4":
            raise ValueError("DeepSeek-V4.1 currently supports only FP4 experts.")
        return cast("DeepseekV4FP8Config", super().from_config(config))

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if prefix.rsplit(".", 1)[-1] == "wo_a":
                return DeepseekV41WoALinearMethod()
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            layer._mxfp8_model_profile = "deepseek_v41"
            return ModelOptLinearMethod(
                QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
                CkptCtx(scale_block_size=(32, 32)),
            )
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return DeepseekV41Mxfp4MoEMethod(layer.moe_config)
        return super().get_quant_method(layer, prefix)
