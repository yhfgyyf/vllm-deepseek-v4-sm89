# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import importlib.metadata

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_DEEPSEEK_V41_ENGRAM_MXFP8_SHAPE = (25600, 6144)
_DEEPSEEK_V41_ENGRAM_MXFP8_TACTICS = {
    1: 1,
    4: 1,
    8: 1,
    16: 1,
    32: 1,
    128: 2,
    1024: 4,
}
_DEEPSEEK_V41_MXFP8_TACTIC_FLASHINFER_VERSION = (
    "0.6.18+glm53.dsv4.vision1.sm89sm120.cu130.pt213"
)
_FLASHINFER_MXFP8_WORKSPACE_KEY = "mm_mxfp8_workspace"


def _deepseek_v41_engram_mxfp8_tactic(rows: int, n: int, k: int) -> int | None:
    if (n, k) != _DEEPSEEK_V41_ENGRAM_MXFP8_SHAPE:
        return None
    return _DEEPSEEK_V41_ENGRAM_MXFP8_TACTICS.get(rows)


@functools.cache
def _has_deepseek_v41_sm120_mxfp8_tactics() -> bool:
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return False
    return version == _DEEPSEEK_V41_MXFP8_TACTIC_FLASHINFER_VERSION


def _deepseek_v41_sm120_mxfp8_impl(
    input_mxfp8: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    tactic = _deepseek_v41_engram_mxfp8_tactic(
        input_mxfp8.shape[0], weight.shape[0], weight.shape[1]
    )
    if tactic is None:
        return vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )

    with torch.cuda.device(input_mxfp8.device):
        from flashinfer.gemm.gemm_base import (
            DEFAULT_WORKSPACE_SIZE,
            _load_gemm_sm120_mxfp8_module,
        )
        from flashinfer.utils import _get_cache_buf

        output = torch.empty(
            (input_mxfp8.shape[0], weight.shape[0]),
            dtype=out_dtype,
            device=input_mxfp8.device,
        )
        workspace = _get_cache_buf(
            _FLASHINFER_MXFP8_WORKSPACE_KEY,
            DEFAULT_WORKSPACE_SIZE,
            input_mxfp8.device,
        )
        module = _load_gemm_sm120_mxfp8_module()
        if module.mxfp8_gemm_tactic_num() != 10:
            raise RuntimeError(
                "DeepSeek-V4.1 SM120 MXFP8 tactics require the pinned ten-tactic "
                "FlashInfer CUTLASS module."
            )
        module.mxfp8_gemm(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            output,
            workspace,
            tactic,
        )
    return output


def _deepseek_v41_sm120_mxfp8_fake(
    input_mxfp8: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    del input_scale, weight_scale
    return torch.empty(
        (input_mxfp8.shape[0], weight.shape[0]),
        dtype=out_dtype,
        device=input_mxfp8.device,
    )


direct_register_custom_op(
    op_name="deepseek_v41_sm120_mxfp8",
    op_func=_deepseek_v41_sm120_mxfp8_impl,
    fake_impl=_deepseek_v41_sm120_mxfp8_fake,
)


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        self._deepseek_v41_engram_tactics_enabled = (
            c.model_profile == "deepseek_v41"
            and current_platform.is_device_capability(120)
            and _has_deepseek_v41_sm120_mxfp8_tactics()
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            return False, "requires >=sm_100 (Blackwell)"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        out_dtype = x.dtype
        N, K = weight.shape

        input_shape = x.shape
        input_2d = x.view(-1, K)
        min_dim = 128

        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,
            min_amax=(1e-4 if self.config.model_profile == "deepseek_v41" else 0.0),
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        use_deepseek_v41_engram_tactics = (
            self._deepseek_v41_engram_tactics_enabled
            and (N, K) == _DEEPSEEK_V41_ENGRAM_MXFP8_SHAPE
        )
        if not use_deepseek_v41_engram_tactics:
            output = vllm_flashinfer.mm_mxfp8(
                input_mxfp8,
                weight.t(),
                input_scale,
                weight_scale,
                out_dtype=out_dtype,
                backend="cutlass",
            )
        else:
            output = torch.ops.vllm.deepseek_v41_sm120_mxfp8(
                input_mxfp8,
                weight,
                input_scale,
                weight_scale,
                out_dtype,
            )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires sm_100/sm_103 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        layer.weight = Parameter(weight.contiguous().t(), requires_grad=False)
        layer.weight_scale = Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N], column-major
        weight_scale = layer.weight_scale
        out_dtype = x.dtype
        K, N = weight.shape

        input_shape = x.shape
        input_2d = x.view(-1, K)
        min_dim = 128

        assert min_dim <= K, (
            f"mm_mxfp8 requires K >= {min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert K % MXFP8_BLOCK_SIZE == 0, (
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}."
        )
        assert min_dim <= N, (
            f"mm_mxfp8 requires N >= {min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d, is_sf_swizzled_layout=True
        )

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(
                f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}."
            )

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        padded_n = ((N + 127) // 128) * 128
        if padded_n != N:
            padded_weight = weight.new_zeros((padded_n, K))
            padded_weight[:N] = weight
            weight = padded_weight

            padded_scale = weight_scale.new_zeros((padded_n, scale_k))
            padded_scale[:N] = weight_scale
            weight_scale = padded_scale
        else:
            weight = weight.contiguous()

        layer.weight = Parameter(
            shuffle_matrix_a(weight, 128).reshape(padded_n, K),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
            requires_grad=False,
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (
            f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."
        )

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        input_mxfp8, input_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(
            input_2d
        )
        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=x.dtype,
            backend="trtllm",
            use_8x4_sf_layout=True,
        )
        if output.shape[-1] != output_size:
            output = output[:, :output_size].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
