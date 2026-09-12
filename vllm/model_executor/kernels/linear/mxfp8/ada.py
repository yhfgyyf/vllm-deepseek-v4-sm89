# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native K32 MXFP8 linear kernel for Ada (SM89)."""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


@triton.jit
def _ada_mxfp8_k32_linear_kernel(
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(split_id * 32, K, SPLIT_K * 32, num_stages=2):
        offs_k = k_start + tl.arange(0, 32)
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        weight = tl.load(
            w_ptr + offs_k[:, None] * N + offs_n[None, :],
            mask=mask_n[None, :],
            other=0.0,
        )
        group = k_start // 32
        x_scale_exp = tl.load(
            x_scale_ptr + offs_m * (K // 32) + group,
            mask=mask_m,
            other=0,
        ).to(tl.float32)
        w_scale_exp = tl.load(
            w_scale_ptr + offs_n * (K // 32) + group,
            mask=mask_n,
            other=0,
        ).to(tl.float32)
        block = tl.dot(x, weight, out_dtype=tl.float32)
        scale = tl.exp2(x_scale_exp[:, None] + w_scale_exp[None, :] - 254.0)
        accumulator += block * scale

    if HAS_BIAS and SPLIT_K == 1:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator += bias[None, :]
    output_offset = split_id * M * N
    tl.store(
        output_ptr + output_offset + offs_m[:, None] * N + offs_n[None, :],
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _ada_mxfp8_splitk_reduce_kernel(
    partial_ptr,
    bias_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    partial_offset = offs_m[:, None] * N + offs_n[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in tl.range(0, SPLIT_K):
        accumulator += tl.load(
            partial_ptr + split_id * M * N + partial_offset,
            mask=mask,
            other=0.0,
        )
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    tl.store(output_ptr + partial_offset, accumulator, mask=mask)


class AdaMxfp8LinearKernel(Mxfp8LinearKernel):
    """SM89 FP8 tensor-core GEMM with exact per-K32 MXFP8 scaling.

    Activations remain FP8 through the GEMM. Weight and activation scales are
    applied to each K32 partial accumulator, so this is not Marlin W8A16 and
    never expands the checkpoint weight to BF16.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA"
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = (
                None if capability is None else capability.major * 10 + capability.minor
            )
        if compute_capability != 89:
            return False, "requires sm_89"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.model_profile != "deepseek_v41":
            return False, "reserved for the DeepSeek-V4.1 K32 contract"
        return True, None

    def input_quant_key(self):
        return kMxfp8Dynamic

    @staticmethod
    def _split_k(rows: int, output_size: int) -> int:
        output_tiles = triton.cdiv(rows, 32) * triton.cdiv(output_size, 64)
        if output_tiles >= 128:
            return 1
        if output_tiles >= 64:
            return 2
        if output_tiles >= 32:
            return 4
        if output_tiles >= 16:
            return 8
        if output_tiles >= 8:
            return 16
        return 32

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        output_size, input_size = weight.shape
        if input_size % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"Ada MXFP8 requires input size divisible by 32, got {input_size}."
            )
        expected_scale_shape = (output_size, input_size // MXFP8_BLOCK_SIZE)
        if tuple(layer.weight_scale.shape) != expected_scale_shape:
            raise ValueError(
                "Ada MXFP8 expects expanded row-major weight scales with shape "
                f"{expected_scale_shape}, got {tuple(layer.weight_scale.shape)}."
            )
        if weight.dtype != MXFP8_VALUE_DTYPE:
            raise TypeError(
                f"Ada MXFP8 expects {MXFP8_VALUE_DTYPE} weights, got {weight.dtype}."
            )
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise TypeError(
                "Ada MXFP8 expects uint8 UE8M0 weight scales, got "
                f"{layer.weight_scale.dtype}."
            )
        layer.weight = Parameter(weight.t().contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data.contiguous(), requires_grad=False
        )
        layer._ada_mxfp8_input_size = input_size
        layer._ada_mxfp8_output_size = output_size

    @staticmethod
    def _quantize_input(
        x: torch.Tensor | QuantizedActivation,
        input_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.dtype, torch.Size]:
        quantized = as_quantized_activation(x, kMxfp8Dynamic)
        if quantized is None:
            input_shape = x.shape
            output_dtype = x.dtype
            x_2d = x.reshape(-1, input_size)
            xq, x_scale = mxfp8_e4m3_quantize(
                x_2d,
                is_sf_swizzled_layout=False,
                min_amax=1e-4,
            )
            return xq, x_scale, output_dtype, input_shape

        xq = quantized.data.reshape(-1, input_size)
        x_scale = quantized.scale.reshape(-1, input_size // MXFP8_BLOCK_SIZE)
        if xq.dtype != MXFP8_VALUE_DTYPE or x_scale.dtype != MXFP8_SCALE_DTYPE:
            raise TypeError(
                "Ada MXFP8 pre-quantized input requires float8_e4m3fn data "
                "and uint8 UE8M0 scales."
            )
        if not xq.is_contiguous() or not x_scale.is_contiguous():
            raise ValueError(
                "Ada MXFP8 pre-quantized data and scales must be contiguous."
            )
        return xq, x_scale, quantized.orig_dtype, quantized.orig_shape

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_size = layer._ada_mxfp8_input_size
        output_size = layer._ada_mxfp8_output_size
        xq, x_scale, output_dtype, input_shape = self._quantize_input(x, input_size)
        rows = xq.shape[0]
        output = torch.empty((rows, output_size), dtype=output_dtype, device=xq.device)
        block_m = 32
        block_n = 64
        output_tiles = triton.cdiv(rows, block_m) * triton.cdiv(output_size, block_n)
        split_k = self._split_k(rows, output_size)
        partial = (
            output
            if split_k == 1
            else torch.empty(
                (split_k, rows, output_size),
                dtype=torch.float32,
                device=xq.device,
            )
        )
        grid = (output_tiles, split_k)
        _ada_mxfp8_k32_linear_kernel[grid](
            xq,
            layer.weight,
            x_scale,
            layer.weight_scale,
            bias if bias is not None else output,
            partial,
            rows,
            N=output_size,
            K=input_size,
            HAS_BIAS=bias is not None,
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
        if split_k > 1:
            _ada_mxfp8_splitk_reduce_kernel[(output_tiles,)](
                partial,
                bias if bias is not None else output,
                output,
                rows,
                N=output_size,
                HAS_BIAS=bias is not None,
                SPLIT_K=split_k,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
            )
        return output.view(*input_shape[:-1], output_size)
