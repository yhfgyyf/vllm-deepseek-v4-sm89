# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 cache codecs; E2M1 conversion also targets Ada without FP4 instructions."""

from vllm.triton_utils import tl, triton

MXFP4_BLOCK_SIZE = 32
SWA_ROW_BYTES = 528
GLOBAL_ROW_BYTES = 288


@triton.jit
def _fp32_to_e2m1(x):
    a = tl.abs(x)
    # Adjacent midpoint ties choose an even significand, including signed zero.
    code = tl.where(a > 0.25, 1, 0)
    code = tl.where(a >= 0.75, 2, code)
    code = tl.where(a > 1.25, 3, code)
    code = tl.where(a >= 1.75, 4, code)
    code = tl.where(a > 2.5, 5, code)
    code = tl.where(a >= 3.5, 6, code)
    code = tl.where(a > 5.0, 7, code)
    sign = (x.to(tl.uint32, bitcast=True) >> 28) & 8
    return (code | sign).to(tl.uint8)


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi, NATIVE_FP4: tl.constexpr = False):
    if NATIVE_FP4:
        return tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $1, $2; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f",
            args=[x_hi, x_lo],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ).to(tl.uint8)
    return _fp32_to_e2m1(x_lo) | (_fp32_to_e2m1(x_hi) << 4)


@triton.jit
def _ceil_log2_scale(x):
    bits = x.to(tl.uint32, bitcast=True)
    return (
        ((bits >> 23) & 255).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    )


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi, NATIVE_FP4: tl.constexpr):
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2.0**-126))
    exponent = _ceil_log2_scale(amax * (1.0 / 6.0))
    inv_scale = tl.exp2(-exponent.to(tl.float32))
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale, NATIVE_FP4)
    return packed, (exponent + 127).to(tl.uint8)


@triton.jit
def _rotate_gptj_512(values, positions, cos_sin, COS_STRIDE: tl.constexpr):
    even, odd = tl.split(tl.reshape(values, (256, 2)))
    pair = tl.arange(0, 256) - 224
    base = cos_sin + positions.to(tl.int64) * COS_STRIDE
    c = tl.load(base + tl.maximum(pair, 0), pair >= 0, other=1).to(tl.float32)
    s = tl.load(base + 32 + tl.maximum(pair, 0), pair >= 0, other=0).to(tl.float32)
    return tl.interleave(even * c - odd * s, odd * c + even * s).to(tl.bfloat16)


@triton.jit
def _store_swa_row(values, dst):
    groups = tl.reshape(values.to(tl.float32), (16, 32))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 1e-4)
    exponent = _ceil_log2_scale(amax * (1.0 / 448.0))
    scaled = groups * tl.exp2(-exponent.to(tl.float32))[:, None]
    data = tl.clamp(scaled, -448, 448).to(tl.float8e4nv)
    tl.store(
        dst + tl.arange(0, 512), tl.reshape(data, (512,)).to(tl.uint8, bitcast=True)
    )
    tl.store(dst + 512 + tl.arange(0, 16), (exponent + 127).to(tl.uint8))


@triton.jit
def _store_global_row(values, dst, NATIVE_FP4: tl.constexpr):
    groups = tl.reshape(values.to(tl.float32), (32, 16))
    amax = tl.maximum(tl.max(tl.abs(groups), 1), 6.0 * (2.0**-9))
    scales = (amax * (1.0 / 6.0)).to(tl.float8e4nv)
    # Approximate reciprocal multiplication can cross exact E2M1 midpoint ties.
    scaled = tl.clamp(tl.div_rn(groups, scales.to(tl.float32)[:, None]), -6.0, 6.0)
    lo, hi = tl.split(tl.reshape(scaled, (256, 2)))
    packed = _fp32x2_to_fp4x2(lo, hi, NATIVE_FP4)
    tl.store(dst + tl.arange(0, 256), packed)
    tl.store(dst + 256 + tl.arange(0, 32), scales.to(tl.uint8, bitcast=True))
