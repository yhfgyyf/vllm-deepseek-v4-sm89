# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    dequantize_fp8ds_mla656_global_slots,
)


def _dequantize_cache_entry(
    cache_entry: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_dim = 512
    rope_dim = 64
    out = torch.empty(latent_dim + rope_dim, dtype=dtype, device=cache_entry.device)
    scales = cache_entry.view(torch.float32)[latent_dim // 4 : latent_dim // 4 + 4]
    for tile in range(4):
        start = tile * 128
        end = start + 128
        ops.convert_fp8(
            out[start:end],
            cache_entry[start:end],
            scales[tile].item(),
            kv_dtype="fp8",
        )
    rope_offset = latent_dim // 2 + 8
    out[latent_dim:] = cache_entry.view(dtype)[rope_offset : rope_offset + rope_dim]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(current_platform.is_rocm(), reason="fp8_ds_mla is CUDA-only")
@torch.inference_mode()
def test_dequantize_fp8ds_mla656_global_slots() -> None:
    device = "cuda"
    dtype = torch.bfloat16
    num_blocks = 3
    block_size = 8
    num_tokens = 17

    torch.manual_seed(0)
    kv_c = torch.randn(num_tokens, 512, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, 64, dtype=dtype, device=device)
    slot_mapping = torch.tensor(
        [0, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 3],
        dtype=torch.long,
        device=device,
    )
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        656,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype="fp8_ds_mla",
        scale=scale,
    )

    slot_ids = torch.tensor(
        [
            [0, 5, 7, -1, -1],
            [16, 17, 3, 20, -1],
            [23, 21, 11, 9, 8],
        ],
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty(slot_ids.shape + (576,), dtype=dtype, device=device)

    dequantize_fp8ds_mla656_global_slots(out, kv_cache, slot_ids, block_size)

    expected = torch.zeros_like(out)
    for row in range(slot_ids.shape[0]):
        for col in range(slot_ids.shape[1]):
            slot = int(slot_ids[row, col].item())
            if slot < 0:
                continue
            expected[row, col] = _dequantize_cache_entry(
                kv_cache[slot // block_size, slot % block_size],
                dtype,
            )

    torch.testing.assert_close(
        out[:, :, :512],
        expected[:, :, :512],
        atol=1e-3,
        rtol=1e-2,
    )
    assert torch.equal(out[:, :, 512:], expected[:, :, 512:])
