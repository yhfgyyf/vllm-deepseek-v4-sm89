# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical contracts for V4.1 native codecs, compression and delayed mHC."""

import pytest
import torch

from vllm.models.deepseek_v4_1.common.ops.quant_utils import _fp32x2_to_fp4x2
from vllm.triton_utils import tl, triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _rope(x, positions, cache, inverse=False):
    out = x.float().clone()
    cs = cache[positions.long()].float()
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    c, s = cs[:, :32].reshape(shape), cs[:, 32:].reshape(shape)
    if inverse:
        s = -s
    even, odd = x[..., -64::2].float(), x[..., -63::2].float()
    out[..., -64::2] = even * c - odd * s
    out[..., -63::2] = odd * c + even * s
    return out.to(x.dtype)


def _e2m1(x):
    # Independent nearest-value reference; ties prioritize the even encoding.
    values = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    distance = (x.abs()[..., None] - values).abs()
    tied = distance == distance.amin(-1, keepdim=True)
    order = torch.arange(8, device=x.device)
    priority = order + (order % 2) * 8
    code = torch.where(tied, priority, 32).argmin(-1)
    return (code | (torch.signbit(x).to(torch.int64) * 8)).to(torch.uint8)


def _fp4_pack(x, group, e4m3_scale):
    groups = x.float().reshape(*x.shape[:-1], -1, group)
    amax = groups.abs().amax(-1).clamp_min(6 * 2 ** (-9 if e4m3_scale else -126))
    if e4m3_scale:
        scales = (amax / 6).to(torch.float8_e4m3fn)
        codes = _e2m1((groups / scales.float()[..., None]).clamp(-6, 6))
        scale_bytes = scales.view(torch.uint8)
    else:
        exp = torch.ceil(torch.log2(amax / 6))
        codes = _e2m1(groups * torch.exp2(-exp)[..., None])
        scale_bytes = (exp + 127).to(torch.uint8)
    codes = codes.reshape(x.shape)
    return codes[..., ::2] | (codes[..., 1::2] << 4), scale_bytes


def _fp8_pack(x):
    groups = x.float().reshape(*x.shape[:-1], -1, 32)
    exp = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    data = (groups * torch.exp2(-exp)[..., None]).clamp(-448, 448)
    return data.reshape(x.shape).to(torch.float8_e4m3fn), (exp + 127).to(torch.uint8)


def _rms(x, w):
    x = x.float()
    return (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w.float()
    ).bfloat16()


def _cache(pages, page_size, row_bytes):
    # Page padding is deliberately not divisible by the row size.
    storage = torch.full(
        (pages, page_size * row_bytes + 80), 165, device="cuda", dtype=torch.uint8
    )
    cache = storage.as_strided(
        (pages, page_size, row_bytes), (storage.stride(0), row_bytes, 1)
    )
    return storage, cache


def _cos(size=128):
    angles = torch.randn(size, 32, device="cuda")
    return torch.cat((angles.cos(), angles.sin()), -1)


@triton.jit
def _convert_pairs(
    x, out, SIZE: tl.constexpr, NATIVE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    lo = tl.load(x + 2 * i, 2 * i < SIZE, 0)
    hi = tl.load(x + 2 * i + 1, 2 * i + 1 < SIZE, 0)
    tl.store(out + i, _fp32x2_to_fp4x2(lo, hi, NATIVE), i < SIZE // 2)


@pytest.mark.parametrize("sign", [1, -1])
def test_e2m1_reference_midpoints(sign):
    x = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]) * sign
    assert (_e2m1(x) & 7).tolist() == [0, 0, 2, 2, 4, 4, 6, 6]


@cuda
def test_e2m1_native_and_ada_codec_match_at_rounding_boundaries():
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    positive = torch.cat(
        (
            mid,
            torch.nextafter(mid, mid * 0),
            torch.nextafter(mid, mid * 2),
            mid.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 100]),
        )
    )
    x = torch.cat((positive, -positive))
    ref = _e2m1(x)
    expected = ref[::2] | (ref[1::2] << 4)
    modes = [False, True] if torch.cuda.get_device_capability()[0] >= 10 else [False]
    for native in modes:
        out = torch.empty(x.numel() // 2, device="cuda", dtype=torch.uint8)
        _convert_pairs[(1,)](
            x, out, x.numel(), native, triton.next_power_of_2(out.numel())
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_query_rope_and_swa_insert_preserve_padded_pages(heads):
    from vllm.models.deepseek_v4_1.common.ops import fused_q_rope_swa_insert

    torch.manual_seed(4)
    tokens = 7
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    kv[0].zero_()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 7, 8, 15, 16, -1, 24], device="cuda")
    cs = _cos()
    storage, cache = _cache(3, 8, 528)
    expected_storage = storage.clone()
    expected_cache = expected_storage.as_strided(cache.shape, cache.stride())
    packed, scales = _fp8_pack(_rope(kv, positions, cs))
    for row, slot in enumerate(slots.tolist()):
        if 0 <= slot < 24:
            expected_cache[slot // 8, slot % 8] = torch.cat(
                (packed[row].view(torch.uint8), scales[row])
            )
    out = fused_q_rope_swa_insert(q, kv, positions, cs, cache, slots)
    torch.testing.assert_close(out, _rope(q, positions, cs), rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("ratio", [1, 2])
def test_global_fp4_insert_uses_group16_and_group_start_rope(ratio):
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    torch.manual_seed(8)
    latent = torch.randn(9, 512, device="cuda", dtype=torch.bfloat16)
    latent[1].zero_()
    positions = torch.arange(9, device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 1, 2, 3, 4, 7, 8, -1, 12], device="cuda")
    cs = _cos()
    storage, cache = _cache(3, 4, 288)
    expected = storage.clone()
    view = expected.as_strided(cache.shape, cache.stride())
    data, scales = _fp4_pack(_rope(latent, positions // ratio * ratio, cs), 16, True)
    for i, slot in enumerate(slots.tolist()):
        if 0 <= slot < 12 and (i + 1) % ratio == 0:
            view[slot // 4, slot % 4] = torch.cat((data[i], scales[i]))
    rope_quant_insert(latent, positions, cs, cache, slots, ratio)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("heads", [8, 16, 32])
def test_inverse_rope_retains_bf16_grouped_projection_contract(heads):
    from vllm.models.deepseek_v4_1.common.ops import fused_inv_rope_fp8_quant

    x = torch.randn(7, heads, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(7, device="cuda", dtype=torch.int64)
    cs = _cos()
    groups = heads // 8
    out, scale = fused_inv_rope_fp8_quant(x, positions, cs, groups, 8, quantize=False)
    torch.testing.assert_close(
        out, _rope(x, positions, cs, True).reshape(7, groups, 4096), rtol=0, atol=0
    )
    assert out.transpose(0, 1).is_contiguous()
    assert scale.numel() == 0


@cuda
def test_indexer_query_codec_matches_official_group32():
    from vllm.models.deepseek_v4_1.common.ops import fused_indexer_q_rope_quant

    q = torch.randn(7, 32, 128, device="cuda", dtype=torch.bfloat16)
    q[0].zero_()
    w = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(7, device="cuda", dtype=torch.int64)
    cs = _cos()
    (actual, scales), weights = fused_indexer_q_rope_quant(
        pos, q, cs, w, 128**-0.5, 32**-0.5
    )
    ref, ref_scale = _fp4_pack(_rope(q, pos, cs), 32, False)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    torch.testing.assert_close(scales, ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        weights, w.float() * 128**-0.5 * 32**-0.5, rtol=1e-6, atol=1e-7
    )


@cuda
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("page", [64, 128])
def test_indexer_key_cache_is_segregated_not_interleaved(ratio, page):
    from vllm.models.deepseek_v4_1.common.ops import indexer_k_norm_rope_store

    k = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(7, device="cuda", dtype=torch.int64)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor(
        [0, page - 1, page, page + 1, -1, 2 * page, 3 * page], device="cuda"
    )
    cs = _cos()
    storage, cache = _cache(3, page, 68)
    expected = storage.clone()
    data, scale = _fp4_pack(_rope(_rms(k, w), pos // ratio * ratio, cs), 32, False)
    for i, slot in enumerate(slots.tolist()):
        if 0 <= slot < 3 * page and (i + 1) % ratio == 0:
            p, r = divmod(slot, page)
            expected[p, r * 64 : (r + 1) * 64] = data[i]
            expected[p, page * 64 + r * 4 : page * 64 + (r + 1) * 4] = scale[i]
    indexer_k_norm_rope_store(k, pos, cs, w, 1e-6, cache, slots, ratio, True)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("row_major", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 129])
def test_query_quant_scale_layout_matches_consumer(monkeypatch, row_major, tokens):
    from vllm.models.deepseek_v4_1.common.ops import query_quant

    monkeypatch.setattr(
        query_quant.current_platform,
        "is_device_capability",
        lambda cap: row_major and cap == 89,
    )
    monkeypatch.setattr(
        query_quant.current_platform,
        "is_device_capability_family",
        lambda cap: not row_major and cap == 120,
    )
    qr = torch.randn(tokens, 1280, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    qr[0].zero_()
    qw = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    out, kvo = query_quant.fused_q_kv_rmsnorm_quant(qr, kv, qw, kw, 1e-6)
    data, scales = _fp8_pack(_rms(qr, qw))
    if row_major:
        actual = out.scale
    else:
        rows = torch.arange(tokens, device="cuda")[:, None]
        cols = torch.arange(40, device="cuda")[None, :]
        offsets = (
            rows // 128 * (128 * 40)
            + cols // 4 * 512
            + rows % 32 * 16
            + rows % 128 // 32 * 4
            + cols % 4
        )
        actual = out.scale[offsets]
    torch.testing.assert_close(
        out.data.view(torch.uint8), data.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual, scales, rtol=0, atol=0)
    torch.testing.assert_close(kvo, _rms(kv, kw), rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("ratio", [1, 2])
def test_compression_handles_chunk_start_ring_read_before_tail_write(ratio):
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm,
    )

    torch.manual_seed(12)
    positions = torch.tensor([1, 2, 3, 4, 0, 1, 2], device="cuda", dtype=torch.int64)
    req = torch.tensor([0, 0, 0, 0, 1, 1, 1], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 4, 7], device="cuda", dtype=torch.int32)
    state = torch.randn(2, 4, 1024, device="cuda", dtype=torch.float32)
    raw = torch.randn(7, 512 * ratio, device="cuda", dtype=torch.float32)
    w = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    slots = req.long() * 4 + positions % 4
    latent = torch.full((7, 512), 37.0, device="cuda", dtype=torch.bfloat16)
    expected = latent.clone()
    expected_state = state.clone()
    for i, (p, r) in enumerate(zip(positions.tolist(), req.tolist())):
        if ratio == 1:
            expected[i] = _rms(raw[i], w)
        else:
            if (p + 1) % 2 == 0:
                pair = torch.stack((expected_state[r, (p - 1) % 4], raw[i]))
                pooled = (pair[:, :512] * pair[:, 512:].softmax(0)).sum(0)
                expected[i] = _rms(pooled, w)
            expected_state[r, p % 4] = raw[i]
    fused_save_compress_norm(
        raw,
        positions,
        state if ratio == 2 else None,
        slots,
        starts if ratio == 2 else None,
        req if ratio == 2 else None,
        w,
        1e-6,
        ratio,
        latent,
    )
    torch.testing.assert_close(latent, expected, rtol=1e-2, atol=1e-2)
    if ratio == 2:
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("graph_replay", [False, True])
def test_compressor_padding_does_not_write_latents_or_ring(graph_replay):
    """Padded FULL-graph capacity is not the live token count."""
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
        fused_save_compress_norm,
    )
    from vllm.models.deepseek_v4_1.compressor import CompressorMetadataBuilder
    from vllm.v1.attention.backend import CommonAttentionMetadata

    torch.manual_seed(13)
    device = "cuda"
    builder = object.__new__(CompressorMetadataBuilder)
    builder.capacity = 8
    builder.token_to_req_indices = torch.zeros(8, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
    starts = torch.zeros(5, dtype=torch.int32, device=device)
    positions = torch.zeros(8, dtype=torch.int64, device=device)
    source_slots = torch.full((8,), -1, dtype=torch.int64, device=device)
    blocks = torch.arange(4, dtype=torch.int32, device=device).view(4, 1)
    initial_state = torch.randn(4, 8, 1024, device=device)
    state = initial_state.clone()
    raw = torch.randn(8, 1024, device=device)
    weight = torch.randn(512, dtype=torch.bfloat16, device=device)
    latent = torch.full((8, 512), 37, dtype=torch.bfloat16, device=device)
    graph = None

    # Uneven verification plus prefill, then a zero-draft verification budget.
    for lengths, cpu_lengths in [
        ([3, 1, 3, 0], [2, 2, 3, 0]),
        ([1, 1, 3, 0], [1, 1, 3, 0]),
    ]:
        live = sum(lengths)
        cpu_starts = torch.tensor([0, *cpu_lengths], dtype=torch.int32).cumsum(0)
        starts.copy_(torch.tensor([0, *lengths], device=device).cumsum(0))
        pos = [p + i for n, p in zip(lengths, [1, 4, 0, 0]) for i in range(n)]
        positions.copy_(torch.tensor(pos + [1] * (8 - live), device=device))
        source_slots.fill_(-1)
        source_slots[:live] = 0
        cm = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=cpu_starts,
            seq_lens=torch.tensor(
                [p + n for p, n in zip([1, 4, 0, 0], lengths)],
                dtype=torch.int32,
                device=device,
            ),
            num_reqs=4,
            num_actual_tokens=8,
            max_query_len=3,
            max_seq_len=5,
            block_table_tensor=blocks,
            slot_mapping=source_slots,
            positions=positions,
        )
        metadata = builder.build(0, cm)
        assert metadata.slot_mapping[live:].tolist() == [-1] * (8 - live)
        state.copy_(initial_state)
        latent.fill_(37)
        expected_state, expected = state.clone(), latent.clone()
        token = 0
        for req, length in enumerate(lengths):
            for _ in range(length):
                p = pos[token]
                if (p + 1) % 2 == 0:
                    pair = torch.stack((expected_state[req, (p - 1) % 8], raw[token]))
                    pooled = (pair[:, :512] * pair[:, 512:].softmax(0)).sum(0)
                    expected[token] = _rms(pooled, weight)
                expected_state[req, p % 8] = raw[token]
                token += 1

        def forward(metadata=metadata):
            fused_save_compress_norm(
                raw,
                positions,
                state,
                metadata.slot_mapping,
                starts,
                metadata.token_to_req_indices,
                weight,
                1e-6,
                2,
                latent,
            )

        if graph_replay:
            if graph is None:
                forward()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
            state.copy_(initial_state)
            latent.fill_(37)
            graph.replay()
        else:
            forward()
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(latent, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(latent[live:], expected[live:], rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("tokens", [1, 17, 32, 33, 129])
@pytest.mark.parametrize("carried", [False, True])
@pytest.mark.parametrize("with_norm", [False, True])
def test_delayed_mhc_uses_previous_pre_mix(monkeypatch, tokens, carried, with_norm):
    from tests.kernels.test_mhc_kernels import sinkhorn_normalize_ref
    from vllm.model_executor.kernels.mhc.deepseek_v41 import mhc_pre_delayed_tilelang

    torch.manual_seed(7)
    residual = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    x = residual[:, 0].contiguous() if with_norm else residual.flatten(1)
    fn = torch.randn(24, x.shape[1], device="cuda", dtype=torch.float32) * 0.001
    scale = torch.randn(3, device="cuda") * 0.1
    base = torch.randn(24, device="cuda") * 0.1
    pre = torch.rand(tokens, 4, device="cuda") if carried else None
    norm = torch.randn(5120, device="cuda", dtype=torch.bfloat16) if with_norm else None
    post, comb, layer, next_pre = mhc_pre_delayed_tilelang(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20, pre, x, norm
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    mixed = x.float() @ fn.T
    mixed *= torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    mixed *= torch.cat(
        (scale[:1].expand(4), scale[1:2].expand(4), scale[2:].expand(16))
    )
    mixed += base
    torch.testing.assert_close(
        next_pre, mixed[:, :4].sigmoid() + 1e-6, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        post.squeeze(-1), mixed[:, 4:8].sigmoid() * 2, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        comb,
        sinkhorn_normalize_ref(mixed[:, 8:].reshape(tokens, 4, 4), 20, 1e-6),
        rtol=1e-5,
        atol=1e-6,
    )
    ref_layer = (
        residual[:, 0]
        if pre is None
        else (residual.float() * pre[..., None]).sum(1).bfloat16()
    )
    if norm is not None:
        ref_layer = _rms(ref_layer, norm)
    torch.testing.assert_close(layer, ref_layer, rtol=1e-2, atol=1e-2)


@cuda
@pytest.mark.parametrize("tokens", [1, 16, 32, 513])
def test_hc_collapse_matches_fp32_reference(tokens):
    from vllm.model_executor.kernels.mhc.deepseek_v41 import hc_collapse_triton

    x = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
    coeff = torch.randn(tokens, 4, device="cuda")
    actual = hc_collapse_triton(x, coeff)
    expected = sum(x[:, i].float() * coeff[:, i, None] for i in range(4)).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cuda
def test_global_slot_mapping_rejects_padding_and_invalid_pages():
    from vllm.models.deepseek_v4_1.common.ops import (
        compute_global_topk_indices_and_lens,
    )

    local = torch.tensor(
        [[0, 63, 64, 128, -1, 512]] * 4, device="cuda", dtype=torch.int32
    )
    req = torch.tensor([0, 1, 1, 3], device="cuda", dtype=torch.int32)
    valid = torch.tensor([True, True, False, True], device="cuda")
    table = torch.tensor([[2, -1], [0, 1]], device="cuda", dtype=torch.int32)
    slots, lengths = compute_global_topk_indices_and_lens(local, req, table, 64, valid)
    expected = torch.tensor(
        [[128, 191, -1, -1, -1, -1], [0, 63, 64, -1, -1, -1], [-1] * 6, [-1] * 6],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(slots, expected, rtol=0, atol=0)
    assert lengths.tolist() == [2, 3, 0, 0]
