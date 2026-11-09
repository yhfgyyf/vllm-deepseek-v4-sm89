# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-size arguments of the DeepSeek V4.1 Triton kernels are runtime values.

Three kernels run eagerly at every prefill or decode size: the mHC prenorm
projection, the Engram lookup and the global top-k slot mapping. If a token,
row or request count is a ``tl.constexpr`` (or value-specialized), every new
size compiles a new variant: a stall of several hundred ms per unseen prompt
length and an unbounded Triton cache. Each kernel must serve every size from
one compiled variant and still match its torch reference.
"""

import pytest
import torch

from tests.kernels.test_engram import _make_embedding, _reference_lookup
from vllm.model_executor.kernels.mhc import deepseek_v41 as mhc_native
from vllm.models.deepseek_v4_1.common import engram as engram_module
from vllm.models.deepseek_v4_1.common.ops import cache_utils

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

RUNTIME_BATCH_ARGS = [
    (mhc_native._prenorm_project_kernel, "tokens"),
    (engram_module._engram_lookup_kernel, "num_rows"),
    (engram_module._engram_lookup_kernel, "grid_size"),
    (cache_utils._compute_global_topk_indices_and_lens_kernel, "num_reqs"),
]


def _compiled_variants(kernel) -> int:
    caches = kernel.device_caches.get(torch.cuda.current_device())
    return 0 if caches is None else len(caches[0])


@pytest.mark.parametrize(
    ("kernel", "name"),
    RUNTIME_BATCH_ARGS,
    ids=[f"{k.fn.__name__}:{n}" for k, n in RUNTIME_BATCH_ARGS],
)
def test_batch_size_argument_is_not_specialized(kernel, name):
    param = next(p for p in kernel.params if p.name == name)
    assert not param.is_constexpr
    assert param.do_not_specialize


@cuda
def test_prenorm_gemm_reuses_one_variant_per_split_count():
    torch.manual_seed(0)
    kernel = mhc_native._prenorm_project_kernel
    width, mix_size = 4 * 1024, 24
    fn = torch.randn(mix_size, width, device="cuda") * 1e-4
    before = _compiled_variants(kernel)
    split_counts = set()
    for tokens in (33, 47, 64, 100, 128, 129, 256, 511, 1000, 1024, 1025, 2049, 4100):
        splits = 16 if tokens <= 128 else 4 if tokens <= 1024 else 1
        split_counts.add(splits)
        x = torch.randn(tokens, width, device="cuda").to(torch.bfloat16)
        mixes = torch.empty(splits, tokens, mix_size, device="cuda")
        sqrsum = torch.empty(splits, tokens, device="cuda")
        mhc_native._native_hc_prenorm_gemm(x, fn, mixes, sqrsum)
        x_splits = x.float().view(tokens, splits, width // splits).transpose(0, 1)
        fn_splits = fn.view(mix_size, splits, width // splits).permute(1, 2, 0)
        torch.testing.assert_close(mixes, x_splits @ fn_splits, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            sqrsum, x_splits.square().sum(-1), atol=8.0, rtol=5e-4
        )
    assert _compiled_variants(kernel) - before <= len(split_counts)


@cuda
def test_engram_lookup_reuses_one_variant_across_token_counts():
    layer = _make_embedding(cpu_offload=False)
    kernel = engram_module._engram_lookup_kernel
    before = _compiled_variants(kernel)
    cols, rows = layer.part_n_hash_cols, layer.part_num_embeddings
    for num_tokens in (1, 2, 7, 16, 17, 31, 100, 255, 256, 431, 1000, 2047, 4093):
        for background in (False, True):
            ids = torch.randint(
                0, rows, (num_tokens, cols), dtype=torch.int32, device="cuda"
            )
            expected = _reference_lookup(
                layer.weight.cuda(),
                layer.weight_scale_inv.cuda(),
                ids,
                layer.vocab_start_idx,
                layer.vocab_end_idx,
            )
            out = torch.empty(
                num_tokens, cols, layer.dim, dtype=torch.bfloat16, device="cuda"
            )
            layer.lookup(ids, out, background=background)
            assert torch.equal(out, expected), (num_tokens, background)
    assert _compiled_variants(kernel) - before <= 1


def _reference_global_slots(topk, token_to_req, table, block_size, valid):
    num_reqs, width = table.shape
    req = token_to_req.long()
    token_ok = valid & (req >= 0) & (req < num_reqs)
    blocks = torch.div(topk, block_size, rounding_mode="floor")
    ok = (topk >= 0) & token_ok[:, None] & (blocks < width)
    numbers = table[req.clamp(0, num_reqs - 1)[:, None], blocks.clamp(0, width - 1)]
    ok &= numbers >= 0
    slots = torch.where(ok, numbers * block_size + topk % block_size, -1)
    return slots.to(torch.int32), ok.sum(-1, dtype=torch.int32)


@cuda
def test_global_slot_mapping_reuses_one_variant_across_batch_sizes():
    torch.manual_seed(0)
    kernel = cache_utils._compute_global_topk_indices_and_lens_kernel
    width, topk, block_size = 300, 512, 64
    # The block table is a row slice of one persistent buffer, as in serving.
    persistent = torch.randint(-1, 5000, (32, width), device="cuda", dtype=torch.int32)
    before = _compiled_variants(kernel)
    for num_reqs in (1, 2, 3, 5, 8, 13, 16, 17, 31, 32):
        for num_tokens in (1, 16, 33, 500):
            table = persistent[:num_reqs]
            local = torch.randint(
                -1,
                width * block_size + 50,
                (num_tokens, topk),
                device="cuda",
                dtype=torch.int32,
            )
            token_to_req = torch.randint(
                -1, num_reqs + 1, (num_tokens,), device="cuda", dtype=torch.int32
            )
            valid = torch.rand(num_tokens, device="cuda") > 0.1
            slots, lens = cache_utils.compute_global_topk_indices_and_lens(
                local, token_to_req, table, block_size, valid
            )
            ref_slots, ref_lens = _reference_global_slots(
                local, token_to_req, table, block_size, valid
            )
            torch.testing.assert_close(slots, ref_slots, rtol=0, atol=0)
            torch.testing.assert_close(lens, ref_lens, rtol=0, atol=0)
    assert _compiled_variants(kernel) - before <= 1
