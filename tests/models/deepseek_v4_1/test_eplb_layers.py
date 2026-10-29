# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB wiring of the DeepSeek V4.1 MoE layers; CPU only.

Target layers keep the global EPLB settings, DSpark drafter layers run without
EPLB, the expert mapping lists the redundant physical slots, and EPLB moves
expert rows through uint8 views so dtypes NCCL and gloo reject
(float8_e8m0fnu Marlin scales) can still be gathered and sent.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.config import ParallelConfig
from vllm.config.parallel import EPLBConfig
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV4Model,
    _eplb_byte_view,
    _eplb_for_layer,
    _moe_parallel_config,
)

NUM_HIDDEN_LAYERS = 40
DRAFT_LAYERS = range(NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS + 3)
LOCAL_EXPERTS = 5


def _eplb_parallel_config(num_redundant_experts: int) -> ParallelConfig:
    pc = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
    pc.enable_eplb = True
    pc.eplb_config = EPLBConfig(
        num_redundant_experts=num_redundant_experts,
        window_size=100,
        step_interval=300,
        log_balancedness=True,
    )
    return pc


def test_eplb_for_layer_turns_off_for_drafter_layers():
    for r in (0, 2, 8):
        for i in range(NUM_HIDDEN_LAYERS):
            assert _eplb_for_layer(i, NUM_HIDDEN_LAYERS, True, r) == (True, r)
        for i in DRAFT_LAYERS:
            assert _eplb_for_layer(i, NUM_HIDDEN_LAYERS, True, r) == (False, 0)
    for i in [*range(NUM_HIDDEN_LAYERS), *DRAFT_LAYERS]:
        assert _eplb_for_layer(i, NUM_HIDDEN_LAYERS, False, 0) == (False, 0)


def test_target_layers_keep_the_global_config_object():
    pc = _eplb_parallel_config(8)
    for i in range(NUM_HIDDEN_LAYERS):
        assert _moe_parallel_config(pc, i, NUM_HIDDEN_LAYERS) is pc


def test_drafter_layers_get_an_eplb_off_copy():
    pc = _eplb_parallel_config(8)
    eplb_before = pc.eplb_config
    for i in DRAFT_LAYERS:
        d = _moe_parallel_config(pc, i, NUM_HIDDEN_LAYERS)
        assert d is not pc and d.eplb_config is not pc.eplb_config
        assert d.enable_eplb is False
        assert d.eplb_config.num_redundant_experts == 0
        assert d.tensor_parallel_size == 8 and d.enable_expert_parallel is True
        assert d.eplb_config.window_size == 100
        assert d.eplb_config.step_interval == 300
        assert d.eplb_config.communicator == pc.eplb_config.communicator
    assert pc.enable_eplb is True
    assert pc.eplb_config is eplb_before
    assert pc.eplb_config.num_redundant_experts == 8


def test_eplb_off_globally_is_identity():
    pc = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=True)
    assert pc.enable_eplb is False
    for i in [0, NUM_HIDDEN_LAYERS - 1, *DRAFT_LAYERS]:
        assert _moe_parallel_config(pc, i, NUM_HIDDEN_LAYERS) is pc


@pytest.mark.parametrize("num_redundant_experts", [0, 8])
def test_expert_mapping_lists_redundant_slots(num_redundant_experts):
    """Physical slots n..n+R-1 must be in the mapping or they are never loaded
    and EPLB serves zero weights from them."""
    n_routed_experts = 384
    layer = SimpleNamespace(
        ffn=SimpleNamespace(
            use_mega_moe=False, n_redundant_experts=num_redundant_experts
        )
    )
    model = SimpleNamespace(
        layers=[layer],
        start_layer=0,
        end_layer=1,
        config=SimpleNamespace(n_routed_experts=n_routed_experts),
        named_parameters=lambda: iter(()),
    )
    mapping = DeepseekV4Model.get_expert_mapping(model)
    expert_ids = {expert_id for _, _, expert_id, _ in mapping}
    assert expert_ids == set(range(n_routed_experts + num_redundant_experts))
    for shard in ("w1", "w2", "w3"):
        assert sum(1 for *_, shard_id in mapping if shard_id == shard) == (
            n_routed_experts + num_redundant_experts
        )


def _expert_tensors(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 255, (LOCAL_EXPERTS, 96), dtype=torch.uint8, generator=g)
    return {
        "int32_marlin_weight": torch.randint(
            -(2**31), 2**31 - 1, (LOCAL_EXPERTS, 48), dtype=torch.int32, generator=g
        ),
        "e8m0_scale": raw[:, :40].contiguous().view(torch.float8_e8m0fnu),
        "bf16": torch.randn(LOCAL_EXPERTS, 24, generator=g).to(torch.bfloat16),
        "fp32": torch.randn(LOCAL_EXPERTS, 12, generator=g),
        "uint8": raw[:, 40:].contiguous(),
    }


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_byte_view_aliases_storage():
    for name, w in _expert_tensors(0).items():
        raw = _eplb_byte_view(w)
        assert raw.dtype == torch.uint8, name
        assert raw.shape == (LOCAL_EXPERTS, w.shape[1] * w.element_size()), name
        assert raw.data_ptr() == w.data_ptr(), name
        assert raw.is_contiguous(), name


def test_byte_view_row_copy_moves_one_expert():
    src, dst = _expert_tensors(1), _expert_tensors(2)
    for name in src:
        before = dst[name].clone()
        _eplb_byte_view(dst[name])[3].copy_(_eplb_byte_view(src[name])[1])
        assert _same_bits(dst[name][3], src[name][1]), name
        for row in (0, 1, 2, 4):
            assert _same_bits(dst[name][row], before[row]), name


def test_byte_view_rejects_bad_layouts():
    w = torch.zeros(LOCAL_EXPERTS, 8, dtype=torch.float8_e8m0fnu)
    for bad in (w.t(), w.view(LOCAL_EXPERTS, 2, 4), w[:, ::2]):
        with pytest.raises(ValueError):
            _eplb_byte_view(bad)


def _gloo_worker(rank: int, init_file: str, result_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    mine = _expert_tensors(10 + rank)
    peer = 1 - rank
    recv = {k: torch.empty_like(v) for k, v in mine.items()}
    ops = []
    for k in mine:
        ops.append(dist.P2POp(dist.isend, _eplb_byte_view(mine[k]), peer))
        ops.append(dist.P2POp(dist.irecv, _eplb_byte_view(recv[k]), peer))
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    expected = _expert_tensors(10 + peer)
    ok = all(_same_bits(recv[k], expected[k]) for k in mine)
    for k, w in mine.items():
        gathered = [torch.empty_like(_eplb_byte_view(w)) for _ in range(2)]
        dist.all_gather(gathered, _eplb_byte_view(w))
        ok = ok and all(
            _same_bits(gathered[r].view(w.dtype), _expert_tensors(10 + r)[k])
            for r in range(2)
        )
    if rank == 0:
        with open(result_file, "w") as f:
            f.write("ok" if ok else "mismatch")
    dist.destroy_process_group()
    assert ok


def test_gloo_transfers_byte_views_of_every_expert_dtype():
    """The staged gloo communicator and the profile all_gather move the byte
    views; the bytes read back as the original tensors on the other side."""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    with tempfile.TemporaryDirectory() as d:
        init_file, result_file = f"{d}/init", f"{d}/result"
        mp.spawn(_gloo_worker, args=(init_file, result_file), nprocs=2, join=True)
        with open(result_file) as f:
            assert f.read() == "ok"
