# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import vllm.models.deepseek_v4.vision_model as vision_model_module
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.deepseek_v4.vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEWLINE,
    IMAGE_PAD,
    IMAGE_START,
    DeepseekV4Aligner,
    DeepseekV4VisionMLP,
    DeepseekV4VisionTransformer,
    build_image_sentinel_embeddings,
)
from vllm.models.deepseek_v4.vision_model import (
    DeepseekV4ForConditionalGeneration,
    DeepseekV4VisionForCausalLM,
)


def _patch_tp(monkeypatch):
    monkeypatch.setattr("vllm.models.deepseek_v4.vision.get_tp_size", lambda: 1)
    monkeypatch.setattr("vllm.models.deepseek_v4.vision.get_tp_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _vision_config(**overrides):
    values = dict(
        vision_n_layers=1,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=16,
        vision_patch_size=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        hidden_size=12,
        dim=12,
        rms_norm_eps=1e-6,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _load_official_vision_module():
    path = Path(
        "/home/yyf/data2/models/deepseek-ai/"
        "DeepSeek-V4-Flash-Vision-Exp/inference/vision.py"
    )
    if not path.exists():
        pytest.skip("DeepSeek-V4 Vision official inference code is unavailable")
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_official_vision",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deepseek_v4_vision_tower_and_aligner_shapes(
    monkeypatch,
    default_vllm_config,
):
    if not torch.cuda.is_available():
        pytest.skip("DeepSeek V4 vision forward test requires CUDA")
    del default_vllm_config
    _patch_tp(monkeypatch)
    config = _vision_config()
    vision = DeepseekV4VisionTransformer(
        config,
        quant_config=None,
        prefix="vision",
    ).cuda()
    aligner = DeepseekV4Aligner(config, quant_config=None, prefix="aligner").cuda()

    patches = torch.randn(6, 3, 2, 2, device="cuda")
    hidden_states = vision(patches, n_vit_h=2, n_vit_w=3)
    aligned = aligner(hidden_states, n_vit_h=2, n_vit_w=3)

    assert hidden_states.shape == (6, 8)
    assert aligned.shape == (2, 12)


def test_deepseek_v4_vision_tower_matches_official_bf16(
    monkeypatch,
    default_vllm_config,
):
    if not torch.cuda.is_available():
        pytest.skip("DeepSeek V4 vision numerical test requires CUDA")
    del default_vllm_config
    _patch_tp(monkeypatch)
    official = _load_official_vision_module()
    official_get_cos_sin = official.get_vision_cos_sin
    monkeypatch.setattr(
        official,
        "get_vision_cos_sin",
        lambda *args: tuple(tensor.cuda() for tensor in official_get_cos_sin(*args)),
    )
    config = _vision_config(
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=128,
        hidden_size=96,
        dim=96,
    )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        torch.manual_seed(123)
        reference_vision = official.ViT(config).cuda()
        reference_aligner = official.Aligner(config).cuda()
        vision = DeepseekV4VisionTransformer(
            config,
            quant_config=None,
            prefix="vision",
        ).cuda()
        aligner = DeepseekV4Aligner(
            config,
            quant_config=None,
            prefix="aligner",
        ).cuda()
    finally:
        torch.set_default_dtype(previous_dtype)

    vision.load_state_dict(reference_vision.state_dict())
    aligner.load_state_dict(reference_aligner.state_dict())
    patches = torch.randn(
        12,
        3,
        2,
        2,
        dtype=torch.bfloat16,
        device="cuda",
    )

    with torch.inference_mode():
        expected_hidden = reference_vision(patches, 3, 4)
        actual_hidden = vision(patches, 3, 4)
        expected_aligned = reference_aligner(expected_hidden, 3, 4)
        actual_aligned = aligner(actual_hidden, 3, 4)

    torch.testing.assert_close(
        actual_hidden,
        expected_hidden,
        rtol=3e-2,
        atol=4e-2,
    )
    torch.testing.assert_close(
        actual_aligned,
        expected_aligned,
        rtol=2e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_deepseek_v4_vision_mlp_tp_matches_unsharded(
    monkeypatch,
    default_vllm_config,
    tp_size,
):
    del default_vllm_config
    config = _vision_config(vision_dim=8, vision_inter_dim=2816)
    x = torch.randn(5, config.vision_dim)
    w1 = torch.randn(2 * config.vision_inter_dim, config.vision_dim)
    w2 = torch.randn(config.vision_dim, config.vision_inter_dim)
    gate, up = torch.nn.functional.linear(x, w1).chunk(2, dim=-1)
    expected = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2)

    monkeypatch.setattr(
        "vllm.models.deepseek_v4.vision.is_vit_use_data_parallel",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
        lambda output: output,
    )

    partial_outputs: list[torch.Tensor] = []
    for tp_rank in range(tp_size):
        monkeypatch.setattr(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            lambda rank=tp_rank: rank,
        )
        monkeypatch.setattr(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            lambda rank=tp_rank: rank,
        )
        mlp = DeepseekV4VisionMLP(config, prefix="vision.blocks.0.mlp")
        mlp.w1.weight.weight_loader(mlp.w1.weight, w1)
        mlp.w2.weight.weight_loader(mlp.w2.weight, w2)
        shard_size = config.vision_inter_dim // tp_size
        shard_start = tp_rank * shard_size
        expected_w1 = torch.cat(
            [
                w1[shard_start : shard_start + shard_size],
                w1[
                    config.vision_inter_dim + shard_start : config.vision_inter_dim
                    + shard_start
                    + shard_size
                ],
            ]
        )
        torch.testing.assert_close(mlp.w1.weight, expected_w1)
        torch.testing.assert_close(
            mlp.w2.weight,
            w2[:, shard_start : shard_start + shard_size],
        )
        partial_outputs.append(mlp(x))

    torch.testing.assert_close(
        sum(partial_outputs),
        expected,
        rtol=1e-5,
        atol=2e-4,
    )


def test_deepseek_v4_sentinel_embeddings_replace_image_slots():
    image_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    image_start = torch.tensor([10.0, 11.0])
    image_end = torch.tensor([20.0, 21.0])
    image_newline = torch.tensor([30.0, 31.0])
    image_pad = torch.tensor([40.0, 41.0])
    token_types = torch.tensor(
        [IMAGE_PAD, IMAGE_START, IMAGE, IMAGE_NEWLINE, IMAGE, IMAGE_END]
    )
    image_indices = torch.tensor([0, 1])

    embeddings = build_image_sentinel_embeddings(
        image_features=image_features,
        token_types=token_types,
        image_indices=image_indices,
        image_start=image_start,
        image_end=image_end,
        image_newline=image_newline,
        image_pad=image_pad,
    )

    assert torch.equal(embeddings[0], image_pad)
    assert torch.equal(embeddings[1], image_start)
    assert torch.equal(embeddings[2], image_features[0])
    assert torch.equal(embeddings[3], image_newline)
    assert torch.equal(embeddings[4], image_features[1])
    assert torch.equal(embeddings[5], image_end)


def test_deepseek_v4_vision_embed_input_ids_masks_oov_and_preserves_raw_ids():
    class FakeLanguageModel:
        def __init__(self):
            self.seen_input_ids = None

        def embed_input_ids(self, input_ids):
            self.seen_input_ids = input_ids.clone()
            return torch.zeros(input_ids.numel(), 2)

    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100)
    model._has_oov_mm_tokens = True
    model.language_model = FakeLanguageModel()
    model.get_language_model = lambda: model.language_model

    raw_input_ids = torch.tensor([5, 100, 101, 6])
    is_multimodal = raw_input_ids >= 100
    multimodal_embeddings = (torch.ones(2, 2),)
    inputs_embeds = model.embed_input_ids(
        raw_input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    assert torch.equal(
        model.language_model.seen_input_ids,
        torch.tensor([5, 0, 0, 6]),
    )
    assert torch.equal(raw_input_ids, torch.tensor([5, 100, 101, 6]))
    assert torch.equal(inputs_embeds[is_multimodal], torch.ones(2, 2))


def test_deepseek_v4_vision_parse_processor_fields():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100, vision_downsample_ratio=2)

    image_inputs = model._parse_image_inputs(
        pixel_values=torch.arange(24).view(6, 1, 2, 2),
        image_grid_thw=torch.tensor([[1, 2, 3]]),
        num_image_patches=torch.tensor([6]),
        image_replacement_ids=torch.tensor(
            [101, 100, 102, 103, 102, 104],
        ),
        image_indices=torch.tensor([0, 1]),
    )

    assert len(image_inputs) == 1
    assert image_inputs[0].n_vit_h == 2
    assert image_inputs[0].n_vit_w == 3
    assert torch.equal(
        image_inputs[0].token_types,
        torch.tensor(
            [IMAGE_PAD, IMAGE_START, IMAGE, IMAGE_NEWLINE, IMAGE, IMAGE_END],
        ),
    )
    assert torch.equal(image_inputs[0].image_indices, torch.tensor([0, 1]))


def test_deepseek_v4_vision_parse_video_processor_fields():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100)

    video_inputs = model._parse_video_inputs(
        video_pixel_values=torch.arange(28).view(7, 1, 2, 2),
        video_n_vit_h=torch.tensor([1, 2]),
        video_n_vit_w=torch.tensor([3, 2]),
        num_video_frame_patches=torch.tensor([3, 4]),
        video_num_frames=torch.tensor([2]),
        video_token_types=torch.tensor(
            [
                IMAGE_START,
                IMAGE,
                IMAGE_NEWLINE,
                IMAGE,
                IMAGE_END,
                IMAGE_START,
                IMAGE,
                IMAGE_END,
            ]
        ),
        video_image_indices=torch.tensor([0, 1, 2]),
    )

    assert len(video_inputs) == 1
    assert len(video_inputs[0].frames) == 2
    frame_shapes = [
        (frame.n_vit_h, frame.n_vit_w, frame.patches.shape[0])
        for frame in video_inputs[0].frames
    ]
    assert frame_shapes == [
        (1, 3, 3),
        (2, 2, 4),
    ]
    assert torch.equal(
        video_inputs[0].token_types,
        torch.tensor(
            [
                IMAGE_START,
                IMAGE,
                IMAGE_NEWLINE,
                IMAGE,
                IMAGE_END,
                IMAGE_START,
                IMAGE,
                IMAGE_END,
            ]
        ),
    )
    assert torch.equal(video_inputs[0].image_indices, torch.tensor([0, 1, 2]))


def test_deepseek_v4_conditional_generation_alias_matches_registry_name():
    assert DeepseekV4ForConditionalGeneration.__name__ == (
        "DeepseekV4ForConditionalGeneration"
    )
    assert DeepseekV4VisionForCausalLM is DeepseekV4ForConditionalGeneration
    assert hasattr(DeepseekV4ForConditionalGeneration, "_processor_factory")


def test_deepseek_v4_vision_advertises_eagle3_support():
    assert supports_eagle3(DeepseekV4ForConditionalGeneration)


def test_deepseek_v4_vision_placeholder_matches_tokenizer():
    assert (
        DeepseekV4VisionForCausalLM.get_placeholder_str("image", 0)
        == "<｜deepseek_image｜>"
    )
    assert (
        DeepseekV4VisionForCausalLM.get_placeholder_str("video", 0)
        == "<|place_holder_mm_span_0436|>"
    )


def test_deepseek_v4_vision_video_embeddings_concat_frames_per_item():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vocab_size=100)
    model.image_start = torch.nn.Parameter(torch.tensor([10.0, 11.0]))
    model.image_end = torch.nn.Parameter(torch.tensor([20.0, 21.0]))
    model.image_newline = torch.nn.Parameter(torch.tensor([30.0, 31.0]))
    model.image_pad = torch.nn.Parameter(torch.tensor([40.0, 41.0]))

    def fake_encode_image(patches, n_vit_h, n_vit_w):
        del n_vit_h, n_vit_w
        return patches[:, 0, 0, :]

    model.encode_image = fake_encode_image
    embeddings = model.embed_multimodal(
        video_pixel_values=torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
                [[[5.0, 6.0]]],
            ]
        ),
        video_n_vit_h=torch.tensor([1, 1]),
        video_n_vit_w=torch.tensor([2, 1]),
        num_video_frame_patches=torch.tensor([2, 1]),
        video_num_frames=torch.tensor([2]),
        video_token_types=torch.tensor(
            [
                IMAGE_START,
                IMAGE,
                IMAGE,
                IMAGE_END,
                IMAGE_START,
                IMAGE,
                IMAGE_END,
            ]
        ),
        video_image_indices=torch.tensor([0, 1, 2]),
    )

    assert len(embeddings) == 1
    torch.testing.assert_close(
        embeddings[0],
        torch.tensor(
            [
                [10.0, 11.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [20.0, 21.0],
                [10.0, 11.0],
                [5.0, 6.0],
                [20.0, 21.0],
            ]
        ),
    )


def test_deepseek_v4_vision_tower_ignores_global_quant_config(monkeypatch):
    seen_quant_configs = []

    class FakeVision(torch.nn.Module):
        def __init__(self, config, quant_config=None, prefix=""):
            super().__init__()
            del config, prefix
            seen_quant_configs.append(("vision", quant_config))

    class FakeAligner(torch.nn.Module):
        def __init__(self, config, quant_config=None, prefix=""):
            super().__init__()
            del config, prefix
            seen_quant_configs.append(("aligner", quant_config))

    class FakeLanguageModel(torch.nn.Module):
        packed_modules_mapping = {}
        hf_to_vllm_mapper = None

        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            del vllm_config, prefix
            self.make_empty_intermediate_tensors = lambda *args, **kwargs: None

    monkeypatch.setattr(
        vision_model_module,
        "DeepseekV4VisionTransformer",
        FakeVision,
    )
    monkeypatch.setattr(vision_model_module, "DeepseekV4Aligner", FakeAligner)
    monkeypatch.setattr(
        vision_model_module,
        "DeepseekV4ForCausalLM",
        FakeLanguageModel,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.get_offloader",
        lambda: SimpleNamespace(supports_tower_offload=False),
    )

    hf_config = SimpleNamespace(vocab_size=100, hidden_size=8)
    mm_config = SimpleNamespace(
        get_limit_per_prompt=lambda modality: 1,
        mm_encoder_only=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            get_multimodal_config=lambda: mm_config,
        ),
        quant_config=object(),
    )

    DeepseekV4ForConditionalGeneration(vllm_config=vllm_config)

    assert seen_quant_configs == [("vision", None), ("aligner", None)]
    assert (
        vllm_config.model_config.get_multimodal_config().get_limit_per_prompt("video")
        == 1
    )


def test_deepseek_v4_vision_load_weights_splits_vision_and_language():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.image_start = torch.nn.Parameter(torch.zeros(2))
    model.image_end = torch.nn.Parameter(torch.zeros(2))
    model.image_newline = torch.nn.Parameter(torch.zeros(2))
    model.image_pad = torch.nn.Parameter(torch.zeros(2))
    model.process_weights_after_loading = lambda: None

    loaded_vision = set()
    delegated_names: list[str] = []

    class FakeVision:
        def load_weights(self, weights):
            names = [name for name, _ in weights]
            loaded_vision.update(names)
            return set(names)

    class FakeAligner:
        def load_weights(self, weights):
            names = [name for name, _ in weights]
            loaded_vision.update(names)
            return set(names)

    class FakeLanguage:
        def load_weights(self, weights):
            delegated_names.extend(name for name, _ in weights)
            return set(delegated_names)

    model.vision = FakeVision()
    model.aligner = FakeAligner()
    model.language_model = FakeLanguage()

    loaded = model.load_weights(
        [
            ("vision.blocks.0.norm1.weight", torch.ones(2)),
            ("aligner.w1.weight", torch.ones(2, 2)),
            ("image_start", torch.ones(2)),
            ("image_end", torch.ones(2) * 2),
            ("model.layers.0.norm.weight", torch.ones(2)),
            ("lm_head.weight", torch.ones(2, 2)),
        ]
    )

    assert "blocks.0.norm1.weight" in loaded_vision
    assert "w1.weight" in loaded_vision
    assert torch.equal(model.image_start, torch.ones(2))
    assert torch.equal(model.image_end, torch.ones(2) * 2)
    assert delegated_names == ["model.layers.0.norm.weight", "lm_head.weight"]
    assert loaded == {
        "vision.blocks.0.norm1.weight",
        "aligner.w1.weight",
        "image_start",
        "image_end",
        "language_model.model.layers.0.norm.weight",
        "language_model.lm_head.weight",
    }


def test_deepseek_v4_vision_load_weights_rejects_bad_image_sentinel_shape():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.image_start = torch.nn.Parameter(torch.zeros(2))
    model.vision = SimpleNamespace(load_weights=lambda weights: set())
    model.aligner = SimpleNamespace(load_weights=lambda weights: set())
    model.language_model = SimpleNamespace(load_weights=lambda weights: set())
    model.process_weights_after_loading = lambda: None

    with pytest.raises(AssertionError, match="image_start"):
        model.load_weights([("image_start", torch.ones(3))])


def test_deepseek_v4_vision_rejects_precomputed_image_embeddings():
    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)

    with pytest.raises(NotImplementedError, match="layout metadata"):
        model._parse_image_inputs(image_embeds=torch.zeros(1, 4, 8))


def test_deepseek_v4_vision_load_weights_real_index_subset():
    model_dir = "/home/yyf/data2/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
    shard_path = f"{model_dir}/model-00001-of-00048.safetensors"

    try:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            names = [
                "vision.norm.weight",
                "aligner.w1.bias",
                "image_start",
                "embed.weight",
            ]
            available_names = set(f.keys())
            missing = [name for name in names if name not in available_names]
            if missing:
                pytest.skip(f"Missing expected Vision shard tensors: {missing}")
            weights = [(name, f.get_tensor(name)) for name in names]
    except FileNotFoundError:
        pytest.skip("DeepSeek-V4-Flash-Vision-Exp weights are not available locally")

    model = DeepseekV4VisionForCausalLM.__new__(DeepseekV4VisionForCausalLM)
    torch.nn.Module.__init__(model)
    model.image_start = torch.nn.Parameter(
        torch.zeros_like(dict(weights)["image_start"])
    )
    model.process_weights_after_loading = lambda: None

    seen_vision: list[str] = []
    seen_aligner: list[str] = []
    seen_language: list[str] = []

    class FakeVision:
        def load_weights(self, weights):
            seen_vision.extend(name for name, _ in weights)
            return set(seen_vision)

    class FakeAligner:
        def load_weights(self, weights):
            seen_aligner.extend(name for name, _ in weights)
            return set(seen_aligner)

    class FakeLanguage:
        def load_weights(self, weights):
            seen_language.extend(name for name, _ in weights)
            return set(seen_language)

    model.vision = FakeVision()
    model.aligner = FakeAligner()
    model.language_model = FakeLanguage()

    loaded = model.load_weights(weights)

    assert seen_vision == ["norm.weight"]
    assert seen_aligner == ["w1.bias"]
    assert seen_language == ["embed.weight"]
    assert torch.equal(model.image_start, dict(weights)["image_start"])
    assert loaded == {
        "vision.norm.weight",
        "aligner.w1.bias",
        "image_start",
        "language_model.embed.weight",
    }
