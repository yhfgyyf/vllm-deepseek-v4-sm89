# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.model_loader.utils import process_weights_after_loading


class _ModelWithPostLoadHook(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_load_calls = 0

    def process_weights_after_loading(self) -> None:
        self.post_load_calls += 1


def test_process_weights_after_loading_calls_model_hook() -> None:
    model = _ModelWithPostLoadHook()
    model_config = SimpleNamespace(dtype=torch.float32, quantization=None)

    process_weights_after_loading(model, model_config, torch.device("cpu"))

    assert model.post_load_calls == 1
