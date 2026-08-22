# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer


class FakeHfTokenizer:
    vocab_size = 100

    def __len__(self) -> int:
        return self.vocab_size

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        return [len(text)]


@pytest.mark.parametrize(
    "factory",
    [get_deepseek_v4_tokenizer, get_deepseek_v32_tokenizer],
)
def test_deepseek_tokenizer_len_matches_base(factory) -> None:
    tokenizer = factory(FakeHfTokenizer())

    assert len(tokenizer) == 100
    assert tokenizer.get_added_vocab() == {"</think>": 100}
