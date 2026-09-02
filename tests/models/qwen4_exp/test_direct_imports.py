# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm.models.qwen4_exp.nvidia.qsa",
        "vllm.models.qwen4_exp.nvidia.model",
    ],
)
def test_qwen4_exp_nvidia_modules_support_direct_import(module_name: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                f"importlib.import_module({module_name!r}); "
                "print('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
