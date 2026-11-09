# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 hardware-isolated model entry point."""

from importlib import import_module

from vllm.platforms import current_platform

capability = (
    current_platform.get_device_capability() if current_platform.is_cuda() else None
)
if capability is not None and (
    (capability.major, capability.minor) == (8, 9) or capability.major == 12
):
    _exports = dict.fromkeys(
        (
            "DeepseekV4FP8Config",
            "DeepseekV41ForCausalLM",
            "DSparkDeepseekV4ForCausalLM",
        ),
        "vllm.models.deepseek_v4_1",
    )
else:
    _backend = "amd" if current_platform.is_rocm() else "nvidia"
    _exports = {
        "DeepseekV4FP8Config": f"{__name__}.quant_config",
        "DeepseekV41ForCausalLM": f"{__name__}.{_backend}.vl_model",
        "DSparkDeepseekV4ForCausalLM": f"{__name__}.{_backend}.dspark",
    }


def __getattr__(name: str):
    # Native models also import the shared common package. Defer model imports
    # so importing a common helper does not cycle back into a partial model.
    if name not in _exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_exports[name]), name)
    globals()[name] = value
    return value


__all__ = [
    "DSparkDeepseekV4ForCausalLM",
    "DeepseekV4FP8Config",
    "DeepseekV41ForCausalLM",
]
