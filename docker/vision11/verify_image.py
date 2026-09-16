#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the installed image payload, toolchain, and optional GPU loaders."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import regex as re

EXPECTED_PREFIXES = {"vllm": "vllm/", "flashinfer": "flashinfer/"}
EXPECTED_VERSIONS = {
    "torch": "2.13.0+cu130",
    "triton": "3.7.1",
    "transformers": "5.16.1",
}
NATIVE_PAYLOADS = {
    "vllm/_C_stable_libtorch.abi3.so",
    "vllm/_flashkda_C.abi3.so",
    "vllm/_moe_C_stable_libtorch.abi3.so",
    "vllm/_qutlass_C.abi3.so",
    "vllm/_rust_tool_parser.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/fs_io_C.abi3.so",
    "vllm/spinloop.abi3.so",
    "vllm/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so",
    "vllm/vllm-rs",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_REVISION = re.compile(r"[0-9a-f]{40}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="also initialize CUDA and load the vLLM custom ops and DeepGEMM",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def module_path(module: object) -> Path:
    path = getattr(module, "__file__", None)
    require(path is not None, f"module has no file: {module!r}")
    return Path(path).resolve()


def distribution(name: str) -> importlib.metadata.Distribution:
    try:
        return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        if name == "flashinfer-python":
            return importlib.metadata.distribution("flashinfer_python")
        raise


def validate_manifest(manifest: Any) -> dict[str, Any]:
    require(isinstance(manifest, dict), "manifest must be a JSON object")
    required = {
        "vllm_version",
        "flashinfer_version",
        "source_revision",
        "wheel_source_revision",
        "vllm_wheel_sha256",
        "flashinfer_wheel_sha256",
        "payloads",
    }
    require(required <= manifest.keys(), "manifest is missing required fields")
    for field in ("source_revision", "wheel_source_revision"):
        require(
            isinstance(manifest[field], str)
            and HEX_REVISION.fullmatch(manifest[field]) is not None,
            f"invalid {field}",
        )
    for field in ("vllm_wheel_sha256", "flashinfer_wheel_sha256"):
        require(
            isinstance(manifest[field], str)
            and HEX_SHA256.fullmatch(manifest[field]) is not None,
            f"invalid {field}",
        )
    payloads = manifest["payloads"]
    require(
        isinstance(payloads, dict) and set(payloads) == set(EXPECTED_PREFIXES),
        "manifest payloads must contain exactly vllm and flashinfer",
    )
    for package, prefix in EXPECTED_PREFIXES.items():
        entries = payloads[package]
        require(isinstance(entries, dict) and entries, f"empty {package} payload")
        for name, digest in entries.items():
            path = PurePosixPath(name)
            require(
                isinstance(name, str)
                and name.startswith(prefix)
                and not path.is_absolute()
                and ".." not in path.parts,
                f"unsafe or misplaced {package} payload: {name!r}",
            )
            require(
                isinstance(digest, str) and HEX_SHA256.fullmatch(digest) is not None,
                f"invalid SHA256 for {name}",
            )
    return manifest


def record_payload(dist: importlib.metadata.Distribution, prefix: str) -> set[str]:
    files = dist.files
    require(files is not None, f"{dist.metadata['Name']} has no installed RECORD")
    return {str(path) for path in files if str(path).startswith(prefix)}


def physical_payload(site: Path, prefix: str) -> set[str]:
    root = site / prefix.removesuffix("/")
    require(root.is_dir(), f"missing installed package directory: {root}")
    return {
        path.relative_to(site).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def verify_payloads(
    manifest: dict[str, Any], site: Path
) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    distributions = {
        "vllm": distribution("vllm"),
        "flashinfer": distribution("flashinfer-python"),
    }
    for package, prefix in EXPECTED_PREFIXES.items():
        expected = manifest["payloads"][package]
        expected_names = set(expected)
        recorded = record_payload(distributions[package], prefix)
        require(
            recorded == expected_names,
            f"{package} RECORD payload mismatch: "
            f"missing={sorted(expected_names - recorded)[:20]!r}, "
            f"unexpected={sorted(recorded - expected_names)[:20]!r}",
        )
        physical = physical_payload(site, prefix)
        require(
            physical == expected_names,
            f"{package} installed payload mismatch: "
            f"missing={sorted(expected_names - physical)[:20]!r}, "
            f"unexpected={sorted(physical - expected_names)[:20]!r}",
        )
        for name, expected_digest in expected.items():
            path = (site / name).resolve()
            require(path.is_relative_to(site), f"payload escapes site-packages: {name}")
            require(path.is_file(), f"missing payload: {name}")
            require(
                file_sha256(path) == expected_digest,
                f"payload hash mismatch: {name}",
            )
        counts[package] = len(expected)

    vllm_payload = manifest["payloads"]["vllm"]
    actual_native = {
        name for name in vllm_payload if name.endswith(".so") or name == "vllm/vllm-rs"
    }
    require(
        actual_native == NATIVE_PAYLOADS,
        "audited native payload mismatch: "
        f"missing={sorted(NATIVE_PAYLOADS - actual_native)!r}, "
        f"unexpected={sorted(actual_native - NATIVE_PAYLOADS)!r}",
    )
    require(
        "vllm/vllm_flashmla_C.abi3.so" not in vllm_payload,
        "unexpected generic vllm_flashmla_C payload",
    )
    deep_gemm_count = sum(
        name.startswith("vllm/third_party/deep_gemm/") for name in vllm_payload
    )
    return counts, {
        "native_artifacts": len(actual_native),
        "deep_gemm_payloads": deep_gemm_count,
    }


def verify_not_editable(site: Path, distributions: list[str]) -> None:
    editable = sorted(site.glob("__editable__*.pth")) + sorted(site.glob("*.egg-link"))
    require(not editable, f"editable-install artifacts found: {editable!r}")
    for pth in site.glob("*.pth"):
        text = pth.read_text(errors="replace").lower()
        require("__editable__" not in text, f"editable path hook found: {pth}")
        require("/home/" not in text, f"host path found in path hook: {pth}")
        require("/workspace" not in text, f"workspace path found in path hook: {pth}")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "import ")):
                continue
            path_entry = (pth.parent / line).resolve()
            require(
                path_entry.is_relative_to(site),
                f"external path entry found in {pth}: {line}",
            )
    for name in distributions:
        dist = distribution(name)
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            require(
                not direct_url.get("dir_info", {}).get("editable", False),
                f"editable distribution: {name}",
            )


def require_tool(name: str) -> Path:
    path = shutil.which(name)
    require(path is not None, f"required tool not found: {name}")
    return Path(path).resolve()


def command_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return " | ".join(lines[-3:])


def verify_toolchain(site: Path) -> dict[str, Any]:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    nvcc = cuda_home / "bin/nvcc"
    require(nvcc.is_file(), f"missing nvcc: {nvcc}")
    cxx = require_tool("c++")
    tools = {
        name: str(require_tool(name))
        for name in ("cmake", "ninja", "cuobjdump", "ffmpeg")
    }
    tools.update({"nvcc": str(nvcc), "cxx": str(cxx)})

    cuda_include = cuda_home / "include"
    for header in (
        "cuda_runtime.h",
        "curand.h",
        "curand_kernel.h",
        "cublas_v2.h",
        "nvrtc.h",
    ):
        require((cuda_include / header).is_file(), f"missing CUDA header: {header}")
    cuda_lib_roots = [cuda_home / "lib64", cuda_home / "targets/x86_64-linux/lib"]
    for library in ("libcurand.so*", "libcublas.so*", "libnvrtc.so*"):
        require(
            any(root.exists() and any(root.glob(library)) for root in cuda_lib_roots),
            f"missing CUDA library: {library}",
        )

    python_include = Path(sysconfig.get_path("include")).resolve()
    require((python_include / "Python.h").is_file(), "missing Python.h")
    tvm_include = site / "tvm_ffi/include"
    require((tvm_include / "tvm/ffi/tvm_ffi.h").is_file(), "missing TVM FFI headers")
    require((tvm_include / "dlpack/dlpack.h").is_file(), "missing DLPack headers")

    flashinfer_data = site / "flashinfer/data"
    include_dirs = [
        python_include,
        tvm_include,
        flashinfer_data / "include",
        flashinfer_data / "cutlass/include",
        flashinfer_data / "cccl/cub",
        flashinfer_data / "cccl/libcudacxx/include",
        flashinfer_data / "cccl/thrust",
        flashinfer_data / "spdlog/include",
    ]
    required_headers = [
        flashinfer_data / "include/flashinfer/attention/hopper.cuh",
        flashinfer_data / "cutlass/include/cutlass/cutlass.h",
        flashinfer_data / "cccl/cub/cub/cub.cuh",
        flashinfer_data / "spdlog/include/spdlog/spdlog.h",
    ]
    for header in required_headers:
        require(header.is_file(), f"missing vendored JIT header: {header}")
    for include_dir in include_dirs:
        require(include_dir.is_dir(), f"missing include directory: {include_dir}")

    source = r"""
#include <Python.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nvrtc.h>

int main(int argc, char**) {
  curandStatePhilox4_32_10_t state{};
  if (argc == 99) {
    cublasHandle_t cublas;
    curandGenerator_t curand;
    nvrtcProgram program;
    cublasCreate(&cublas);
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    nvrtcCreateProgram(&program, "", "probe.cu", 0, nullptr, nullptr);
  }
  return sizeof(state) == 0 || PY_MAJOR_VERSION != 3;
}
"""
    with tempfile.TemporaryDirectory(prefix="vllm-image-toolchain-") as temp:
        temp_path = Path(temp)
        source_path = temp_path / "probe.cu"
        output_path = temp_path / "probe"
        source_path.write_text(source)
        command = [
            str(nvcc),
            "-std=c++17",
            "-ccbin",
            str(cxx),
            str(source_path),
            "-o",
            str(output_path),
            f"-I{python_include}",
        ]
        command.extend(("-lcublas", "-lcurand", "-lnvrtc"))
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        require(output_path.is_file(), "nvcc compile/link probe produced no binary")

    return {
        "cuda_home": str(cuda_home),
        "nvcc_version": command_version([str(nvcc), "--version"]),
        "cxx_version": command_version([str(cxx), "--version"]),
        "tools": tools,
        "cuda_compile_link_probe": "passed",
        "python_header": str(python_include / "Python.h"),
        "ffi_include": str(tvm_include),
        "flashinfer_data": str(flashinfer_data),
    }


def verify_imports(site: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    modules = {
        name: importlib.import_module(name)
        for name in ("vllm", "flashinfer", "torch", "triton", "transformers")
    }
    for name, module in modules.items():
        path = module_path(module)
        require(
            path.is_relative_to(site),
            f"{name} imported outside image venv: {path}",
        )

    versions = {
        "vllm": distribution("vllm").version,
        "flashinfer": distribution("flashinfer-python").version,
        **{
            package: importlib.metadata.version(package)
            for package in EXPECTED_VERSIONS
        },
    }
    require(versions["vllm"] == manifest["vllm_version"], "vLLM version mismatch")
    require(
        versions["flashinfer"] == manifest["flashinfer_version"],
        "FlashInfer version mismatch",
    )
    for package, expected in EXPECTED_VERSIONS.items():
        require(versions[package] == expected, f"{package} version mismatch")
    require(
        getattr(modules["vllm"], "__version__", None) == versions["vllm"],
        "vLLM module and metadata versions differ",
    )
    require(
        getattr(modules["flashinfer"], "__version__", None) == versions["flashinfer"],
        "FlashInfer module and metadata versions differ",
    )
    require(
        getattr(modules["torch"].version, "cuda", None) == "13.0",
        "torch CUDA mismatch",
    )
    return {
        "module_paths": {
            name: str(module_path(module)) for name, module in modules.items()
        },
        "versions": versions,
    }


def verify_gpu_loaders(site: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    require(torch.cuda.is_available(), "--gpu requested but CUDA is unavailable")
    device_count = torch.accelerator.device_count()
    require(device_count > 0, "--gpu requested but no CUDA device is visible")
    capability = torch.cuda.get_device_capability(0)

    custom_ops = importlib.import_module("vllm._custom_ops")
    stable_extension = importlib.import_module("vllm._C_stable_libtorch")
    deep_gemm = importlib.import_module("vllm.third_party.deep_gemm")
    deep_gemm_extension = importlib.import_module("vllm.third_party.deep_gemm._C")
    for module in (custom_ops, stable_extension, deep_gemm, deep_gemm_extension):
        path = module_path(module)
        require(path.is_relative_to(site), f"GPU module loaded outside venv: {path}")
    require(hasattr(torch.ops._C, "rms_norm"), "vLLM RMSNorm op is absent")
    inputs = torch.arange(512, device="cuda", dtype=torch.float32).reshape(4, 128)
    inputs = (inputs / 17).to(torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(inputs)
    custom_ops.rms_norm(output, inputs, weight, 1e-6)
    reference = inputs.float() * torch.rsqrt(
        inputs.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)
    torch.accelerator.synchronize()
    return {
        "device_count": device_count,
        "device_name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "custom_ops": str(module_path(custom_ops)),
        "stable_extension": str(module_path(stable_extension)),
        "deep_gemm": str(module_path(deep_gemm)),
        "deep_gemm_extension": str(module_path(deep_gemm_extension)),
        "native_rms_norm_matches_reference": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(sys.version_info[:2] == (3, 12), "image must use Python 3.12")
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(sys.flags.no_user_site == 1, "user site-packages must be disabled")
    prefix = Path(sys.prefix).resolve()
    require(prefix == Path("/opt/venv"), f"unexpected Python environment: {prefix}")
    site = Path(sysconfig.get_path("purelib")).resolve()
    require(site.is_relative_to(prefix), f"site-packages escapes image venv: {site}")
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if (path / "vllm").is_dir():
            require(
                path == site,
                f"source checkout is importable outside image venv: {path}",
            )

    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    payload_counts, payload_details = verify_payloads(manifest, site)
    verify_not_editable(site, ["vllm", "flashinfer-python"])
    toolchain = verify_toolchain(site)

    # Avoid FlashInfer probing CUDA while performing the default CPU-only audit.
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0")
    imports = verify_imports(site, manifest)
    gpu = verify_gpu_loaders(site) if args.gpu else None
    return {
        "passed": True,
        "manifest": str(args.manifest.resolve()),
        "python": sys.version.split()[0],
        "environment": str(prefix),
        "site_packages": str(site),
        "source_revision": manifest["source_revision"],
        "wheel_source_revision": manifest["wheel_source_revision"],
        "wheel_sha256": {
            "vllm": manifest["vllm_wheel_sha256"],
            "flashinfer": manifest["flashinfer_wheel_sha256"],
        },
        "payload_files_checked": payload_counts,
        "payload_details": payload_details,
        "imports_and_versions": imports,
        "editable_or_host_source_imports": False,
        "toolchain": toolchain,
        "gpu": gpu,
    }


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except Exception as error:
        report = {
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if isinstance(
            error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)
        ):
            output = error.output
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            report["compiler_output"] = output
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
