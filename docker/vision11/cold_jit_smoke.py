#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run cold SM120 FlashInfer/Triton and vLLM media smoke checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

tl: Any = None
active_cache_root: Path | None = None


def _vector_add_kernel(x_ptr, y_ptr, output_ptr, count, BLOCK):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="new or empty directory in which cold caches are retained",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="run only the CUDA/JIT checks (video checks are mandatory by default)",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def snapshot(root: Path, suffix: str | None = None) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or (suffix is not None and path.suffix != suffix):
            continue
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = {
            "sha256": file_sha256(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def prepare_cache_root(requested: Path | None) -> tuple[Path, dict[str, Path]]:
    global active_cache_root
    if requested is None:
        root = Path(tempfile.mkdtemp(prefix="vllm-cold-jit-smoke-"))
    else:
        root = requested.resolve()
        if root.exists():
            require(root.is_dir(), f"cache root is not a directory: {root}")
            require(not any(root.iterdir()), f"cache root is not empty: {root}")
        else:
            root.mkdir(parents=True)
    active_cache_root = root

    caches = {
        "flashinfer": root / "flashinfer-workspace",
        "triton": root / "triton",
        "torch_inductor": root / "torch-inductor",
        "torch_extensions": root / "torch-extensions",
        "vllm": root / "vllm",
        "deep_gemm": root / "deep-gemm",
        "cuda": root / "cuda",
    }
    require(not any(path.exists() for path in caches.values()), "caches are not cold")
    env_paths = {
        "FLASHINFER_WORKSPACE_BASE": caches["flashinfer"],
        "TRITON_CACHE_DIR": caches["triton"],
        "TORCHINDUCTOR_CACHE_DIR": caches["torch_inductor"],
        "TORCH_EXTENSIONS_DIR": caches["torch_extensions"],
        "VLLM_CACHE_ROOT": caches["vllm"],
        "DG_JIT_CACHE_DIR": caches["deep_gemm"],
        "CUDA_CACHE_PATH": caches["cuda"],
    }
    for name, path in env_paths.items():
        os.environ[name] = str(path)
    os.environ["MAX_JOBS"] = "4"
    os.environ["FLASHINFER_NVCC_THREADS"] = "2"
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "12.0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    return root, caches


def configure_bundled_ptxas(site: Path) -> Path:
    ptxas = site / "triton/backends/nvidia/bin/ptxas-blackwell"
    require(ptxas.is_file() and os.access(ptxas, os.X_OK), f"missing {ptxas}")
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = str(ptxas)
    return ptxas


def verify_cublas(torch: Any) -> dict[str, Any]:
    left = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 17
    right = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 13
    expected = left @ right
    actual = left.cuda() @ right.cuda()
    torch.accelerator.synchronize()
    actual_cpu = actual.cpu()
    torch.testing.assert_close(actual_cpu, expected, rtol=1e-5, atol=1e-5)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float((actual_cpu - expected).abs().max()),
    }


def top_k_reference(torch: Any, logits: Any, top_k: Any) -> Any:
    expected = torch.full_like(logits, -torch.inf)
    for row, count in enumerate(top_k.cpu().tolist()):
        indices = torch.topk(logits[row], int(count)).indices
        expected[row, indices] = logits[row, indices]
    return expected


def samples_are_in_top_k(torch: Any, logits: Any, top_k: Any, samples: Any) -> bool:
    for row, count in enumerate(top_k.cpu().tolist()):
        allowed = torch.topk(logits[row], int(count)).indices
        if not bool((allowed == samples[row]).any().item()):
            return False
    return True


def verify_flashinfer(torch: Any, cache: Path) -> dict[str, Any]:
    flashinfer = importlib.import_module("flashinfer")
    sampling = importlib.import_module("flashinfer.sampling")
    jit_sampling = importlib.import_module("flashinfer.jit.sampling")
    spec = jit_sampling.gen_sampling_module()
    require(spec.is_aot is False, "sampling unexpectedly resolved to an AOT module")
    require(not spec.jit_library_path.exists(), "sampling JIT artifact was not cold")
    initial_files = snapshot(cache)
    require(
        all(
            Path(name).name == "flashinfer_jit.log" and details["size"] == 0
            for name, details in initial_files.items()
        ),
        "FlashInfer compiled cache was populated before first call",
    )

    batch, vocab = 4, 257
    logits = torch.linspace(-4.0, 7.0, batch * vocab, device="cuda").reshape(
        batch, vocab
    )
    top_k = torch.tensor([1, 3, 7, 16], dtype=torch.int32, device="cuda")
    expected_mask = top_k_reference(torch, logits, top_k)

    masked = sampling.top_k_mask_logits(logits, top_k)
    torch.accelerator.synchronize()
    require(torch.equal(masked, expected_mask), "FlashInfer top-k mask mismatch")

    sample = sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        1.0,
        deterministic=True,
        seed=20260916,
        offset=0,
    )
    torch.accelerator.synchronize()
    require(tuple(sample.shape) == (batch,), f"invalid sampling shape: {sample.shape}")
    require(sample.is_cuda, "sampling output is not a CUDA tensor")
    require(sample.dtype == torch.int32, f"invalid sampling dtype: {sample.dtype}")
    require(
        samples_are_in_top_k(torch, logits, top_k, sample),
        "sample fell outside its top-k set",
    )

    first = snapshot(cache, ".so")
    require(first, "FlashInfer cold calls produced no cached shared library")
    require(spec.jit_library_path.is_file(), "sampling JIT library is absent")

    masked_again = sampling.top_k_mask_logits(logits, top_k)
    sample_again = sampling.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        1.0,
        deterministic=True,
        seed=20260916,
        offset=0,
    )
    torch.accelerator.synchronize()
    require(torch.equal(masked_again, expected_mask), "reused top-k mask mismatch")
    require(torch.equal(sample_again, sample), "seeded sampling is not deterministic")
    second = snapshot(cache, ".so")
    require(second == first, "FlashInfer second call rebuilt or changed JIT artifacts")

    return {
        "version": flashinfer.__version__,
        "jit_is_aot": spec.is_aot,
        "sampling_library": str(spec.jit_library_path),
        "cached_shared_objects": first,
        "masking_matches_reference": True,
        "samples": sample.cpu().tolist(),
        "seeded_repeat_matches": True,
    }


def verify_triton(torch: Any, cache: Path, ptxas: Path) -> dict[str, Any]:
    global tl
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    require(not snapshot(cache), "Triton cache was populated before first launch")

    _vector_add_kernel.__annotations__["BLOCK"] = tl.constexpr
    vector_add = triton.jit(_vector_add_kernel)

    count = 98_432
    x = torch.arange(count, dtype=torch.float32, device="cuda") / 101
    y = torch.flip(x, dims=(0,))
    output = torch.empty_like(x)
    grid = (triton.cdiv(count, 256),)
    vector_add[grid](x, y, output, count, BLOCK=256)
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    first = snapshot(cache)
    require(first, "Triton launch produced no cache artifacts")
    require(
        any(name.endswith(".cubin") for name in first),
        "Triton cache contains no compiled cubin",
    )

    output.fill_(float("nan"))
    vector_add[grid](x, y, output, count, BLOCK=256)
    torch.accelerator.synchronize()
    torch.testing.assert_close(output, x + y, rtol=0, atol=0)
    second = snapshot(cache)
    require(second == first, "Triton second launch changed its compiled cache")

    version = subprocess.run(
        [str(ptxas), "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    ).stdout.strip()
    return {
        "version": triton.__version__,
        "elements": count,
        "block": 256,
        "masked_tail_elements": count % 256,
        "matches_reference": True,
        "cache_files": first,
        "bundled_ptxas_blackwell": str(ptxas),
        "ptxas_version": version,
    }


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    require(result.returncode == 0, f"ffmpeg failed: {result.stdout.strip()}")


def verify_media() -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    require(ffmpeg is not None, "ffmpeg is not installed")
    import numpy as np

    torchcodec = importlib.import_module("torchcodec")
    from vllm.multimodal.media import ImageMediaIO
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoBackend

    with tempfile.TemporaryDirectory(prefix="vllm-image-video-smoke-") as temp:
        temp_path = Path(temp)
        video_path = temp_path / "eight-frames.mp4"
        image_path = temp_path / "first-frame.png"
        run_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "nullsrc=size=48x32:rate=8,geq=lum='N*20':cb=128:cr=128",
                "-frames:v",
                "8",
                "-c:v",
                "mpeg4",
                "-q:v",
                "1",
                str(video_path),
            ]
        )
        run_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(image_path),
            ]
        )

        image = ImageMediaIO().load_file(image_path).media
        require(image.size == (48, 32), f"unexpected decoded image size: {image.size}")
        require(image.mode == "RGB", f"unexpected decoded image mode: {image.mode}")

        data = video_path.read_bytes()
        loader = VIDEO_LOADER_REGISTRY.load("opencv")
        require(
            isinstance(loader, VideoBackend),
            "opencv registry entry is not a VideoBackend instance",
        )
        default_frames, default_metadata = loader.load_bytes(data, num_frames=4)
        opencv_frames, opencv_metadata = loader.load_bytes(
            data, num_frames=4, backend="opencv"
        )
        torchcodec_frames, torchcodec_metadata = loader.load_bytes(
            data,
            num_frames=4,
            backend="torchcodec",
            num_ffmpeg_threads=2,
        )

        expected_shape = (4, 32, 48, 3)
        for name, frames in (
            ("default", default_frames),
            ("opencv", opencv_frames),
            ("torchcodec", torchcodec_frames),
        ):
            require(tuple(frames.shape) == expected_shape, f"{name}: {frames.shape}")
        require(
            default_metadata["frames_indices"]
            == opencv_metadata["frames_indices"]
            == torchcodec_metadata["frames_indices"],
            "video backends sampled different frame indices",
        )
        require(
            np.array_equal(default_frames, opencv_frames),
            "default video decoder does not match explicit OpenCV",
        )
        np.testing.assert_allclose(
            torchcodec_frames.astype(np.int16),
            opencv_frames.astype(np.int16),
            rtol=0,
            atol=3,
        )
        max_delta = int(
            np.abs(
                torchcodec_frames.astype(np.int16) - opencv_frames.astype(np.int16)
            ).max()
        )
        return {
            "ffmpeg": ffmpeg,
            "image": {"size": list(image.size), "mode": image.mode},
            "video_shape": list(expected_shape),
            "frame_indices": opencv_metadata["frames_indices"],
            "default_backend": default_metadata["video_backend"],
            "opencv_backend": opencv_metadata["video_backend"],
            "torchcodec_backend": torchcodec_metadata["video_backend"],
            "torchcodec_version": getattr(torchcodec, "__version__", "unknown"),
            "opencv_torchcodec_max_abs_delta": max_delta,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(sys.version_info[:2] == (3, 12), "image must use Python 3.12")
    require(not os.environ.get("PYTHONPATH"), "PYTHONPATH must be unset")
    require(sys.flags.no_user_site == 1, "run with user site-packages disabled")
    prefix = Path(sys.prefix).resolve()
    require(prefix == Path("/opt/venv"), f"unexpected Python environment: {prefix}")
    site = Path(sysconfig.get_path("purelib")).resolve()

    cache_root, caches = prepare_cache_root(args.cache_root)
    ptxas = configure_bundled_ptxas(site)

    torch = importlib.import_module("torch")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(
        torch.accelerator.device_count() == 1,
        "exactly one target GPU must be visible",
    )
    capability = torch.cuda.get_device_capability(0)
    require(capability == (12, 0), f"expected SM120, got {capability}")

    report = {
        "passed": True,
        "environment": str(prefix),
        "cache_root": str(cache_root),
        "cache_environment": {
            name: os.environ[name]
            for name in (
                "FLASHINFER_WORKSPACE_BASE",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
                "VLLM_CACHE_ROOT",
                "DG_JIT_CACHE_DIR",
                "CUDA_CACHE_PATH",
                "MAX_JOBS",
                "FLASHINFER_NVCC_THREADS",
                "FLASHINFER_CUDA_ARCH_LIST",
            )
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
        },
        "cublas": verify_cublas(torch),
        "flashinfer": verify_flashinfer(torch, caches["flashinfer"]),
        "triton": verify_triton(torch, caches["triton"], ptxas),
        "media": None if args.skip_video else verify_media(),
    }
    report["cache_manifest"] = snapshot(cache_root)
    return report


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
        if active_cache_root is not None:
            report["cache_root"] = str(active_cache_root)
            report["partial_cache_manifest"] = snapshot(active_cache_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
