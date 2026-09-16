# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the built image, without host packages, CUDA, caches, or a GPU."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import regex as re

IMAGE = "ghcr.io/yhfgyyf/vllm-deepseek-v4-sm89:0.28.1rc1-vision11-sm89-sm120-cu130"
REVISION = "678d8f531e5e498e4060b21df254d27467e0cef9"
SCRIPTS = Path(__file__).resolve().parent
MEDIA_CHECK = """
import json, runpy
namespace = runpy.run_path('/opt/vllm-image/cold_jit_smoke.py')
print(json.dumps({'passed': True, 'media': namespace['verify_media']()}))
"""
SM89_CHECK = """
import json, subprocess
import regex as re
from flashinfer.jit.sampling import gen_sampling_module
spec = gen_sampling_module()
assert not spec.jit_library_path.exists()
spec.build(verbose=True)
compiled = subprocess.check_output(
    ['cuobjdump', '--list-elf', str(spec.jit_library_path)], text=True)
assert 'sm_89' in compiled
native = subprocess.check_output([
    'cuobjdump', '--list-elf',
    '/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so'
], text=True)
arches = sorted(set(re.findall(r'sm_[0-9]+[a-z]?', native)))
assert 'sm_89' in arches and 'sm_120' in arches
print(json.dumps({'passed': True, 'sampling_arch': 'sm_89',
                  'native_arches': arches, 'gpu_execution': False}))
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence
    image_id = (evidence / "image.iid").read_text().strip()
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id), "Invalid image ID")
    raw = subprocess.check_output(["docker", "image", "inspect", IMAGE])
    image = json.loads(raw)[0]
    require(image["Id"] == image_id, "Built image/tag mismatch")
    require(image["Architecture"] == "amd64", "Unexpected image architecture")
    require(image["Os"] == "linux", "Unexpected image OS")
    require(image["Config"]["Entrypoint"] == ["vllm", "serve"], "Wrong entrypoint")
    labels = image["Config"]["Labels"]
    require(labels["org.opencontainers.image.revision"] == REVISION, "Source drift")
    require(
        labels["org.opencontainers.image.source"]
        == "https://github.com/yhfgyyf/vllm-deepseek-v4-sm89",
        "Wrong repository label",
    )
    (evidence / "image-inspect.json").write_bytes(raw)

    def checked(log, *command):
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPTS / "run_logged.py"),
                str(evidence / log),
                "--",
                *command,
            ],
            check=True,
        )

    container = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "4",
        "--entrypoint",
        "/opt/venv/bin/python",
    ]
    checked(
        "cpu-image-check.log",
        *container,
        image_id,
        "-I",
        "/opt/vllm-image/verify_image.py",
        "--manifest",
        "/opt/vllm-image/manifest.json",
    )
    checked("media-decode.log", *container, image_id, "-I", "-c", MEDIA_CHECK)
    checked(
        "sm89-offline-jit.log",
        *container,
        "--env",
        "FLASHINFER_CUDA_ARCH_LIST=8.9",
        "--env",
        "FLASHINFER_WORKSPACE_BASE=/tmp/vision11-ci-jit",
        "--env",
        "MAX_JOBS=2",
        "--env",
        "FLASHINFER_NVCC_THREADS=2",
        image_id,
        "-I",
        "-c",
        SM89_CHECK,
    )
    packages = subprocess.check_output(
        [
            *container,
            image_id,
            "-I",
            "-c",
            (
                "import importlib.metadata as m,json; "
                "print(json.dumps({d.metadata['Name']:d.version "
                "for d in m.distributions()},sort_keys=True,indent=2))"
            ),
        ]
    )
    json.loads(packages)
    (evidence / "installed-packages.json").write_bytes(packages)
    result = {
        "passed": True,
        "image_id": image_id,
        "tag": IMAGE,
        "source_revision": REVISION,
        "payload_files": {"vllm": 4931, "flashinfer": 6246},
        "cold_sm89_sampling_compile_passed": True,
        "media_decode_passed": True,
        "gpu_execution": False,
        "full_model_serving": False,
        "validation_network": "none",
        "host_cuda_or_cache_mounts": False,
    }
    (evidence / "ci-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Optimized Python is not permitted for verification")
    main()
