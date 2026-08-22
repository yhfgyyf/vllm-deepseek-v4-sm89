#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/home/yyf/.cargo/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.2
if [[ -z "${VLLM_VERSION_OVERRIDE:-}" ]]; then
  BASE_VERSION=$(
    .venv/bin/python <<'PY'
from setuptools_scm import get_version

print(
    get_version(
        root=".",
        git_describe_command=(
            "git describe --dirty --tags --long --match 'v[0-9]*' "
            "--exclude '*-cu*' --exclude 'vtest'"
        ),
    )
)
PY
  )
  VERSION_SEPARATOR="+"
  if [[ "$BASE_VERSION" == *+* ]]; then
    VERSION_SEPARATOR="."
  fi
  export VLLM_VERSION_OVERRIDE="${BASE_VERSION}${VERSION_SEPARATOR}cu132"
fi
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-2}

echo "=== START $(date +%T) | nvcc $(nvcc --version | tail -1) | arch=$TORCH_CUDA_ARCH_LIST | version=$VLLM_VERSION_OVERRIDE ==="
.venv/bin/python -m build --wheel --no-isolation --outdir dist-sm89
echo "=== END $(date +%T) ==="
ls -lh dist-sm89/*.whl
echo "WHEEL_BUILD_OK"
