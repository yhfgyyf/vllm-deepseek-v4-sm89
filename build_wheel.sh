#!/bin/bash
set -e
set -o pipefail
cd /home/yyf/vllm

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/home/yyf/.cargo/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0
SHORT_SHA=$(git rev-parse --short=9 HEAD)
export VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-0.23.1rc1.dev145+g${SHORT_SHA}.cu130}
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=${MAX_JOBS:-16}
export NVCC_THREADS=${NVCC_THREADS:-2}

echo "=== START $(date +%T) | nvcc $(nvcc --version | tail -1) | arch=$TORCH_CUDA_ARCH_LIST ==="
rm -rf build/ dist/*.whl
.venv/bin/python -m build --wheel --no-isolation
RC=$?
echo "=== END $(date +%T) rc=$RC ==="
[ $RC -eq 0 ] && ls -lh dist/*.whl && echo "WHEEL_BUILD_OK" || echo "WHEEL_BUILD_FAILED rc=$RC"
