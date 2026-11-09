# DeepSeek V4.1 on SM89 and SM120

## Framework baseline

The source incorporates official vLLM main commit
`3213796cd32df625a0669e9423acb397b627b790` (checked 2026-11-09).
It retains this fork's SM89/SM120 native attention, MXFP4/MXFP8 MoE, CED,
adaptive verification and PP/DSpark support. Other architectures use the
upstream DeepSeek V4.1 implementation. SM100/SM103 kernels are not selected
for SM120 merely because both devices belong to the Blackwell family.

The historical `vision11` wheels and container use the older vLLM 0.28 / Torch
2.13 / FlashInfer 0.6.18 stack. They do not contain this framework upgrade.
Their full-model validation and benchmark results do not validate this source.

## Build the current source

Use a new Linux x86_64 Python 3.12 environment, CUDA 13.2 toolkit, GCC 13 or
another compiler supported by that toolkit, and Rust/Cargo supporting edition
2024. The current requirements select Torch 2.14, Triton 3.8, TileLang 0.1.12
and FlashInfer 0.7.3. Do not pin the older Triton 3.7.1 or reuse extensions
compiled against Torch 2.13.

From this vLLM checkout:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
uv pip install -r requirements/cuda.txt --torch-backend=cu130

export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST='8.9;12.0'
export MAX_JOBS=4
uv pip install --no-build-isolation -e . --torch-backend=cu130
```

The complete candidate C++/CUDA build used CUDA 13.2 and GCC 13 on the SM120
host, with both `8.9` and `12.0` targets. Its extensions also ran on L20/SM89;
L20's local JIT kernels used CUDA 13.0. This does not establish that the whole
current source builds with CUDA 13.0. The Rust frontend was built in the
`dev` profile; a release Rust build and distributable vLLM wheel were not
validated in this upgrade.

Upstream FlashInfer 0.7.3 needs the additive native V4.1 extension below for
SM89/SM120. It preserves the native 528-byte SWA and 288-byte global-cache
row layouts; the upstream SM100 cache layout cannot replace these formats.
Keep this custom FlashInfer installation when updating dependencies.

```bash
vllm_source="$PWD"
git clone --branch v0.7.3 --recurse-submodules \
  https://github.com/flashinfer-ai/flashinfer.git ../flashinfer-v41-native
cd ../flashinfer-v41-native
git checkout 775207deb34055667df89ed618fb4b06fc84de74
git submodule update --init --recursive
git apply --check \
  "$vllm_source/tools/recipes/patches/flashinfer-0.7.3-deepseek-v41-native.patch"
git apply \
  "$vllm_source/tools/recipes/patches/flashinfer-0.7.3-deepseek-v41-native.patch"
git apply --check \
  "$vllm_source/tools/recipes/patches/flashinfer-0.7.3-sm89-glm-compat.patch"
git apply \
  "$vllm_source/tools/recipes/patches/flashinfer-0.7.3-sm89-glm-compat.patch"
uv pip install --python "$vllm_source/.venv/bin/python" -r requirements.txt
BUILD_NVEP=0 FLASHINFER_LOCAL_VERSION=dsv41.sm89sm120.3 \
  uv build --wheel --no-build-isolation \
  --python "$vllm_source/.venv/bin/python" --out-dir dist
uv pip install --python "$vllm_source/.venv/bin/python" --no-deps \
  dist/flashinfer_python-0.7.3+dsv41.sm89sm120.3-py3-none-any.whl
cd "$vllm_source"
uv pip check
uv run --no-project python -c \
  'import vllm, vllm.models.deepseek_v41; from flashinfer.mla import deepseek_v41; print(vllm.__file__, deepseek_v41.__file__)'
```

Build the FlashInfer wheel before installing it: its build backend invokes
`uv` for compiler wheels, which can wait on the parent installer's environment
lock during a combined `uv pip install --no-build-isolation .` invocation.
`BUILD_NVEP=0` disables an optional, unrelated NVEP build. The nine additive
files come from this fork's FlashInfer source revision
`b89cb3bad8f375f2dafcdb58620dcd4e70ad9c31` and include their kernel tests. A second additive patch retains 30 files for
SM89 legacy DeepSeek sparse MLA and GLM NoPE cache/attention compatibility
from the same revision. Kernel symbols and JIT paths use an isolated namespace;
SM120 continues to use the official 0.7.3 attention kernels. The vLLM adapter
translates the former NoPE scale-format name to the current API.

## Serving configuration

Keep the documented default target TP4 / PP2, DSpark TP4, and
`--max-num-seqs 4`. Optional target TP2 / PP4 uses
`VLLM_PP_LAYER_PARTITION=10,10,10,10` and
`"draft_tensor_parallel_size":4` in `--speculative-config`; its draft group
spans the final four GPUs. Clear the partition override when switching back
to TP4 / PP2. The native attention selector accepts the existing
`FLASHINFER_MLA_SPARSE_DSV41` name and the explicit
`FLASHINFER_MLA_SPARSE_DSV41_NATIVE` name.

The current framework uses the GPU model runner directly;
`VLLM_USE_V2_MODEL_RUNNER` is no longer a supported switch. CED, image limits,
adaptive verification and FULL graph settings remain in the repository's
DeepSeek V4.1 launch examples. Recheck the actual context capacity and memory
budget when deploying the new dependency stack.

## Validation scope

The 2026-11-09 validation used L20 (SM89) and RTX PRO 6000 (SM120), both with
Torch `2.14.0+cu130`, Triton `3.8.0`, Transformers `5.22.2`, TileLang `0.1.12`
and the patched FlashInfer source. The final FlashInfer wheel is
`0.7.3+dsv41.sm89sm120.3`; SHA256:
`452441ada692081da39c61ababc88884acd6337c22a543e6bcf3c52bf7b0233a`.
All 38 added runtime files were compared byte-for-byte with the wheel.

| Check | L20 / SM89 | RTX PRO 6000 / SM120 |
| --- | --- | --- |
| Native model, CED, indexer, MoE and adaptive suites | 505 passed, 44 skipped | 507 passed, 42 skipped |
| Persistent top-k boundary and graph regressions | 7 passed | 7 passed |
| Cache planning, frontend and configuration suites | 389 passed after a missing-config rerun | 389 passed |
| Engram and graph-break boundaries | 109 passed, 5 skipped | 109 passed, 5 skipped |
| Worker, GLM processor, activation fusion and CED budget | 106 passed | 106 passed after a missing-config rerun |
| Legacy sparse MLA API and mixed FULL graph suites | 265 passed, 70 skipped after constructor-test migration | 259 passed, 70 skipped, plus 6 new NoPE cases |
| GLM NoPE cache helper tests | 9 passed | 9 passed |

These suites overlap and must not be summed. NoPE cases compare real packed
cache output and replay after changing queries and indices, for 1/64/96 tokens
and 8/32 heads. They use retained SM89 attention and official 0.7.3 SM120
attention. Hardware/optional-reference skips are not successful validation.
The constructor-test migration checks that unsupported devices reject a
Blackwell-only CUTLASS kernel; it does not relax a numerical tolerance.

Eight-rank NCCL checks passed for TP4/PP2 and TP2/PP4 with a TP4 draft group,
including group membership, collectives, broadcast source and restoration of
the target TP context. These checks construct real configurations and groups
without loading a complete model. Both test hosts matched all 5,788 Python
source/test file hashes. Build-generated third-party Python files are recorded
separately from repository sources.

Full-model validation also passed on 8x L20 with target TP4/PP2, draft TP4,
DSpark 5, adaptive verification, FULL_AND_PIECEWISE graphs, CED, Engram CPU
offload, an 8192-token batch budget and 2 GiB KV allocation per worker.
The benchmark uses eight sequence slots to match the historical baseline;
the launch examples retain their four-sequence default.

| Input / output | Previous TTFT (s) | Updated TTFT (s) | Previous decode (tok/s) | Updated decode (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| 8192 / 512 | 1.224 | 1.202 ± 0.005 | 106.73 | 118.79 ± 11.61 |
| 32768 / 512 | 4.775 | 4.719 ± 0.019 | 102.48 | 115.80 ± 12.48 |
| 131072 / 512 | 21.798 | 22.025 ± 0.021 | 107.10 | 116.93 ± 14.85 |

Each length uses one warmup and four measured, concurrency-one requests with
identical saved prompt token IDs and a cold prefix for every request. Values
are means; updated ± values are sample standard deviations across four
document topics. Every measured response contained 512 tokens, with zero
prefix hits and preemptions. Decode counts actual SSE token IDs, excluding
all tokens delivered in the first event. This historical whole-stack
comparison does not isolate one optimization or establish output equivalence.
Server prefill means changed by -1.94%, -0.94% and +1.11%, respectively.

One Chinese response, one two-image shape/OCR request and one complete tool
round trip passed. A cold 1,048,064-token request completed in 440.46 seconds
and correctly retrieved a unique code placed near the middle; the configured
context limit was 1,048,576. These are bounded smoke checks, not a broad
quality evaluation or a full-model SM120 validation.

Full-model testing exposed two migration defects: obsolete native MoE
loading hooks and a compressor that treated disabled generic ring mappings
as invalid real tokens. The native compressor now derives ring addresses
from request blocks and positions. Its extended regression failed before
the fix and passes afterward, preserving null-block and padding behavior.
Actual adaptive startup calibration and normal graphs were enabled for the
reported serving measurements.

The [migration ledger](deepseek_v41_upgrade_audit.md) maps all 24 correctness
and 21 performance findings from the preceding audit to the active source
route. Shared V4.1 frontends and Engram use current upstream code; native
SM89/SM120 attention, routing and expert quantization retain their hardware
and numerical contracts.

Numerical selection used 64 checkpoint tensors (84.9 MB) with seeded synthetic
inputs. Two packed experts favor retaining MXFP8 activation QDQ and applying
routing weights before FC2 quantization: normalized RMSE against the staged
reference is at most `3.50e-4`, versus roughly `4.18–4.49%` without MXFP8 QDQ.
The FP32 staged reference was checked against FP64 at one token only. These
errors measure the intended quantized operator contract, not model accuracy.

Across four real mHC weight sets, small post/pre fusion at 1/8/32 tokens reduced
median operator latency by 30–41% on SM120 and 27–45% on L20. At 64/256 tokens
the native path was faster, so fusion is restricted to 1–32 tokens. Native
TF32x3 prenorm, delayed collapse, and Engram injection boundaries remain.
Both paths were compared to FP64 equations; partial operator comparisons do
not establish full-model output equivalence.
