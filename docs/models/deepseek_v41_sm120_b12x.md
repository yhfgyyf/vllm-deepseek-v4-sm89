# DeepSeek V4.1 SM120 B12X adapters

Current source includes an **experimental, explicit opt-in** for the SM120
components measured on one RTX PRO 6000 Blackwell. The new serving adapters
have not completed GPU integration or full-model qualification. The November
12 host was occupied by another model; no GPU tests or service restarts were
performed for this source update. Leaving the switch unset retains the native
SM120 paths. SM89 K1/K2/K3 launch tuning remains architecture-specific.

## Scope and numerical contract

Set `VLLM_DEEPSEEK_V41_B12X=1` only in the pinned environment below. The switch
does not select the generic `--linear-backend b12x` or `--moe-backend b12x`.

| Component | Selected cases | Preserved contract and fallback |
| --- | --- | --- |
| Mixed Attention | SM120, BF16 Q, 1–32 rows, 16/32 heads, SWA width 128 and nonempty global indices, global page size 64/128 | Native mixed FP8/FP4 KV bytes; physical slot indices; BF16 attention; inverse RoPE and grouped output. Pure SWA, larger batches and other layouts retain native kernels. |
| Delayed mHC + norm | SM120, 1–32 rows, hidden 5120, full-width projection, incoming pre-mix and norm, equal pre/Sinkhorn epsilon, post scale 2, 20 Sinkhorn iterations | FP32 FN and coefficients; incoming mix collapses the current input, new mix is returned separately. Broadcast/draft inputs without incoming mix and larger batches retain native kernels. |
| MXFP8 Linear | SM120 V4.1, 1–32 rows, `(N,K)=(1152,5120)` or `(1792,5120)`, BF16 output | Existing V4.1 activation quantization including `min_amax=1e-4`; identical quantized values/scales; borrowed weight storage. Other shapes and larger batches retain FlashInfer. |

FP8 Attention and B12X MoE are not enabled. The earlier MoE experiment did not
preserve the V4.1 routed-weight/activation-quantization boundary sufficiently
to qualify that route. The existing native MoE and expert parallelism remain.

Plans are prepared outside CUDA Graph capture and reused by geometry. Scratch
and output storage are owned by each invocation, rather than caching request
inputs. A previously unseen shape during capture uses native execution and
logs a warning. The Attention adapter adds inverse RoPE after the B12X BF16
output; its additional rounding and runtime cost need integration testing.

## Pinned optional environment

The measurements used
[B12X e9ce547767ff9ee6509faf294fa1b4e2380dfbf5](https://github.com/local-inference-lab/B12X/commit/e9ce547767ff9ee6509faf294fa1b4e2380dfbf5),
package version 1.3.0, CUTLASS DSL 4.6.2, Torch 2.14 and the fork's patched
FlashInfer 0.7.3. The fork's generic `vllm[b12x]` extra currently pins 1.5.1;
that is not the measured adapter API. Do not replace a running environment.

In a separate environment built from the
[current vLLM/FlashInfer sources](deepseek_v41_sm89_sm120.md#build-the-current-source):

```bash
uv pip install 'nvidia-cutlass-dsl==4.6.2'
uv pip install --no-deps \
  'b12x @ git+https://github.com/local-inference-lab/B12X.git@e9ce547767ff9ee6509faf294fa1b4e2380dfbf5'
uv pip check
export VLLM_DEEPSEEK_V41_B12X=1
```

This is an optional source adapter, not a new mandatory package dependency or
a published wheel. Keep `--linear-backend auto --moe-backend auto` in the
SM120 TP4 launch template. To restore native selection, unset
`VLLM_DEEPSEEK_V41_B12X` before starting a new process.

The existing `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` switch is independent.
Its module compiled on SM120, but only single-rank rejection was tested there;
multi-GPU SM120 IPC correctness and performance are still unverified. L20 TP8
IPC results do not qualify a PRO 6000 topology.

## Reused component evidence

On November 11, 19 Attention, 3 mHC and 3 Dense captured-input cases completed
numerical and changed-input CUDA Graph checks on one RTX PRO 6000. Representative
median GPU times in microseconds were:

| Case | Native | B12X |
| --- | --- | --- |
| Mixed Attention M6/H16 | 20.35–20.40 | 12.76–12.85 |
| Mixed Attention M6/H32 | 30.15–30.24 | 13.17–13.20 |
| Delayed mHC M6 | 8.867 | 6.999 |
| Delayed mHC M128, retained native | 21.949 | 55.584 |
| MXFP8 M6/N1792/K5120 | 30.475 | 12.643 |
| MXFP8 M6/N1152/K5120 | 19.774 | 12.222 |

Attention timings exclude inverse RoPE, projections and communication. Dense
timings exclude activation quantization and scale swizzle. M128 snapshots may
be truncated prefill inputs; they do not measure complete prefill. These
numbers describe the earlier isolated components, not this adapter's speed or
whole-model improvement. No serving timeline or Top10 is claimed for the new
adapter.

## Qualification still required

When an SM120 GPU is available, run the adapter regression through its actual
vLLM entry points in the pinned environment:

```bash
.venv/bin/python -m pytest -q \
  tests/models/deepseek_v4_1/test_sm120_b12x.py
```

The file includes CPU scale-layout checks plus GPU numerical, changed-input
Graph, inverse-RoPE/output-offset and native-fallback cases. It is selected by
the existing native-component CI directory; that CI uses L4, so its SM120 GPU
cases require a separate target-host run. GPU skips are not a qualification.
Full-model text/image/tool quality, C1 and complete C4, all worker imports,
actual kernel selection, and target-topology IPC must also be checked before
this opt-in can become a recommended default.
