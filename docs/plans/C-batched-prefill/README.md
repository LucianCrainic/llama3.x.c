# Track C: batched prefill, and with it AMX (one socket)

## Goal

Replace per-token prompt processing with microbatched prefill. Each weight byte read
from memory is then reused across M prompt tokens, and prefill becomes compute-bound.
Prefill gets faster by climbing the precision ladder the user chose:

1. **AVX-512 FP32 GEMM.** Exact FP32 semantics.
2. **AMX-BF16.** An explicit BF16 precision mode.
3. **AMX-INT8.** The existing Q8_0 semantics, W8A8.

Each stage can be selected on its own and is validated against the reference.

The track runs on **one socket**, in parallel with track A. It reuses the F1/B1
tensor/shard descriptors and the F2 worker teams; there is no separate scheduler and no
third-party dependency. Two-socket prefill comes later, in D1.

## Why it matters

These numbers come from `docs/hbm-roofline.md` §4, a per-socket model that ignores
attention and overheads:

| 8B | Throughput |
| --- | ---: |
| Today, sequential prefill, Cache mode (**measured**) | 2.7 tok/s |
| llama.cpp FP32, batched (**measured**) | 89 tok/s |
| AVX-512 FP32 at ~5.5 TFLOP/s (**modeled**) | ~370 tok/s |
| AMX-BF16 at 27 TFLOP/s (**modeled**, oneDNN-measured rate) | ~1,800 tok/s |
| AMX-INT8 at 69 TOPS (**modeled**) | ~4,600 tok/s |

The ridge points (**modeled**) are B≈32 tokens for FP32 AVX-512 and B≈50 tokens for
AMX-BF16. AMX does not help single-stream decode (`docs/hbm-roofline.md` §3), so this
track does not change decode.

## Phases

| ID | Phase | Output | Gate |
| --- | --- | --- | --- |
| C1 | [Batched forward API](C1-batched-forward-api.md) | `engine_forward_batch`; microbatched `run_sequence`; prompt logits skipped | **Bitwise** equal to per-token forward |
| C2 | [AVX-512 FP32 GEMM](C2-avx512-fp32-gemm.md) | Team-tiled GEMM with STORE/ADD/SWIGLU/PARTIAL epilogues | Kernel budget; 8B within 1e-3 of per-token |
| C3 | [Batched causal attention and KV](C3-batched-causal-attention-and-kv.md) | M-query causal attention; batched KV append | Matches per-token attention within budget |
| C4 | [AMX enablement](C4-amx-enablement.md) | ISA detection, XTILEDATA permission, tile config, aligned workspaces | AMX microkernel tests on hardware |
| C5 | [AMX-BF16 precision mode](C5-amx-bf16-precision-mode.md) | `LLAMA_PRECISION=bf16`; packed BF16 weights per shard | Error recorded; provisional gate fixed |
| C6 | [AMX-INT8 for Q8_0](C6-amx-int8-q8.md) | W8A8 AMX with per-group rescale | Agreement with the AVX-512 Q8 batched path |
| C7 | [One-socket prefill campaign](C7-one-socket-prefill-campaign.md) | Prefill and decode measured per stage | Speedup and crossover report |

## Status (7 November 2026)

| Phase | Implemented | Tested locally | Open (needs the cluster, not run) |
| --- | --- | --- | --- |
| C1-C3 | yes | yes (bitwise, kernel budgets; AVX-512 microkernel only on an emulation) | `STAGE=check PHASE=C1`, `C2`, `C3` |
| C4 | `KERNELS_AMX`, permission, tile config, BF16 / int8 tile kernels, packing helpers | packing on every platform; tile kernels on the software tile model (exact int32, BF16 within 2e-6); `-arch x86_64` build | `STAGE=check PHASE=C4`: real `tdpbf16ps` / `tdpbssd`, `arch_prctl`, GCC 11 build, tile TFLOP/s |
| C5 | `LLAMA_PRECISION=bf16`, packing, driver, references, report | RNE conversion bitwise; GEMM vs double; engine: decode unchanged, split-independent, tile model == scalar reference | `STAGE=check PHASE=C5`: error vs FP32 and HF, the provisional gate, budget on HBM |
| C6 | batched Q8 forward, AVX-512 VNNI tile kernel, AMX-INT8, packing, W8A32 rule | exact int32 dots on the tile model; kernels vs double; scalar batched forward bitwise == per-token | `STAGE=check PHASE=C6`: VNNI and AMX kernels on hardware, agreement on 1B / 8B, rescale overhead |
| C7 | `phase_C7` (and `C4`-`C6`) in `cresco8-prefill.sbatch` | `DRY_RUN=1` only | the campaign (`STAGE=bench PHASE=C7`, about 5 h) after C4-C6 pass |

## Track exit gate (milestone M3, prefill half)

- **Correctness.** All four prefill kernels pass their gates:

  | Kernel | Selected by |
  | --- | --- |
  | AVX-512 FP32 | `LLAMA_KERNELS=avx512` |
  | AMX-BF16 | `LLAMA_KERNELS=amx` with `LLAMA_PRECISION=bf16` |
  | AVX-512 Q8 | `runq` with `LLAMA_KERNELS=avx512` |
  | AMX-INT8 | `runq` with `LLAMA_KERNELS=amx` |

- **Performance report.** C7 reports prefill throughput and TTFT per stage, model,
  prompt length and M, plus decode measured separately with no regression.

## Selectability

| Knob | Effect |
| --- | --- |
| `LLAMA_PREFILL_BATCH=1` | Today's per-token path. The default until C7 decides otherwise. |
| `LLAMA_PREFILL_BATCH=M` with `LLAMA_KERNELS=avx512` | AVX-512 GEMM, FP32 or Q8 |
| `LLAMA_KERNELS=amx` | AMX for batched prefill. FP32 checkpoints require `LLAMA_PRECISION=bf16`. Without it, `amx` on FP32 is an error, because FP32 execution never silently changes precision. |
| `LLAMA_Q8_ACTIVATIONS=fp32` | W8A32 prefill stays on the AVX-512 path, even when `amx` is selected. This is logged. |
