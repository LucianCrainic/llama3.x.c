# Compute kernels

The numerical hot paths of both engines live in
[src/kernels_simd.c](../src/kernels_simd.c), declared in
[include/kernels.h](../include/kernels.h). The engines (`src/run.c`, `src/runq.c`)
now only lay out weights, caches and the order of operations. The motivation
and the roofline behind each kernel are in [hbm-roofline.md](hbm-roofline.md).

## Instruction set selection

`kernels_init()` runs once at model load and selects:

- `avx512`: AVX-512 F/BW/DQ/VL + VNNI + FMA, when CPU and OS support it
  (Xeon Max 9480 / Sapphire Rapids and later).
- `scalar`: the original loops with their original operation order. On CPUs
  without AVX-512, results are bitwise identical to the previous engine.

`LLAMA_KERNELS=auto|scalar|avx512` overrides the choice. Requesting `avx512` on a
CPU without it fails rather than falling back silently. Reports record the active
set as `build.kernels`. No compiler flag is required: AVX-512 functions use
`target` attributes and are dispatched at runtime, so the default
`make CC=gcc OPENMP=1` binary is portable. OpenMP regions live only in plain C
drivers, which call one ISA-specific function per row block.

SIMD reductions change FP32 summation order. Agreement with the scalar
reference is therefore within tolerance, not bitwise. The exceptions are RoPE
and activation quantization, which are bitwise identical by construction.

## Kernels

| Kernel | Scalar path | AVX-512 path |
| --- | --- | --- |
| FP32 GEMV `k_matmul*` | original row loop | 4 rows per pass sharing each `x` load, 8 independent zmm FMA chains, masked tails |
| Fused GEMV variants | – | `_qkv` (Wq/Wk/Wv in one region), `_swiglu` (W1 and W3 with the SiLU gate as epilogue), `_add` (Wo/W2 accumulate into the residual) |
| Q8_0 W8A8 `k_matmul_q8*` | original grouped int32 loop | VNNI `vpdpbusd` on \|x\| and sign-adjusted w; per-group FP32 scaling in vector accumulators (group size multiple of 64) |
| Q8_0 W8A32 | grouped FP32 loop | int8 widened in registers, FMA against FP32 activations (group size multiple of 16) |
| Quantization `k_quantize` | original | vector max/divide/clamp; `roundf` reproduced exactly (bitwise equal) |
| RMSNorm `k_rmsnorm` | `rmsnorm_eps` | 4 accumulators, masked tail |
| RoPE `k_rope_table` + `k_rope_apply` | table, original arithmetic | pair swap with `vpermilps`, no FMA (bitwise equal) |
| Attention `k_attention` | original per-head loop | GQA group per task; positions split into 32–256-position chunks merged with a stable softmax; vectorized `exp` |

Work distribution: each GEMV gives every thread one contiguous, balanced row
range. Each thread therefore streams weight rows sequentially, which keeps the
hardware prefetchers effective. Fusion reduces OpenMP parallel regions per layer
from 8 to 5.

RoPE frequencies and `cosf`/`sinf` are computed once per position, not once per
layer and head. For 8B, that replaces 65,536 `powf`+`cosf`+`sinf` evaluations
per token with 64.

Attention runs one task per (KV head, position chunk), so the 3–4 query heads of
a group read K/V once, from L2. The number of tasks scales with the thread
count instead of stopping at the head count.

### W8A32 option

`LLAMA_Q8_ACTIVATIONS=fp32` makes `runq` multiply dequantized weights by FP32
activations instead of quantizing activations per matmul. The weight bytes
streamed per token are the same; activation quantization error disappears. The
report's `model.precision` becomes `int8_q8_0_w8a32`. The default remains the
repository's W8A8 Q8_0 rule.

## Validation

`make test` builds and runs `tests/kernels.c`. It checks every available
instruction set against double-precision references:

- matmul variants, Q8 (both modes, group sizes 4–128, ties for `roundf`, an
  all-zero group), RMSNorm (including in place), RoPE (both scaling types),
  attention up to 4,096 positions, GQA ratios 1–4 and head sizes 2–128
- bitwise equality of RoPE and quantization between `scalar` and `avx512`

The Phase 2 fixed-logit and CLI regressions also run with the selected kernels.

## Results on CRESCO8 flat mode

Measured on 2026-11-01 on cresco8-hbm14 (Xeon Max 9480, flat mode, GCC 11.4.1,
`make CC=gcc OPENMP=1`) with
[scripts/cresco8-kernels.sbatch](../scripts/cresco8-kernels.sbatch), Slurm job
5621182. Each run generates 128 tokens after an 86-token prompt. Weights are
staged into `/dev/shm` with `numactl --membind=2` (HBM) or `--membind=0` (DDR),
and threads are bound to socket 0 unless noted. `base` is `develop` at
`2a16a18` (the scalar engine), and `simd` is this branch. "Decode GB/s" is
weight bytes per token × decode tokens/s. HBM and DDR traffic come from uncore
`hbm_cas_rd`/`ddr_cas_rd` counters over the whole run.

### Base vs SIMD, one socket, 56 threads

Decode tok/s, mean ± sd of 3 runs:

| Model | Format | Tier | base | simd | Speedup | simd GB/s | Share of tier ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | HBM | 41.69 ± 0.50 | 86.43 ± 0.25 | 2.07× | 427 | 77% of 552 |
| 1B | FP32 | DDR | 38.65 ± 0.03 | 48.14 ± 0.13 | 1.25× | 238 | 90% of 265 |
| 1B | Q8_0 | HBM | 77.12 ± 9.00 | 222.65 ± 0.61 | 2.89× | 292 | 53% |
| 1B | Q8_0 | DDR | 80.38 ± 0.23 | 159.16 ± 0.55 | 1.98× | 209 | 79% |
| 8B | FP32 | HBM | 7.52 ± 0.00 | 14.90 ± 0.01 | 1.98× | 447 | 81% |
| 8B | FP32 | DDR | 6.93 ± 0.01 | 8.31 ± 0.00 | 1.20× | 249 | 94% |
| 8B | Q8_0 | HBM | 18.17 ± 0.05 | 44.49 ± 0.16 | 2.45× | 355 | 64% |
| 8B | Q8_0 | DDR | 17.40 ± 0.04 | 28.51 ± 0.13 | 1.64× | 227 | 86% |

- **The old engine was compute-bound at a full socket.** At 56 threads, `base`
  runs within 4–9% of the same speed on DDR and HBM (8B FP32: 6.93 vs 7.52 tok/s,
  about 4 GB/s per core).
- **The new kernels make the memory tier matter.** HBM is now 1.8× faster than
  DDR for FP32 and 1.4–1.6× for Q8_0.
- **DDR is saturated.** FP32 on DDR reaches 90–94% of the measured DDR ceiling,
  so the remaining DDR gain is 6–10%.
- **8B FP32 on HBM reaches 81% of the stream-read ceiling.** That is
  14.9 tok/s against the modeled 18.3 for one socket's HBM, and above the
  327–358 GB/s of the oneDNN GEMV reference.
- **Placement is clean.** HBM runs read about 2,075 GB (1B FP32) and 12,780 GB
  (8B FP32) from HBM, with under 25 GB from DDR. `base` and `simd` move the same
  bytes; the gain is per-core throughput only.
- **Prefill uses the same kernels.** It still runs one GEMV per prompt token,
  so its tok/s tracks decode (batched prefill is listed under
  "Not in this change").

### Thread scaling on HBM

| Threads | 8B FP32 GB/s | HBM read ceiling | Share | 8B Q8_0 GB/s | Share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 199 | 223 | 89% | 143 | 64% |
| 28 | 333 | 387 | 86% | 246 | 64% |
| 56 | 447 | 552 | 81% | 355 | 64% |

FP32 stays close to the ceiling at every thread count. Q8_0 holds a constant
64%, so its limit is per-core instruction throughput, not memory. The
per-token work is activation quantization, VNNI dots and one scale FMA per
64-weight group. The Q8_0 kernel is therefore the next kernel target. 1B
reaches lower shares (53% for Q8_0) because its matrices are small: 16 layers
of 2,048-wide GEMVs at more than 200 tok/s, where per-region OpenMP
synchronization is a visible part of the 4.5 ms per token.

### Other configurations

| Configuration | 8B FP32 tok/s | 8B Q8_0 tok/s | UPI rx data flits (M) |
| --- | ---: | ---: | ---: |
| 56 threads, socket 0, HBM node 2 | 14.90 | 44.49 | 43 / 20 |
| 112 threads, weights on HBM node 2 | 6.42 | 21.28 | 456,259 / 122,566 |
| 112 threads, weights interleaved on HBM 2,3 | 8.17 | 26.16 | 454,209 / 122,118 |
| 56 threads, `OMP_WAIT_POLICY=active` | 14.91 | 44.52 | – |
| 56 threads, W8A32 (`LLAMA_Q8_ACTIVATIONS=fp32`) | – | 40.40 ± 0.08 | 20 |

- **Two sockets are slower than one.** With row-split GEMVs, half of every
  matrix is read across UPI, from remote HBM at 50–110 GB/s. This needs
  socket-local weight shards (tensor parallelism), not a kernel change.
- **The OpenMP wait policy makes no difference.**
- **W8A32 costs 9% on 8B for the same HBM bytes.** Widening int8 to FP32
  gives 16 lanes per FMA, against 64 per VNNI instruction. On 1B it costs about
  20%, with high run-to-run variance (179 ± 13 tok/s).

### Accuracy against Hugging Face FP32

Results are on the four Phase 2 validation prompts, plus greedy generation.

| Model | Mode | Budget | max\|Δlogit\| | Passed |
| --- | --- | --- | ---: | --- |
| 1B | FP32 | atol 1e-3 | 2.0e-5 | yes |
| 8B | FP32 | atol 1e-3 | 1.6e-4 | yes |
| 1B | Q8_0 W8A32 | atol 1.0 | 0.76 | yes |
| 8B | Q8_0 W8A32 | atol 1.0 | 0.64 | yes |
| 1B | Q8_0 W8A8 | atol 1.0 | 1.32 | no |
| 8B | Q8_0 W8A8 | atol 1.0 | 1.42 | no |

- **Generated tokens match the reference in every mode.**
- **FP32 is more accurate than before.** The 1B max is 2.0e-5, against 8.3e-5
  for the Phase 2 scalar path.
- **8B W8A8 failed in Phase 2 as well.** Phase 2 measured 1.411 at position 0.
  An independent HF FP32 model with groupwise W8A8 fake quantization gets 1.410,
  so the failure belongs to the quantization scheme.
- **1B W8A8 moves with summation order.** It passed in Phase 2 at 0.993, just
  under the 1.0 budget. The same binary gives 0.993 with `LLAMA_KERNELS=scalar`
  and 1.32 with `avx512`.
- **The AVX-512 kernels themselves agree with scalar.** On the failing prompt,
  scalar and AVX-512 differ by 8e-5 in FP32 and 6e-5 in W8A32. Under W8A8 the
  difference is 0.60, because a tiny change in summation order flips activation
  rounding at every matmul.
- **The W8A32 option fixes both models.** It passes the budget on both models
  with lower RMS error.

The campaign gates benchmarks on FP32 validation only; INT8 results are
recorded with the budget unchanged.

## Not in this change

The following items from the roofline analysis are not kernel work:

- explicit NUMA placement of weights (anonymous memory + `mbind`)
- two-socket tensor parallelism
- batched prefill, and with it AMX GEMM
- skipping vocabulary logits for prompt positions
- BF16 weight format
