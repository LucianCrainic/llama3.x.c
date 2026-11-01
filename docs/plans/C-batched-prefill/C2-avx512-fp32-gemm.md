# C2. AVX-512 FP32 batched GEMM

## 1. Goal and status

Implement team-parallel, cache-blocked FP32 GEMM, `C[M×N] (op)= A[M×K] · W[N×K]ᵀ`,
directly on the existing `W[out][in]` layout. Each weight panel is streamed once per
microbatch. The GEMM feeds every projection of the batched forward. The epilogues
match the GEMV kinds (STORE, ADD, SWIGLU, PARTIAL), so the same kernel later serves the
two-socket shards (D1).

Status: **implemented locally; cluster check pending**. Branch: `feature/prefill-c2-avx512-gemm` (stacked on C1).

Implementation notes:

- `src/gemm.c`, `include/gemm.h`: `k_gemm_f32(A, lda, m, k, segments, count)` with up to three segments sharing A
  (QKV) in one parallel region, each with its own ldc, weight pointer, leading dimension and epilogue
  (`GEMM_STORE`, `ADD`, `SWIGLU`, `PARTIAL`). Segments are split into chunks of 16 weight rows distributed over the team
  or OpenMP threads (`kernels_parallel_segments`, so a worker's share never straddles two segments).
- **Microkernel: the plan's second alternative.** A 4×4 tile of dot products (4 A rows × 4 W rows, 16 independent zmm
  accumulators, 8 loads per 16 FMAs, horizontal reduce at the end of a K block) reads A as it is, with no transposed
  packing. K is blocked by 1024 (`LLAMA_GEMM_KC` tunes it), M by 256, and a chunk's tile sums go through a small scratch
  so SWIGLU sees the complete K sum of both W1 and W3 before the nonlinearity. The broadcast form with packed A remains
  an option if the sweep shows the horizontal reductions cost too much.
- **Scalar instruction set** uses one K block and ascending k: it is bitwise equal to the per-row GEMV, which the tests
  assert for STORE/ADD/SWIGLU.
- `forward_layer_batch` (FP32 engine) runs a whole layer for `count` rows through the GEMM and writes K/V straight into
  the cache. `LLAMA_PREFILL_GEMM=auto|on|off` (auto: AVX-512 kernels active and batch above one) selects it; the Q8
  engine still runs the per-row reference (its AVX-512/AMX GEMM is C6). `runtime.kernels_prefill` reports the choice.
- **Local evidence.** 12.9 M checks against the double reference over shapes with padded leading dimensions, all four
  epilogues, M above one block, several chunks and K blocks, team sizes 1/3/7; the same driver with a portable
  microkernel at K blocks 7/16/100/1024; and the AVX-512 microkernel's own code (macros, masks, tails) executed on a
  scalar emulation (`tests/avx512_emul.h`, 21.6 M checks). Mutation checks (dropped masked tail, swapped tile index,
  wrong edge index, lost K-block accumulation) are all caught. The microkernel also cross-compiles for x86-64.
  **Not yet run:** the real AVX-512 instructions and any timing; both are the cluster `check`.
- `build/release/gemm-bench` (`tests/gemm_bench.c`) prints GFLOP/s and weight GB/s per shape and M, GEMM and per-row GEMV,
  with the weights bound to a node; `tools/summarize_gemm.py` turns it into the table.

## 2. Depends on / unblocks

- Depends on: C1 (oracle and buffers), F2 (`team_parallel_rows`) and B2's epilogue
  enum. B2 adds `PARTIAL`; if C2 lands first, C2 adds `PARTIAL` itself.
- Unblocks: C3, C4, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `dots_f32_avx512` | `src/kernels_simd.c:112` | 4-row register blocking pattern, `tail_mask` (:88) |
| Epilogues and `silu_gate` | `src/kernels_simd.c:409-457`, `:412` | Same order of operations per element |
| `team_parallel_rows` | F2 | Distribution over N blocks |
| Kernel tests | `tests/kernels.c:64-85` | Budget formula reused with M rows |
| AVX-512 target attribute | `src/kernels_simd.c:14-20` | Same feature set; no new compiler flags |

## 4. Changes

### 4.1 Where the code goes

`src/gemm.c` and `include/gemm.h`. Kernels stay in the existing style: the parallel
driver is plain C, and target-attributed microkernels are called per tile.

```c
/* Proposed. */
typedef struct { const float *a; int lda; } GemmA;        /* M x K activations, row-major */
void k_gemm_f32(float *c, int ldc, GemmA a, const float *w, size_t ldw,
                int m, int n, int k, int kind, const float *w3 /* SWIGLU */);
```

### 4.2 Blocking

These are starting values, tuned in the C2 `check` sweep:

| Level | Starting value |
| --- | --- |
| Microkernel | 8 N rows × 2 zmm of M (`m_r` = 32 tokens is too many for registers): broadcast the W element, FMA into 16 accumulators of 16 floats (8 W rows × 2 M lanes). Alternative to measure: `m_r` = 4 tokens × 4 W rows with horizontal reduction, matching the GEMV style. |
| K | Blocks of 256 so A and W panels stay in L1/L2. A is packed once per microbatch into a transposed `K × M` panel so it can be broadcast. |
| N | Blocks of 64 rows distributed across the team (`team_parallel_rows` over N/64). Each block streams its W rows exactly once per microbatch. |

### 4.3 Epilogues

The epilogues are applied per output element after the full K loop:

- `STORE` and `PARTIAL`: `c = acc`.
- `ADD`: `c += acc`.
- `SWIGLU`: computes W1 and W3 tiles on the same N block, then applies `silu_gate`.

### 4.4 QKV

QKV is three N-segments that share A, split per segment as in F2.

### 4.5 Integration

The batched forward (C1) switches each projection from the per-row reference to
`k_gemm_f32` when `LLAMA_KERNELS` resolves to AVX-512 and M > 1. The classifier runs as
GEMM only for `LOGITS_ALL`; otherwise it is a GEMV on the last row.

### 4.6 Scalar fallback

A simple triple loop with the same accumulation order per element as the GEMV scalar
path, used when AVX-512 is unavailable (the macOS arm64 build).

## 5. Local tests

These run on macOS (scalar) and on Linux AVX-512 hosts.

- **Kernel budget.** For random shapes M ∈ {1, 3, 16, 31, 64}, N ∈ {1, 17, 64, 1000}
  and K ∈ {16, 100, 4096}, every element is within `2e-5·Σ|w·x| + 1e-6` of the double
  reference. Every epilogue kind and team sizes 1, 3 and 7 are covered.
- **SWIGLU.** Error-propagated budget, as `tests/kernels.c:71`.
- **Fixture forward.** Batched GEMM forward vs the C1 reference, all positions, within
  the fixture tolerance (`rel 2e-5 / abs 2e-6`).

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C2`. **Estimated** at 60 minutes. It needs
the user's go-ahead to submit.

| Run | Gate or record |
| --- | --- |
| `test-kernels` at 1, 28 and 56 workers | Pass |
| `compare_engines.py` 8B FP32: M = 64 GEMM vs per-token, `all` logits, 4 prompts | max\|Δlogit\| ≤ 1e-3, argmax 100% |
| GEMM microbenchmark: 8B layer shapes (4096×4096, 1024×4096, 14336×4096, 4096×14336) at M ∈ {1, 8, 16, 32, 64, 128, 256}, 56 workers, HBM | **Measured** GFLOP/s and GB/s, compared with the **modeled** ridge (B≈32) and the **measured** FP32 FMA peak of 8.8 TF (`docs/hbm-roofline.md:25-31`) |
| Short prefill sweep: 8B FP32, 2048-token prompt, M ∈ {16, 32, 64, 128} | Prefill tok/s recorded (C7 is the full campaign) |

## 7. Exit criteria

- The GEMM passes the kernel budget, and 8B batched logits agree with per-token within
  1e-3.
- The microbenchmark shows compute-bound behaviour above the ridge, or the gap is
  explained (for example L2 blocking or A-packing overhead).

## 8. Risks and open questions

- **Register pressure.** AVX-512 has 32 zmm registers, so the microkernel shape needs
  measurement. Begin with the simpler 4×4 form if 8×32 spills.
- **SWIGLU A-tile reuse.** Running W1 and W3 together doubles the live accumulators.
  The fallback is two GEMMs with an `M × hidden/…` temporary and then the gate.

## 9. Not in this phase

- Batched attention (C3).
- AMX (C4–C6).
- Q8 AVX-512 batched GEMM. It is a loop of the existing W8A8/W8A32 dots per row inside
  the same tiling, added here only as the per-row reference. It is upgraded in C6
  alongside AMX-INT8, where it is the oracle.
