# B2. Partial-output and column-slice kernels

## 1. Goal and status

Add the kernel variants that a shard needs:

- **Projections without a fused residual.** A `PARTIAL` epilogue writes `out = W·x`
  into a socket-local partial buffer.
- **Input-column-slice GEMV.** Wo and W2 run on packed shards.
- **Team-local entry points** for every projection, in FP32, W8A8 and W8A32.

The residual is added exactly once, after the cross-socket sum (B4). Without this, a
per-socket `ADD` epilogue would apply the residual twice (study §6.2).

Status: **implemented; cluster check passed** (see Cluster results).

Implementation notes:

- **Entry points** (`include/kernels.h`, `src/kernels_simd.c`): `k_matmul_ld`, `k_matmul_swiglu_ld`, `k_matmul_q8_ld`,
  `k_matmul_q8_swiglu_ld`, the `KernelEpilogue` kinds `K_STORE`, `K_ADD`, `K_PARTIAL`, `K_SWIGLU`, and
  `kernels_set_thread_team`. `K_PARTIAL` is the `PARTIAL` epilogue: it stores like `K_STORE`. The `_ld` forms
  validate `ld >= n` (and `ld % gs == 0` for Q8) and reject `K_SWIGLU` outside the pair form.
- **Existing entry points are untouched.** Their jobs carry `ld = 0` (meaning `n`), so the same row stride and the
  same AVX-512 `dots_*` calls result. The AVX-512 Q8 bodies gained an `ld` parameter next to `n` (weight rows use
  `ld`, scales use `ld / gs`, the input length stays `n`); the FP32 body already took a stride. With `ld == n` every
  address expression equals the old one.
- **Team-local.** `kernels_set_thread_team(team)` binds the calling thread's kernels to a team, overriding the
  process-wide `kernels_set_team` for that thread, because two socket coordinators must run kernels on two teams at
  the same time. The B3 attention entry points use the same selection. QKV on a shard needs nothing new: the three
  segments already take their own row counts (`dim/S`, `kv_dim/S`, `kv_dim/S`) and split per segment.
- **Tests** (`tests/kernels.c`, scalar, team sizes none/1/2/3/5/7; the AVX-512 half runs on the cluster).
  - Column slices of random `W[d][n]` at `n/2` with `K_PARTIAL`, sum in double: within `2e-5·Σ|w·x| + 1e-6`, with
    padded (`ld = n/2 + 16`) and packed rows; padding is NaN (FP32) or 127 and NaN scales (Q8), so a read of
    padding fails.
  - Q8 W8A8 and W8A32 for gs 4, 16, 64, 128: each half's activations are quantized from the owned slice and must equal
    the corresponding slice of the full quantization **bitwise** (values and scales); the halves sum to the double
    reference within the budget; `K_ADD` equals `K_PARTIAL` plus the base.
  - Row slices: the shard's rows equal the full product's rows **bitwise for the scalar kernels**. Specification
    said bitwise for every ISA; the AVX-512 GEMV reduces a 4-row block differently from a 1-3-row tail, so a row
    computed in another block position can differ in the last bits (already true across team splits). The test asserts
    bitwise on scalar and the budget otherwise.
  - Two coordinator threads, each bound to its own team (3 and 2 workers), run 300 GEMVs each concurrently.
  - ASAN/UBSAN and TSAN clean on `test-kernels`; a mutation of the scale stride is caught.
- **Cross-compile.** `cc -arch x86_64 -fsyntax-only` compiles `kernels_simd.c`, `tests/kernels.c`, `loader.c` and
  `shards.c` without warnings. There is no AVX-512 emulation of the Q8 `dots` bodies (the emulator in
  `tests/avx512_emul.h` covers only the GEMM microkernel), so the changed AVX-512 Q8 code is reviewed, not executed.
- **Open (cluster).** `test-kernels` on AVX-512 with team sizes 1, 28 and 56 (`LLAMA_TEST_TEAMS`), looking for
  `avx512 ok` and 0 failures, which also covers the new Q8 `ld` argument; the existing Q8 results on AVX-512 (logits
  of the 1B/8B Q8 gate) must be unchanged, since `ld == n` there; `cresco8-tp.sbatch PHASE=B2` (dry-run verified; it runs `LLAMA_TEST_TEAMS=1,28,56` and requires `avx512 ok` and `avx512 teams:56 ok`).

## 2. Depends on / unblocks

- Depends on: B1 (shard layout) and F2 (team-local `team_parallel_rows`; per-segment
  splitting).
- Unblocks: B3, B5 and C2 (C2 uses the same epilogue kinds for GEMM).

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| Epilogue kinds | `src/kernels_simd.c:409` `enum { STORE, ADD, SWIGLU }` | Add `PARTIAL`. It is the same as `STORE` but targets a partial buffer; it stays separate so call sites state their intent. |
| Fused ADD | `src/kernels_simd.c:455` (FP32), `:565` (Q8) | Must not be used per socket when S = 2 |
| `dots_f32_avx512` | `src/kernels_simd.c:112` | Takes a `stride` already, so a packed `[dim][cols]` shard with `ld = cols` works unchanged |
| `dots_q8_avx512` / `dots_q8f_avx512` | `src/kernels_simd.c:163`, `:190` | Need the per-row scale pointer for a column slice: `ws + r·(cols/gs)` in the packed shard. The input `xq`/`xs` covers only the owned slice. |
| Kernel tests | `tests/kernels.c:64-167` | Double reference and budget |

## 4. Changes

- **New `kernels.h` entry points.** They take explicit `ld` and team context. The
  public names are separate; existing calls are untouched.

  ```c
  /* Proposed. */
  void k_matmul_ld(float *out, const float *x, const float *w, size_t ld, int n, int d, int kind);
  void k_matmul_q8_ld(float *out, const K8Input *x, const K8Weight *w, size_t ld, int n, int d, int gs, int kind);
  ```

  - `n` is the input length the shard sees, which is `cols` for a `COLS` shard.
  - `ld` is the destination row stride from B1.
  - `kind` is `STORE`, `ADD` or `PARTIAL`. `SWIGLU` keeps its pair form, renamed
    `k_matmul_swiglu_ld`.
- **QKV on a shard.** It is three segments on the local row counts (`dim/S`,
  `kv_dim/S`, `kv_dim/S`), split per segment across the team (F2).
- **`ADD` stays available** for `S = 1`, so the one-socket path keeps its fused
  residual and its speed.
- **Scalar paths** get the same `ld` parameter. The scalar Q8 fallback
  (`src/kernels_simd.c:511-519`) must handle a column slice with `cols % gs == 0`.

## 5. Local tests

New cases in `tests/kernels.c`, for every ISA (scalar and AVX-512) and team sizes 1, 3
and 7:

- **Column-slice equivalence.** For a random `W[d][n]` split at `n/2`, the two
  `PARTIAL` results summed with a double accumulator must be within the existing budget
  of the full `matmul`.
- **Q8 column slices.**
  - **W8A8:** each half's activation is quantized from the owned slice only, at the
    original group boundaries.
  - **W8A32:** activations stay in FP32.
  - Gs values 16, 64 and 128 are checked against the same budget formula
    (`tests/kernels.c:126-167`).
- **Row slices** produce exactly the rows of the full product, bitwise, because the
  arithmetic per row is unchanged.
- **`ld != n` with padding.** A guard pattern after each row must stay untouched.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B2`. **Estimated** at 15 minutes. It needs the
user's go-ahead to submit. It runs `test-kernels` on AVX-512 with team sizes 1, 28 and
56, and checks for `avx512 ok`.

### Cluster results (2026-11-08)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Jobs 5694397 and 5694402. `test-kernels` passes scalar, AVX-512 and AMX levels on teams of 1, 28 and 56 (7,920,258 checks,
0 failures), including the `PARTIAL` epilogue, the column-slice and padded-`ld` cases; the AVX-512 half of these tests
had never run before this check. `make test` is green on Linux (job 5694402), as are the AMX kernels (`kernels: amx ok`).

## 7. Exit criteria

- Every new kernel passes the existing budget, for every ISA and team size.
- The one-socket path is bitwise unchanged.

## 8. Risks and open questions

- **Summation order of the sum `p0+p1`.** It differs from a full-row dot product.
  Errors stay within the FP32 budget, but they are not bitwise. That is why B5 gates on
  tolerance (study §6.2).

## 9. Not in this phase

- The reduction itself (B4).
- Attention (B3).
- Batched (GEMM) variants (C2).
