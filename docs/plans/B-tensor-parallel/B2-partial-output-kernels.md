# B2. Partial-output and column-slice kernels

## 1. Goal and status

Add the kernel variants that a shard needs:

- **Projections without a fused residual.** A `PARTIAL` epilogue writes `out = W·x`
  into a socket-local partial buffer.
- **Input-column-slice GEMV.** Wo and W2 run on packed shards.
- **Team-local entry points** for every projection, in FP32, W8A8 and W8A32.

The residual is added exactly once, after the cross-socket sum (B4). Without this, a
per-socket `ADD` epilogue would apply the residual twice (study §6.2).

Status: **planned**. Branch: `feature/tp-b2-partial-kernels`.

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
