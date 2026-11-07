# C4. AMX enablement

## 1. Goal and status

Make AMX available as a runtime-selected ISA level without adding dependencies:

- CPU feature detection;
- the Linux permission request for tile data state;
- tile configuration per worker thread;
- a 64-byte-aligned workspace;
- tested BF16 and INT8 tile microkernels.

Nothing in the forward pass uses AMX yet. That comes in C5 and C6.

Status: **implemented locally; AMX hardware check pending** (2026-11-07). Branch: `feature/prefill-c4-amx-enable`
(worked on in the `tp-local-c` worktree). Everything below is in; what has run is listed in §5, and what needs a
real AMX node is the open cluster check in §6.

## 2. Depends on / unblocks

- Depends on: C2 (GEMM driver and tiling skeleton) and F2 (per-worker initialization
  hook).
- Unblocks: C5, C6.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| ISA enum and selection | `include/kernels.h:12-15`; `src/kernels_simd.c:22-65` | Static `isa` global, `supported()` via `__builtin_cpu_supports`, `LLAMA_KERNELS` env |
| Target-attribute pattern | `src/kernels_simd.c:14-20` | One attribute macro per ISA level |
| Test ISA loop | `tests/kernels.c:281-298` | Runs each available ISA with a reseeded RNG |
| Peak data | `docs/hbm-roofline.md:25-37` | **Measured** AMX-BF16 112 TF and AMX-INT8 225 TOPS (56 cores), oneDNN 27 TF and 69 TOPS at M = 256 |

## 4. Changes

### 4.1 ISA level

- **Enum.** `KERNELS_AMX` is added above `KERNELS_AVX512`. AMX implies that the
  AVX-512 kernels stay in use for everything except the batched GEMM tiles.
- **Detection.** `__builtin_cpu_supports("amx-tile")`, `"amx-bf16"` and `"amx-int8"`.
  GCC 11 supports these names; the cluster build is GCC 11.4.1. If the builtin is
  unavailable, the fallback is CPUID leaf 7 subleaf 0, EDX bits 22, 24 and 25.
- **Selection.** `LLAMA_KERNELS=amx` is an explicit request: it fails if AMX or its
  permission is unavailable. `auto` never selects AMX until C7 decides the defaults.

### 4.2 OS permission (Linux)

- **Request.** Call `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`
  once per process, before any worker thread executes a tile instruction:
  - `ARCH_REQ_XCOMP_PERM` is 0x1023 and `XFEATURE_XTILEDATA` is 18.
  - The values come from `<asm/prctl.h>` when it is available, otherwise from
    documented constants guarded by a static check.
- **Verification.** `ARCH_GET_XCOMP_PERM`, then the result goes into the report
  (`build.amx_permission`).
- **Failure.** If the kernel denies the request (older kernels, seccomp), `amx` fails
  with a message.

### 4.3 Tile configuration

- **Per worker.** Each worker thread calls `_tile_loadconfig` with a 64-byte
  configuration once, at its first AMX job (a per-thread `configured` flag), and
  `_tile_release` at pool stop.
- **BF16 tiles.** A `16 × 32` BF16 tile (64 bytes per row); B as VNNI pairs, so
  `K/2` rows × `N·2` BF16 values; C as `16 × 16` FP32.
- **INT8 tiles.** A `16 × 64` int8; B as VNNI quads (`K/4` × `N·4`); C as `16 × 16`
  int32.

### 4.4 Workspace

`memory_alloc` gives only 16-byte alignment (`src/runtime.c:22-28`). AMX buffers come
from `NumaRegion` (page-aligned, A2), or from an `aligned_alloc(64, …)` wrapper with
accounting before A2 lands.

### 4.5 Microkernels in `src/gemm.c`

The `amx` target attribute is used only here:

- `amx_bf16_tile_gemm(C, A_bf16, B_packed, m ≤ 16·2, n ≤ 16·2, k)`: 2 × 2 C tiles,
  with two A tiles and two B tiles in flight.
- `amx_s8_tile_gemm(C_i32, A_s8, B_packed, …)`: the same with `_tile_dpbssd`.

### 4.6 Packing helpers

`pack_bf16_vnni(dst, W, rows, cols, ld)` and `pack_s8_vnni(dst, Wq, rows, cols, ld)`.
Both take a `Shard` (B1), so one-socket and two-socket packing share the same code.

## 5. Local tests

- **macOS / non-AMX.** The AMX tests skip with the message `amx unavailable`, and the
  build still succeeds. The packing helpers are tested on every platform, by unpacking
  and comparing bitwise.
- **AMX hosts.**
  - BF16 tile GEMM vs a double reference, computed on the BF16-rounded inputs, within
    `2e-6·Σ|w·x|`. That isolates accumulation error from conversion error.
  - INT8 tile GEMM: exact int32 equality with a scalar int32 reference.
  - Shapes cover tails (`m`, `n` not multiples of 16; `k` not a multiple of 32 or 64)
    through padded packing.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C4`. **Estimated** at 20 minutes. It needs
the user's go-ahead to submit. **Open, not submitted**: `phase_C4` is written and verified with `DRY_RUN=1`; no AMX
instruction has run yet.

`phase_C4` runs `test-amx`, `test-amx-emul`, `test-kernels` and `test-gemm` at 1 and 56 workers (both now repeat their suite
at the `amx` level) and `amx-bench` (`tests/amx_bench.c`); it records PASS only when the output shows the hardware tiles
(`amx: hardware tiles, permission granted`) and `kernels: amx teams:56 ok`, so an `amx unavailable` skip cannot pass.

| Run | Gate or record |
| --- | --- |
| `test-kernels` with `LLAMA_KERNELS=amx` at 1 and 56 workers | `amx ok` |
| Permission report | `ARCH_GET_XCOMP_PERM` shows XTILEDATA |
| Tile microbenchmark: one core and 56 cores, BF16 and INT8, L2-resident operands | **Measured** TFLOP/s and TOPS vs the **measured** 112 TF / 225 TOPS peaks |

### Local results (2026-11-07, macOS arm64, no AMX)

| Check | Result |
| --- | --- |
| `make test` (`test-amx`, `test-amx-emul`, Python suites) | green; `test-amx` prints `amx unavailable: ...` and skips the tile kernels |
| `test-amx-emul`: BF16 tile GEMM, 7 x 6 x 11 shapes, tails in m, n and k, a leading dimension wider than the padded K | worst error 1.1e-7 of Σ\|w·x\| against a 2e-6 budget |
| `test-amx-emul`: INT8 tile GEMM, same shapes, values in [-128, 127] | exact int32 equality |
| `test-amx-emul`: 6 worker threads, 3 rounds, each configuring its own tiles | pass |
| Packing helpers, every platform | unpacked through the tile geometry, bitwise equal, padding zero, `_tiles` range forms identical |
| `-arch x86_64` build of `src/gemm.c`, `src/kernels_simd.c` and `tests/amx.c`, run under Rosetta | compiles clean; the CPUID path reports `this CPU does not report AMX-TILE, AMX-BF16 and AMX-INT8` |

Not exercised locally, because they need an AMX CPU on Linux: the real `_tile_*` instructions, the
`arch_prctl(ARCH_REQ_XCOMP_PERM)` request and `ARCH_GET_XCOMP_PERM` read-back, and GCC 11.4.1 compiling the
intrinsics. The emulation model (`tests/amx_emul.h`) accumulates each K pair as even product then odd product with one
FP32 rounding per addition; the hardware's order inside a pair is not specified, which is why the BF16 budget is
2e-6 and not zero.

### Implementation notes

- **ISA level.** `KERNELS_AMX` sits above `KERNELS_AVX512` in `include/kernels.h`. The `isa` the other kernels branch
  on stays `KERNELS_AVX512` when AMX is selected; AMX state lives in `gemm.c` (`gemm_amx_active()`), so no AVX-512 kernel
  changed. Edits to `src/kernels_simd.c` are limited to `supported()`, `kernels_select()`, `kernels_init()`,
  `kernels_name()`, `kernels_amx_active()` and one `gemm_amx_thread_release()` call in `kernels_backend_free()`.
- **Detection.** CPUID leaf 7 subleaf 0 EDX bits 22, 24 and 25 directly, not `__builtin_cpu_supports`, so the result
  does not depend on the compiler's list of feature names. AVX-512 F/BW/DQ/VL/VNNI is required too.
- **Permission.** `gemm_amx_unavailable()` issues the request once (in `kernels_init`, before the worker team exists)
  and reads it back with `ARCH_GET_XCOMP_PERM`. The result is `build.amx_permission` in the report: `not requested`,
  `granted`, `denied` or `unsupported`.
- **Tile configuration.** All eight tiles are 16 rows of 64 bytes (the same configuration serves BF16 and int8). A
  thread configures at its first AMX call; the key destructor releases the tiles when a worker exits, and
  `kernels_backend_free()` releases the coordinator's. No change to `src/workers.c` was needed.
- **Packing layout.** A tile of 16 output channels is stored K-pair-major (BF16) or K-quad-major (int8), so one tile
  load is a contiguous 1 KiB: BF16 element `nt * kp * 16 + (k / 32) * 512`, int8 byte `nt * kp * 16 + (k / 64) * 1024`,
  with K and the channels zero-padded (`kp` = K rounded up to 32 or 64). The helpers take plain `(rows, cols, ld)`
  pointers. A B1 `Shard` is a (row range, column range) copy of the same `W[out][in]` matrix with its own stride
  (`ld` = `cols`), so the call is `pack_*(dst, shard->q, shard->rows, shard->cols, shard->ld)`;
  `pack_bf16_job_from_shard` and `pack_s8_job_from_shard` build that job (with a check that the shard is a bound ROWS or
  COLS shard of the right dtype), and `pack_*_vnni_many` takes one job per shard. Added on the integration branch
  `feature/tp-merge-b-c`; `tests/shards.c` checks, for every ROWS and COLS shard of 96 plans (FP32 and Q8, one and two
  sockets), that the job's bytes equal packing the matching slice of the source tensor in place. Not done: the engines
  still pack whole matrices (`run.c`, `runq.c`), and the two-socket path (`src/tp.c`) does not pack at all, so
  `LLAMA_SOCKETS=2` with `LLAMA_PRECISION=bf16` is rejected until D1 wires a packed forward per socket. A Q8 shard's
  group scales stay in `shard->s` and are read in place, as `K8Weight` does.
- **Microkernel contract.** `A` must be zero-padded and readable up to whole tiles, `C` has room for whole 16 x 16
  tiles. The C5 and C6 drivers own that padding in their aligned A panels.
- **Workspace.** `aligned_buffer_alloc()` carves a 64-byte-aligned range out of an accounted `memory_alloc`; the C5
  packed weights use a bound region instead.

## 7. Exit criteria

- AMX is detected, permitted, configured per worker and released.
- The microkernels are exact (INT8) or within budget (BF16).
- Non-AMX builds and hosts are unaffected.

## 8. Risks and open questions

- **Frequency licence.** AMX use can lower core frequency. C7 records frequency where
  possible, and any decode slowdown after a prefill must be seen in the separate decode
  measurement.
- **Tile state cost.** XTILEDATA adds 8 KB of signal and context-switch state per
  thread. It is harmless here, but documented.

## 9. Not in this phase

- Forward-pass integration (C5, C6).
- AMX attention.
- AMX decode, which the roofline shows does not help (`docs/hbm-roofline.md` §3).
