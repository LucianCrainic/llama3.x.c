# F2. Persistent worker teams

## 1. Goal and status

Add a dependency-free backend of **persistent pthread worker teams**. It runs the
existing kernels without opening OpenMP regions. With one team, it must match the
OpenMP path numerically and in speed. The same interface later carries:

- B: two socket teams;
- C: batched GEMM tiles;
- D: two-socket prefill.

The user decision requires that no separate scheduler be introduced for prefill, so
this is the only scheduler the plans use.

Status: **implemented; cluster check done except the Hugging Face gate** (see Cluster results). Branch: `feature/found-f2-worker-teams` (stacked on A3).

Implementation notes:

- **`include/workers.h`, `src/workers.c`.** One pool, one or two teams, the caller is worker
  0. Counters (`generation`, `done`, `ready`, `stop`, barrier count and generation) are
  64 bytes apart; waiters spin 2048 times with `pause`/`yield`, then `sched_yield`.
  `team_affinity_verified` and `team_worker_cpu` carry the read-back for F3.
- **`fail()` in a worker** exits the process from that thread; the leader (spinning on
  the job counter) cannot hang it, and `pool_stop` returns immediately when called from a
  pool thread. Both are tested with subprocess timeouts.
- **Kernels.** `kernels_set_team` routes `parallel_rows`, the F32/Q8 drivers and
  attention to the team. `f32_team` / `q8_team` split every segment (Q, K, V) on its own
  inside **one** `team_run`, so no extra barrier is added; the OpenMP path keeps its
  concatenated split unchanged (it is not moved to per-segment splitting here, so its
  results are bit-for-bit what they were). Attention uses a team-local atomic task queue,
  `team_barrier`, then the head merge; the OpenMP path shares the same
  `attention_task`/`attention_merge` helpers. `AttentionWork.workers` is set at init, so
  the team must be set before `k_attention_init`.
- **Backend.** `kernels_backend_init`/`kernels_backend_free` (called from
  `engine_load_context`/`engine_free`) read `LLAMA_BACKEND=omp|teams`, `LLAMA_THREADS`
  and `LLAMA_TOPOLOGY`. Without a topology the team pins its workers to the allowed
  CPUs in order (Linux) or runs unpinned; with one, team 0 uses socket 0's CPUs
  (virtual: unpinned). A team of two sockets is B's.
- **Build.** `-pthread`, `MODE=tsan`, `workers.c` in `COMMON`; `OPENMP=0` is the default
  and is what the teams tests run under.
- **Tests (macOS, scalar).** `tests/kernels.c` runs every kernel on teams of 1, 2, 3, 5 and
  7 against the double references, plus attention at 31/32/255/256/4095 positions;
  `tests/workers.c` covers lifecycle, row coverage and a randomly delayed barrier stress;
  `test_runtime.py` runs the fixture logits under teams of 1, 2, 3 and 7 within the
  baseline tolerance. TSAN reports nothing on `test-workers`, `test-kernels` and the
  engine fixtures under teams.
- **Not run.** This Mac has no AVX-512, so the AVX-512 team paths are only type-checked
  (`-arch x86_64 -fsyntax-only`, also with a stub `omp.h` for the `-fopenmp` build). Pinning
  and read-back are Linux-only. `test-workers bench-barrier` measures the per-barrier cost
  for the cluster `check`.

## 2. Depends on / unblocks

- Depends on: nothing.
- Unblocks:
  - A5: owner-team first touch.
  - B2, B3, B4: team-local kernels and the reduction.
  - C2: GEMM tiles.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `parallel_rows` | `src/kernels_simd.c:70` | Bare `omp parallel`, contiguous `[rows*t/nt, rows*(t+1)/nt)`. Keep it as the `omp` backend. |
| `F32Job`, `f32_range`, `f32_rows`, `f32_run` | `src/kernels_simd.c:446-481` | Row-range helpers. Already independent of the threading model. |
| `Q8Job`, `q8_range`, `q8_rows`, `q8_run` | `src/kernels_simd.c:556-591` | Same, for Q8 |
| Epilogue kinds | `src/kernels_simd.c:409` `enum { STORE, ADD, SWIGLU }` | B2 adds `PARTIAL` |
| `max_threads`, `attention_chunks`, `k_attention_init` | `src/kernels_simd.c:651, 662, 675` | Must take a team size instead of `omp_get_max_threads()` |
| `k_attention` task loop | `src/kernels_simd.c:735` | `omp for schedule(dynamic,1)` over `n_kv_heads*chunks`, then the merge |
| Rule: OpenMP regions only in plain C drivers | `src/kernels_simd.c:1-4`, `docs/kernels.md:20-23` | The same rule applies to team entry points |

## 4. Changes

### 4.1 Worker API

`include/workers.h` and `src/workers.c`:

```c
/* Proposed. */
typedef struct Team Team;               /* opaque: threads, barrier, job slot */
typedef struct {
  int teams;                            /* 1 or 2 */
  int threads[2];                       /* per team */
  const int *cpus[2];                   /* pinned CPU ids, NULL = unpinned */
} PoolConfig;

void pool_start(const PoolConfig *cfg); /* threads created once, at load */
void pool_stop(void);
Team *pool_team(int index);
int team_size(const Team *t);
int team_index(const Team *t);          /* socket/team id */

typedef void (*TeamFn)(const void *job, int worker, int workers);
void team_run(Team *t, TeamFn fn, const void *job);        /* leader + workers, returns after barrier */
void team_parallel_rows(Team *t, int rows, RowFn fn, const void *job);
void team_barrier(Team *t);             /* usable inside a TeamFn */
```

How the pool behaves:

- **Leader.** The leader (the caller) is a member of its team and does not spin as an
  extra thread (study §9).
- **Pinning.** Each worker is pinned with `pthread_setaffinity_np` on Linux, and its
  affinity is read back and reported (F3). On macOS, workers are unpinned.
- **Synchronization.** A generation-counted barrier using C11 atomics. Workers spin
  for a bounded time, then call `sched_yield`; futex parking is deferred until
  measurements justify it (study §9). Counters sit on separate 64-byte cache lines.
- **Startup.** The pool starts in `engine_load_context` (`src/run.c:211`,
  `src/runq.c:294`) when `LLAMA_BACKEND=teams`.
- **Thread count.** `LLAMA_THREADS` gives the total; otherwise it comes from the
  allowed CPU set (A1 makes this topology-aware). `LLAMA_TOPOLOGY=virtual:SxT` gives
  S unpinned teams of T threads.

### 4.2 Kernel changes in `src/kernels_simd.c`

- **Per-segment splitting.** `f32_run` / `q8_run` stop concatenating the Q/K/V row
  spaces (:476, :586). Each segment is divided across the team on its own, so a
  per-team or per-socket range never straddles projections. The OpenMP path keeps
  today's behaviour until parity is shown, and is then moved to per-segment splitting
  too, so that the omp and teams backends compute identical rows per thread count.
- **Team-aware attention.** `AttentionWork` stores `workers` instead of calling
  `omp_get_max_threads()`. `attention_chunks(positions, n_kv_heads, limit, workers)`
  takes the same input.
- **Attention task queue.** The `k_attention` task loop becomes a team-local atomic
  counter queue over `(kv group, chunk)` tasks, then `team_barrier`, then the merge
  over heads split by worker.
- **Dispatch.** `k_matmul*` and `k_attention` dispatch to `team_parallel_rows` when a
  current team is set (`kernels_set_team(Team *)`), and to `parallel_rows` otherwise.
  The public kernel signatures are unchanged.

### 4.3 Build

- The `Makefile` adds `-pthread` to `CFLAGS` and `LDLIBS`, and adds `MODE=tsan`
  (`-fsanitize=thread -O1 -g`).
- `workers.c` joins `COMMON`.
- The `teams` backend works with `OPENMP=0`. Combining `OPENMP=1` with
  `LLAMA_BACKEND=teams` is allowed, but the engine logs that OpenMP regions are unused.

## 5. Local tests

All of these run on macOS.

- **`tests/kernels.c`:** a new loop runs every kernel with `LLAMA_BACKEND=teams` at
  1, 2, 3 and 7 workers (odd counts stress range splitting) and checks it against the
  double reference with the existing budget.
- **Attention:** 1 and 5 workers on positions 0, 31, 32, 255, 256 and 4095. These
  cover the chunk boundaries at 32 and 256 (`attention_chunks`).
- **`tests/test_runtime.py`:** the fixture logits under `LLAMA_BACKEND=teams` match
  `baseline_logits.json` within the existing tolerance. They are bitwise equal when the
  omp and teams backends use the same per-segment split and thread count.
- **`make MODE=tsan OPENMP=0 test-kernels`** shows no reports. A barrier stress test
  runs 10⁶ generations with random worker delays.
- **Pool lifecycle:** start/stop leaves `memory_current` at zero, and `pool_stop`
  after a `fail()` in a worker does not hang.

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check` (the F3 skeleton), run on the Flat node.
**Estimated** at 30 minutes. It needs the user's go-ahead to submit.

| Run | Config | Purpose |
| --- | --- | --- |
| `test-kernels` | teams, 1 team × {1, 56} threads, AVX-512 | ISA × backend parity on hardware |
| `validate.py` 1B, 8B FP32 | `LLAMA_BACKEND=teams`, 56 pinned on socket 0, staged HBM | Same HF gate as today |
| Decode benchmark 8B FP32/Q8 | omp vs teams, 56 threads, staged HBM, 3 repeats | No regression |
| Affinity dump | per worker | Pinning read-back recorded |

The workload is identical to `scripts/cresco8-kernels.sbatch:116-136` (`short.txt`,
`--context-length 2048 --generation-length 128 -t 0 --seed 42 --chat-template`).

### Cluster results (2026-11-01)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`. Job 5622314, commit `58f091a`,
`results/m1-check-20261101-1..-3` under `~/PFS/lpor` (first job 5622296 failed and is kept as the record of two bugs:
an `OMP_PROC_BIND` launcher pin and a topology allocation counted as a leak, both fixed). Weights staged in `/dev/shm`
on HBM node 2, `numactl --cpunodebind=0`, 56 threads, `short.txt`, 128 generated tokens, 3 launches (median [range]).

| Model, precision | omp decode tok/s | teams decode tok/s | teams / omp |
| --- | ---: | ---: | ---: |
| 8B FP32 | 15.37 [15.32-15.37] | 15.96 [15.82-15.97] | +3.8% |
| 8B Q8_0 | 45.24 [45.18-45.47] | 50.04 [50.00-50.17] | +10.6% |
| 1B FP32 | 86.97 [86.12-87.69] | 90.82 [90.70-91.23] | +4.4% |
| 1B Q8_0 | 249.99 [249.97-252.81] | 268.75 [265.68-268.90] | +7.5% |

- **Kernel parity on hardware.** `test-kernels` passes scalar and AVX-512 on teams of 1, 2, 3, 5, 7 (`make test`)
  and 1 and 56 (`LLAMA_TEST_TEAMS=1,56`): 0 failures.
- **Agreement, C against C** (`tools/compare_engines.py`, 53 positions, omp vs teams): FP32 max |Δlogit| 9.5e-6 (1B) and
  2.5e-5 (8B), argmax 100%; Q8_0 bitwise identical on both models.
- **Pinning.** 56 workers pinned to cores 0-55; every read-back equals its assignment (`affinity_verified`).
- **Barrier cost** (unpinned team, 200000 rounds): 0.36 µs at 2 threads, 1.1 at 4, 2.1 at 8, 2.6 at 16, 4.7 at 32,
  3.5 at 56. `barrier 1000000 56` (random delays) passes.
- **Not run.** The Hugging Face gate (`validate.py`) under `teams`: there is no torch/transformers environment on the
  cluster any more, so `HF_GATE=0` skipped it. The transitive argument (teams agrees with omp within 2.5e-5, and omp
  passed the gate in the earlier campaign) is recorded, not substituted for the gate. TSAN is clean locally only.
## 7. Exit criteria

- **Kernel parity:** every kernel passes at all tested worker counts, scalar and
  AVX-512.
- **TSAN:** clean.
- **Real model gate:** 1B and 8B FP32 pass `validate.py` under `teams`.
- **Speed:** one team of 56 decodes within the run-to-run range of OpenMP (**measured**
  baseline 14.90 tok/s FP32 and 44.49 tok/s Q8_0, 8B), or the regression is understood
  and recorded with a per-region timing breakdown.
- **Affinity:** every worker's read-back affinity equals its assigned core.

## 8. Risks and open questions

- **Synchronization cost.** OpenMP already uses 5 regions per layer
  (`docs/kernels.md:42-45`), and the 1B model is partly bound by region
  synchronization (`docs/kernels.md:131-133`). A naive pthread barrier could be
  slower. Measure the per-barrier cost in the `check` job before tuning.
- **`fail()` in worker threads.** `fail` uses a single-shot flag, and other failing
  threads `pause()` (`src/runtime.c:40-55`). The pool must not wait forever on a
  worker that called `fail`.
- **Changed thread-to-row mapping.** Per-segment splitting changes which thread
  computes which rows on the OpenMP path. Results stay deterministic for a fixed
  thread count, but logits can change in the last bits compared with the recorded
  baseline. Re-record only if the change is understood.

## 9. Not in this phase

- Two teams with real placement (B).
- Cross-socket reduction (B4).
- GEMM tiling (C2).
- Futex parking.
- Overlapping communication with compute.
