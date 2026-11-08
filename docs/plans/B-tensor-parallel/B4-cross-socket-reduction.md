# B4. Cross-socket shared-memory reduction

## 1. Goal and status

Implement the exchange that combines the two sockets' partial outputs and adds the
residual **once**. Use the release/acquire protocol of study §9. It must be:

- race-free;
- safe for buffer lifetimes;
- free of false sharing;
- sized for a vector of `M × dim` FP32 values, so D1's batched prefill can reuse it
  unchanged.

Status: **implemented; cluster check passed, latency above the per-token budget** (see Cluster results).

Implementation notes:

- **API** (`include/workers.h`, `src/workers.c`). `exchange_create`, `exchange_destroy`, `exchange_partial`,
  `exchange_reduce_add`, `exchange_abort`, `exchange_region_bytes`, `exchange_set_timing`/`exchange_timing`.
  The proposed signatures changed in two places:
  - `exchange_partial(e, socket)` is called **once per reduction by the socket's coordinator, before the projection
    job**, not by every worker. It numbers the generation and performs step 5 (waits for `consumed_peer ≥ g - 2`),
    so no worker races on a generation counter. It returns `NULL` after an abort.
  - The team of each socket comes from the pool (`pool_team(s)`), so `exchange_reduce_add` takes no `Team *`.
- **Protocol.** Exactly §4.2: local barrier, `ready_s` (release), wait on `ready_peer` (acquire), each worker adds its
  range `residual[i] += p0[i] + p1[i]`, local barrier, `consumed_s` (release). `ready` and `consumed` are on separate
  64-byte lines; the partial buffers are cache-line aligned, in a caller-owned `ROLE_WORK` region or on the heap.
- **Reuse rule is a safety net here.** With a strictly alternating `exchange_partial`/`exchange_reduce_add` per
  socket, `ready_peer ≥ g + 1` already implies that the peer finished reading generation g, so removing the
  `consumed` wait does not change any test result (mutation-checked). It is kept because the protocol states it and
  because it is the guard once B5 decouples the add from the next projection. The abort test covers it directly.
- **Abort.** `workers_abort()` sets a process-wide flag. `fail()` sets it before exit handlers run; `exchange_abort`
  is the same flag. Every exchange wait loop and `team_barrier` return on it. Workers that are still computing when
  `fail()` unmaps regions in its exit handlers can fault; that is not specific to the exchange and not addressed.
- **Two coordinators.** Team 0's `team_run` is called by the engine's main thread, which is worker 0 of team 0. Team 1
  therefore needs a coordinator thread of its own to call `team_run`, concurrently. The test harness uses a pthread;
  B5 must provide the engine's equivalent (see B5, open item).
- **Timing.** With `exchange_set_timing`, the leader of each socket accumulates `reduce` (step 3 add) and
  `barrier_wait` (steps 1, 3 wait, 4, and the reuse wait). Wiring them into the report's `phase_seconds` is B5.
- **Tests** (`tests/workers.c`, run by `make test` through `test-workers` and `tests/test_numa.py`). 2 × T workers
  for T = 1, 3, 7 and E = 1, 7, 4096, 4096·64 (1000 generations; 200 for the largest), team sizes (3, 1) and (2, 5),
  a slow producer or a slow consumer on either socket (random sleeps at three protocol steps), regions and heap
  buffers; the residuals are compared **bitwise** with a serial reference after the last generation. Abort: an
  in-process abort releases workers waiting for `ready` and a coordinator waiting on the reuse rule
  (`exchange_partial` returns `NULL`); `fail()` in a socket-1 worker and in the socket-0 coordinator end the process
  with exit 1 (subprocess, 30 s guard). `test-workers bench-exchange` is the cluster microbenchmark (§6); it lives in
  `tests/workers.c` instead of a separate `exchange_bench.c`.
- **Verified locally.** `make test`, `make MODE=sanitize` and `make MODE=tsan OPENMP=0` on `test-workers`: no reports.
  Mutations (ready published before the local barrier, swapped operand order) are caught by the exactness check.
- **Open (cluster).** Pinned latency for E = 4096 and 4096 × {16, 64, 256} on 2 × {14, 28, 56} workers with the
  buffers on HBM 2 and 3, against the per-token budget of §7; remote-push alternative of §8 (not implemented); `cresco8-tp.sbatch PHASE=B4`
  (dry-run verified: stress at 2 × 14/28/56 workers, `test-workers bench-exchange`, and a summary that prints the
  64-reduction cost against 10% of 27 ms). This Mac's median of 4 µs for E = 4096 (4+4 unpinned workers) is not evidence for CRESCO8.

## 2. Depends on / unblocks

- Depends on: F2 (teams, barriers, atomics conventions).
- Unblocks: B5, B6, D1.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Team barrier and generation counter | F2 `workers.c` | Local completion |
| C11 `<stdatomic.h>` | — | Already in use: `fail` uses `atomic_flag` (`src/runtime.c:40-55`) |
| Protocol | study §9, steps 1–5 | Normative |

## 4. Changes

### 4.1 API

```c
/* Proposed, include/workers.h. */
typedef struct Exchange Exchange;     /* per pair of teams; owns 2 partial buffers per socket */
Exchange *exchange_create(int elements_max, NumaRegion *socket_regions[2]);
float *exchange_partial(Exchange *e, int socket);   /* where local PARTIAL output is written */
/* Called by every worker of socket s inside a TeamFn after the local projection: */
void exchange_reduce_add(Exchange *e, int socket, int worker, int workers,
                         float *residual, int elements);
```

### 4.2 Protocol for one reduction (generation g)

1. **Local completion.** Workers of socket s finish writing `partial_s[g & 1]`, then
   meet at the local team barrier.
2. **Publish.** The leader of s stores `ready_s = g` with `memory_order_release`.
3. **Read and add.** Every worker of s waits until `ready_peer ≥ g` (acquire). It then
   reads its disjoint element range `[w·E/W, (w+1)·E/W)` of **both** partial buffers
   and writes `residual[i] += partial_0[i] + partial_1[i]`.
   - The operand order `p0 + p1` is fixed on both sockets, so both residual copies stay
     bitwise identical.
   - Each socket updates only **its own** residual copy. No shared residual is written
     across sockets.
4. **Acknowledge.** After a local barrier, the leader stores `consumed_s = g`
   (release).
5. **Reuse.** Before writing `partial_s[g & 1]` again (at generation g + 2), socket s
   waits for `consumed_peer ≥ g` (acquire). Double buffering removes the wait in the
   common case; the lifetime rule still holds.

### 4.3 Layout and waiting

- `ready` and `consumed` for each socket are on their own 64-byte lines.
- Partial buffers are in the producing socket's HBM `ROLE_WORK` region, so remote
  reads cross UPI once per element per reduction.
- Waiting is a bounded spin, then `sched_yield`, in the same style as the F2 barrier.
  Futex parking only comes after measurement.
- `LLAMA_PROFILE=1` records `reduce` (step 3) and `barrier_wait` (steps 1 and 5) per
  socket in the report's `phase_seconds`.

### 4.4 Lifetime and errors

- A `fail()` in either team sets a shared abort flag that every wait loop checks, so
  there is no deadlock on error exit.
- Teardown waits for both sockets' last `consumed` before releasing the regions.

## 5. Local tests

These run on macOS, and under TSAN where available.

- **Exactness.** With 2 × T workers (T = 1, 3, 7) and E ∈ {1, 7, 4096, 4096·64},
  1000 generations of random partials. Check the residuals against a serial reference
  **bitwise**, since the operand order is fixed and each element is one addition.
- **Delay injection.** Random sleeps at each protocol step in one socket. Partials of
  generation g must never be overwritten before the peer acknowledges them. A canary
  generation tag stored in each buffer is checked on read.
- **Abort.** `fail()` in socket 1 mid-generation. The process exits with the report
  written and no hang (timeout guard).
- **TSAN.** `make MODE=tsan OPENMP=0` on the stress test with zero reports.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B4`. **Estimated** at 20 minutes. It needs the
user's go-ahead to submit.

A microbenchmark (`tests/exchange_bench.c`) runs with 2 × {14, 28, 56} pinned workers
and partial buffers on HBM 2 and 3. It measures one reduction's latency for E = 4096
(decode) and E = 4096 × {16, 64, 256} (prefill microbatches), reporting the median and
p99 over 10⁵ generations. The model to compare against:

- `32 layers × 2 = 64` reduction phases per token for 8B decode (study §8.3);
- the per-token budget is the gap between the one-socket token time and the target
  two-socket token time.

### Cluster results (2026-11-08)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Job 5694397 (commit `44ce282`), partial buffers on HBM nodes 2 and 3, workers pinned.

- **Correctness.** The stress test (bitwise exactness against the serial reference, delay injection, canary generations)
  passes on 2 × 14, 2 × 28 and 2 × 56 workers, and the abort test (`fail()` in one socket) exits without a hang.
- **Latency of one reduction** (median / p99 over 10^5 generations):

  | Workers per socket | E = 4096 (decode) | E = 65536 | E = 262144 | E = 1048576 |
  | --- | ---: | ---: | ---: | ---: |
  | 14 | 47.3 / 95.2 µs | | | |
  | 28 | 51.3 / 103.4 µs | | | |
  | 56 | 60.0 / 121.0 µs | 64.3 / 127.1 µs | 85.5 / 156.0 µs | 171.5 / 249.2 µs |

- **Budget.** 64 reductions per 8B token cost 3.0 ms (14 workers), 3.3 ms (28) or 3.8 ms (56) at the median, above the
  42.2 µs per-reduction budget of this phase (about 10% of the modeled 27 ms two-socket token). This is the exit-criterion
  case that must be flagged before B5; see B5 for the measured decode, where the reduction is a small part of the loss.

## 7. Exit criteria

- The reduction is bitwise exact and TSAN-clean, and survives delay injection and
  abort.
- Latency is **measured** and compared with the per-token budget.
- If 64 reductions per token cost more than about 10% of the **modeled** two-socket
  FP32 token time (27 ms, study §8.3), flag it before B5 and consider a reduction
  scheme with fewer barriers.

## 8. Risks and open questions

- **Synchronization cost for small models.** On the 1B INT8 model, the two
  sockets' synchronization can cost more than the weight reads it saves (study §9).
  B7 reports where this crossover happens instead of tuning it away.
- **Remote reads are coherent loads.** Remote reads of `partial_peer` are coherent UPI
  loads. If they are the bottleneck, an alternative is for each socket to push its
  partial into the peer's region (a remote store) and read only locally. Measure both
  in the microbenchmark.

## 9. Not in this phase

- Overlapping the reduction with the next projection.
- Reduce-scatter variants.
- More than two sockets.
