# B4. Cross-socket shared-memory reduction

## 1. Goal and status

Implement the exchange that combines the two sockets' partial outputs and adds the
residual **once**. Use the release/acquire protocol of study §9. It must be:

- race-free;
- safe for buffer lifetimes;
- free of false sharing;
- sized for a vector of `M × dim` FP32 values, so D1's batched prefill can reuse it
  unchanged.

Status: **planned**. Branch: `feature/tp-b4-reduction`.

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
