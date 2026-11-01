# A2. Explicitly bound memory regions

## 1. Goal and status

Add an owned-region allocator that places a large region on **exactly one** NUMA node
before any page is touched. The owning team populates it, and it is freed exactly once.
This is the only route by which weights, KV and workspaces get explicit placement.

Status: **implemented; cluster check passed**. Branch: `feature/numa-a2-bound-regions` (stacked on A1).

Implementation notes:

- **`maxnode` is `max_node + 2`, not `max_node + 1`.** The kernel reads `maxnode - 1`
  bits of the mask (as libnuma passes `size + 1`), so a plain bit count would drop the
  highest node id and make `mbind` to that node fail with `EINVAL`. This corrects the
  wording in §4.2 step 3.
- Descriptors gain `data_bytes` (rounded extent that is bound) and `populated`. The
  registry keeps its own copies, so it can release regions whose descriptors are gone;
  it is registered with `atexit` on the first reservation, which makes the `fail()` path
  (`exit`) release everything.
- `region_reserve_try` returns a message; `region_reserve` calls `fail()`. A zero or
  overflowing size is an error. The per-node budget is checked against the node's
  `MemTotal` and the bytes already reserved on it.
- `MemoryKind` gains `MEM_WEIGHTS_OWNED` and `MEM_WEIGHTS_PACKED` (report keys
  `owned_weights_peak`, `packed_weights_peak`); `memory_track` / `memory_untrack`
  account for memory that is not `memory_alloc`ed.
- The Linux code compiles against stub headers (`-fsyntax-only`) but has not run.
  `LLAMA_NUMA_FAULT` injection is tested only on Linux (`tests/test_numa.py`).
- The report's `regions[]` list is F3's; the registry exposes `region_live_at`.

## 2. Depends on / unblocks

- Depends on: A1 (node IDs, `max_node`, the allowed set).
- Unblocks: A3, A4, A5, and every per-socket buffer in B, C5, C6 and D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `memory_alloc` / `memory_free` | `src/runtime.c:67-91` | `calloc` with an inline header. Keep it for small and default allocations. |
| `MemoryKind`, counters | `include/runtime.h:34-43`, `src/runtime.c:16-20` | Shared accounting. Ordinary globals, so updates must stay on the coordinator. |
| `checked_mul` / `checked_add` | `src/runtime.c:57-66` | Size and offset arithmetic |
| Allocation order | study §5.2 steps 1–7 | Normative for this phase |

## 4. Changes

### 4.1 Region type and API

The proposed descriptor comes from study §5.2:

```c
/* Proposed. */
typedef enum { BACKEND_PORTABLE, BACKEND_LINUX_BOUND } RegionBackend;
typedef enum { ROLE_WEIGHTS, ROLE_PACKED, ROLE_EMBEDDING, ROLE_KV_K, ROLE_KV_V,
               ROLE_WORK, ROLE_SCRATCH } RegionRole;
typedef struct {
  void *mapping_base; size_t mapping_bytes;   /* what munmap receives */
  void *data; size_t logical_bytes;
  int socket, node;                           /* node -1 = portable */
  RegionRole role; RegionBackend backend;
  int thp_requested, read_only;
} NumaRegion;

int  region_reserve(NumaRegion *r, size_t bytes, int socket, int node, RegionRole role, int thp);
void region_mark_populated(NumaRegion *r);    /* coordinator accounting after owner fill */
void region_protect(NumaRegion *r);           /* mprotect PROT_READ for weights */
void region_release(NumaRegion *r);           /* idempotent, exactly-once */
```

### 4.2 Allocation order (Linux bound backend)

1. **Compute sizes.** Check them for overflow, round up to the page size (or the THP
   size when THP is requested), and check them against the socket budget.
2. **Reserve.** `mmap(NULL, n, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)`.
   Do not use `MAP_POPULATE` and do not write anything into the region. Metadata lives
   in the `NumaRegion` struct, outside the payload.
3. **Bind.** `syscall(SYS_mbind, addr, n, MPOL_BIND, mask, maxnode, 0)`:
   - The mask has a single node bit.
   - `maxnode` is a bit count (`topology.max_node + 1`), not a byte count.
   - Use `<linux/mempolicy.h>` constants and `<sys/syscall.h>` `SYS_mbind`; syscall
     numbers are not hardcoded.
4. **Optional THP.** With `LLAMA_THP=on`, call `madvise(MADV_HUGEPAGE)` before
   population. Starts and sizes are aligned to 2 MiB.
5. **Populate.** The owning team writes the region; this is the caller's
   responsibility. For weights, the loader's copy provides the writes, with no extra
   memset (A4).
6. **Mark populated.** After population, the coordinator updates accounting
   (`region_mark_populated`). Workers never touch the global counters.
7. **Protect.** `region_protect` makes weight regions read-only after loading.
8. **Release.** `munmap(mapping_base, mapping_bytes)`. A per-process registry of live
   regions lets `fail()` paths and partial startup release everything exactly once.

### 4.3 Failure semantics

- **Bind failure.** If `LLAMA_PLACEMENT=hbm` or `ddr` was requested and `mbind` fails
  (ENOSYS, EPERM, EINVAL, or the node is not allowed), the engine calls `fail()`. The
  message names the node, the errno, and the `mems_allowed` value from A1. It never
  falls back to another node.
- **Portable backend.** On macOS, or with `LLAMA_PLACEMENT=file`, regions use
  `mmap`/`munmap` without binding and report `node:-1`.
- **Capacity.** An oversubscribed node can still kill the process at fault time (study
  §5.2). The guard is the per-socket budget check in step 1, not a recovery path.

### 4.4 Accounting and report

- `MemoryKind` adds `MEM_WEIGHTS_OWNED` and `MEM_WEIGHTS_PACKED`, and
  `src/report.c:265` `names[]` grows to match.
- Every live region appears in the report's `regions[]` (F3).

## 5. Local tests

These run on macOS with the portable backend and on Linux CI-like hosts with the bound
backend.

- Size arithmetic: overflow near `SIZE_MAX`, page and THP rounding, zero bytes.
- Double release is a no-op, and releasing an unreserved descriptor is safe.
- A simulated `fail()` during a 3-region startup leaves no live registry entries and
  `memory_current` at zero.
- Linux only, with a test-only `LLAMA_NUMA_FAULT=EPERM` injection: a bind failure under
  `LLAMA_PLACEMENT=hbm` exits with the documented message, and the JSON report has
  status `error`.

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check PHASE=A2`. **Estimated** at 10 minutes. It needs
the user's go-ahead to submit.

| Test | Expectation |
| --- | --- |
| Reserve and bind 1 GiB on each of nodes 0, 1, 2, 3 from a socket-0-pinned thread | `mbind` returns 0 on all four |
| Same, under `numactl --membind=0,2` | Nodes 1 and 3 fail with the documented message |
| THP request on node 2 | `madvise` returns 0. Actual backing is checked in A3. |
| Startup failure injection | No leaked mappings (`/proc/self/maps` before and after) |

### Cluster results (2026-11-01)

cresco8-hbm14, job 5622320, commit `7faa81d` (`results/m1-check-20261101-3/A2`).

- `mbind` returns 0 for 1 GiB on each of nodes 0, 1, 2 and 3 from a socket-0-pinned process.
- **Under `numactl --membind=0,2` nodes 1 and 3 also bind**: a VMA policy set by `mbind` overrides the task policy, so
  the plan's expectation that they fail was wrong (A3 confirms the pages land on node 1). The consequence is good for
  A4: a job-wide `--membind` cannot misplace a region. A node outside the cpuset-allowed set is refused by the kernel,
  which this check cannot reproduce; the documented message is exercised through `LLAMA_NUMA_FAULT=EPERM`.
- THP request on node 2: `madvise(MADV_HUGEPAGE)` succeeds (`thp_requested=1`).
- Startup failure injection: after a failed second reservation and `region_release_all`, `/proc/self/maps` has the same
  number of mappings and the same total mapped bytes as before (no leak).
## 7. Exit criteria

- Regions bind on every allowed node, and a disallowed node fails clearly.
- Every mapping is released exactly once, including on error paths.
- Accounting stays on the coordinator, and the report lists every region.

## 8. Risks and open questions

- **Huge and mixed regions.** VMA counts can grow with many separate regions (one per
  layer per role). Prefer one region per {socket, role} with suballocation that never
  crosses roles with different node or lifetime (study §5.5).
- **Interaction with the job's memory policy.** A job-wide `numactl --membind` from
  the sbatch script would conflict with per-region policy. The new scripts launch
  `teams` runs with CPU binding only and leave memory policy to the engine.

## 9. Not in this phase

- Verifying placement (A3).
- Loading weights into regions (A4).
- Migrating pages, which this design deliberately avoids (study §5.2).
