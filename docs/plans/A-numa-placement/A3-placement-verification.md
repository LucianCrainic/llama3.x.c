# A3. Placement verification

## 1. Goal and status

Check where populated pages actually reside, and report it. This is how a run earns
the label "local HBM". The check distinguishes between:

- policy and observed residency;
- full and sampled coverage;
- unpopulated and wrong-node pages.

It also records actual huge-page backing (study §5.5, §10.2).

Status: **implemented; cluster check passed** (see Cluster results). Branch:
`feature/numa-a3-verification` (stacked on A2).

Implementation notes:

- `region_verify` queries `ceil(logical_bytes / page)` pages, so the unpopulated tail of a THP-rounded region is not
  counted as missing. It returns -1 for a portable region or when `move_pages` is blocked (then use
  `region_numa_maps` and label the run `unverified-numa_maps`).
- `smaps` `AnonHugePages` is per VMA and VMAs merge, so a partly overlapping VMA contributes in proportion to the
  overlap: `thp_bytes` is an estimate there and exact where the region covers whole VMAs.
- Sampling: both ends of every 2 MiB extent (evenly thinned when `max_samples` is smaller than the endpoint count),
  then a deterministic stride; `PageCheck.scheme` records it and a sampled check has `full = 0`.
- `LLAMA_VERIFY_SAMPLES` and `LLAMA_PLACEMENT_STRICT` are read through `region_verify_samples_default` and
  `placement_strict`; the callers (A4 onwards) own the timing split and the fatal path.
- `test-regions verify NODE BYTES none|half|all [thp] [samples=N] [expect=M] [threads=T]` is the cluster probe: it
  reports populate bandwidth (optionally from a pinned team), verification cost, `thp_bytes` and the `numa_maps`
  policy and per-node pages. `expect=M` is the negative control.
- Local tests cover status classification, sampling, and the `smaps`/`numa_maps` parsers on fixture text; the 64 MiB
  half-touched region test runs on Linux only.

## 2. Depends on / unblocks

- Depends on: A2.
- Unblocks: A4, A5 and A6, plus every campaign that claims a placement (B7, C7, D2).

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| `NumaRegion` registry | A2 | The engine's own extent manifest. VMAs can merge, so `/proc` alone is not enough. |
| Report `regions[]` | F3 | `verified`, `pages_queried`, `pages_mismatched`, `pages_missing`, `thp_bytes` |

## 4. Changes

### 4.1 Page query

```c
/* Proposed. */
typedef struct { size_t queried, on_node, wrong_node, missing, errors; int full; } PageCheck;
int region_verify(const NumaRegion *r, size_t max_samples, PageCheck *out);
```

- **The call.** `syscall(SYS_move_pages, 0, count, pages, NULL, status, 0)` requests
  location only, with no migration. It runs in batches of 4096 addresses.
- **Status classification.**

  | Status | Classified as |
  | --- | --- |
  | `>= 0` | the page's node, compared with `r->node` |
  | `-ENOENT` | `missing` (not populated) |
  | other negative | `errors` |

- **Coverage.**
  - **Full** when `pages <= max_samples`. The default `max_samples` is 0, meaning
    full; `LLAMA_VERIFY_SAMPLES=N` limits it.
  - **Sampled** otherwise. Take a deterministic stride plus the first and last page of
    every 2 MiB extent, and record the scheme in the report. A sampled run is never
    labelled "full".

### 4.2 Huge-page and policy cross-check

- **THP backing.** Parse `/proc/self/smaps` for VMAs that overlap each region and sum
  `AnonHugePages`. Report it as `thp_bytes`. A successful `madvise` call is not
  evidence of huge pages.
- **Policy.** Parse `/proc/self/numa_maps` for the policy string (`bind:2`) and the
  per-node page counts as a second, independent signal. Keep this as a diagnostic only.

### 4.3 When verification runs

| When | What |
| --- | --- |
| After weight population, before warm-up | All weight regions |
| After the prompt, and at the end of the run | KV regions (A5); their populated extent grows with context |
| Always | Not inside timed decode or prefill. The time goes to `timing_seconds.placement_verify`. |

### 4.4 Strict mode

`LLAMA_PLACEMENT_STRICT=1` (the default for `hbm` and `ddr`) makes any `wrong_node` or
`errors` count fatal before timing starts. Otherwise the run is labelled `unverified`.

## 5. Local tests

- **Unit tests** of status classification and sampling on a synthetic status array.
  These run on all platforms.
- **Linux only:** bind a 64 MiB region to node 0 on any NUMA Linux host. Touch half of
  it, then expect `missing` to be about half and `wrong_node` to be 0.
- **`smaps` parser:** a fixture text with merged VMAs and partial overlaps.

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check PHASE=A3`. **Estimated** at 20 minutes. It needs
the user's go-ahead to submit.

| Test | Expectation |
| --- | --- |
| 8 GiB on each of nodes 0–3, populated by a socket-0 team, fully queried | 100% `on_node`, 0 missing |
| Same with THP on | `thp_bytes` reported; mostly 2 MiB backed if THP is enabled system-wide |
| Populate bandwidth per node | **measured** GB/s for startup cost modelling |
| Deliberately wrong policy (control): `numactl --membind=0` around a portable allocation | Verification reports node 0, not 2 |
| 8 GiB on node 2 populated by a socket-1 team | Still node 2. Binding wins over first touch, and the region is remote for S1. |

### Cluster results (2026-11-01)

cresco8-hbm14, job 5622320, commit `7faa81d` (`results/m1-check-20261101-3/A3`). 8 GiB per region, populated by a
56-thread team pinned to socket 0, every page queried (2,097,152 pages, `move_pages` in batches of 4096).

| Node | On node | Missing | Populate GB/s (**measured**) | Verify s | `numa_maps` policy |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 (DDR, local) | 100% | 0 | 95.6 | 0.12 | bind:0 |
| 1 (DDR, remote) | 100% | 0 | 33.5 | 0.12 | bind:1 |
| 2 (HBM, local) | 100% | 0 | 190.7 | 0.12 | bind:2 |
| 3 (HBM, remote) | 100% | 0 | 45.7 | 0.12 | bind:3 |

- With THP requested: the same 100% on node, populate 102 / 27 / 189 / 63 GB/s, `thp_bytes` 8 GiB. Transparent huge
  pages are `[always]` on this node, so the runs without the request also report 8 GiB huge-page backing; `madvise`
  success alone is never taken as evidence.
- Node 2 populated by a **socket-1** team: still 100% node 2, populate 62.4 GB/s (binding wins over first touch).
- **Negative control:** a portable region first-touched under `numactl --membind=0` and checked against node 2 is
  reported 100% off node 2 (`numa_maps` N0 = 2,097,152 pages).
- **mbind beats a job-wide membind:** `numactl --membind=0` around a region bound to node 1 leaves all pages on node 1.
- Sampling (4096 pages) is labelled `stride+extent-endpoints`, not full.
- **Kernel detail found here:** untouched anonymous pages report `-EFAULT` from `move_pages`, not `-ENOENT`; both now
  count as `missing`. The full 8 GiB query costs about 0.12 s, so full verification is affordable in `bench` runs too.
## 7. Exit criteria

- Full verification of 8 GiB per node reports 100% on the requested node.
- The negative control is detected.
- THP backing is reported from `smaps`.
- The verification cost is measured and reported separately from inference.

## 8. Risks and open questions

- **Permissions.** `move_pages` on the process's own pages needs no special
  privileges. If a seccomp or container policy blocks it, fall back to `numa_maps`
  counts and label the run `verified: unverified-numa_maps`.
- **Cost on 8B.** Full verification of 14 GiB FP32 at 4 KiB pages is about 3.6 M
  pages, roughly 900 batched calls (**estimated**). Measure it. If it is too slow,
  use sampling for `bench` and full verification for `check`.

## 9. Not in this phase

- Loading real weights (A4).
- Hardware counter evidence of actual traffic. That comes from perf in campaigns, and
  study §10.3 notes that page placement does not prove access locality.
