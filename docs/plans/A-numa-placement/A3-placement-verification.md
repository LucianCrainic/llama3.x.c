# A3. Placement verification

## 1. Goal and status

Check where populated pages actually reside, and report it. This is how a run earns
the label "local HBM". The check distinguishes between:

- policy and observed residency;
- full and sampled coverage;
- unpopulated and wrong-node pages.

It also records actual huge-page backing (study §5.5, §10.2).

Status: **planned**. Branch: `feature/numa-a3-verification`.

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
