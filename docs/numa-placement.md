# One-socket placement: engine-owned memory against the staged baseline

> **Correction (2026-11-21):** HBM byte figures derived from `hbm_cas_rd` counts were computed at 64 B per count; the counter counts 32 B (calibrated in job 5892172, see [xengine-results-validation.md](xengine-results-validation.md)). Affected figures below are halved and marked †.

**Status: run (9 November 2026); the long-context point is covered by the B7 campaign (17 November 2026).** Job 5713411 on the Flat-mode node, tree 52d35c1, 31 minutes, exit 0, 107 reports,
no failed run. The sections below carry the measured numbers; the 8192-token point of this job is invalid (cut at 4096 tokens by the engine's default step
cap, fixed in 16b7332); the same configuration was measured in B7, see below. Provenance and gates go in
[the validation record](numa-placement-validation.md). Plan: [A6](plans/A-numa-placement/A6-one-socket-placement-campaign.md). Branch: `feature/numa-a6-campaign`.

Question: what does engine-owned placement (`LLAMA_PLACEMENT=ddr|hbm`, optional `LLAMA_THP=on`) achieve on one socket,
compared with the existing staged-file baseline (`file`, a checkpoint staged in `/dev/shm` under `numactl --membind`), split into
the loader change, DDR against HBM, normal pages against THP, and local against remote. The result is the engine's local-HBM
kernel rate, the baseline every two-socket claim (B7, D2) is judged against.

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` (paste it here once the run is done); the sections after it are
the curated views.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| — | | | | | | | | | | | | | | | | | |

## Decode by placement (one socket, 56 threads)

Median [range] of 3 interleaved repeats, tok/s, chat prompt of 86 tokens and 128 generated tokens. The earlier staged references are the **measured** values of
`docs/kernels.md` (job 5621182, older tree): 8B FP32 14.90 tok/s on staged HBM and 8.31 on staged DDR; 8B Q8_0 44.49 and 28.51.

| Model | Precision | `file-shm-hbm` (teams) | `anon-ddr` | `anon-hbm` | `anon-hbm-thp` | `remote-hbm` (control) | `omp-file-shm-hbm` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | 97.16 [96.71-97.87] | 49.99 [49.48-50.40] | 91.94 [91.83-92.59] | 92.58 [91.85-93.36] | 22.07 [21.98-22.10] | 87.51 [87.41-89.30] |
| 1B | W8A8 | 306.62 [303.86-310.71] | 177.61 [176.80-178.43] | 304.97 [301.54-306.15] | 305.87 [303.83-308.83] | 74.03 [73.49-74.06] | 249.45 [249.14-249.72] |
| 8B | FP32 | 16.07 [15.79-16.16] | 8.41 [7.88-8.42] | 15.69 [15.66-15.71] | 15.68 [15.66-15.71] | 3.59 [3.59-3.59] | 15.12 [14.07-15.43] |
| 8B | W8A8 | 50.91 [50.30-51.30] | 29.69 [29.66-29.70] | 50.76 [50.69-50.85] | 50.74 [50.73-50.76] | 12.84 [12.55-12.86] | 46.71 [45.49-46.73] |
| 1B | W8A32 (`anon-hbm` only) | | | 265.15 [261.79-265.80] | | | |
| 8B | W8A32 (`anon-hbm` only) | | | 45.76 [45.27-45.98] | | | |

## Placement evidence

Every owned run's weights and KV caches are one region per kind on the requested node, fully verified, 0 pages elsewhere or missing (8B FP32: 7,329,028
pages queried):

| Configuration | Weights region | Verified | THP bytes | Startup: populate s | Startup: verify s |
| --- | --- | --- | ---: | ---: | ---: |
| `anon-hbm` (8B FP32) | node 2 | full | 28.0 GiB | 0.3 | 0.48 |
| `anon-ddr` | node 0 | full | 28.0 GiB | 0.4 | 0.46 |
| `remote-hbm` | node 3 (as asked) | full | 28.0 GiB | 0.5 | 0.47 |
| `anon-hbm` (8B W8A8) | node 2 | full | 7.4 GiB | 0.1 | 0.15 |

## Thread scaling (`anon-hbm`, 14 / 28 / 56 threads)

| Model | Precision | 14 | 28 | 56 |
| --- | --- | ---: | ---: | ---: |
| 1B | FP32 | 41.50 | 71.07 | 91.94 |
| 1B | W8A8 | 140.67 | 234.80 | 304.97 |
| 8B | FP32 | 6.71 | 11.30 | 15.69 |
| 8B | W8A8 | 21.12 | 34.54 | 50.76 |

**Long-context point (8B FP32, 8192-token prompt): invalid in this job, measured in B7.** The run stopped at 4096 forward passes (`stop: step_limit`, 4096 of 8226 prompt tokens processed, no decode) because
`run_engine` did not pass `-n`; its figure (16.5 tok/s per-token prefill over 4096 tokens) must not be quoted. Fixed in 16b7332 (`-n 0`). The B7 campaign (jobs 5839003 and 5839004, commit 2aad124, [tensor-parallel.md](tensor-parallel.md)) ran the same configuration with the fix: 8B FP32, engine-owned HBM (`teams`, `LLAMA_PLACEMENT=hbm`), 56 threads, one socket, an 8192-token retained context and 128 timed tokens. Weights (30.0 GB) and KV (2.2 GB) are verified `full` on node 2 with 0 wrong pages. Decode is 13.29 [13.24-13.42] tok/s at that context (16.42 at 512 tokens, 9.66 at 32768); W8A8 28.54 [28.48-28.57]. No separate A6 run was made, since the configuration is identical.

## Memory traffic (perf, first repeat, whole process)

| Configuration (8B FP32) | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | ---: |
| `file-shm-hbm` | 6208 † | 19.7 | 314 |
| `anon-hbm` | 6219 † | 18.1 | 26 |
| `anon-hbm-thp` | 6215 † | 18.5 | 28 |
| `anon-ddr` | 5.2 † | 6242 | 40 |
| `remote-hbm` | 6288 † | 33.1 | **890,339** |

The counters are as `perf` reports them over the whole process (loading, prefill and decode); read them as ratios between configurations. The HBM figures (†) are about 207 times the 30 GB weight set for the whole process, which also includes loading and prefill, so they are not
one pass per token; the excess is unexplained and does not affect the comparison: owned HBM reads HBM and almost nothing from DDR or across UPI, owned DDR reads only DDR,
and the remote control moves the same HBM volume across UPI (34,000 times the flits of local HBM).

## Modeled against measured

One socket reads 447 of 552 GB/s for FP32 decode (**measured**, 81%, `docs/kernels.md`). `anon-hbm` 8B FP32 at 15.69 tok/s streams 15.69 x 32.1 GB = **504 GB/s**
(91% of the 552 GB/s stream figure, 84% of the 600 GB/s that a pure stream reaches in `tests/stream_bench.c`); the staged copy reaches 16.07 tok/s = 516 GB/s.


## Exit criteria

- [~] `anon-hbm` 8B decode is at least `file-shm-hbm` within the observed ranges: **Q8 yes** (50.76 [50.69-50.85] inside the staged 50.30-51.30), **FP32 marginally not** (15.69 [15.66-15.71] against 16.07 [15.79-16.16]: 2.4% below the median and 0.5% below the staged minimum, ranges disjoint). Both clear the earlier reference (14.90 and 44.49).
- [x] `anon-ddr` is close to the staged DDR numbers (8.31 and 28.51 tok/s), so the loader adds no overhead: 8.41 [7.88-8.42] and 29.69 [29.66-29.70].
- [x] Every `anon-*` run reports 100% verified weight placement; `remote-hbm` shows the penalty (4.4x slower on 8B FP32, 4.0x on W8A8) and reports node 3.
- [x] THP: neutral, 15.68 against 15.69 on 8B FP32, 50.74 against 50.76 on W8A8, 1B within the ranges. The node already backs the weights with huge pages whatever the knob says (28.0 GiB `thp_bytes` with `LLAMA_THP` off, THP being `[always]`), so the knob changes nothing here: `B7_THP` and `D_THP` stay `off`.
- [x] The correctness gate (`anon-hbm` bitwise equal to `file-shm-hbm`, 1B and 8B, FP32 and INT8) passed before any timing (4 of 4).
- [x] The long-context point: measured in B7 (8B FP32 anon-hbm t56, 8192-token context: 13.29 tok/s decode, placement verified), see above.

## Follow-ups for a later plan

- A staged baseline already near the bandwidth ceiling is a valid result (the value of track A is then verified ownership and per-socket shards).
