# One-socket placement: engine-owned memory against the staged baseline

**Status: not run.** The campaign of phase A6 is written and dry-run verified but has not been submitted; every table below is
an empty skeleton for the numbers the run produces. Nothing in this document is a measurement. Provenance and gates go in
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

Medians with the full range of the repeats; the staged references are the **measured** values of `docs/kernels.md` (job 5621182):
8B FP32 14.90 tok/s on staged HBM and 8.31 on staged DDR; 8B Q8_0 44.49 and 28.51.

| Model | Precision | `file-shm-hbm` | `anon-ddr` | `anon-hbm` | `anon-hbm-thp` | `remote-hbm` (control) | `omp-file-shm-hbm` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | — | — | — | — | — | — |
| 1B | W8A8 | — | — | — | — | — | — |
| 8B | FP32 | — | — | — | — | — | — |
| 8B | W8A8 | — | — | — | — | — | — |
| 8B | W8A32 (`anon-hbm` only) | | | — | | | |

## Placement evidence

| Configuration | Weights region | Verified | Pages queried | Pages elsewhere or missing | THP bytes | Startup: populate s | Startup: verify s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `anon-hbm` | — | — | — | — | — | — | — |
| `anon-ddr` | — | — | — | — | — | — | — |
| `remote-hbm` | node 3 expected | — | — | — | — | — | — |

## Thread scaling (`anon-hbm`, 14 / 28 / 56 threads) and the long-context point

| Model | Precision | 14 | 28 | 56 | 8B FP32 56 threads, 8192-token prompt |
| --- | --- | ---: | ---: | ---: | ---: |
| — | — | — | — | — | — |

## Memory traffic (perf, first repeat)

| Configuration | HBM read GB | DDR read GB | UPI rx data flits (M) | Decode GB/s implied |
| --- | ---: | ---: | ---: | ---: |
| — | — | — | — | — |

## Modeled against measured

One socket reads 447 of 552 GB/s for FP32 decode (**measured**, 81%, `docs/kernels.md`). The ceiling for `anon-hbm`: —.


## Exit criteria

Each is ticked only with the observed value written next to it.

- [ ] `anon-hbm` 8B decode is at least `file-shm-hbm` within the observed ranges, for FP32 and Q8 (references 14.90 and 44.49 tok/s): —
- [ ] `anon-ddr` is close to the staged DDR numbers (8.31 and 28.51 tok/s), so the loader adds no overhead: —
- [ ] Every `anon-*` run reports 100% verified weight placement; `remote-hbm` shows the penalty and reports node 3: —
- [ ] THP: median and range reported; adopted as the B7 default (`B7_THP`, `D_THP`) only if not slower beyond the range: —
- [ ] The correctness gate (`anon-hbm` bitwise equal to `file-shm-hbm`) passed before any timing: —

## Follow-ups for a later plan

- A staged baseline already near the bandwidth ceiling is a valid result (the value of track A is then verified ownership and per-socket shards).
