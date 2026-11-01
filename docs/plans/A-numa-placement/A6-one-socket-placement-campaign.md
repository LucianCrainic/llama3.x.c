# A6. One-socket placement campaign

## 1. Goal and status

Measure what engine-owned placement achieves on one socket, compared with the existing
staged-file baseline. Placement is separated into four effects:

- the loader change itself;
- DDR vs HBM;
- normal pages vs THP;
- remote vs local.

The result establishes the engine's local-HBM kernel rate, which is the baseline every
later two-socket claim is judged against (study §10.3).

Status: **planned**. Branch: `feature/numa-a6-campaign` (scripts and results doc only).

## 2. Depends on / unblocks

- Depends on: A5 and F3.
- Unblocks: B7 (its one-socket baseline) and C7 (C7 runs on anon HBM if A6 is done).

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Run matrix pattern and workload | `scripts/cresco8-kernels.sbatch:116-176` | Same prompt, flags and labels style |
| Library | F3 `scripts/lib/cresco8-common.sh` | Build, flat check, perf, gate |
| Summary | F3 `tools/summarize_campaign.py` | Medians, ranges, placement columns |

## 4. Changes

- Add a `PHASE=A6` matrix to `scripts/cresco8-placement.sbatch`.
- Add a results document, `docs/numa-placement.md`, and its companion
  `docs/numa-placement-validation.md`.

### 4.1 Run matrix

All runs are on socket 0, pinned with `LLAMA_BACKEND=teams`, with 3 repeats.

| Label key | Weights | KV/work | Purpose |
| --- | --- | --- | --- |
| `file-shm-hbm` | Staged `/dev/shm` with `--membind=2`; `LLAMA_PLACEMENT=file` | default | Existing baseline, now with residency observed by A3 |
| `anon-ddr` | `LLAMA_PLACEMENT=ddr` | DDR node 0 | Isolates the loader change from the HBM benefit |
| `anon-hbm` | `LLAMA_PLACEMENT=hbm` | HBM node 2 | Engine's local-HBM rate |
| `anon-hbm-thp` | `hbm` + `LLAMA_THP=on` | HBM + THP | Page-size effect |
| `remote-hbm` (control) | explicit map pointing socket 0 at HBM node 3 | node 3 | Remote penalty; a diagnostic, not a deployment choice |
| `omp-file-shm-hbm` | `LLAMA_BACKEND=omp`, staged | default | Ties back to the `docs/kernels.md` numbers |

| Dimension | Values |
| --- | --- |
| Models | 1B, 8B |
| Precisions | FP32, INT8 (W8A8); W8A32 on `anon-hbm` only |
| Threads | 14, 28, 56 |
| Workload | `short.txt`, `--context-length 2048 --generation-length 128 -t 0 --seed 42 --chat-template` |
| Long-context point | 8B FP32 `anon-hbm` t56 with a 8192-token prompt and 128 generated tokens |
| Perf | HBM/DDR CAS reads and UPI flits per socket |

**Estimated** duration is 3–4 hours. That is 2 models × 2 precisions × 6 placements ×
3 thread counts × 3 repeats ≈ 216 runs, trimmed by running the 14 and 28 thread points
only for `anon-hbm`. It needs the user's go-ahead to submit.

### 4.2 Correctness gate (before benchmarks)

- `compare_engines.py` shows bitwise equality of `anon-hbm` vs `file-shm-hbm`.
- `validate.py` gates FP32 as today; INT8 is recorded.

## 5. Local tests

None beyond A1–A5. Run `bash -n` and a dry-run mode (`DRY_RUN=1` prints the matrix) to
check the script locally.

## 6. Cluster run

As in §4.1. Outputs:

- `summary.md` with columns: placement, verified, THP bytes, threads, decode tok/s,
  decode GB/s, prefill tok/s, startup breakdown, HBM/DDR/UPI counters.
- The raw JSON files.
- `environment.txt` and `flat-mode.json`.

## 7. Exit criteria

These are the track A exit gate:

- `anon-hbm` 8B decode is ≥ `file-shm-hbm` within the observed ranges, for FP32 and
  Q8. The **measured** reference is 14.90 and 44.49 tok/s.
- `anon-ddr` is close to the staged DDR numbers (8.31 and 28.51 tok/s), which confirms
  that the loader adds no overhead.
- All `anon-*` runs report 100% verified weight placement. `remote-hbm` shows the
  expected penalty and reports node 3.
- THP effect: report the median and range, and adopt THP as the B7 default only if it
  is not slower beyond the observed range.

## 8. Risks and open questions

- **Staged baseline already near the ceiling.** If `file-shm-hbm` is already at the
  bandwidth limit, `anon-hbm` may show no gain. That is a valid result: the value of
  track A is then verified ownership plus per-socket shard capability, not speed.
- **Clock and power variability.** The **measured** 1.7 GHz under PL1 (hw-bench docs)
  can make repeats noisy. Record frequency where available and interleave
  configurations across repeats rather than running them in blocks.

## 9. Not in this phase

- Two sockets (B7).
- Prefill batching (C7).
- KV tiering.
