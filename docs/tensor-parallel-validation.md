# Two-socket tensor-parallel decode: validation record

**Status: run (2026-11-17).** Provenance and gates of phase B7. Results: [tensor-parallel.md](tensor-parallel.md).

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-17. 1B job 5839003: 32 min 39 s, COMPLETED, exit code 0. 8B job 5839004: 2 h 33 min 4 s, COMPLETED, exit code 0, started when the 1B job ended |
| Node, memory mode | cresco8-hbm14, Flat mode (`flat-mode.json`: DDR nodes 0 and 1, HBM nodes 2 and 3) |
| Slurm job ID(s) | 5839003 (1B), 5839004 (8B) |
| Commit | 2aad124 (`develop`, `COMMIT` file in the staged tree `repo-2aad124`; `source-repo-2aad124.sha256`) |
| Compiler, flags | gcc 11.4.1 20231218 (Red Hat 11.4.1-3), release mode, `-O3 -fopenmp -pthread` (`build-config-repo-2aad124.txt`, `build-repo-2aad124.log`) |
| Campaign directory | `b7-20261117-1b` and `b7-20261117-8b` under the results directory (1.5 MB each; 349 files each, 98 JSON reports and 46 perf csv files for 8B) |
| Script and stage | `scripts/cresco8-tp.sbatch`, `STAGE=bench`, `PHASE=B7`, `SKIP_TEST=1`, `HF_GATE=0`, `B7_THP=off` |
| Models | Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct artifacts (FP32 and Q8_0 v3), one job per model; checkpoint SHA-256 in `environment.txt` |
| Repeats | 3 (`REPEATS=3`), interleaved; the profiled runs (`-xprof`, `-xp4prof`) once |
| Exclusivity | hbm14 is exclusive; no other job ran on it (the two jobs ran back to back) |
| perf counters | `perf_uncore=1` (`environment.txt`) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`, absent). All 8 passed (`phase-results.txt`:
4 PASS per model); 53 prompt positions each.

| Gate | Criterion | Observed (max over positions) | Result |
| --- | --- | --- | --- |
| One socket against two, FP32, batched prefill (the warm-up path), M = 64 | max abs <= 1e-3, argmax equal | 1B 8.1e-6 (rms 2.2e-6); 8B 7.1e-5 (rms 1.1e-5); argmax 100% | pass |
| One socket against two, W8A8, batched | statistical gate (`W8A8_GATE`: rms <= 0.2, argmax >= 0.9, max <= 16) | 1B max 0.41, rms 0.131, argmax 100%; 8B max 0.58, rms 0.127, argmax 98.1% (52 of 53) | pass |
| One socket against two, W8A32 | max abs <= 1e-3 | 1B 8.3e-6; 8B 6.7e-5; argmax 100% | pass |
| packed4 against rows, FP32, two sockets, `LLAMA_GEMV_SCHED=dynamic` | bitwise equal logits | 1B and 8B: bitwise on all 53 positions, max abs 0.0, on the real AVX-512 kernels | pass |
| Placement of both sockets (weights, packed4 copy, KV) | 100% verified on the socket's node | weights, packed4 copy, KV and both work regions `full` on nodes 2 and 3; embedding on node 0 | pass |

The Hugging Face gate with two sockets was not part of this job (`HF_GATE=0`); it passed in K2 and K4 (jobs 5830325, 5837867, see
[B5](plans/B-tensor-parallel/B5-two-socket-fp32-forward.md), [B6](plans/B-tensor-parallel/B6-two-socket-q8.md),
[D1](plans/D-integration/D1-two-socket-batched-prefill.md)). `make test` was not run in this job (`SKIP_TEST=1`, `test-skipped.txt`); it
passed on the hardware in jobs 5704671 and 5713378 on trees that differ from this one by test and script changes only.

## How the run was submitted

Standing authorization of the owner (16 November 2026): jobs are submitted without a per-job go-ahead. Shape of the command
(placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<4 h for 1B, 14 h for 8B> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF",STAGE=bench,PHASE=B7,REPEATS=3,B7_THP=off,SKIP_TEST=1,HF_GATE=0 \
  scripts/cresco8-tp.sbatch
```

The matrix is `phase_B7` (`DRY_RUN=1` prints it). The 8B job was chained behind the 1B job (`--dependency`); the 8B job used 2 h 33 min of its
limit.

## Deviations from the plan

- The worker-count sweep (2 x 14, 28, 42) and the control run at context 512 only, and W8A32 skips context 32768 (its warm-up is per token).
- The `LLAMA_TP_PARTITION=rows` comparator was not built (decision L8) and is not needed: reductions plus waits are below 0.1% of the token.
- `make test` and the Hugging Face gate were not run in this job (above).
- The one-socket long-context runs are labelled `hbm (unverified)` by the summarizer because their 10-40 MB work region is not verified;
  weights and KV are `full`.
- Three single repeats are low outliers (1B FP32 2x56 at 512: 132.0 against a median of 168.9; 1B FP32 2x56p4 at 32768: 48.4 against 69.4; 8B FP32 control: 6.18
  against 8.24); the medians are reported with the ranges. No repeat was rerun.

## Raw data

`B7/summary.md` (generated, kept in [tensor-parallel-summary-table.md](tensor-parallel-summary-table.md)), the JSON reports (`<label>-<rep>.json`),
the `.stderr` files of the profiled runs, `*.perf.csv`, `phase-results.txt`, `B7/compare-*.json` (the 8 gate reports), `exit-status.txt` (exit code
0), `environment.txt`, `flat-mode.json`, `job.stdout`, `no-hf-gate.txt`, `test-skipped.txt`.
