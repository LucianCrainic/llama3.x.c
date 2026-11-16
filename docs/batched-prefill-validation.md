# Batched prefill on one socket: validation record

**Status: run (2026-11-09).** Provenance and gates of phase C7. Results: [batched-prefill.md](batched-prefill.md).

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-09 (job ran 3 h 26 min 52 s, state COMPLETED, exit code 0) |
| Node, memory mode | cresco8-hbm14, Flat mode (`flat-mode.json`) |
| Slurm job ID(s) | 5713857 (final run). 5713471 stopped at its gates on the same day: 8 AMX-INT8 comparisons applied the strict 1e-3 gate, fixed in ca2f7f8 |
| Commit | ca2f7f8 (`develop` locally, unpushed); `COMMIT` file in the staged tree |
| Compiler, flags | `build-config-repo-ca2f7f8.txt`, `build-repo-ca2f7f8.log` in the campaign directory |
| Campaign directory | `c7-20261109-2` under the results directory (12 MB, 106 files in `C7/`; the 491 run stderr files are empty) |
| Script and stage | `scripts/cresco8-prefill.sbatch`, `STAGE=bench`, `PHASE=C7`, `SKIP_TEST=1`, `HF_GATE=0` |
| Models | Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct artifacts (FP32 and Q8_0 v3); checkpoint SHA-256 in `environment.txt` |
| Repeats | 3 (`REPEATS=3`), per-token baselines at 8192 tokens once |
| Exclusivity | hbm14 is exclusive; no other job ran on it |
| perf counters | `perf_uncore=1` |
| Weights placement | staged files (placement `file`), not anonymous HBM |

## Gates

Correctness gates run before any benchmark and a failure stops it. The AMX-BF16 comparison is recorded (it does not stop the run).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| AVX-512 FP32 GEMM against per-token (2 models x M 64, 256 x 86 and 2048 tokens) | max abs 1e-3 | 8 of 8 passed | pass |
| AVX-512 Q8 against per-token (same grid) | max abs 1e-3 | 8 of 8 passed | pass |
| AMX-INT8 against AVX-512 Q8 (same grid) | W8A8 statistical gate | 8 of 8 passed | pass |
| AMX-BF16 against AVX-512 FP32 (same grid) | recorded, provisional argmax 1.0 | 6 of 8 passed; 1B at 86 tokens (M=64 and M=256): max 0.057, rms 0.0096, argmax 98.1%, the flip at position 0 (BOS) | 2 recorded deviations |
| Hugging Face comparison of AMX-BF16 | Hugging Face numbers | not run (`HF_GATE=0`) | open (K1) |
| Decode after prefill | within the same-context per-token run | 6 of 6 measured inside or within 1.5% (8B W8A32 has no same-context reference) | pass, see results document |

`make test` was not run in this job (`SKIP_TEST=1`); the tool test suites passed locally on this commit before it was staged.

## How the run was submitted

Each submission needed the owner's go-ahead. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=06:00:00 \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=C7,SKIP_TEST=1,HF_GATE=0,REPEATS=3 \
  scripts/cresco8-prefill.sbatch
```

The matrix is `phase_C7` (`DRY_RUN=1` prints it); the run used 3 h 27 min of the 6 h limit.

## Deviations from the plan

- The per-token baselines run once (not three times) at 8192 tokens and on 14 and 28 threads.
- At 2048 tokens the gates compare the last position only (1 GB of logits per side otherwise).
- The AMX-INT8 gates use the W8A8 statistical gate, not 1e-3 (a gate revision: the noise floor of W8A8 requantization is a property
  of the kernel switch, shown by D3; the first attempt of C7 failed on it).
- The Hugging Face gate was not run; the weights were staged files, not anonymous HBM (A6: 2.4% on 8B FP32 decode).

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`, with the profile fields of the `-xprof` runs), `*.perf.csv`, `C7/phase-shares.txt`,
the 32 `C7/compare-*.json` gate reports, `exit-status.txt` (exit code 0), `environment.txt`, `flat-mode.json`. The generated summary
is kept in [batched-prefill-summary-table.md](batched-prefill-summary-table.md).
