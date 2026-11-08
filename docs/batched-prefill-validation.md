# Batched prefill on one socket: validation record

**Status: not run.** This is the skeleton of the validation record of phase C7. Every cell that says "—" is filled from
the campaign directory after the run, with the job ID, commit and date; nothing here is a measurement yet.

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | — |
| Node, memory mode | — (Flat mode, checked by `flat-mode.json`) |
| Slurm job ID(s) | — |
| Commit | — (`git rev-parse HEAD` of the tree under test; `COMMIT` file in the staged tree) |
| Compiler, flags | — (`environment.txt`, `build-config-*.txt`) |
| Campaign directory | — (`OUT`) |
| Script and stage | `scripts/cresco8-prefill.sbatch`, `STAGE=bench`, `PHASE=C7` |
| Models | — (`MODELS`, checkpoint SHA-256 in `environment.txt`) |
| Repeats | — (`REPEATS`; the plan asks for 3 interleaved) |
| Exclusivity | — (`--exclusive`; any other job on the node is a deviation) |
| perf counters | — (`perf_uncore=` in `environment.txt`) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| AVX-512 FP32 GEMM against per-token (8B, 2048 tokens, last position) | max |dlogit| <= 1e-3 | — | — |
| AMX-BF16 | recorded against FP32 until C5 fixes its gate; Hugging Face comparison | — | — |
| AVX-512 Q8 and AMX-INT8 against per-token | W8A8 statistical gate (last position) | — | — |
| Decode after prefill | within the one-socket range | — | — |

The `make test` result of the job (`test.log`) is: —.

## How the run was submitted

A submission needs the owner's go-ahead for that job. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<see the script header> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=C7 \
  scripts/cresco8-prefill.sbatch
```

The matrix is `phase_C7` (about 245 launches per model plus 16 correctness gates; `DRY_RUN=1` prints it). Submit with `--time=06:00:00`: the per-token baselines dominate (8B FP32 per token is about 85 minutes of it). C7 runs on anon HBM if A6 is done, otherwise on staged `/dev/shm` HBM, and records which: —.

## Deviations from the plan

- The per-token baselines run once (not three times) at 8192 tokens and on 14 and 28 threads.
- At 2048 tokens the gates compare the last position only (1 GB of logits per side otherwise).
- —

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`), the `.stderr` files of profiled runs, `*.perf.csv`,
`phase-results.txt`, `correctness-status.txt`, `environment.txt`, `flat-mode.json`: —.
