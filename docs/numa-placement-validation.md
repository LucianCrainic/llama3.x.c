# One-socket placement: validation record

**Status: not run.** This is the skeleton of the validation record of phase A6. Every cell that says "—" is filled from
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
| Script and stage | `scripts/cresco8-placement.sbatch`, `STAGE=bench`, `PHASE=A6` |
| Models | — (`MODELS`, checkpoint SHA-256 in `environment.txt`) |
| Repeats | — (`REPEATS`; the plan asks for 3 interleaved) |
| Exclusivity | — (`--exclusive`; any other job on the node is a deviation) |
| perf counters | — (`perf_uncore=` in `environment.txt`) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| `anon-hbm` against `file-shm-hbm` | bitwise equal logits | — | — |
| FP32 against Hugging Face | `validate.py` default budget | — | — |
| INT8 | recorded, not gating | — | — |

The `make test` result of the job (`test.log`) is: —.

## How the run was submitted

A submission needs the owner's go-ahead for that job. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<see the script header> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=A6 \
  scripts/cresco8-placement.sbatch
```

The matrix is `phase_A6` (121 launches for 1B and 8B at three repeats; `DRY_RUN=1` prints it). Configurations are interleaved across repeats (`REP=n`); `--time` of 04:00:00 may need extending.

## Deviations from the plan

- THP was `[always]` on the Flat-mode node (A3), so "THP off" runs may already be huge-page backed: `thp_bytes` in `regions[]` is the evidence, not the knob.
- The 14 and 28 thread points run for `anon-hbm` only (plan §4.1).
- —

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`), the `.stderr` files of profiled runs, `*.perf.csv`,
`phase-results.txt`, `correctness-status.txt`, `environment.txt`, `flat-mode.json`: —.
