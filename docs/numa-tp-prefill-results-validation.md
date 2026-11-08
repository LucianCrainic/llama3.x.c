# Final campaign: validation record

**Status: not run.** This is the skeleton of the validation record of phase D2. Every cell that says "—" is filled from
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
| Script and stage | `scripts/cresco8-prefill.sbatch`, `STAGE=bench`, `PHASE=D2` |
| Models | — (`MODELS`, checkpoint SHA-256 in `environment.txt`) |
| Repeats | — (`REPEATS`; the plan asks for 3 interleaved) |
| Exclusivity | — (`--exclusive`; any other job on the node is a deviation) |
| perf counters | — (`perf_uncore=` in `environment.txt`) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| Everything above is re-run at the headline points only | from one commit and one node state | — | — |
| Placement | verified in every report | — | — |
| Gates of A6, B7, C7, D1 | passed on the same commit | — | — |

The `make test` result of the job (`test.log`) is: —.

## How the run was submitted

A submission needs the owner's go-ahead for that job. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<see the script header> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=D2 \
  scripts/cresco8-prefill.sbatch
```

The matrix is `phase_D2` (144 launches per model; `DRY_RUN=1` prints it); estimated at 5-6 hours, submit with `--time=07:00:00` or split MODELS. If node time is short, run 8B at 2048 tokens first and record what was skipped: —.

## Deviations from the plan

- Numbers of A6, B7, C7 and D1 are not reused: D2 reruns the headline points.
- —

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`), the `.stderr` files of profiled runs, `*.perf.csv`,
`phase-results.txt`, `correctness-status.txt`, `environment.txt`, `flat-mode.json`: —.
