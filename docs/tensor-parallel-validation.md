# Two-socket tensor-parallel decode: validation record

**Status: not run.** This is the skeleton of the validation record of phase B7. Every cell that says "—" is filled from
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
| Script and stage | `scripts/cresco8-tp.sbatch`, `STAGE=bench`, `PHASE=B7` |
| Models | — (`MODELS`, checkpoint SHA-256 in `environment.txt`) |
| Repeats | — (`REPEATS`; the plan asks for 3 interleaved) |
| Exclusivity | — (`--exclusive`; any other job on the node is a deviation) |
| perf counters | — (`perf_uncore=` in `environment.txt`) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| One socket against two, FP32, batched prefill (the warm-up path) | max |dlogit| <= 1e-3, residual copies identical | — | — |
| One socket against two, W8A8, batched | statistical gate (`W8A8_GATE`) | — | — |
| One socket against two, W8A32 | max |dlogit| <= 1e-3 | — | — |
| packed4 against rows, FP32, two sockets, `LLAMA_GEMV_SCHED=dynamic` | bitwise equal logits | — | — |
| Placement of both sockets (weights, packed4 copy, KV) | 100% verified on the socket's node | — | — |

The `make test` result of the job (`test.log`) is: —.

## How the run was submitted

A submission needs the owner's go-ahead for that job. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<see the script header> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=B7 \
  scripts/cresco8-tp.sbatch
```

The matrix is `phase_B7` (96 launches per model; `DRY_RUN=1` prints it). Submit with `--time=06:00:00` or one model per job. `B7_THP` follows A6.

## Deviations from the plan

- The worker-count sweep (2 x 14, 28, 42) and the control run at context 512 only, and W8A32 skips context 32768 (its warm-up is per token).
- The `LLAMA_TP_PARTITION=rows` comparator was not built (decision L8).
- —

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`), the `.stderr` files of profiled runs, `*.perf.csv`,
`phase-results.txt`, `correctness-status.txt`, `environment.txt`, `flat-mode.json`: —.
