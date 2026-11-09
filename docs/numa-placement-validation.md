# One-socket placement: validation record

**Status: run (9 November 2026), long-context point excepted.** Job 5713402 (tree 75761d4) passed the four gates and stopped at the remote-HBM control (the engine refuses a map pointing a socket at the other socket's HBM, 'node 0 is not nearer to HBM node 3 than to node 2'); `LLAMA_TOPOLOGY_ALLOW_FAR_HBM=1` and non-fatal launch failures were added, and **job 5713411 (tree 52d35c1) is the record below.**

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-09 |
| Node, memory mode | cresco8-hbm14, Intel Xeon Max 9480, Flat mode (`flat-mode.json`) |
| Slurm job ID(s) | 5713411 (record); 5713402 (aborted at the remote control) |
| Commit | 52d35c1cb94833a4a15a5062dc604902350207bd (develop) |
| Compiler, flags | GCC 11.4.1 (Red Hat 11.4.1-3), `-O3 -fopenmp`, C11 (`build-config-repo-52d35c1.txt`) |
| Campaign directory | `~/PFS/lpor/results/a6-20261109-2` (cluster) |
| Script and stage | `scripts/cresco8-placement.sbatch`, `STAGE=bench`, `PHASE=A6` |
| Models | Llama 3.2 1B Instruct and Llama 3.1 8B Instruct, FP32 and Q8_0 checkpoints |
| Repeats | 3, interleaved across configurations |
| Exclusivity | `--exclusive`; the queue held no other job |
| perf counters | yes (first repeat of every configuration) |

## Gates

Correctness gates run before any benchmark, and a failure stops it (`gate-failed.txt`).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| `anon-hbm` against `file-shm-hbm` | bitwise equal logits | 1B and 8B, FP32 and INT8: identical | PASS (4 of 4) |
| FP32 against Hugging Face | `validate.py` default budget | not run (`HF_GATE=0`) | — |
| INT8 | recorded, not gating | not run (`HF_GATE=0`) | — |

`make test` was not run in this job (`SKIP_TEST=1`; it passed on the same code in job 5713378).

## How the run was submitted

A submission needs the owner's go-ahead for that job. Shape of the command (placeholders for what is not kept in the repository):

```bash
sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=<see the script header> \
  --export=ALL,ROOT=<tree>,OUT=<new directory>,PYTHON=<validation python>,MODELS="NAME:ARTIFACTS:HF ...",STAGE=bench,PHASE=A6 \
  scripts/cresco8-placement.sbatch
```

The matrix is `phase_A6` (121 launches for 1B and 8B at three repeats; `DRY_RUN=1` prints it). Configurations are interleaved across repeats (`REP=n`); `--time` of 04:00:00 may need extending.

## Deviations from the plan

- THP was `[always]` on the node (A3): runs with `LLAMA_THP` off are already huge-page backed (28.0 GiB `thp_bytes` for 8B FP32 in both), so the THP comparison measures no difference by construction.
- The 14 and 28 thread points ran for `anon-hbm` only (plan §4.1).
- **The 8192-token long-context point is invalid**: `run_engine` did not pass `-n`, the engine stopped at its default of 4096 forward passes, and no decode ran. Fixed (`-n 0`, commit 16b7332); to be re-run.
- The remote control needed `LLAMA_TOPOLOGY_ALLOW_FAR_HBM=1` (an override added for it); every other topology check still applied.
- The Hugging Face gate was not run (`HF_GATE=0`).

## Raw data

`summary.md`, the JSON reports (`<label>-<rep>.json`), `*.perf.csv`, `phase-results.txt`, `environment.txt`, `flat-mode.json`: in the campaign directory above.
