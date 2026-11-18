# Final campaign: validation record

**Status: run (2026-11-17).** Provenance, gates and deviations of phase D2. Results:
[numa-tp-prefill-results.md](numa-tp-prefill-results.md).

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-17 (1B job started 20:58, ran 27 min 9 s; 8B job started 21:25, ran 2 h 0 min 39 s; both COMPLETED, exit code 0) |
| Node, memory mode | cresco8-hbm14, Flat mode (`flat-mode.json`: DDR nodes 0 and 1, HBM nodes 2 and 3) |
| Slurm job ID(s) | 5845031 (1B), 5845032 (8B, `--dependency=afterany` on the first) |
| Commit | 8f3114f (`develop`, pushed; `COMMIT` file in the staged tree). Source changes since the D1 job 5713378 (75761d4): `src/numa.c` (the far-HBM diagnostic override of A6, 8 lines) and the campaign scripts |
| Compiler, flags | gcc 11.4.1 20231218 (Red Hat 11.4.1-3), `-O3 -fopenmp -std=c11 -pthread` (`build-config-repo-8f3114f.txt`, `build-repo-8f3114f.log` in each campaign directory) |
| Campaign directories | `d2-20261117-1b` and `d2-20261117-8b` under the results directory (3.3 and 3.4 MB; 144 reports each, with the `.perf.csv` of the first repeat, plus `flat-mode.json` and `prompts.json`) |
| Script and stage | `scripts/cresco8-prefill.sbatch`, `STAGE=bench`, `PHASE=D2`, `SKIP_TEST=1`, `HF_GATE=0`, `SKIP_PRELUDE_HF=1`, `REPEATS=3`, `B7_THP=off`, `D_THP=off` |
| Models | Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct artifacts (FP32 and Q8_0 v3); checkpoint SHA-256 in `environment.txt` |
| Repeats | 3 (`REPEATS=3`), interleaved per cell in the order of `phase_D2`; per-token baselines at 8192 tokens also ran 3 times (C7 ran them once) |
| Exclusivity | hbm14 is exclusive; the 8B job started after the 1B job ended; no other job ran on it |
| perf counters | `perf_uncore=1` (first repeat of every configuration) |
| M per rung | `D2_M1="avx512_fp32=128 amx_bf16=256 avx512_q8=256 amx_int8=256"` (one socket), `D2_M2="avx512_fp32=64 amx_bf16=256 avx512_q8=256 amx_int8=256"` (two sockets), from C7 and D1; the dry run printed the values and 144 launches per model |

## Gates

D2 runs benchmarks only: its phase has no correctness gate of its own (`SKIP_TEST=1`, `HF_GATE=0`, no `compare_engines.py` runs). The record below says what was
checked in this run and what was carried from the earlier phases, which ran on earlier commits (listed in the results document, section 5).

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| Everything above is re-run at the headline points only | from one commit and one node state | 288 reports from commit 8f3114f, both jobs on the same node, status `completed` in every report (285 reached the generation limit, 3 stopped on a stop token, see Deviations) | pass |
| Placement | verified in every report | every report has its region table; weights, embedding, KV and packed regions are `full` with 0 mismatched pages (432 weights, 144 embedding, 432 + 432 KV, 108 packed, 576 work); 144 one-socket batched work regions (1-70 MB) are `unverified` by design | pass, with that exception |
| Gates of A6, B7, C7, D1 | passed on the same commit | passed on earlier commits: A6 jobs 5713402 and 5713411 (52d35c1), D1 jobs 5704689, 5713378 and 5837867, B7 gates job 5704682 and jobs 5839003, 5839004 (2aad124), C7 job 5713857 (ca2f7f8), K1 and K2 jobs 5830324, 5830325. The source differs from the D1 job by `src/numa.c` only | carried, not re-run |

The `make test` result of the job: not run (`SKIP_TEST=1`, `test-skipped.txt`); the tool test suites passed locally on this commit before it was staged.

## How the run was submitted

Shape of the command (placeholders for what is not kept in the repository; the variables were exported in the submitting shell and passed with `--export=ALL`):

```bash
export ROOT=<tree> PYTHON=<validation python> STAGE=bench PHASE=D2 HF_GATE=0 SKIP_TEST=1 SKIP_PRELUDE_HF=1 REPEATS=3 B7_THP=off D_THP=off
export D2_M1="avx512_fp32=128 amx_bf16=256 avx512_q8=256 amx_int8=256" D2_M2="avx512_fp32=64 amx_bf16=256 avx512_q8=256 amx_int8=256"
MODELS="1b:<artifacts>:<HF snapshot>" OUT=<new directory> sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=04:00:00 \
  scripts/cresco8-prefill.sbatch
MODELS="8b:<artifacts>:<HF snapshot>" OUT=<new directory> sbatch --partition=<flat-mode partition> --nodelist=<verified Flat-mode node> --time=12:00:00 \
  --dependency=afterany:<first job> scripts/cresco8-prefill.sbatch
```

The matrix is `phase_D2` (144 launches per model; `DRY_RUN=1` prints it). The plan estimated 5-6 hours; the run took 2 h 28 min in all (27 min for 1B, 2 h 1 min for 8B).

## Deviations from the plan

- Numbers of A6, B7, C7 and D1 are not reused: D2 reruns the headline points. Comparisons with them (the one-socket rungs are 9-17% faster than C7's) are stated as such.
- No placement comparison against staged weights is part of D2 (every launch uses engine-owned placement, `LLAMA_PLACEMENT=hbm` from a copy staged on DDR node 0); the staged comparison is A6's.
- The campaign was submitted as two jobs, one per model, because of the node's time limits; both used the same tree and the same node.
- The three two-socket 1B W8A8 runs from a retained context of 8192 tokens (`1b-int8-teams-hbm-s2-t56-m64-p8192-kavx512_q8-xdec`) stopped on a stop token after 29 of 128 generated tokens; the other 285 reports reached the generation limit. Their decode rate (195.96 tok/s) is measured over 29 tokens. The one-socket runs of the same configuration generate 128 tokens: the two paths diverge in the continuation, which the W8A8 statistical gate allows.
- The one-socket batched runs do not verify their work region (1-70 MB); the generated tables label those rows `hbm (unverified)`. Two-socket reports state `placement: file` (the shard loader's staging) while their regions are verified on nodes 2 and 3.
- 1B prefill at 86 tokens is noise-limited (68-120 ms); two sockets do not win beyond variability for AMX-BF16 and AVX-512 FP32 there.
- The Hugging Face gate and `make test` were not part of the job; K1, K2 and K4 ran them on 17 November 2026. K3 (the F3 acceptance comparison with the kernels script) was not run.

## Raw data

`D2/summary.md` and the JSON reports (`<label>-<rep>.json`) of both campaign directories, the `*.perf.csv` files (first repeat), `environment.txt`,
`flat-mode.json`, `prompts.json`, `exit-status.txt` (`exit_code=0`), `no-hf-gate.txt`, `test-skipped.txt`. The generated tables of both jobs are kept in
[numa-tp-prefill-results-summary-table.md](numa-tp-prefill-results-summary-table.md). The `.stdout` and `.stderr` files stay on the cluster.
