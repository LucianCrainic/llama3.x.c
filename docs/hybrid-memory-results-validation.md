# Hybrid HBM/DDR memory baseline campaign: validation record

Companion of [hybrid-memory-results.md](hybrid-memory-results.md). Jobs, trees, elapsed times, node-hours, checks, and the way every median is re-derived. Result directories are given by
name only (each is the per-job directory the script creates). A "tree" is a staged copy of the tracked files of the commit named. Every job ran `scripts/hybrid-campaign.sbatch`
exclusively on one Flat node of the HBM partition, one job per node at a time. Labels: **measured** (a job), **derived** (arithmetic on measured values), **modelled** (a limits-model ceiling).

## 1. Jobs

Elapsed times of the first four jobs are from the scheduler's accounting; the later ones are from the job logs (the first to the last log line of the script, because the accounting was failing with
authentication errors at the time; the scheduler's own figure is about half a minute longer per job).

| job | step(s) | tree | elapsed | runs | result directory | result |
|---|---|---|---|---:|---|---|
| 5950120 | C0: 8B, one and two sockets | 47c08fb | 1:37:00 | 240 | p4-20261126-c0-8b | 0 failed; 6 trace groups of 40, 0 differ; all placement checks PASS |
| 5950121 | C1, C2, C7 (first run) | 47c08fb | 2:02:25 | 64 | p4-20261126-c1-c2-c7 | 0 failed. **Used:** the real-fill decode cells at 4096 and 32768 (c1, c2). **Void:** every synthetic cell and every TTFT cell (the start-up stall of the node lies in the windows), replaced below |
| 5962575 | plan gate of C3, C4, C6, C1, C2 | 787a714 | under 1 min | 0 | p4-20261127-plan | `--plan-only` of every cell; `c1 hyb80` at 65536 and `c4 rows-auto-c20` at 131072 are not admitted |
| 5962577 | C3, C4, reruns of the C1 and C2 32768 synthetic cells and of the C2 131072 cell | 787a714 | 3:36:26 | 106 | p4-20261127-c3-c4 | 0 failed; 5 trace groups (16, 18, 12, 42, 18 traces), 0 differ; every placement check PASS; slowest synthetic decode step at most 0.47 s; one cell not admitted (`rows-auto-c20` at 131072, in `skipped.tsv`) |
| 5962578 | C6 (70B, one socket), first set | 787a714 | 1:02:27 | 28 | p4-20261127-c6 | 0 failed; nothing skipped; slowest synthetic step at most 0.42 s |
| 5962625 | plan gate of the C6 follow-up | c834754 | under 1 min | 0 | p4-20261127-plan2 | all cells admitted |
| 5962626 | C6 follow-up (C6b): kvmatch cells, serial twice, 4096 positions | c834754 | 1:53:11 | 84 | p4-20261127-c6b | 0 failed; nothing skipped; 2 trace groups, 0 differ |
| 5963861 | plan gate of the TTFT job | ae89d0e | under 1 min | 0 | p4-20261127-plan3 | all cells admitted |
| 5963862 | TTFT with prewarm (C1, C2, C7), cold variants of C1, harness check h1 | ae89d0e | 2:19:14 | 70 | p4-20261127-ttft | 1 run recorded as failed (see 3); 6 trace groups, 0 differ |
| 5963868 | plan gate and K/V-layer search of the grid job | 0221a93 | 0:41 | 0 | p4-20261128-plan4 | the four `@KVFIT@` cells resolved to 54, 46 (98304) and 40, 34 (131072) layers |
| 5963869 | grid job c4g | 0221a93 | 1:46:34 | 52 | p4-20261128-c4g | 0 failed; one trace group per context (26 traces each), 0 differ; slowest synthetic step at most 0.47 s |

**Node time (derived, elapsed x one node):** 1.617 + 2.040 + 3.607 + 1.041 + 1.886 + 2.321 + 1.776 + about 0.05 (five plan jobs) = **14.3 node-hours** of the 40 allowed. Nothing was cancelled, no
job ran beside another on the same node.

## 2. Tools and trees

* Summary: `tools/hybrid_campaign_summary.py DIR [DIR ...]` (medians [min-max] per cell, CAS bytes per token and per second, UPI as derived GB, package J per token, load time, resident set, placement
  PASS/FAIL, the ceiling of the plan of each report from `tools/hybrid_limits.py`, a pause scan on each run's measured window, synthetic against real, profile phases, token traces across directories).
  Version used for this page: feature/hybrid-campaign at the commit of this document; the ceilings are those of the same checkout (the planner capacity model of commit 96a20ed and the corrected concurrent formula).
* The jobs ran trees 47c08fb (C0, C1/C2/C7 first run), 787a714 (C3/C4, C6), c834754 (C6b), ae89d0e (TTFT) and 0221a93 (grid): each is the campaign branch on develop 0dd18db to 5e14327. The engine code
  is develop's in every job; the later trees add the synthetic primer guard (`LLAMA_SYNTHETIC_KV_MIN_SECONDS`, develop 5f28871) and the prewarm knob (`LLAMA_BENCH_PREWARM_SECONDS`, develop 5e14327), which only
  the cells that name them use. No engine default was changed.
* Setup of every cell, from the job's environment record: `LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384 LLAMA_KV_VERIFY=1 LLAMA_GEMV_STAGGER=1`,
  explicit topology (`explicit:0/2/0,1/3/1`), one socket under `numactl --cpunodebind=0`, every `LLAMA_*` variable of the caller cleared before each cell. Command: `--benchmark --chat-template --no-text -t 0 -n 0`
  with `--context-length` warm + 256 (131072 at the limit), prime-number prompt, 96 generated tokens (8B 256, TTFT cells 32 and a prompt file of the stated token count with the chat template).
* Counters: `perf stat -a --per-socket -I 100` with the CAS reads and writes of the DDR and HBM controllers, UPI data flits, package and RAM energy, cycles (decode cells); the TTFT cells count the CAS reads and the
  cycles. 32 bytes per HBM CAS, 64 per DDR CAS. The perf window is cut to each phase with the report's `phase_epoch_ns` (`tools/hbm_ddr_results.py`).

## 3. Checks

* **Every medians table is regenerated from the raw `--json` reports of its directory** by the summary tool; the manager's independent derivation from the same files agrees to the digits shown for every cell reported
  here. The only disagreement is the run count of the C3/C4 job: 106 runs (the manager's note said 126; the 106 report files and the 106 traces in the five groups agree).
  Rounding: the C6b 4096 gain of the 14-core split is +10.35% from the raw medians (5.4831 against 4.9688 tok/s), +10.3% when computed from the rounded 5.483 and 4.969, and is quoted as +10.4%.
* **Placement:** every run of the jobs above, except one, passed `tools/check_placement.py --planned --allow-sampled` inside the job. The exception is run `r5-c7-amx-t2048` of job 5963862: the engine run completed
  (31 decode steps, 5.80 s prefill) and the script's post-run check failed because the Python of the shared file system could not be started ("No such device"); the check run afterwards on the saved report
  passes (26 PASS, 0 FAIL). The run is kept in the medians (its TTFT, 5.16 s, is the median of its cell).
* **Pause scan:** idle spans of 0.5 s or more (and holes of dropped samples) of the joint HBM+DDR read bins, below 10% of the window's 90th percentile, in the decode window of every decode run and in the prefill window of every
  TTFT run. Counts per job: C0 0 of 240 flagged; C3/C4 0 of 106; C6 and C6b 0 of 112; grid job 0 of 52; TTFT job: 0 of 47 prewarmed runs, 7 of 10 cold runs of C1 (2 at 2048 tokens, 5 at 8192).
  Job 5950121 (cold processes): every synthetic run flagged (voided), the TTFT runs at 8192 and with the AVX-512 kernels flagged (voided); its real-fill decode cells are clean.
* **Token traces:** equal inside every (model, sockets, measurement, kernels) group of every job; across the C1/C2/C7 and C3/C4 jobs the 70B 2S d32768 trace is the same for the all-HBM hybrid, the DDR layout and the
  row-split cells.
* **Synthetic fill:** synthetic against real at 32768: +0.8% (70B 2S hybrid), +0.4% (70B 2S ddr), +0.8% (70B 1S serial); the slowest decode step of every synthetic run in the jobs from 787a714 on is at most 0.47 s.
* **Stop rule** (30% against `tools/hybrid_limits.py`): no decode cell exceeds it with the corrected formula (section 7 of the results page). The first formula would have flagged two 1S cells of C0.
  The planner capacity model of the library reproduces the layer counts of the plans of jobs 5962577, 5962578, 5962626 and 5963868 (tests in `tests/test_hybrid_limits.py`).

## 4. Re-deriving a median

```
python3 tools/hybrid_campaign_summary.py <results directory>            # every table of the results page for that directory
python3 tools/hybrid_campaign_summary.py <directory> --json              # the per-cell statistics as JSON
python3 tools/hybrid_campaign_summary.py --predict <plan.json> --warm N  # the limits-model prediction of a cell from its --plan-only report
python3 tests/test_hybrid_campaign.py; python3 tests/test_hybrid_limits.py
```

Each directory holds `runs.tsv` (run, step, setting, measurement, repetition, group, start and end epochs), the reports (`<run>.json`), the perf CSV and clock files, the token traces, the placement
check output, `plan/` (the `--plan-only` report, resolved environment and prediction of every cell), `skipped.tsv`, `failures.tsv`, `environment.txt` (job, tree, host record, `numactl -H`, THP and governor)
and `source.sha256` (the hash of every tracked source file of the tree).
