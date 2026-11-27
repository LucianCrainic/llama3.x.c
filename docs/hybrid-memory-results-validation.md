# Hybrid HBM/DDR memory baseline campaign: validation record

Companion of [hybrid-memory-results.md](hybrid-memory-results.md). Partial: the jobs of the sections finished so far. Result directories are given by name only (each is the per-job
directory the script creates); a "tree" is a staged copy of the tracked files of the commit named. Every job ran `scripts/hybrid-campaign.sbatch` exclusively on one Flat node
of the HBM partition. Labels: **measured** (a job), **derived** (arithmetic on measured values), **modelled** (a limits-model ceiling).

## 1. Jobs

| job | steps | tree | elapsed | runs | result directory | result |
|---|---|---|---|---:|---|---|
| 5950120 | C0 (8B, 1S and 2S) | 47c08fb | 1:37:00 | 240 | p4-20261126-c0-8b | 0 failed; 6 trace groups of 40, 0 differ |
| 5950121 | C1, C2, C7 | 47c08fb | 2:02:25 | 64 | p4-20261126-c1-c2-c7 | 0 failed; the synthetic cells and the cold TTFT runs contain the startup stall (void for the results) |
| 5962577 | C3, C4, reruns of the C1 and C2 32768 and C2 131072 synthetic cells | 787a714 | 3:36:26 | 126 | p4-20261127-c3-c4 | 0 failed; traces equal in 5 groups (16, 18, 12, 42, 18); every placement check PASS; one cell not admitted (section 2) |
| 5962578 | C6 (70B 1S, first set) | 787a714 | 1:02:27 | 28 | p4-20261127-c6 | 0 failed; nothing skipped |

Node time so far (derived, elapsed x one node): 1.62 + 2.04 + 3.61 + 1.04 = 8.31 node-hours of the 40 allowed, one job per node, nothing cancelled.

## 2. C3 and C4 (job 5962577)

* **Cells:** C3 at 32768 (real fill) serial, rows f0.1 c8, rows f0.2 c14, 5 repetitions; at 65536 (synthetic) serial, serial-kv80, rows-auto-c14, rows-auto-c20, 5 repetitions;
  the synthetic check of the serial 32768 cell (3 repetitions). C4 at 65536, 98304 and 131072 (synthetic): serial-w80, rows-auto-c20, rows-kvmatch-c20, 5 repetitions.
  Reruns: C1 hyb80 and C2 ddr at 32768 synthetic, C2 ddr at 130816 synthetic, 3 repetitions each. One profiled run per cell.
* **Plan gate:** `rows-auto-c20` at 131072 positions is not admitted (socket 0 HBM node needs 69,259,980,800 bytes, 9,130,438,656 over its 60,129,542,144-byte budget);
  it is in the job's `skipped.tsv`. The 80/80 hybrid is not admitted at 65536 either, so the C1 65536 cell does not exist.
* **Synthetic fill:** build of tree 787a714 (develop 5f28871 with the 30 s primer guard); the slowest synthetic decode step of every synthetic run is at most 0.47 s (limit 1 s).
  Synthetic against real at 32768 for the serial cell: 7.261 against 7.199 tok/s (+0.9%, 3 against 5 runs).
* **Pause scan:** 0 of the 126 runs have an idle span of 0.5 s or more in the decode window, none a hole of dropped samples.
* **Medians:** re-derivable from the raw `--json` files of the directory by `python3 tools/hybrid_campaign_summary.py <directory>`; the manager's independent derivation agrees to the digits shown.
* **Ceilings:** `tools/hybrid_limits.py` with the concurrent formula of 27 November (K/V read after the split GEMV at full-socket rates); the earlier formula is described in
  docs/plans/hybrid-limits.md section 3.1.
