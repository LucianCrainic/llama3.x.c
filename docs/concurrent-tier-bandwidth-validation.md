# Concurrent HBM + DDR bandwidth: validation record

Companion of [concurrent-tier-bandwidth.md](concurrent-tier-bandwidth.md). It lists the jobs, where the raw files are, the placement proof, the counter-against-computed-bytes
checks (with every exception), and what is and is not recorded about the data content. No host names or cluster paths are given; a "results directory" is the per-job directory
the job script creates on the cluster's results file system.

## 1. Jobs

All jobs ran `scripts/concurrent-tier.sbatch` exclusively on one Flat/Quadrant node of the HBM partition, from a staged copy of the tracked files of the commit named below
(a `COMMIT` file in the staged tree holds the full hash; the job copies the built binary and its sha256 into its results directory).

| job id | node | commit | stages | repetitions | elapsed |
|---|---|---|---|---|---|
| 5916187 | node A | 2959b6b | smoke (failed: a GEMV-shaped region was only 90% populated; fixed) | 1 | seconds |
| 5916190 | node A | 49c267c | smoke (passed, one and two sockets, both page sizes) | 1 | 0:03 |
| 5916193 | node A | 94b87bb | frequency check with turbostat | 1 | 0:03 |
| 5916195 | node A | f62cad3 | smoke, base (3 reps), e1, e2, e6, lat, freq, e5hbm | 7 | 1:29:08 |
| 5916213 | node B | 6809025 | rep, rep-e5 (the replication points) | 7 | 0:11:21 |
| 5916429 | node A | 6809025 | e3, e4, e5, e5hbm (the e3 and e4 regions of this job were zero-filled) | 7 | 0:50:23 |
| 5923181 | node A | 2f8d329 | fillcmp, e3, e4, e5c, e5off, shuffled order inside every process | 7 | 0:32:34 |
| 5923755 | node A | 63fb770 | freq (turbostat over HBM-only, 48+8, 28+28, DDR-only windows) | 1 | about 0:03 |
| 5924010 | node A | 0f775a9 | hbmx, e6b, e5alias, e5c2 (second-round checks: baseline extension, real splits, stripe aliasing, row-split worker count), shuffled, observed pinning recorded | 7 | 0:39:58 |
| 5924011 | node B | 0f775a9 | repe3, repe4, repe5 (shuffled replication of E3, E4, E5) | 7 | 0:13:02 |
| 5925935 | node A | c9882b1 | e1g, e3g (staggered HBM stripes in splits and interleaved regions), shuffled | 7 | 0:11:46 |
| 5923753, 5923807 | node A | e7cf313, 63fb770 | pooled summary step (tables and figure) | - | seconds |

Node time used by these jobs is about 4.5 node-hours, against the 30-hour limit. Node A is the node on which every headline number was measured; node B repeated the key points.
(The node names are in the job logs, not here.) The committed tree at the end of the work is the branch `perf/concurrent-tier-bandwidth`.

## 2. Raw files

Each results directory holds: `environment.txt` (date, kernel 5.14, lscpu, numactl, THP and NUMA-balancing settings, perf version, sudo rights, uncore frequency limits, job description), `build.log`,
`executables.sha256`, `source-*.sha256` (every source file of the tree), `cfg/<stage>.cfg` (the configurations of each stage, one per line), `raw/<label>.jsonl` (stdout of one process:
a `start` event, `placement` events, one line per window and socket, and, from commit 1e8ba44 on, an `args` event with the command line, the fill and the shuffle option),
`raw/<label>.perf.csv` (the perf interval series, 100 ms, per socket), `raw/<label>.stderr`, `process-status.tsv` (exit status of every process), `job.stdout`, `exit-status.txt`
and, after the analysis step, `summary/{tables.md,summary.json,concurrent-tier-bandwidth.png}`. Labels are STAGE-PAGE-rN (page 2m or 4k, repetition N).

## 3. Placement proof

Every tier region (20 GiB per tier and socket) was queried page by page with `move_pages(2)` (no migration) after it was populated (stage `setup`) and again at the end of the process
(`end`; for 4 KiB regions the end check is sampled, 996480 pages per region: first and last page of every 2 MiB extent plus a stride). Every interleaved E3/E4 region was queried
page by page against the pattern it was built with (event `window`).

* Events: 1356 `placement` events over the first five analysed results directories (536 setup, 536 end, 284 interleaved) and 665 more over jobs 5924010, 5924011 and 5925935.
* Pages queried 5,145,765,632 (first five directories) + 2,190,440,448 (second round); pages on the intended node: all of them; on another node 0; missing 0; query errors 0.
* **Pinning.** From the second round on every process prints a `pinning` event with the CPU each worker's affinity mask was read back as: 133 events, 7448 workers, 7448 on the CPU assigned to them (a mismatch aborts the process). `perf` was started on a different core from the pool leader's in these jobs (the last core of the idle socket in one-socket runs). The first five directories predate this record; their pinning was checked by the same read-back at pool start (`team_affinity_verified`) but is not printed. No run was accepted with a failed check: the program aborts on any mismatch.
* Transparent huge pages: for every 2 MiB region the THP-backed bytes were at least 95% of the region (all of them in practice, 20 GiB); for every 4 KiB region THP-backed bytes were 0
  (`madvise(MADV_NOHUGEPAGE)`, with THP set to `always` on the node).
* NUMA balancing was off (`numa_balancing=0`); the kernel is 5.14.0 and has no weighted-interleave sysfs, so E3 used `set_mempolicy(MPOL_BIND)` per page before the first touch.

## 4. Counter bytes against computed bytes

**Conversion.** DDR CAS reads x 64 B, HBM CAS reads x 32 B (calibrated earlier on this part). Computed bytes of a window are the bytes the workers issued, warm-up included, per tier
(GEMV: exact rows x row bytes per pass x passes; page-interleaved region: the window total split by the page ratio, because the program books all of its bytes under one tier).

**Attributing counters to windows.** The perf series is cut into bursts of consecutive 100 ms intervals whose memory traffic exceeds 2 GB/s; windows are separated by idle gaps of 0.6 s, so a window is one
burst and its counters are the sums over the burst's intervals (the first and last interval hold the idle edges, so partial intervals at the window edges are counted whole but contain only the
traffic that really happened). This method does not depend on the perf clock origin. When a process had as many bursts as windows (5138 of 5455 socket-0 windows) the k-th burst is the k-th window. When it had more
(the kernel zeroing and touching the 32 GiB E3/E4 regions made extra bursts: 306 windows), each window is paired with the burst whose start is closest to its expected start on the perf clock (the perf origin precedes the
`start` event by 0.87 s, the median of 4590 order-attributed windows, range 0.56-0.95 s because bursts begin on a 0.1 s grid), within 0.6 s; 11 windows (E4, 4 KiB) had no burst and have no counters. A check that instead slices
the series at the window's wall-clock times depends on that origin: shifting it by 0.1 s moves per-window ratios by 3-5%, and slices that reach into a neighbouring window's traffic produce DDR ratios of 1.18-1.24 for E4 and fillcmp,
which the burst attribution does not show (below: absolute excess at most 3 GB per window).

**Per window and tier** (ratio = counter bytes / computed bytes; only tiers with at least 0.5 GB computed in the window are compared):

| stage | tier | windows | min | median | max | outside +-5% |
|---|---|---|---|---|---|---|
| base | ddr | 594 | 0.996 | 0.999 | 1.449 | 6 |
| base | hbm | 690 | 0.997 | 0.998 | 1.466 | 6 |
| e1 | ddr | 224 | 0.998 | 1.002 | 1.071 | 7 |
| e1 | hbm | 266 | 0.997 | 0.999 | 1.001 | 0 |
| e1g | ddr | 168 | 1.002 | 1.008 | 1.035 | 0 |
| e1g | hbm | 266 | 0.997 | 0.998 | 1.000 | 0 |
| e1v | ddr | 154 | 0.998 | 1.004 | 1.071 | 7 |
| e1v | hbm | 154 | 0.998 | 0.999 | 1.001 | 0 |
| e2 | ddr | 308 | 0.999 | 1.003 | 1.023 | 0 |
| e2 | hbm | 322 | 0.998 | 1.000 | 1.001 | 0 |
| e3 | ddr | 84 | 1.002 | 1.009 | 1.034 | 0 |
| e3 | hbm | 84 | 0.999 | 1.000 | 1.000 | 0 |
| e3g | ddr | 42 | 1.005 | 1.010 | 1.023 | 0 |
| e3g | hbm | 63 | 0.999 | 0.999 | 1.000 | 0 |
| e4 | ddr | 72 | 0.999 | 1.004 | 1.024 | 0 |
| e4 | hbm | 100 | 0.998 | 0.999 | 1.000 | 0 |
| e5 | ddr | 1092 | 0.999 | 1.007 | 1.092 | 56 |
| e5 | hbm | 1036 | 0.998 | 1.000 | 1.001 | 0 |
| e5alias | hbm | 350 | 0.956 | 0.998 | 1.000 | 0 |
| e5c | ddr | 196 | 0.999 | 1.007 | 1.062 | 14 |
| e5c | hbm | 280 | 0.997 | 0.999 | 1.001 | 0 |
| e5c2 | ddr | 252 | 0.998 | 1.005 | 1.014 | 0 |
| e5c2 | hbm | 308 | 0.998 | 1.000 | 1.001 | 0 |
| e5hbm | hbm | 308 | 0.997 | 0.998 | 1.000 | 0 |
| e5off | hbm | 140 | 0.998 | 0.999 | 0.999 | 0 |
| e6 | ddr | 56 | 1.004 | 1.014 | 1.023 | 0 |
| e6 | hbm | 420 | 0.998 | 0.999 | 1.458 | 14 |
| e6b | ddr | 462 | 1.002 | 1.008 | 1.038 | 0 |
| e6b | hbm | 462 | 0.998 | 0.999 | 1.000 | 0 |
| fcrandom | ddr | 35 | 0.999 | 1.002 | 1.012 | 0 |
| fcrandom | hbm | 70 | 0.998 | 0.999 | 1.000 | 0 |
| fczero | ddr | 35 | 0.999 | 1.002 | 1.011 | 0 |
| fczero | hbm | 70 | 0.998 | 0.999 | 1.000 | 0 |
| freq | ddr | 4 | 0.999 | 1.000 | 1.003 | 0 |
| freq | hbm | 5 | 0.998 | 0.998 | 1.000 | 0 |
| hbmx | hbm | 98 | 0.998 | 0.999 | 1.458 | 14 |
| lat | ddr | 140 | 0.999 | 1.003 | 1.022 | 0 |
| lat | hbm | 168 | 0.998 | 1.000 | 1.037 | 0 |
| smoke | ddr | 58 | 0.997 | 1.004 | 1.019 | 0 |
| smoke | hbm | 64 | 0.993 | 0.999 | 1.002 | 0 |
| zfill | ddr | 158 | 0.999 | 1.008 | 1.033 | 0 |
| zfill | hbm | 186 | 0.998 | 0.999 | 1.000 | 0 |


Aggregate per run (244 processes): HBM counter/computed bytes min 0.9929 median 0.9992 max 1.0311; DDR min 0.9975 median 1.0075 max 1.1695.

Processes with an aggregate ratio outside +-2%: [('e1g-4k-r1', 19, 0.998784934704142, 1.022841988834883), ('e1g-4k-r2', 19, 0.9987919101748984, 1.0226746522851853), ('e1g-4k-r3', 19, 0.9988069991933037, 1.022762682708761), ('e1g-4k-r4', 19, 0.9987857134442493, 1.0226292092486897), ('e1g-4k-r5', 19, 0.9987812994958973, 1.0226419879822215), ('e1g-4k-r6', 19, 0.9988007607419698, 1.0227332736781907), ('e1g-4k-r7', 19, 0.9987724456562006, 1.0227648779497243), ('e3g-2m-r2', 9, 0.9994652844317915, 1.0204076882931805), ('e6-2m-r1', 30, 1.0052027890044297, 1.03039096116712), ('e6-2m-r2', 30, 1.0047478464323034, 1.0307490972002122), ('e6-2m-r3', 30, 1.0050350990988168, 1.029032152918297), ('e6-2m-r4', 30, 1.0048844991534895, 1.028735378047813), ('e6-2m-r5', 30, 1.0050370271718756, 1.030515503641219), ('e6-2m-r6', 30, 1.0049472274217672, 1.028519903484194), ('e6-2m-r7', 30, 1.0051794593771473, 1.0282631682535601), ('e6-4k-r1', 30, 1.0067662765112164, 1.1672899323904489), ('e6-4k-r2', 30, 1.0067100677961434, 1.1660668696186827), ('e6-4k-r3', 30, 1.0067885035431794, 1.1662680433526331), ('e6-4k-r4', 30, 1.006765849905249, 1.1683161299203946), ('e6-4k-r5', 30, 1.006804635452403, 1.1694987687449858), ('e6-4k-r6', 30, 1.0067569048349942, 1.1680062213572122), ('e6-4k-r7', 30, 1.0067889328835113, 1.1681857354828962), ('hbmx-2m-r1', 7, 1.0276811015821232, None), ('hbmx-2m-r2', 7, 1.0290060150403748, None), ('hbmx-2m-r3', 7, 1.029401514863013, None), ('hbmx-2m-r4', 7, 1.0294600719768092, None), ('hbmx-2m-r5', 7, 1.0290203822259705, None), ('hbmx-2m-r6', 7, 1.02862087893884, None), ('hbmx-2m-r7', 7, 1.0286491518281637, None), ('hbmx-4k-r1', 7, 1.0306619926102516, None), ('hbmx-4k-r2', 7, 1.0307767466885551, None), ('hbmx-4k-r3', 7, 1.0307283391815794, None), ('hbmx-4k-r4', 7, 1.0308708926598764, None), ('hbmx-4k-r5', 7, 1.031120394473258, None), ('hbmx-4k-r6', 7, 1.0306048430843946, None), ('hbmx-4k-r7', 7, 1.0305567524817212, None)].

**Closure in absolute GB** (the same windows; for the page-interleaved E3/E4/fillcmp windows the computed split follows the page ratio, so the HBM CAS ratio is about 1, and the sum of both tiers closes to 1.000-1.003.
The ratio 0.72 obtained by comparing the HBM counter with all interleaved bytes booked under "HBM" is an accounting artefact: 0.72 is 1 - the DDR share of the E3 mix at the ratios tested; the sum (HBM+DDR CAS) / total closes at 1.002):

| config or stage | sockets | windows | computed HBM GB | HBM CAS GB | computed DDR GB | DDR CAS GB | DDR excess GB median | DDR excess GB max | (HBM+DDR CAS)/(computed total) |
|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 1284 | 74.2 | 74.0 | 0.0 | 0.9 | +0.11 | +34.77 | 1.000 |
| e1 | 1 | 350 | 427.4 | 426.6 | 116.2 | 117.7 | +0.31 | +1.75 | 1.000 |
| e1g | 1 | 266 | 588.2 | 587.1 | 49.3 | 50.3 | +0.34 | +1.72 | 1.000 |
| e1v | 1 | 168 | 244.8 | 244.6 | 116.0 | 116.5 | +0.33 | +0.93 | 1.001 |
| e2 | 1 | 350 | 432.1 | 431.9 | 205.7 | 207.0 | +0.44 | +1.86 | 1.001 |
| e3-0-1 | 1 | 28 | 0.0 | 0.2 | 312.8 | 314.2 | +1.63 | +2.35 | 1.006 |
| e3-1-0 | 1 | 28 | 689.6 | 689.4 | 0.0 | 1.8 | +1.76 | +2.69 | 1.002 |
| e3-1-1 | 1 | 28 | 274.8 | 274.8 | 274.8 | 276.8 | +1.83 | +2.28 | 1.003 |
| e3-2-1 | 1 | 28 | 435.7 | 435.5 | 217.8 | 219.6 | +1.65 | +2.56 | 1.002 |
| e3-3-1 | 1 | 28 | 521.7 | 521.6 | 173.9 | 175.9 | +1.70 | +2.68 | 1.002 |
| e3-4-1 | 1 | 28 | 567.4 | 567.3 | 141.9 | 143.9 | +1.67 | +2.66 | 1.002 |
| e3-8-1 | 1 | 28 | 623.3 | 623.0 | 77.9 | 79.5 | +1.69 | +2.70 | 1.002 |
| e3g | 1 | 63 | 612.4 | 611.9 | 76.5 | 77.7 | +1.49 | +1.73 | 1.002 |
| e4-ddr56 | 2 | 56 | 0.0 | 0.1 | 313.7 | 314.0 | +0.07 | +0.53 | 1.001 |
| e4-hbm56 | 2 | 56 | 678.8 | 678.1 | 0.0 | 1.0 | +1.01 | +1.79 | 1.000 |
| e4-inter-8-1 | 2 | 34 | 631.5 | 631.1 | 78.9 | 80.0 | +0.81 | +1.89 | 1.001 |
| e4-shared-8-1 | 2 | 56 | 604.6 | 604.0 | 75.6 | 76.5 | +1.04 | +1.84 | 1.001 |
| e4-split-cd0 | 2 | 56 | 682.1 | 680.7 | 0.0 | 1.0 | +1.03 | +1.78 | 1.000 |
| e5 | 1 | 1120 | 245.5 | 245.3 | 86.6 | 87.2 | +0.67 | +1.59 | 1.002 |
| e5alias | 1 | 350 | 346.5 | 331.5 | 0.0 | 0.4 | +0.36 | +1.73 | 1.000 |
| e5c | 1 | 280 | 285.9 | 285.8 | 50.0 | 50.6 | +0.56 | +1.07 | 1.001 |
| e5c2 | 1 | 308 | 259.9 | 259.9 | 81.8 | 81.8 | +0.48 | +0.98 | 1.001 |
| e5hbm | 1 | 308 | 336.9 | 336.3 | 0.0 | 0.6 | +0.57 | +0.99 | 1.000 |
| e5off | 1 | 140 | 336.9 | 336.5 | 0.0 | 0.6 | +0.62 | +1.05 | 1.001 |
| e6 | 1 | 420 | 658.6 | 657.7 | 0.0 | 1.6 | +0.60 | +1.76 | 1.001 |
| e6b | 1 | 462 | 581.8 | 581.1 | 91.8 | 93.0 | +0.97 | +1.70 | 1.001 |
| fcrandom-ddr56 | 1 | 7 | 0.0 | 0.1 | 314.6 | 314.3 | -0.34 | -0.32 | 0.999 |
| fcrandom-gemv32-act52 | 1 | 7 | 346.5 | 345.9 | 0.0 | 0.3 | +0.29 | +0.30 | 0.999 |
| fcrandom-gemv32-act56 | 1 | 7 | 346.5 | 346.1 | 0.0 | 0.3 | +0.29 | +0.33 | 1.000 |
| fcrandom-gemv64-act52 | 1 | 7 | 327.3 | 326.6 | 0.0 | 0.3 | +0.28 | +0.33 | 0.999 |
| fcrandom-gemv64-act56 | 1 | 7 | 327.3 | 326.8 | 0.0 | 0.3 | +0.26 | +0.27 | 0.999 |
| fcrandom-hbm56 | 1 | 7 | 680.9 | 679.7 | 0.0 | 0.3 | +0.32 | +0.36 | 0.999 |
| fcrandom-inter-1-0 | 1 | 7 | 684.6 | 684.2 | 0.0 | 1.7 | +1.72 | +1.76 | 1.002 |
| fcrandom-inter-4-1 | 1 | 7 | 550.9 | 550.7 | 137.7 | 139.3 | +1.49 | +1.59 | 1.002 |
| fcrandom-shared-4-1 | 1 | 7 | 537.1 | 536.5 | 134.3 | 134.6 | +0.33 | +0.36 | 1.000 |
| fcrandom-split28-28 | 1 | 7 | 338.7 | 338.5 | 265.8 | 266.0 | +0.25 | +0.28 | 1.000 |
| fcrandom-split48-8 | 1 | 7 | 580.2 | 579.4 | 91.1 | 91.4 | +0.32 | +0.36 | 0.999 |
| fczero-ddr56 | 1 | 7 | 0.0 | 0.1 | 315.1 | 314.8 | -0.35 | -0.33 | 0.999 |
| fczero-gemv32-act52 | 1 | 7 | 346.5 | 345.9 | 0.0 | 0.3 | +0.29 | +0.29 | 0.999 |
| fczero-gemv32-act56 | 1 | 7 | 346.5 | 346.1 | 0.0 | 0.3 | +0.29 | +0.32 | 0.999 |
| fczero-gemv64-act52 | 1 | 7 | 327.3 | 326.6 | 0.0 | 0.3 | +0.27 | +0.33 | 0.999 |
| fczero-gemv64-act56 | 1 | 7 | 327.3 | 326.8 | 0.0 | 0.3 | +0.26 | +0.27 | 0.999 |
| fczero-hbm56 | 1 | 7 | 720.9 | 719.6 | 0.0 | 0.3 | +0.31 | +0.38 | 0.999 |
| fczero-inter-1-0 | 1 | 7 | 735.5 | 735.1 | 0.0 | 1.6 | +1.65 | +1.69 | 1.002 |
| fczero-inter-4-1 | 1 | 7 | 578.8 | 578.6 | 144.7 | 146.3 | +1.57 | +1.62 | 1.002 |
| fczero-shared-4-1 | 1 | 7 | 581.4 | 580.9 | 145.3 | 145.7 | +0.34 | +0.38 | 1.000 |
| fczero-split28-28 | 1 | 7 | 353.7 | 353.6 | 276.9 | 277.1 | +0.26 | +0.30 | 1.000 |
| fczero-split48-8 | 1 | 7 | 612.2 | 611.3 | 97.9 | 98.2 | +0.33 | +0.34 | 0.999 |
| freq | 1 | 7 | 4150.9 | 4144.0 | 650.2 | 652.3 | +2.02 | +2.08 | 0.999 |
| hbmx | 1 | 98 | 644.1 | 643.4 | 0.0 | 0.6 | +0.56 | +1.70 | 1.001 |
| lat | 1 | 196 | 326.4 | 326.6 | 162.6 | 163.9 | +0.64 | +2.17 | 1.002 |
| smoke | 1 | 22 | 216.6 | 216.4 | 72.2 | 72.6 | +0.29 | +1.13 | 1.001 |
| smoke | 1 | 4 | 176.1 | 174.9 | 88.6 | 88.4 | -0.09 | +0.42 | 0.996 |
| smoke | 2 | 44 | 222.6 | 222.4 | 74.2 | 74.9 | +0.29 | +1.28 | 1.001 |

The DDR excess over the computed bytes is 0.1-1.8 GB per window in the median and at most 3 GB (apart from the exceptions below); it is the same size in windows with 0 computed DDR bytes (E3 1:0: 1.8 GB, E4 HBM-only: 1.0 GB per window, e6: 1.6 GB)
as in windows with 300 GB (DDR-only: -0.3 to +0.1), so it is a roughly constant background of about 1-2 GB per window (the kernel work around the window, `perf` and the shell, other traffic of the node, and for E3/E4 the page zeroing
of the 32 GiB region built just before the window), not hardware over-fetch proportional to the volume.

**Exceptions (outside 0.95-1.05) and reasons.**

1. NTA prefetch hint (config `base-*-h2-*` and `e6-hint2`): HBM and DDR counter bytes 1.44-1.47 times the computed bytes in 40 windows (26 of the base and e6 stages, 14 of the extended sweep). These are real extra reads (the hardware fetches the line a second time; the delivered
   bandwidth halves: 231 GB/s): not an error of the accounting, an effect of the hint (explanation inferred from the 1.45 factor and the halved rate; not examined further).
2. E1 split 54+2 at 4 KiB (`e1-cd2`, `e1v-cd2`), DDR tier: 1.07 (computed 12.9-24.7 GB, excess 1.0-1.7 GB): the constant background above, large against a small DDR volume.
3. E5 and E5c with r = 0.05 (DDR volume 16.4 GB gs64, 17.3 GB gs32), DDR tier: 1.055-1.091, excess 0.9-1.5 GB: the same background on a small volume.
Nothing else is outside 5%.

Runs without counters: 6 [('e4-4k-r1', 'no burst near the window'), ('e4-4k-r2', 'no burst near the window'), ('e4-4k-r3', 'no burst near the window'), ('e4-4k-r4', 'no burst near the window'), ('e4-4k-r5', 'no burst near the window'), ('e4-4k-r6', 'no burst near the window')]

**Aggregate per process** (all attributed windows of a process, no alignment between windows involved): see the first lines above. The DDR maximum of 1.17 belongs to the e6 processes, which contain the NTA-hint windows.

**All 152 windows outside +-5%:**

| run | config | socket | tier | ratio | computed GB | attribution |
|---|---|---|---|---|---|---|
| base-2m-r1 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.466 | 156.1 | order |
| base-2m-r1 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.448 | 71.6 | order |
| base-2m-r2 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.466 | 155.3 | order |
| base-2m-r2 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.448 | 70.5 | order |
| base-2m-r3 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.466 | 156.2 | order |
| base-2m-r3 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.449 | 71.3 | order |
| base-4k-r1 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.449 | 161.1 | order |
| base-4k-r1 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.440 | 79.1 | order |
| base-4k-r2 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.449 | 161.0 | order |
| base-4k-r2 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.441 | 78.6 | order |
| base-4k-r3 | base-hbm-c56-S2-pf4096-h2-nt0 | 0 | hbm | 1.449 | 161.6 | order |
| base-4k-r3 | base-ddr-c28-S2-pf4096-h2-nt0 | 0 | ddr | 1.440 | 78.8 | order |
| e1-4k-r1 | e1-cd2 | 0 | ddr | 1.071 | 24.2 | order |
| e1-4k-r1 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r2 | e1-cd2 | 0 | ddr | 1.070 | 23.8 | order |
| e1-4k-r2 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r3 | e1-cd2 | 0 | ddr | 1.071 | 23.5 | order |
| e1-4k-r3 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r4 | e1-cd2 | 0 | ddr | 1.070 | 24.7 | order |
| e1-4k-r4 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r5 | e1-cd2 | 0 | ddr | 1.070 | 24.0 | order |
| e1-4k-r5 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r6 | e1-cd2 | 0 | ddr | 1.071 | 24.2 | order |
| e1-4k-r6 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e1-4k-r7 | e1-cd2 | 0 | ddr | 1.070 | 24.6 | order |
| e1-4k-r7 | e1v-cd2 | 0 | ddr | 1.071 | 12.9 | order |
| e6-2m-r1 | e6-hint2 | 0 | hbm | 1.455 | 276.4 | order |
| e6-2m-r2 | e6-hint2 | 0 | hbm | 1.458 | 256.0 | order |
| e6-2m-r3 | e6-hint2 | 0 | hbm | 1.455 | 278.1 | order |
| e6-2m-r4 | e6-hint2 | 0 | hbm | 1.456 | 272.9 | order |
| e6-2m-r5 | e6-hint2 | 0 | hbm | 1.456 | 270.3 | order |
| e6-2m-r6 | e6-hint2 | 0 | hbm | 1.456 | 275.1 | order |
| e6-2m-r7 | e6-hint2 | 0 | hbm | 1.455 | 284.8 | order |
| e6-4k-r1 | e6-hint2 | 0 | hbm | 1.446 | 299.1 | order |
| e6-4k-r2 | e6-hint2 | 0 | hbm | 1.446 | 299.5 | order |
| e6-4k-r3 | e6-hint2 | 0 | hbm | 1.446 | 299.7 | order |
| e6-4k-r4 | e6-hint2 | 0 | hbm | 1.446 | 299.2 | order |
| e6-4k-r5 | e6-hint2 | 0 | hbm | 1.446 | 299.3 | order |
| e6-4k-r6 | e6-hint2 | 0 | hbm | 1.446 | 299.5 | order |
| e6-4k-r7 | e6-hint2 | 0 | hbm | 1.446 | 300.0 | order |
| e5-4k-r1 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r1 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.057 | 16.4 | order |
| e5-4k-r1 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.054 | 16.4 | order |
| e5-4k-r1 | e5-gs64-r0.05-shared | 0 | ddr | 1.087 | 16.4 | order |
| e5-4k-r1 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r1 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.057 | 17.3 | order |
| e5-4k-r1 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.054 | 17.3 | order |
| e5-4k-r1 | e5-gs32-r0.05-shared | 0 | ddr | 1.088 | 17.3 | order |
| e5-4k-r2 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.060 | 16.4 | order |
| e5-4k-r2 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r2 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.056 | 16.4 | order |
| e5-4k-r2 | e5-gs64-r0.05-shared | 0 | ddr | 1.091 | 16.4 | order |
| e5-4k-r2 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r2 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r2 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r2 | e5-gs32-r0.05-shared | 0 | ddr | 1.091 | 17.3 | order |
| e5-4k-r3 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.060 | 16.4 | order |
| e5-4k-r3 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.058 | 16.4 | order |
| e5-4k-r3 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.055 | 16.4 | order |
| e5-4k-r3 | e5-gs64-r0.05-shared | 0 | ddr | 1.090 | 16.4 | order |
| e5-4k-r3 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r3 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5-4k-r3 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r3 | e5-gs32-r0.05-shared | 0 | ddr | 1.090 | 17.3 | order |
| e5-4k-r4 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.060 | 16.4 | order |
| e5-4k-r4 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r4 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.055 | 16.4 | order |
| e5-4k-r4 | e5-gs64-r0.05-shared | 0 | ddr | 1.091 | 16.4 | order |
| e5-4k-r4 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r4 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5-4k-r4 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r4 | e5-gs32-r0.05-shared | 0 | ddr | 1.091 | 17.3 | order |
| e5-4k-r5 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.061 | 16.4 | order |
| e5-4k-r5 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r5 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.056 | 16.4 | order |
| e5-4k-r5 | e5-gs64-r0.05-shared | 0 | ddr | 1.091 | 16.4 | order |
| e5-4k-r5 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r5 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5-4k-r5 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r5 | e5-gs32-r0.05-shared | 0 | ddr | 1.091 | 17.3 | order |
| e5-4k-r6 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.061 | 16.4 | order |
| e5-4k-r6 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r6 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.056 | 16.4 | order |
| e5-4k-r6 | e5-gs64-r0.05-shared | 0 | ddr | 1.090 | 16.4 | order |
| e5-4k-r6 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r6 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r6 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r6 | e5-gs32-r0.05-shared | 0 | ddr | 1.091 | 17.3 | order |
| e5-4k-r7 | e5-gs64-r0.05-cdw2 | 0 | ddr | 1.060 | 16.4 | order |
| e5-4k-r7 | e5-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5-4k-r7 | e5-gs64-r0.05-cdw4 | 0 | ddr | 1.056 | 16.4 | order |
| e5-4k-r7 | e5-gs64-r0.05-shared | 0 | ddr | 1.091 | 16.4 | order |
| e5-4k-r7 | e5-gs32-r0.05-cdw2 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r7 | e5-gs32-r0.05-cdw3 | 0 | ddr | 1.059 | 17.3 | order |
| e5-4k-r7 | e5-gs32-r0.05-cdw4 | 0 | ddr | 1.056 | 17.3 | order |
| e5-4k-r7 | e5-gs32-r0.05-shared | 0 | ddr | 1.092 | 17.3 | order |
| e5c-4k-r1 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.062 | 17.3 | order |
| e5c-4k-r1 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.057 | 16.4 | order |
| e5c-4k-r2 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.058 | 16.4 | order |
| e5c-4k-r2 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5c-4k-r3 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5c-4k-r3 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.056 | 16.4 | order |
| e5c-4k-r4 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.061 | 17.3 | order |
| e5c-4k-r4 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.057 | 16.4 | order |
| e5c-4k-r5 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.059 | 16.4 | order |
| e5c-4k-r5 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.059 | 17.3 | order |
| e5c-4k-r6 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.058 | 16.4 | order |
| e5c-4k-r6 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| e5c-4k-r7 | e5c-gs64-r0.05-cdw3 | 0 | ddr | 1.061 | 16.4 | order |
| e5c-4k-r7 | e5c-gs32-r0.05-cdw3 | 0 | ddr | 1.058 | 17.3 | order |
| hbmx-2m-r1 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.458 | 276.9 | order |
| hbmx-2m-r2 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.457 | 287.6 | order |
| hbmx-2m-r3 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.457 | 286.9 | order |
| hbmx-2m-r4 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.456 | 289.0 | order |
| hbmx-2m-r5 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.457 | 281.8 | order |
| hbmx-2m-r6 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.457 | 280.5 | order |
| hbmx-2m-r7 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.457 | 279.5 | order |
| hbmx-4k-r1 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.447 | 299.3 | order |
| hbmx-4k-r2 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.448 | 299.3 | order |
| hbmx-4k-r3 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.447 | 298.4 | order |
| hbmx-4k-r4 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.448 | 298.2 | order |
| hbmx-4k-r5 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.448 | 298.5 | order |
| hbmx-4k-r6 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.448 | 299.1 | order |
| hbmx-4k-r7 | hbmx-S1-pf16384-hint2 | 0 | hbm | 1.448 | 297.4 | order |

## Content of the regions: pseudo-random against all-zero data (same reads, 2m pages, per-socket GB/s)

| config | combined GB/s | core GHz | package W | reps |
|---|---|---|---|---|
| fcrandom-ddr56 | 262.3 [262.1-262.6] | 2.28 [2.17-2.36] | 352 | 7 |
| fczero-ddr56 | 262.8 [262.6-262.9] | 2.40 [2.37-2.56] | 354 | 7 |
| fcrandom-gemv32-act52 | 423.3 [411.9-428.8] | 1.92 [1.84-2.03] | 355 | 7 |
| fczero-gemv32-act52 | 444.4 [443.2-450.0] | 1.99 [1.97-2.07] | 353 | 7 |
| fcrandom-gemv32-act56 | 406.0 [404.3-406.9] | 1.99 [1.99-2.02] | 352 | 7 |
| fczero-gemv32-act56 | 419.1 [415.0-422.5] | 2.16 [2.13-2.20] | 351 | 7 |
| fcrandom-gemv64-act52 | 421.8 [410.1-439.1] | 1.88 [1.86-2.17] | 354 | 7 |
| fczero-gemv64-act52 | 444.1 [439.0-452.7] | 2.04 [1.96-2.08] | 351 | 7 |
| fcrandom-gemv64-act56 | 402.3 [400.5-406.8] | 1.97 [1.95-2.04] | 352 | 7 |
| fczero-gemv64-act56 | 416.7 [415.3-423.8] | 2.17 [2.14-2.20] | 353 | 7 |
| fcrandom-hbm56 | 565.8 [559.6-573.6] | 1.64 [1.61-1.73] | 352 | 7 |
| fczero-hbm56 | 600.9 [595.7-609.3] | 1.75 [1.59-1.95] | 351 | 7 |
| fcrandom-inter-1-0 | 565.2 [551.3-572.9] | 1.80 [1.59-2.00] | 348 | 7 |
| fczero-inter-1-0 | 609.0 [604.7-613.1] | 1.84 [1.78-1.91] | 348 | 7 |
| fcrandom-inter-4-1 | 566.2 [556.8-572.0] | 1.69 [1.54-1.82] | 348 | 7 |
| fczero-inter-4-1 | 598.6 [593.4-605.4] | 1.77 [1.60-1.87] | 347 | 7 |
| fcrandom-shared-4-1 | 559.0 [554.4-564.5] | 1.61 [1.53-1.73] | 351 | 7 |
| fczero-shared-4-1 | 598.5 [591.9-612.0] | 1.70 [1.62-1.74] | 351 | 7 |
| fcrandom-split28-28 | 497.1 [494.2-501.4] | 1.32 [1.23-1.45] | 351 | 7 |
| fczero-split28-28 | 519.4 [513.1-524.8] | 1.40 [1.34-1.51] | 351 | 7 |
| fcrandom-split48-8 | 555.5 [550.3-558.5] | 1.47 [1.41-1.61] | 352 | 7 |
| fczero-split48-8 | 584.9 [578.6-591.0] | 1.47 [1.41-1.58] | 352 | 7 |

## 5. Data content (what is and is not recorded)

* Jobs A, B and C (commits f62cad3 and 6809025) have no `--fill` option: the tier regions of E1, E2, E5, E6 and the latency runs were always filled by the socket's own workers with a pseudo-random pattern
  (`mix64(i) & 0x7f7f...`), and the GEMV shard with int8 values in [-127, 127] from a linear congruential generator and scales 0.002 + 0.0001 x ((row + group) mod 7). The interleaved E3/E4 region of job B was only touched (one byte per page), i.e. zero-filled.
* From commit 1e8ba44 on, every process prints an `args` event with its command line, the fill and the shuffle option: all 119 processes of jobs 5924010, 5924011 and 5925935 recorded `fill = random` and a shuffled configuration order. Job D (commit 2f8d329) took `--fill` from the job script (`FILL`, default `random`) for every process, and `fillcmp` overrode it per process (`--fill random` or `--fill zero`). The exported job environment and
  `environment.txt` of job D do not record `FILL`, and the process command line is not in its files: **the fill of the E3, E4 and E5c runs of job D is therefore not recorded in the results.**
  What supports it: the code default (`fill = 0`, random, at that commit and the script default), and the data: the all-HBM point of the rerun E3 (562.1 [555.3-568.5] GB/s, 2 MiB) equals
  `fcrandom-inter-1-0` (565.2 [551.3-572.9]) and differs from `fczero-inter-1-0` (609.0 [604.7-613.1]) and from the zero-filled E3 of job B (602.7 [598.3-633.3]); the E4 HBM-only points (1127.8 [1102.1-1148.8])
  equal twice the random-data E1 rate. Commit 1e8ba44 added the `args` event; the second-round jobs have it.
* The GEMV runs of every job use random weights (zero only in the `fczero-gemv*` points).

## 6. Other checks

* **Which timer.** GB/s in all tables are the sums of per-worker bytes / worker duration (documented in the main document); bytes / (real_end - real_begin) differs by up to 1-3% because it includes the start lead, warm-up and the join.
* **Package power per socket** (counters): single-socket runs: socket 0 348-355 W, socket 1 160-200 W (no workers); E4: socket 0 / socket 1 352-355 W each. DRAM power per socket 22 W (HBM-only) to 74 W (DDR-only).
* **Core clock**: cycles / reference cycles x 1.9 GHz over the interior intervals of the burst, socket 0 (details and the reconciliation with turbostat in the main document, section 6). Turbostat data are in the results directory of the frequency job (`raw/freq.turbostat.txt`,
  every line prefixed by its wall-clock second).
* **UPI**: flits are counted as `unc_upi_rxl_flits.all_data` and `unc_upi_txl_flits.all_data`; the conversion to bytes assumes 64/9 bytes of payload per data flit (not verified on this part); the main document gives flits per second as well.
* **Replication**: node B (job 5916213, file order) E1 points differ from node A by -1.6% to +0.9%; its E5 all-HBM points agree within 2% (405.7 against 401-405 at 2 MiB, 444.1 against 443.6-447.2 at 4 KiB). The shuffled replication (job 5924011) is in the main document, sections 1, 5 and 8; its counters close at 0.998-1.001 (HBM) and 1.00-1.01 (DDR) with the same attribution method.
* **Which timer** (see the main document, section 5.2): worker-clock rates are in all tables; the wall-clock rate uses the same measured bytes and starting point over the wall time to the end of the window. For old results the measured bytes are rate x mean worker duration and the start offset is the warm-up (two warm-up passes for the GEMV).
* **e5alias HBM CAS ratio**: median 0.998 per window and tier. Two configurations read 2-4% fewer bytes from HBM than the GEMV counted: the 64 KiB-skew windows (0.977 gs64, 0.957 gs32), because the shifted ranges of neighbouring workers overlap and the overlap hits in the L3; their bandwidth is therefore slightly overestimated. The 4 KiB-skew windows close at 0.998.
* **NTA windows**: the 1.45 counter ratio of the NTA-hint windows is the same in 26 windows of the base and E6 processes and 14 of the extended sweep, in both page sizes and every repetition and in processes with different neighbouring windows, so it is not a burst mis-attribution.
* **E4 interleaved UPI**: in the 2 MiB runs the median UPI traffic is 0.75 GB/s (node B 0.39 GB/s) against 0.12-0.2 GB/s for the other configurations. The spike (up to 2242 Mflit/s in one interval) sits in the intervals next to the window, where the 32 GiB regions are built and released (that phase has less than 6 GB/s of memory traffic and is not part of the window's burst), but some repetitions still contain it in the interior; this is unresolved. 11 of 14 E4 4 KiB interleaved windows (node A) have no counters because the window's burst merged with the allocation burst.
* **Alignment test** (`e5off`): 2 MiB medians span 0.4%, 4 KiB medians 2.2% (gs64) and 1.8% (gs32) with the 4 KiB offset lowest; ranges overlap.

## 7. Reproduction

`ROOT=<staged tree> OUT=<new dir> PYTHON=<python> STAGES="smoke base e1 ..." sbatch scripts/concurrent-tier.sbatch` (see the header of the script; `DRY_RUN=1` prints the commands). Summaries:
`python3 tools/concurrent_tier.py summarize DIR [DIR...] [--zero-fill DIR] -o OUT`; replication: `tools/concurrent_tier.py replicate-check DIR_A DIR_B`.
