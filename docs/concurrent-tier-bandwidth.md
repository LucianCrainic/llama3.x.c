# Concurrent HBM and DDR reads on one socket: does combined bandwidth exceed HBM-only?

Measurements on CRESCO8 Flat/Quadrant nodes (dual Xeon Max 9480; per socket 56 cores, 64 GB HBM = NUMA nodes 2 and 3,
about 512 GB DDR5 = NUMA nodes 0 and 1). Program: `tests/stream_bench.c` in `--exp` mode; job script
`scripts/concurrent-tier.sbatch`; analysis, tables and figure `tools/concurrent_tier.py`. Evidence, job ids and checks are in
[concurrent-tier-bandwidth-validation.md](concurrent-tier-bandwidth-validation.md). GB/s means 1e9 bytes per second. Every
number is labelled **measured**, **derived** (arithmetic on measured numbers), **modelled** or **estimated**. "Median [min-max]" is over
7 repetitions (the base sweep uses 3). The reference for every gain is the best HBM-only measurement made with the same kind of data
(pseudo-random bytes).

## 1. Answer

Scope: core-issued loads (ordinary and prefetched AVX-512 loads) on this node, whose sockets run at a package power limit of about 350 W; nothing here says what a DMA engine or a node with a higher power limit would do.

1. **Reading both tiers at once did not raise the combined bandwidth above HBM-only by more than about 1-3%, and not robustly.** The best HBM-only figure is **569.1 [554.2-571.2] GB/s per socket** (measured, 2 MiB pages, 56 cores, one stream per core, 16 KiB software prefetch; 7 repetitions;
   no variant of prefetch distance, hint or non-temporal loads beat it in the extended sweep, section 3). Mixed configurations (per socket, 2 MiB pages unless stated; p = one-sided exact permutation p-value that the median is above e1-cd0 by chance, shown only when the median is higher):
   * E1 split teams: best 562.4 [553.9-568.0] (54 HBM + 2 DDR cores) = 1.2% below; no split is above it; combined falls to 501 at 28+28.
   * E2 shared cores: best 561.2 [555.0-568.5] (line interleave 4:1) = 1.4% below.
   * E3 interleaved pages: 8:1 573.4 [567.0-577.4] (+0.8%, p = 0.029), 4 KiB 4:1 570.1 [565.9-581.2] (+2.1% over the 4 KiB HBM-only 558.5, p = 0.006); its own all-HBM region reads 562.1.
   * E4 both sockets: node 1167.8 [1113.1-1180.2] for interleaved 8:1 against 1127.8 [1102.1-1148.8] HBM-only (+3.5%, p = 0.029); at 4 KiB 1130.4 against 1120.3 (p = 0.244).
   These are small, same-sign excesses in the interleaved-page configurations (E3, E4) that do not appear in the split-team or shared-core configurations, whose ranges overlap the HBM-only range; with the run-to-run spread of the reference at about 3% and several dozen comparisons they are **not robust**.
   The shuffled replication on a second node is in section 8.
2. **The engine's own decode GEMV (an upper bound for the weight-stream part of decode, section 5) gains 1.4-2.4%.** Over the 70B per-socket shard (gs64), the best all-HBM point (4 KiB pages, 56 workers) reads 447.2 [440.6-451.7] and, in a second independent run, 459.0 [449.7-463.3] GB/s; the best row split (r = 0.2, 14 DDR workers, 4 KiB pages) reads
   453.4 [442.5-459.8] (+1.4%, p = 0.017) and 469.8 [460.4-471.8] (+2.3%, p = 0.006) GB/s: 12.44 to 12.73 decode tok/s-equivalent (gs32: +2.1 to +2.4%). It is a small gain that needs the right worker split and is a fraction of a tok/s.
3. **Why** (section 6): the socket that runs the workers sits at its power limit in every run (348-355 W, measured, including DDR-only; the idle other socket draws 160-200 W; both sockets draw 352-355 W in E4), and
   the average core clock falls from 2.28 GHz (DDR only) to 1.4-1.7 GHz as soon as cores read HBM while the uncore clock rises with the DDR
   share (1.7 to 2.1 GHz). DDR itself saturates at 28 cores (266 GB/s) and adds latency (up to 366 ns under load). The data are consistent with a
   power-limited rather than a controller-limited socket; they do not prove it, because no frequency or power knob could be set, and the clock difference between
   data contents is not resolved (section 6).
4. **Side findings that matter for bandwidth work:** (a) all-zero memory reads 6% faster than pseudo-random data under the same power cap (565.8 against 600.9 GB/s HBM-only);
   (b) the engine's Q8 GEMV on 2 MiB pages loses 8-10% at 56 and 28 workers to a stripe-aliasing effect that a 4 KiB stagger of the workers' addresses removes (411 to 447.5 GB/s, and a raw 56-stream lockstep read 403 against 554-577 GB/s): a hypothesis the tests support (section 5.1); (c) the NTA prefetch hint halves HBM bandwidth.

![combined, HBM and DDR bandwidth against DDR share, and the E5 GEMV](figures/concurrent-tier-bandwidth.png)

Figure (measured, medians with min-max bands): E1 against DDR cores, E2 and E3 against the DDR share of the bytes, E5 as
tok/s-equivalent against the DDR row fraction r with the best all-HBM level of each group size. The black line is the best HBM-only
reference. Page-interleaved (E3) HBM and DDR parts are the combined rate split by the share of the counter bytes seen on each tier.

## 2. Method

* **Windows.** Every measurement is a window in which all workers of a socket start at a shared absolute time, run 0.2 s unmeasured,
  then 1.0 s measured (the base sweep 0.15 + 0.5 s). Each worker counts the bytes it read and its own duration; a tier's rate is the
  sum over its workers of bytes / duration, so for the timed windows both groups are active for the whole measured period and the combined
  rate is the whole-window rate. For a fixed-volume window (`e1v`) every worker reads the same volume (6 GiB); then the
  total volume / total time (last finisher) and the rate over the interval in which all workers were still running are reported
  separately. GEMV windows are 2 warm-up and 16 measured passes over 40 layers (18.2 GB at gs64, 19.3 GB at gs32 per tier copy).
* **Memory.** (The content of every region is pseudo-random unless stated; for the last job this is the code default and the job script default, see the validation document, section 5, for what is and is not recorded.) Each socket's HBM region and DDR region are 20 GiB (above the 16 GiB floor), bound to the local nodes before first
  touch and filled with pseudo-random bytes by the socket's own workers. Two page sizes: 2 MiB (THP, madvise) and 4 KiB (THP
  disabled for the region). Every page was verified with `move_pages` at setup and again at the end (5.1e9 page queries, all on the intended node;
  validation document). E3 regions (32 GiB) are built page by page with `set_mempolicy` and verified page by page; the kernel (5.14) has
  no weighted interleave.
* **Read kernel.** AVX-512 64-byte loads, `S` lockstep streams per core, software prefetch distance, hint and optional non-temporal loads,
  all swept in the base stage. The GEMV is the engine's `k_matmul_q8_ld_worker` (VNNI W8A8), called by each worker for its share of rows
  of its tier's copy, with the engine's phase structure (qkv | wo | w1,w3 | w2) and a team barrier after each. Weights are int8 in
  [-127, 127] from a linear congruential generator, scales 0.002 + 0.0001 x ((row + group) mod 7).
* **Which clock defines GB/s.** A worker's rate is its bytes counted between its first block boundary after the warm-up and its stop, divided by its own
  monotonic-clock duration; tier and combined rates are sums of these (a GEMV rate is bytes per pass divided by the mean measured pass time of the team leader). This is not the same as total bytes
  divided by the wall time between the window's start and end events, which includes the 0.08 s start lead, the 0.2 s warm-up and the final join: for e4-hbm56 the two are 1127.8 and 1137.9 GB/s (0.9%), and
  other windows differ by up to 1-3%. All tables use the worker-clock rates.
* **Counters.** `perf stat -a --per-socket -I 100` ran during every process: DDR CAS reads (x 64 B), HBM CAS reads (x 32 B), UPI
  data flits, RAPL package and DRAM energy, cycles and reference cycles. Each window is a burst of traffic between idle gaps; the
  measured bytes of a window are the counter bytes of its burst, to be compared with the bytes the workers issued (validation document).
* **Order and interleaving.** Repetitions loop outside the process and the two page sizes alternate, so page sizes and repetitions
  are interleaved in time. Inside one process the configurations ran in fixed file order, except the final job (E3, E4, the data
  comparison and the E5 confirmation), which shuffled the order inside each process.
* **Nodes.** All numbers above were measured on one node (node A); the key E1 and E5 points were repeated on a second Flat node
  (node B) and agree within 1.6% (section 8).

## 3. Reference: best HBM-only and DDR-only

Base sweep (3 repetitions, 214 configurations, both page sizes): cores x streams x prefetch distance x hint x non-temporal.

| quantity (measured, per socket) | median [min-max] GB/s |
|---|---|
| best HBM-only, 56 cores, S=1, pf 16 KiB, 2 MiB pages (base, 3 reps) | 575.5 [563.7-579.9] |
| same, 7 reps in E1 | 569.1 [554.2-571.2] |
| same, 4 KiB pages, 7 reps | 558.5 [547.8-570.0] |
| HBM-only at the same settings, E6 (7 reps) | 565.1 [557.4-586.7] |
| best DDR-only, 28 cores, no prefetch | 266.4 [265.3-268.1] (base); 265.8 [264.8-266.4] (E1) |
| DDR-only, 56 cores | 262.5 [262.1-262.8] |

HBM-only against cores (E1 reference points, 2 MiB pages, measured): 16 cores 219.6, 24 312.5, 28 354.0, 32 395.9, 40 460.3, 44 489.4,
48 514.7, 52 542.6, 56 569.1 GB/s. Per core (derived) that is 13.7 GB/s at 16 cores falling to 10.2 GB/s at 56, and the curve is still
rising at 56 cores. DDR-only: 16 cores 202.7, 24 249.6, 28 265.8, 32 263.2, 40 262.0 (derived per core 12.7 at 16, 9.5 at 28): DDR saturates at
about 28 cores, as the earlier controller measurements say.

E6 sensitivity of the HBM-only rate (2 MiB pages, 7 reps; measured): one stream per core is best (563 against 542 with 4 streams and
515 with 8); a longer prefetch is slightly better (16 KiB 563-566, 8 KiB 566, none 542-550); non-temporal loads 568 against 560, no effect; the T1 hint loses 6%
(531) and the NTA hint halves the rate (231 GB/s, and the counters show 1.45 times the bytes the program asked for, so the hardware fetches each line
twice). Spreading the DDR cores across the team (`alt`) instead of a block makes no difference. Hardware prefetchers and the uncore
frequency cannot be changed with this account (sudo is limited to `perf stat`, `turbostat` and `ipmitool sensor`; no MSR tools;
uncore limits not writable): not tested.

**Extended HBM-only sweep and E6 at real splits** (shuffled order inside each process, 7 reps, per socket). Software prefetch distance 32, 64 and 128 KiB with one stream: 526.7 [519.0-564.6], 532.4 [528.9-545.9], 529.5 [526.3-554.2]
(2 MiB), worse than 16 KiB (561.7 [528.6-567.1] in that run); the T1 hint 533.3, the NTA hint 238.2 (the counters read 1.45 times the requested bytes in every one of these windows, in both page sizes and every repetition: the hardware fetches each line twice; it is not an attribution artefact);
non-temporal loads 567.3 [562.0-582.6] (4 KiB 560.1). No variant exceeds the e1-cd0 median of 569.1, so it remains the reference. At real splits (cd = 4, 8, 16; 2 MiB pages) spreading the DDR cores over the team instead of giving them a contiguous block gains 1-2%
(cd 4: 566.2 [550.9-576.1] against 556.0 [550.7-558.6]; cd 8: 563.5 [558.6-569.5] against 558.0 [554.7-563.7]; cd 16: 544.2 against 542.2) and stays at or below the HBM-only reference; the DDR group's streams (1, 2, 4), prefetch distance (0, 4 KiB, 16 KiB) and non-temporal loads change nothing (all within 1%);
a 64 KiB prefetch on the HBM group costs 5% (cd 8: 531.0 against 558.0); 4 KiB pages behave the same.

## 4. Results of the mixing experiments (per socket, measured)

### E1 split teams (c_h cores read HBM, c_d cores read DDR, c_h + c_d = 56)

| HBM + DDR cores | HBM GB/s | DDR GB/s | combined 2 MiB | combined 4 KiB | fixed-volume run: total volume / total time (2 MiB) | core GHz | package W (socket 0) |
|---|---|---|---|---|---|---|---|
| 56 + 0 | 569.1 | 0 | 569.1 [554.2-571.2] | 558.5 [547.8-570.0] | 549.1 [524.8-564.3] | 1.68 | 354 |
| 54 + 2 | 542.1 | 20.3 | 562.4 [553.9-568.0] | 557.5 [544.6-562.3] | 550.6 [528.7-556.6] | 1.62 | 353 |
| 52 + 4 | 518.7 | 37.0 | 555.9 [551.8-557.5] | 546.5 [542.2-551.0] | 535.9 [514.8-543.0] | 1.44 | 353 |
| 48 + 8 | 479.3 | 75.5 | 554.8 [548.2-556.8] | 550.8 [545.5-554.6] | 515.7 [488.6-523.7] | 1.42 | 351 |
| 44 + 12 | 442.3 | 95.4 | 537.1 [530.9-543.9] | 529.4 [526.3-534.8] | 453.4 [439.7-464.4] | 1.43 | 354 |
| 40 + 16 | 407.6 | 135.5 | 542.2 [535.9-552.4] | 527.9 [524.4-538.6] | 492.1 [469.6-495.4] | 1.41 | 352 |
| 36 + 20 | 363.5 | 156.9 | 521.2 [513.6-529.1] | 510.8 [504.4-514.1] | 461.4 [451.9-464.9] | 1.36 | 353 |
| 32 + 24 | 301.1 | 180.4 | 483.4 [442.7-491.4] | 477.8 [467.5-487.2] | 440.5 [437.3-443.5] | 1.21 | 353 |
| 28 + 28 | 283.3 | 219.7 | 501.1 [496.9-505.4] | 488.4 [485.0-494.8] | 455.0 [450.0-457.4] | 1.36 | 353 |
| 24 + 32 | 210.7 | 224.3 | 434.7 [425.9-442.2] | 432.3 [430.1-435.5] | 407.7 [400.6-414.2] | 1.17 | 352 |
| 16 + 40 | 123.7 | 251.4 | 375.1 [372.8-381.2] | 376.2 [374.3-379.6] | 355.3 [354.1-359.1] | 1.75 | 352 |
| 0 + 56 | 0 | 262.5 | 262.5 [262.1-262.8] | 261.7 [260.9-261.8] | 260.8 [260.2-261.2] | 2.28 | 352 |

HBM and DDR columns are 2 MiB pages. Core GHz is the median over repetitions of cycles / reference cycles x 1.9 GHz over the interior
intervals of each window, all 56 cores of the socket (including idle spinning workers), so it is an average over cores, not the clock of one core.
Best split: 56 + 0, i.e. no DDR at all; the best window with DDR is 54 + 2, 1.2% below the HBM-only median (ranges overlap). The
fixed-volume column is lower than the fixed-time column wherever the two groups finish at different times (28 + 28: 455 against 501 GB/s);
during the interval in which both were running the groups read at the fixed-time rates (for 28+28, 287.9 HBM and 223.2 DDR GB/s in the
common interval of the volume run). The doc's reported combined value is the fixed-time one.

### E2 shared cores (all 56 cores read both tiers in a fixed byte ratio)

| HBM:DDR bytes | DDR share | combined, line interleave | combined, chunks of 128 lines (8 KiB) | HBM part / DDR part (chunks) |
|---|---|---|---|---|
| 1:0 | 0 | 544.9 [543.5-547.9] | 553.9 [545.7-576.3] | 553.9 / 0 |
| 8:1 | 0.11 | 558.2 [553.7-566.1] | 561.8 [555.5-563.9] | 499.4 / 62.4 |
| 4:1 | 0.20 | 561.2 [555.0-568.5] | 557.2 [555.5-560.5] | 445.7 / 111.4 |
| 3:1 | 0.25 | 560.1 [555.9-565.4] | 556.6 [554.1-569.0] | 417.5 / 139.2 |
| 2:1 | 0.33 | 534.5 [532.7-539.4] | 513.4 [508.1-519.2] | 342.2 / 171.1 |
| 3:2 | 0.40 | 497.6 [495.8-502.2] | 478.9 [473.6-484.3] | 287.3 / 191.5 |
| 1:1 | 0.50 | 444.3 [440.1-447.2] | 435.8 [431.8-439.1] | 217.9 / 217.9 |
| 1:2 | 0.67 | 368.0 [365.8-370.2] | 367.4 [361.6-369.1] | 122.5 / 244.9 |
| 0:1 | 1 | 264.3 [264.0-264.4] | 262.2 [261.9-262.4] | 0 / 262.2 |

Chunk sizes of 8, 32 and 512 lines give the same combined rate at 3:1 (544.9, 551.5, 554.1) and 1:1 (440.9, 439.4, 433.7). 4 KiB pages: maximum 559.0
(4:1 line). The HBM-only reference is exceeded by none (561.2 against 569.1 [554.2-571.2]).

### E3 page-interleaved region (one 32 GiB region, pages placed HBM:DDR; all cores read it uniformly; pseudo-random data)

| HBM:DDR pages | page share on DDR | DDR share of the counter bytes | combined 2 MiB | combined 4 KiB |
|---|---|---|---|---|
| 1:0 | 0 | 0.002 | 562.1 [555.3-568.5] | 561.0 [551.2-566.2] |
| 8:1 | 0.111 | 0.113 | 573.4 [567.0-577.4] | 565.1 [556.1-571.6] |
| 4:1 | 0.200 | 0.202 | 565.5 [561.2-570.2] | 570.1 [565.9-581.2] |
| 3:1 | 0.250 | 0.252 | 547.3 [543.4-550.7] | 568.6 [566.2-574.3] |
| 2:1 | 0.333 | 0.335 | 512.1 [506.1-521.7] | 539.1 [536.2-543.4] |
| 1:1 | 0.500 | 0.501 | 434.7 [430.7-440.6] | 451.8 [448.6-454.9] |
| 0:1 | 1 | 0.999 | 261.7 [261.6-262.1] | 260.9 [260.5-261.0] |

The counters confirm the placement (the DDR share of the bytes equals the page share). The best point (8:1, 2 MiB: 573.4 [567.0-577.4]) is 0.7% above the E1
HBM-only median and 2% above this region's own all-HBM point; its range overlaps the HBM-only range (554-571), so it is not
a demonstrated gain. The first E3 run (job B) used a region whose pages were touched but never written (all zeros): it read
602.7 GB/s all-HBM and 611.7 at 8:1; those numbers are **not comparable** with the E1 reference and are only used in section 7.

### E4 both sockets at once (each socket reads only its own HBM and DDR; node totals; 7 reps)

| config | 2 MiB node GB/s | 4 KiB node GB/s |
|---|---|---|
| HBM-only, 56 + 56 cores | 1127.8 [1102.1-1148.8] | 1120.3 [1096.4-1140.3] |
| best E1 split (the E1 optimum is 56+0, so identical) | 1141.4 [1120.8-1234.4] | 1119.4 [1106.8-1128.9] |
| shared 8:1 | 1131.7 [1114.5-1146.1] | 1119.1 [1093.2-1135.7] |
| interleaved pages 8:1 | 1167.8 [1113.1-1180.2] | 1130.4 [1108.2-1151.6] |
| DDR-only | 524.8 [524.5-525.6] | 523.8 [522.9-524.3] |

Per node the best HBM-only is 1127.8 GB/s (2 x 563.9, derived from two sockets running together: 0.99 of 2 x 569.1), and the best mixed point 1167.8
(+3.5%, ranges overlapping). UPI traffic is 17-29 Mflit/s of data flits received plus transmitted (socket 0, per second of burst; with the assumed 64/9 bytes of payload per data flit that is 0.12-0.21 GB/s, an estimate: the
flit size is recalled from Intel's tools, not verified on this part) for HBM-only, DDR-only and shared runs, and 220 Mflit/s (1.6 GB/s, 0.13% of the node's memory traffic) in the interleaved 2 MiB runs: essentially zero cross-socket traffic. Both sockets draw 352-355 W package power in E4.

## 5. E5: the engine's Q8 decode GEMV over the 70B per-socket shard

**Scope.** This is the weight-stream part of a decode step only: the engine's `k_matmul_q8_ld_worker` on the seven matrices of a layer (W1 and W3 as two separate GEMVs, not the fused
SwiGLU pair), with a team barrier after each phase, over 40 layers per tier copy. There is no attention, KV cache, classifier, norm or cross-socket reduction. Every tok/s-equivalent below is therefore an **upper bound**
on what a decode token can gain, not a decode speed.

Shard (per socket, 2-socket tensor parallel): Wq 4096x8192, Wk and Wv 512x8192, Wo 8192x4096 (column shard), W1 and W3 14336x8192, W2 8192x14336, 40 layers per tier copy (18.2 GB gs64,
19.3 GB gs32). Rows of every matrix are split between an HBM copy and a DDR copy in the ratio r (DDR fraction); a group of workers streams each part. tok/s-equivalent (derived) = GB/s / 36.9 GB (gs64) or / 39.1 GB (gs32) per socket per token.

### 5.1 All-HBM GEMV and the 2 MiB page effect

All-HBM against workers (measured, 7 reps, gs64, GB/s): 14 workers 157, 28 272-277, 32 317, 40 363, 42 367, 48 398-403, 52 418-428, 56 401-411 with 2 MiB pages; with 4 KiB pages 326, 373, 405, 433, 444-459. With 4 KiB pages the rate rises
to 56 workers; with 2 MiB pages it **drops** at 56 workers (and at 28) by 8-10% (411 against 444-447 at 52-55 workers). The kernel reaches 11.2 GB/s per worker at 14 workers and 7.3-8.2 at 56 (derived), i.e. 72-80% of the stream rate, at 350-353 W
and 1.86-1.99 GHz (counters), while the stream at 52-56 cores runs at 1.4-1.7 GHz.

**Stripe aliasing (a hypothesis that the data support).** With 56 (or 28) workers, W1 and W3 give every worker exactly 256 (or 512) rows of 8192 bytes = 2 MiB (4 MiB), so all workers read at addresses that differ by multiples of 2 MiB at the same time, and with
2 MiB pages the physical addresses are as contiguous as the virtual ones. Tests (2 MiB pages, all-HBM, shuffled order, 7 reps, GB/s):

| test | gs64 | gs32 |
|---|---|---|
| 53 / 54 / 55 workers | 436.7 / 443.9 / 446.4 | 438.2 / 442.5 / 447.4 |
| 56 workers | **411.1 [410.2-415.0]** | **414.3 [413.1-416.9]** |
| 56 workers, worker i reads 4 KiB x i higher | **447.5 [443.1-452.3]** | **443.1 [439.8-450.5]** |
| 56 workers, 64 KiB x i higher | 441.8 [441.2-447.9] | 437.4 [432.1-440.9] |
| 27 / 28 / 29 workers | 292.9 / **277.3** / 309.6 | 278.4 / **272.4** / 295.5 |
| 56 workers, 4 KiB pages (no skew) | 456.7 [451.7-462.6] | 454.3 [452.4-462.1] |

The dip at 56 and 28 workers disappears when the workers' addresses are staggered by 4 KiB, and does not occur with 4 KiB pages. A raw read confirms it with no arithmetic: 56 workers each stream their own 360 MiB stripe (a multiple of 2 MiB) in barrier lockstep, one 1 MiB block per barrier:
stripes aligned (all start at the stripe base) **403.4 [397.4-411.1] GB/s**, started 4 KiB x i in **553.6 [518.2-562.2]**, started (1 MiB + 4 KiB) x i in **576.7 [541.8-586.6]**; without barriers (time-based) the aligned case reads 559.2 [546.3-567.9]; sliced evenly like E1 554.8. With 4 KiB pages the aligned lockstep case is 543.6 [537.4-575.6].
So 2 MiB-aligned simultaneous streams lose 27-30% in the raw read and 8-9% in the GEMV, and only when the workers advance in lockstep on 2 MiB pages. The mechanism (the address interleave across HBM channels and banks) was not examined; this is an inference from where the loss appears and disappears.
The earlier test that shifted all workers by the same offset did not change this (it shifted every address equally, which keeps the 2 MiB relations between workers). **This is a finding beyond this study: an engine that keeps its weights on 2 MiB pages and runs 56 (or 28) workers over such shapes loses about 8-10% of the weight-stream bandwidth, and a per-worker stagger of 4 KiB recovers it** (estimate: not tried in the engine).

### 5.2 Row split against the best all-HBM point

The reference is the best all-HBM point over both page sizes and all worker counts in the same shuffled run (4 KiB pages, 56 workers). Two independent shuffled runs on node A (the first scanned r and the DDR worker count; the second repeated the neighbourhood of the best point, DDR worker count +-2):

| gs64, 4 KiB pages | run 1 GB/s | vs all-HBM (p) | run 2 GB/s | vs all-HBM (p) |
|---|---|---|---|---|
| all-HBM, 56 workers | 447.2 [440.6-451.7] | 0 | 459.0 [449.7-463.3] | 0 |
| r = 0.20, 12 DDR workers | - | - | 436.6 [434.8-440.3] | -4.9% |
| r = 0.20, 14 DDR workers | 453.4 [442.5-459.8] | +1.4% (0.017) | **469.8 [460.4-471.8]** (12.73 tok/s-eq.) | **+2.3% (0.006)** |
| r = 0.20, 16 DDR workers | - | - | 446.1 [438.7-449.1] | -2.8% |
| r = 0.25, 16 DDR workers | - | - | 464.0 [456.0-476.6] | +1.1% (0.047) |
| r = 0.25, 18 DDR workers | 453.4 [442.7-456.5] | +1.4% (0.017) | 460.8 [458.5-472.1] | +0.4% (0.238) |
| r = 0.30, 21 DDR workers | 446.7 [443.0-458.1] | -0.1% | 455.3 [447.7-464.6] | -0.8% |

gs32: all-HBM 4 KiB 445.6 [442.3-446.7] (run 1) and 455.3 [452.7-459.0] (run 2); best split run 1 449.0 (+0.7%); run 2 r = 0.20 with 14 DDR workers 465.0 [459.4-468.5] (+2.1%, p = 0.006) and r = 0.25 with 16 DDR workers 466.1 [462.0-469.7] (+2.4%, p = 0.006). With 2 MiB pages the best split (456.8-461.2) is within +-1.3% of the best all-HBM point, but 11% above the 2 MiB all-HBM point at 56 workers
(410.2), which the aliasing explains. p is the one-sided exact permutation p-value of the difference of the medians over the 7 + 7 repetitions; the smallest value this test can give is 0.0058. Fewer than 14 or more than 16-18 DDR workers loses (r = 0.2 with 12 workers -4.9%, 16 workers -2.8%): the gain exists only when the DDR group's finishing time matches the HBM group's.

[[F5]]

**Reading.** A row split at r = 0.2 - 0.25 with 14-18 DDR workers gains **+1.4% to +2.4%** of weight-stream bandwidth at 4 KiB pages in two independent runs (p = 0.017 and 0.006 for the pre-chosen point), as an upper bound on the gain for a decode token (12.44 to 12.73 tok/s-equivalent in run 2; 0.29 tok/s). It is small, depends on the worker split
(a window of 2-4 workers), is smaller or absent for other points of the scan (many same-sign points have p between 0.006 and 0.4; with about 40 comparisons per table, the smallest uncorrected p of 0.006 would be about 0.2 after a Bonferroni correction, so only the replication of the pre-chosen point carries evidence), and sits on a baseline that
the 2 MiB aliasing lowers by 8-10% on the page size the engine actually uses. Not robust enough to call a result of this study beyond "no gain larger than about 2-3%".

All-DDR reads 250-253 GB/s (6.8 tok/s-eq.). Rows split in the E2 style (every worker reads both parts) is 2-7% worse than the best worker split.

## 6. Why the curve has this shape

Everything below is supported by counters measured in the same windows.

1. **The active socket is at its power limit in every window.** RAPL package power of the socket that runs the workers is 348-355 W (measured; the part's TDP is 350 W) in
   HBM-only, every E1 split, DDR-only and the GEMV runs, with DRAM power rising from 22 W (HBM-only) to 74 W (DDR-only). In the single-socket runs the other socket has no workers
   and draws 160-200 W (measured; node about 530 W), so it is not at its limit. In E4, where both sockets run, both draw 352-355 W. Turbostat on the long windows gives 349 W.
2. **The core clock falls when HBM is read, and the uncore clock rises with the DDR share.** Method (exact): the per-socket perf counters `cycles` and `ref-cycles`
   (fixed counters, 100% running, no multiplexing) of socket 0 summed over the interior intervals of the window's burst (all 0.1 s intervals except the first and the last; this
   includes the 0.2 s warm-up and the measured second); clock = cycles / ref-cycles x 1.9 GHz, which equals cycles / (56 x time) because ref-cycles / (56 x time) is 1.895 GHz. All 56 cores of
   the socket count, spinning idle workers included (they never halt), so this is an average over cores. The edge intervals are left out because there the spinning workers run at other
   clocks: an average over the whole burst gives higher values (a check over whole bursts gives about 2.1 GHz for HBM-only random data where the interior gives 1.4-1.7).
   Turbostat (job with long windows, same configurations, `Bzy_MHz` of cpu 0 and `UncMHz` of package 0, median of the seconds inside the window) agrees with the counters:
   HBM-only 1.69 GHz (counters 1.68), 48+8 1.47 (1.42), 28+28 1.36 (1.36), DDR-only 2.19 (2.28), differences up to 0.09 GHz (4%). The uncore clock is 1.7 GHz [1.5-1.9] HBM-only,
   1.9 [1.3-2.0] at 48+8, 2.0 [1.9-2.2] at 28+28 and 2.1 [2.0-2.3] DDR-only. The drop is large (2.28 to 1.4-1.7 GHz, 25-40%) but it is one window averaged over 56 cores and it moves from window to window by up to
   0.3 GHz depending on what ran before it, so it is an association with the traffic mix, not a demonstrated cause of the lower bandwidth.
3. **Data content changes bandwidth at the same package power; the clock does not explain it.** Over all-zero regions the HBM-only rate is 600.9 [595.7-609.3] against 565.8 [559.6-573.6] GB/s
   over pseudo-random regions (+6.2%, non-overlapping ranges), at 351 and 352 W; the 4:1 shared point 598.5 against 559.0, the 48+8 split 584.9 against 555.5, the all-HBM GEMV about 5%
   (52 workers: 444.1 against 421.8). The medians of the core clock are 1.75 (zero) against 1.64 GHz (random), +6.7%, but the per-repetition values overlap (random 1.41-1.73, zero 1.58-1.90 GHz;
   shuffled order), so the data do not show that the clock carries the effect. What they show is that a content-dependent quantity changes the bandwidth of a part
   whose package power is pinned at its limit, which a purely controller-bound ceiling would not do; the most plausible reading is memory-interface power, an estimate.
4. **HBM is not saturated at 56 cores and is not limited by the controllers.** HBM-only still gains 5% from 52 to 56 cores (542.6 to 569.1) with
   per-core rates falling from 13.7 to 10.2 GB/s: a mix of per-core request concurrency and the clock. Loaded latency (one core chasing pointers while the others stream,
   measured): HBM 156 ns idle, 206 ns with 55 cores streaming HBM; DDR 132 ns idle, 160 ns with 55 cores streaming HBM, 366 ns with 55 cores streaming DDR (DDR saturated),
   241-271 ns with 8-28 DDR streamers. A DDR access costs more queueing than an HBM one and DDR gives at most 262-266 GB/s.
5. **Mixed configurations lose on both sides.** At 48 + 8, the 8 DDR cores read 75.5 GB/s (9.4 GB/s per core) and the 48 HBM cores 479.3 GB/s (10.0 per core), against 12.7 GB/s per core for
   DDR at 16 cores alone and 10.2 per core for 56 HBM cores; the combined rate is the sum of two reduced streams. Fixed-volume runs show the second effect: the group with fewer cores
   per byte finishes last, so total volume / total time falls up to 16% below the fixed-time rate (44+12: 453 against 537 GB/s; 28+28: 455 against 501).
6. **The GEMV is limited by something else as well, and gains little.** It uses 72-80% of the stream rate with 56 workers at 1.9-2.0 GHz, and with 2 MiB pages it loses another 8-10% to the stripe aliasing of section 5.1. DDR workers added to the HBM workers
   draw their share of the same package budget; the best row split is 1.4-2.4% above the best all-HBM point (4 KiB pages).

Whether the limit is power or bandwidth: the constant package power, the lower clocks under HBM traffic, the rising uncore clock with the DDR share and the data-content effect are all
consistent with a package power budget being the binding quantity; the DDR ceiling (262-266 GB/s at 28 cores) bounds the DDR side only. This is an interpretation: no
counterfactual (the same run at a fixed higher clock or a higher power limit) is possible with this account, so a controller or mesh limit that happens to move with data
content is not excluded.

## 7. Modelled 70B decode speed implied (per socket, 36.9 GB per token at gs64)

From the E1 per-tier rates measured under concurrency (random data). t_concurrent = max(B_hbm / BW_hbm(c_h), B_ddr / BW_ddr(c_d)) with the byte
fraction f = BW_ddr / (BW_hbm + BW_ddr) at which both groups finish together; t_serial = B_hbm / BW_hbm(56 cores, HBM-only) + B_ddr / BW_ddr(56 cores, DDR-only); all-HBM is B / 570.6 GB/s (modelled; the engine's kernel reaches 447 GB/s, i.e. 12.1 tok/s, in E5).

| HBM + DDR cores | DDR byte fraction f | tok/s concurrent | tok/s serial tiers | tok/s all-HBM |
|---|---|---|---|---|
| 54 + 2 | 0.036 | 15.24 | 14.84 | 15.46 |
| 52 + 4 | 0.067 | 15.06 | 14.36 | 15.46 |
| 48 + 8 | 0.136 | 15.03 | 13.36 | 15.46 |
| 44 + 12 | 0.177 | 14.57 | 12.83 | 15.46 |
| 40 + 16 | 0.250 | 14.72 | 12.00 | 15.46 |
| 36 + 20 | 0.301 | 14.10 | 11.47 | 15.46 |
| 28 + 28 | 0.437 | 13.63 | 10.27 | 15.46 |
| 16 + 40 | 0.670 | 10.17 | 8.71 | 15.46 |

Concurrent reading is always better than reading the tiers one after the other (the current hybrid placement keeps whole layers in one tier, so it reads them
serially): +2.7% at 54+2 up to +33% at 28+28 (modelled). It is always worse than all-HBM, so it is only useful when part of a model does not fit in HBM: if a
fraction f = 0.25 of the bytes must live in DDR, the concurrent model gives 14.72 tok/s against 12.00 tok/s when the tiers are read one after the other (modelled, both
for gs64 and bytes per token 36.9 GB; the engine's real kernel reaches about 79% of these rates, 12.1 tok/s all-HBM).

## 8. Replication and energy

* Replication on a second Flat node (node B, 7 reps): E1 cd 0, 4, 8, 16, 28, 56 and the fixed-volume point at both page sizes differ from node A by -1.6% to +0.9%
  (for example 56+0: 566.5 against 569.1; 28+28: 493.2 against 501.1; DDR only 262.4 against 262.5). E5 on node B (file order, 7 reps, gs64): all-HBM with 56 workers is again 405.7 [402.7-408.8] GB/s at 2 MiB and 444.1 [437.0-453.1] at 4 KiB; the best split is r = 0.1 with 6 DDR workers, 445.5 [437.5-448.3] (2 MiB) and 453.1 [451.9-457.8] (4 KiB, +2.0%); gs32 gives 455.1 [449.8-456.4] against 444.9 [442.5-452.9] (+2.3%). On this node the split is therefore 2% ahead of the best all-HBM point with ranges that touch; on node A (shuffled run) it is +0.5% (r = 0.1) to +1.4% (r = 0.2). The gain is below the 3% run-to-run spread and is not demonstrated.
* Energy per byte (socket 0, measured RAPL package + DRAM over the interior intervals of the window): HBM-only 569 GB/s at 354 + 22 W = 0.66 J/GB; DDR-only 263 GB/s at 352 + 74 W = 1.62 J/GB; 48+8 0.71 J/GB; 28+28 0.84 J/GB (derived; J per GB read, package + DRAM of one socket, not
  corrected for the 348 W and 22 W that socket draws with the worker teams spinning and no traffic, so the figures are dominated by the pinned package power and say little about the cost of a byte of either tier).

## 9. Limits and uncertainties

* Scope: core-issued loads on this node's ~350 W-per-socket power limit, reads only (writes, attention and KV traffic were not mixed in). The GEMV is the weight stream of decode only (separate W1 and W3, no attention, KV cache or classifier): an upper bound.
* Run-to-run spread of the HBM-only reference is 3% (7 reps); same-sign excesses of 0.7-3.5% (E3, E4, E5) have nominal one-sided permutation p-values of 0.006-0.05, but they come from many comparisons, depend on the worker split, and were replicated only partly; they are reported as "no gain beyond about 2-3%, not robust".
* Core clock is an average of 56 cores of the socket including spinning idle workers; turbostat uncore values are one-second samples of package 0.
* The power-cap explanation is consistent with every counter but unproven (no frequency or power control), and the clock difference between zero and random data is not resolved.
* The stripe-aliasing mechanism is inferred from where the loss appears and disappears; the address-interleave mechanism was not examined, and the stagger fix was not tried inside the engine.
* Open: (a) the UPI traffic of the E4 interleaved 2 MiB runs (median 0.75 GB/s, range 0.14-1.0, against 0.15 elsewhere): the spike sits in the intervals adjacent to the window (allocation and release of the 32 GiB regions), but inside interior intervals some repetitions still show it; (b) 11 of 14 E4 4 KiB interleaved
  windows have no counters (the window's burst merged with the allocation burst); (c) the delivered bandwidth of the NTA hint.
* UPI bytes use an assumed 64/9 bytes per data flit (flits per second are given).
