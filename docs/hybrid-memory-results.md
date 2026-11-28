# Hybrid HBM/DDR memory: baseline campaign results (P4)

Companion: [hybrid-memory-results-validation.md](hybrid-memory-results-validation.md) (every job, tree, elapsed time and result directory). Ceilings:
[plans/hybrid-limits.md](plans/hybrid-limits.md) (`tools/hybrid_limits.py`). Every number is labelled **measured** (a job), **derived** (arithmetic on measured values) or
**modelled** (a ceiling of the limits model). Medians are over interleaved repetitions (3, or 5 for A/B pairs); a range is [min-max].

## 0. What was measured

* **The decode rate follows the bytes read from each tier.** The layer-serial hybrid on the 8B is within 1 to 7% of a straight line in time between its all-HBM and all-DDR cells, taken at the measured DDR share
  of the CAS bytes (section 1.4); the 70B cells are within 0.86 to 1.15 of the calibrated ceilings (section 7).
* **Concurrent HBM and DDR streaming of the weights (`LLAMA_HYBRID_SPLIT=rows`) helps only when weights spill to DDR.** On one socket, 70B (weights do not fit): **+10.4% at 4096 positions**
  (kvmatch, 14 DDR cores, 5 of 5 repetitions above the serial layout of the same repetition), **shrinking to +2.4% at 32768 positions** (5 of 5), against the planner's layer-serial hybrid (measured, section 5).
  With 20 DDR cores the gain is +2.5% and -0.2%.
* **When only the K/V cache spills (70B, two sockets, 65536 to 131072 positions) no row split beats the planner's default layout:** the grid job's eight split cells are -0.6% to -2.0%
  (0 of 5 repetitions above the serial cell of the same repetition, in all eight), and the earlier C3/C4 split cells between -0.2% and -19% (measured, section 4).
* **When everything fits in HBM no split beats the all-HBM layout** (8B one and two sockets at 4096 and 32768: the best split cell is -0.7% to -3.5%, the worst -9.7%; 70B two sockets at 32768: -0.8% and -1.2%; sections 1 and 4).
* **TTFT:** the 70B two-socket AMX prefill of 2048 prompt tokens is 5.15 s warm (after 30 s of untimed compute in the same process) and 6.44 s [5.44-14.37] cold; at 8192 tokens
  24.1 s warm against 34.5 s cold. The AVX-512 prefill is 7.3 times (2048 tokens) and 6.8 times (8192 tokens) slower than AMX (section 3).
* **The cold-process numbers include a once-per-process kernel-mode stall of the node** (6 to 14 s, starting about 9 s after the first forward pass; docs/hbm-ddr-research.md). A decode or
  TTFT window that overlaps it loses up to 46%. Every number below that is not marked cold is outside it (checked by the pause scan, section 7).

## 0.1 Method

* One engine process per (cell, context, repetition) (the engine takes one `--warm-context`, so there is no in-process context ladder); repetitions are interleaved, repetition
  r starting at cell r mod n, 5 repetitions for the A/B steps. One extra `LLAMA_PROFILE=1` run per cell.
* Setup of every cell: `LLAMA_GEMV_STAGGER=1`, strict placement, explicit topology, engine defaults otherwise; the prime-number prompt; 96 generated tokens (256 on the 8B, 32 in the TTFT cells).
* Bytes per tier come from the uncore CAS counters (`perf stat -a --per-socket`, 100 ms) cut to the decode window with the report's `phase_epoch_ns`: 32 bytes per HBM CAS, 64
  per DDR CAS, summed over the sockets (**measured**). They close against the model's weights plus K/V within 1.5% (70B DDR at 32768 positions: 96.1 GB measured against 95.4 GB derived).
* Fills: real forward passes up to 32768 positions on two sockets and up to 4096 on one; synthetic fills (`LLAMA_SYNTHETIC_KV=1`, a build with the 30 s primer guard of develop 5f28871) for the 70B
  contexts from 65536 up and for the one-socket 32768 cells, each checked against a real fill at 32768. A synthetic run is rejected by the summary tool when its slowest decode step exceeds 1 s; none did (largest 0.47 s).
* The pause scan (`tools/hybrid_campaign_summary.py`) counts idle spans of the uncore reads in each run's window: section 7 gives the counts per job.
* Placement: every run passes `check_placement.py --planned`; token traces are equal inside each (model, sockets, context, kernels) group.

## 1. C0: 8B, calibration of the serial and concurrent models against the DDR byte fraction

Cells: `hbm-only`, `ddr`, the serial layer hybrid with n weight layers and n K/V layers in HBM (n = 32 24 16 8 0; classifier in HBM, packed copy in DDR), and the row split
(fractions 0.1, 0.2, 0.3, 14 DDR cores, every K/V layer in HBM). Decode at 4096 and 32768 positions (real fill, 256 steps), TTFT at 2048 tokens. One socket and two sockets, one
job (5950120; 240 runs: 20 cells x 3 repetitions plus one profiled run each, per socket count). All values measured unless marked; the ceiling column is modelled.

### 1.1 One socket

Decode, 4096 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| hbm-only | hbm-only | - | 3 | 44.128 [43.973-44.144] | - | 0.1% (0.0%) | 9.0 / 0.0 | 42.81 | 1.031 |
| ddr | ddr | - | 3 | 25.036 [24.944-25.036] | -43.3% | 99.9% (100.0%) | 0.0 / 9.1 | 26.29 | 0.952 |
| hyb-32 | 32/32 | - | 3 | 44.036 [43.928-44.178] | -0.2% | 0.1% (0.0%) | 9.0 / 0.0 | 42.81 | 1.029 |
| hyb-24 | 24/24 | - | 3 | 38.112 [38.069-38.318] | -13.6% | 23.5% (23.5%) | 6.9 / 2.1 | 37.33 | 1.021 |
| hyb-16 | 16/16 | - | 3 | 33.781 [33.674-33.842] | -23.4% | 47.0% (46.9%) | 4.8 / 4.3 | 33.10 | 1.021 |
| hyb-8 | 8/8 | - | 3 | 29.764 [29.496-29.983] | -32.6% | 70.3% (70.4%) | 2.7 / 6.4 | 29.73 | 1.001 |
| hyb-0 | 0/0 | - | 3 | 26.694 [26.585-26.740] | -39.5% | 93.7% (93.9%) | 0.6 / 8.5 | 26.98 | 0.989 |
| rows-r0.1-c14 | 32/32 | 0.1, 14 | 3 | 39.850 [39.667-40.078] | -9.7% | 7.8% (8.2%) | 8.3 / 0.7 | 36.32 | 1.097 |
| rows-r0.2-c14 | 32/32 | 0.2, 14 | 3 | 42.727 [42.576-42.818] | -3.2% | 16.1% (16.3%) | 7.6 / 1.5 | 39.82 | 1.073 |
| rows-r0.3-c14 | 32/32 | 0.3, 14 | 3 | 40.311 [40.222-40.393] | -8.7% | 24.3% (24.5%) | 6.8 / 2.2 | 39.84 | 1.012 |

Decode, 32768 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| hbm-only | hbm-only | - | 3 | 21.924 [21.881-21.954] | - | 0.1% (0.0%) | 16.7 / 0.0 | 20.10 | 1.091 |
| ddr | ddr | - | 3 | 14.010 [14.004-14.013] | -36.1% | 99.9% (100.0%) | 0.0 / 16.7 | 14.42 | 0.972 |
| hyb-32 | 32/32 | - | 3 | 21.935 [21.888-21.951] | +0.1% | 0.1% (0.0%) | 16.7 / 0.0 | 20.10 | 1.091 |
| hyb-24 | 24/24 | - | 3 | 19.841 [19.750-19.876] | -9.5% | 24.1% (24.2%) | 12.7 / 4.0 | 18.38 | 1.080 |
| hyb-16 | 16/16 | - | 3 | 17.774 [17.766-17.898] | -18.9% | 48.1% (48.3%) | 8.7 / 8.0 | 16.93 | 1.050 |
| hyb-8 | 8/8 | - | 3 | 16.041 [16.032-16.044] | -26.8% | 72.0% (72.5%) | 4.7 / 12.0 | 15.69 | 1.022 |
| hyb-0 | 0/0 | - | 3 | 14.325 [14.316-14.353] | -34.7% | 96.0% (96.6%) | 0.7 / 16.0 | 14.62 | 0.980 |
| rows-r0.1-c14 | 32/32 | 0.1, 14 | 3 | 21.250 [21.224-21.271] | -3.1% | 4.3% (4.5%) | 16.0 / 0.7 | 18.54 | 1.146 |
| rows-r0.2-c14 | 32/32 | 0.2, 14 | 3 | 21.760 [21.744-21.770] | -0.7% | 8.8% (8.9%) | 15.2 / 1.5 | 19.42 | 1.121 |
| rows-r0.3-c14 | 32/32 | 0.3, 14 | 3 | 20.682 [20.680-20.702] | -5.7% | 13.3% (13.4%) | 14.5 / 2.2 | 19.42 | 1.065 |

### 1.2 Two sockets

Decode, 4096 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| hbm-only | hbm-only | - | 3 | 74.754 [74.463-75.169] | - | 0.1% (0.0%) | 8.9 / 0.0 | 82.14 | 0.910 |
| ddr | ddr | - | 3 | 45.077 [44.983-45.150] | -39.7% | 99.9% (100.0%) | 0.0 / 9.0 | 49.68 | 0.907 |
| hyb-32 | 32/32 | - | 3 | 74.755 [74.599-74.824] | +0.0% | 0.1% (0.0%) | 8.9 / 0.0 | 82.14 | 0.910 |
| hyb-24 | 24/24 | - | 3 | 66.141 [66.077-66.220] | -11.5% | 23.5% (23.5%) | 6.8 / 2.1 | 71.27 | 0.928 |
| hyb-16 | 16/16 | - | 3 | 59.450 [59.172-59.483] | -20.5% | 46.9% (46.9%) | 4.8 / 4.2 | 62.95 | 0.944 |
| hyb-8 | 8/8 | - | 3 | 54.809 [54.537-54.837] | -26.7% | 70.3% (70.4%) | 2.7 / 6.3 | 56.36 | 0.972 |
| hyb-0 | 0/0 | - | 3 | 47.101 [46.948-47.285] | -37.0% | 93.7% (93.9%) | 0.6 / 8.4 | 51.02 | 0.923 |
| rows-r0.1-c14 | 32/32 | 0.1, 14 | 3 | 67.557 [67.131-67.597] | -9.6% | 7.6% (8.2%) | 8.1 / 0.7 | 69.60 | 0.971 |
| rows-r0.2-c14 | 32/32 | 0.2, 14 | 3 | 72.163 [72.080-72.677] | -3.5% | 15.9% (16.3%) | 7.4 / 1.4 | 76.37 | 0.945 |
| rows-r0.3-c14 | 32/32 | 0.3, 14 | 3 | 69.302 [69.104-69.532] | -7.3% | 24.3% (24.5%) | 6.7 / 2.2 | 75.45 | 0.919 |

Decode, 32768 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| hbm-only | hbm-only | - | 3 | 39.846 [39.832-39.903] | - | 0.1% (0.0%) | 16.5 / 0.0 | 39.41 | 1.011 |
| ddr | ddr | - | 3 | 25.542 [25.526-25.545] | -35.9% | 100.0% (100.0%) | 0.0 / 16.6 | 27.94 | 0.914 |
| hyb-32 | 32/32 | - | 3 | 39.450 [39.294-40.024] | -1.0% | 0.1% (0.0%) | 16.6 / 0.0 | 39.41 | 1.001 |
| hyb-24 | 24/24 | - | 3 | 35.957 [35.946-36.020] | -9.8% | 24.1% (24.2%) | 12.6 / 4.0 | 35.92 | 1.001 |
| hyb-16 | 16/16 | - | 3 | 32.305 [32.234-32.446] | -18.9% | 48.1% (48.3%) | 8.6 / 8.0 | 32.99 | 0.979 |
| hyb-8 | 8/8 | - | 3 | 29.501 [29.435-29.636] | -26.0% | 72.1% (72.5%) | 4.6 / 12.0 | 30.50 | 0.967 |
| hyb-0 | 0/0 | - | 3 | 26.543 [26.405-26.558] | -33.4% | 96.1% (96.6%) | 0.6 / 16.0 | 28.36 | 0.936 |
| rows-r0.1-c14 | 32/32 | 0.1, 14 | 3 | 37.950 [37.884-37.992] | -4.8% | 4.3% (4.5%) | 15.8 / 0.7 | 36.28 | 1.046 |
| rows-r0.2-c14 | 32/32 | 0.2, 14 | 3 | 39.182 [38.892-39.318] | -1.7% | 8.7% (8.9%) | 15.1 / 1.4 | 38.03 | 1.030 |
| rows-r0.3-c14 | 32/32 | 0.3, 14 | 3 | 37.963 [37.856-38.067] | -4.7% | 13.2% (13.4%) | 14.4 / 2.2 | 37.80 | 1.004 |

### 1.3 TTFT at 2048 prompt tokens (cold process, 3 runs; the prewarm knob did not exist for this job)

One socket:

| setting | TTFT at 2048 prompt tokens, s median [min-max] (3 runs) | measured / AMX peak |
|---|---:|---:|
| hbm-only | 1.183 [1.137-1.306] | 9.0 |
| ddr | 1.242 [1.230-1.243] | 9.4 |
| hyb-32 | 1.165 [1.152-1.307] | 8.8 |
| hyb-24 | 1.171 [1.162-1.178] | 8.9 |
| hyb-16 | 1.201 [1.197-1.305] | 9.1 |
| hyb-8 | 1.295 [1.166-1.334] | 9.8 |
| hyb-0 | 1.172 [1.170-1.315] | 8.9 |
| rows-r0.1-c14 | 1.210 [1.203-1.212] | 9.2 |
| rows-r0.2-c14 | 1.185 [1.177-1.370] | 9.0 |
| rows-r0.3-c14 | 1.211 [1.205-1.342] | 9.2 |

Two sockets:

| setting | TTFT at 2048 prompt tokens, s median [min-max] (3 runs) | measured / AMX peak |
|---|---:|---:|
| hbm-only | 0.822 [0.700-0.879] | 12.5 |
| ddr | 0.797 [0.771-0.866] | 12.1 |
| hyb-32 | 0.729 [0.724-0.777] | 11.0 |
| hyb-24 | 0.735 [0.717-0.831] | 11.1 |
| hyb-16 | 0.724 [0.721-0.735] | 11.0 |
| hyb-8 | 0.754 [0.724-0.775] | 11.4 |
| hyb-0 | 0.856 [0.735-0.919] | 13.0 |
| rows-r0.1-c14 | 0.825 [0.801-1.395] | 12.5 |
| rows-r0.2-c14 | 0.770 [0.769-0.781] | 11.7 |
| rows-r0.3-c14 | 0.777 [0.768-0.795] | 11.8 |

The 8B prefill takes 1 to 2 s, so the node's start-up stall (which comes about 9 s after the first forward) is not in any of these windows (pause scan: 0 flagged windows). The AMX-peak ceiling is
8.8 to 13 times below the measurement, as in the earlier TTFT diagnosis. The hybrid cells keep the packed copy in DDR and are not slower than `hbm-only`, which keeps it in HBM
(2 sockets: 0.729 s for the all-HBM hybrid against 0.822 s for `hbm-only`; the difference is inside the 15% spread of the `hbm-only` runs).

### 1.4 Calibration of the serial model against the DDR byte fraction (derived)

| sockets | context | setting | DDR share of CAS bytes | measured tok/s | tok/s interpolated between the hbm-only and ddr cells of the same job | measured / interpolated |
|---|---|---|---:|---:|---:|---:|
| 1 | 4096 | hyb-24 | 23.5% | 38.11 | 37.43 | 1.018 |
| 1 | 4096 | hyb-16 | 47.0% | 33.78 | 32.49 | 1.040 |
| 1 | 4096 | hyb-8 | 70.3% | 29.76 | 28.73 | 1.036 |
| 1 | 4096 | hyb-0 | 93.7% | 26.69 | 25.74 | 1.037 |
| 1 | 32768 | hyb-24 | 24.1% | 19.84 | 19.30 | 1.028 |
| 1 | 32768 | hyb-16 | 48.1% | 17.77 | 17.24 | 1.031 |
| 1 | 32768 | hyb-8 | 72.0% | 16.04 | 15.58 | 1.029 |
| 1 | 32768 | hyb-0 | 96.0% | 14.32 | 14.22 | 1.008 |
| 2 | 4096 | hyb-24 | 23.5% | 66.14 | 64.72 | 1.022 |
| 2 | 4096 | hyb-16 | 46.9% | 59.45 | 57.11 | 1.041 |
| 2 | 4096 | hyb-8 | 70.3% | 54.81 | 51.11 | 1.072 |
| 2 | 4096 | hyb-0 | 93.7% | 47.10 | 46.23 | 1.019 |
| 2 | 32768 | hyb-24 | 24.1% | 35.96 | 35.11 | 1.024 |
| 2 | 32768 | hyb-16 | 48.1% | 32.30 | 31.39 | 1.029 |
| 2 | 32768 | hyb-8 | 72.1% | 29.50 | 28.38 | 1.039 |
| 2 | 32768 | hyb-0 | 96.1% | 26.54 | 25.90 | 1.025 |

Reading:
* **The measured DDR share of the CAS bytes equals the model's byte fraction within 0.5 percentage points** for every serial cell (the table of section 1.1 and 1.2 gives both).
* **The layer-serial hybrid is 0.8 to 7.2% faster than a straight line between its two anchors** (the byte-additive model of `t_serial` with the two measured cells), at every DDR share from
  23% to 96%. The calibrated ceilings of `hybrid_limits.py` bracket the serial cells within 0.91 to 1.09 of the measured rate (ratio column). The 2S 4096 cells sit 3 to 9% below their
  ceiling, as the 8B 2S warm-4096 rows of docs/plans/hybrid-limits.md already did.
* **The concurrent model (corrected formula of 27 November, K/V read after the split GEMV on all cores) is within 0.92 to 1.15 of every row-split cell** (the ratio column). The first version of the formula
  (K/V charged at the HBM group's rate) put the 1S 32768 cells at 15.6 and 16.2 tok/s against 21.25 and 21.76 measured (ratios 1.36 and 1.35, outside 30%); the corrected one gives 18.5 and 19.4 (1.15 and 1.12).
* **No row split beats the all-HBM layout in the fits case:** one socket, 4096 positions: best 42.73 (r 0.2) against 44.04 (hybrid 32/32) tok/s, -3.0%; 32768: -0.7% (r 0.2); two sockets: -3.5% and -1.7%.
  The all-HBM hybrid equals `hbm-only` (44.04 against 44.13 on one socket, 74.76 against 74.75 on two).
* **Token traces are bitwise equal** in all six groups (40 traces per group).

## 2. C1, C2: the 70B baselines on two sockets

C1 is the HBM baseline: `hybrid` with all 80 weight layers and all 80 K/V layers in HBM and the AMX packed copy in DDR (`LLAMA_HYBRID_PACKED=ddr`); `hbm-only` rejects `LLAMA_HYBRID_PACKED` and
overflows with the packed copy (per socket at 8192 positions weights 34.4 GiB + packed 33.9 + rest 5.0 = 73.2 GiB against the 56 GiB budget), so every "HBM" 70B row of this page is that hybrid. C2 is `LLAMA_PLACEMENT=ddr`.

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| c1 hybrid 80/80, packed in DDR, warm 4096 (real) | 80/80 | - | 3 | 10.515 [10.415-10.516] | - | 0.1% (0.0%) | 76.6 / 0.0 | 11.06 | 0.951 |
| c1 same, warm 32768 (real) | 80/80 | - | 3 | 7.202 [7.150-7.217] | - | 0.1% (0.0%) | 96.0 / 0.1 | 8.10 | 0.889 |
| c1 same, warm 32768 (synthetic check) | 80/80 | - | 3 | 7.259 [7.238-7.271] | - | 0.1% (0.0%) | 95.9 / 0.1 | 8.10 | 0.896 |
| c2 ddr, warm 4096 (real) | ddr | - | 3 | 5.894 [5.881-5.921] | - | 100.0% (100.0%) | 0.0 / 76.9 | 6.03 | 0.977 |
| c2 ddr, warm 32768 (real) | ddr | - | 3 | 4.655 [4.651-4.674] | - | 100.0% (100.0%) | 0.0 / 96.1 | 4.88 | 0.954 |
| c2 ddr, warm 32768 (synthetic check) | ddr | - | 3 | 4.674 [4.673-4.679] | - | 100.0% (100.0%) | 0.0 / 96.1 | 4.88 | 0.958 |
| c2 ddr, warm 130816 (synthetic) | ddr | - | 3 | 2.739 [2.737-2.741] | - | 100.0% (100.0%) | 0.1 / 162.2 | 2.95 | 0.928 |

Reading:
* **Clean cells** (warm 4096 and 32768 real, job 5950121; the 32768 synthetic checks and the 131072 cell, job 5962577 on the same node): the pause scan finds no idle span of 0.5 s or more in any of the 28 runs of these cells.
* **Synthetic fill:** the synthetic check at 32768 is +0.8% (hybrid) and +0.4% (ddr) against the real fill. The first synthetic cells (job 5950121) were void: every run contained the start-up stall and read
  3.87 against 7.20 tok/s (c1) and 2.97 against 4.65 (c2); they are replaced by the reruns above, built with the 30 s primer guard (the slowest decode step of every synthetic run is at most 0.47 s, the tool's limit is 1 s).
* **The 80/80 hybrid is not admitted at 65536 positions** (K/V arenas of three per layer plus the 36.9 GB weights arena exceed 56 GiB by 9.6 GB), so there is no C1 cell at 65536; the planner's layout at that
  context is the `serial` cell of section 4.
* **Bytes:** CAS bytes per token close against the model's weights plus K/V within 1.5% (70B DDR at 32768 positions: 96.07 GB measured, 95.4 GB derived).
* **70B decode against the ceiling:** hybrid 80/80 0.951 (4096) and 0.889 (32768); ddr 0.977 and 0.954; ddr at 130816 positions 0.928.
* **DDR against HBM:** the DDR layout costs 44% of the rate at 4096 (5.89 against 10.51) and 35% at 32768 (4.66 against 7.20).

## 3. TTFT, 70B two sockets (C1, C2, C7 and the harness check h1)

| step | cell | n clean/all | prewarm | prompt tokens | TTFT s median [min-max] | TTFT s clean median | HBM / DDR GB read in the prefill | AMX-peak ceiling s | measured / peak | 8B-scaled prediction s | measured / scaled |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| c1 | c1 hybrid 80/80, prewarmed | 5/5 | 30 s | 2048 | 5.149 [5.139-5.686] | 5.149 | 144 / 189 | 0.635 | 8.1 | 6.15 | 0.84 |
| c1 | c1 same, cold process | 3/5 | 0 s | 2048 | 6.438 [5.438-14.366] | 6.160 | 147 / 190 | 0.635 | 10.1 | 6.15 | 1.05 |
| c1 | c1 same, prewarmed | 5/5 | 30 s | 8192 | 24.115 [23.988-26.627] | 24.115 | 597 / 752 | 2.688 | 9.0 | 29.39 | 0.82 |
| c1 | c1 same, cold process | 0/5 | 0 s | 8192 | 34.489 [30.322-38.626] | - | 598 / 762 | 2.688 | 12.8 | 29.39 | 1.17 |
| c2 | c2 ddr, prewarmed | 5/5 | 30 s | 2048 | 5.412 [5.383-6.012] | 5.412 | 1 / 361 | 0.635 | 8.5 | 6.15 | 0.88 |
| c7 | c7 AMX packed, prewarmed | 5/5 | 30 s | 2048 | 5.160 [5.134-5.871] | 5.160 | 144 / 189 | 0.635 | 8.1 | 6.15 | 0.84 |
| c7 | c7 AVX-512, prewarmed | 5/5 | 30 s | 2048 | 37.501 [37.351-37.541] | 37.501 | 757 / 6 | 0.635 | 59.0 | - | - |
| c7 | c7 AMX packed, prewarmed | 5/5 | 30 s | 8192 | 24.059 [23.943-26.674] | 24.059 | 593 / 753 | 2.688 | 9.0 | 29.39 | 0.82 |
| c7 | c7 AVX-512, prewarmed | 5/5 | 30 s | 8192 | 164.454 [164.260-166.268] | 164.454 | 3229 / 30 | 2.688 | 61.2 | - | - |

Job 5963862, tree ae89d0e, with `LLAMA_BENCH_PREWARM_SECONDS=30` on every prewarmed cell (the engine runs untimed passes of the first 1024 prompt tokens for 30 s of compute, then the timed run
starts from position 0 with the same trace). The AVX-512 prefill has no packed copy and 6 and 30 GB of DDR reads against 189 and 753 GB for AMX. The "8B-scaled" column is the 8B in-situ TOP/s scaled to the 70B, not a ceiling.

Reading:
* **AMX against AVX-512 (prewarmed):** 5.16 against 37.50 s at 2048 tokens (7.3 times) and 24.06 against 164.45 s at 8192 tokens (6.8 times) (derived). The AVX-512 cells are 59 and 61 times the AMX-peak ceiling;
  the 8B-scaled prediction does not apply to them.
* **The three packed placements are equal at 2048 tokens:** hybrid 80/80 5.149 s, c7 5.160 s (the same placement in the A/B job), ddr 5.412 s (+5%); the TTFT of the DDR layout is the DDR read of the packed copy plus the same GEMMs.
* **Cold against prewarmed:** 6.44 s [5.44-14.37] against 5.15 s at 2048 tokens (two of five cold runs contain the stall, 8.96 s idle in the worst; the three clean cold runs have a median of 6.16 s, +20%, and a minimum of 5.44 s),
  and 34.5 s [30.3-38.6] against 24.1 s at 8192 tokens (5 of 5 contain 5.8 to 13.9 s of idle, +43%). A cold TTFT of the 70B in the default build is therefore the stall (when it lands in the prefill) plus about 1 s of
  first-use cost at 2048 tokens; the prewarmed number is the steady-state one.
* **Harness check h1** (explicit topology and 10 perf events, as in job 5950121, against the discovered topology and the 3-event set of the prewarm check, 3 runs each, prewarmed): all four variants are
  between 5.13 and 5.19 s (table above). The harness does not move TTFT. The gap between 6.4 to 7.1 s (job 5950121, cold, 3 runs) and 5.2 s (the prewarm check) is the cold start (the cold cells of this job read 6.44 s, 6.16 s without the paused runs).
* **The run reported as failed** (c7 amx t2048, repetition 5) completed: its report shows 31 decode steps and a 5.8 s prefill. The failure was the post-run check, which could not start Python from the
  shared file system; its placement check run afterwards on the saved report passes (26 PASS, 0 FAIL). The run is kept.
* **Prewarmed spread:** the slowest of each 5-run cell at 2048 tokens is 10 to 14% above its median (5.686, 6.012, 5.871 s; one run per cell); no idle span was found in those runs, so it is not the stall.

## 4. C3, C4 and the grid job: serial hybrid against the concurrent row split, 70B on two sockets

Question: does `LLAMA_HYBRID_SPLIT=rows` beat the layer-serial hybrid when the model fits (C3, 32768 real) and when K/V spills (C3 at 65536; C4 at 65536, 98304 and 131072; the grid job at 98304 and
131072)? The 4096 case is P3's job 5936167 (fits: best rows cell -2.3%) and is not repeated.

### 4.1 Fits in HBM (32768 positions, real K/V, 5 repetitions) (measured)

| setting | decode tok/s median [min-max] | vs serial | DDR share of CAS bytes | HBM / DDR GB per token | modelled ceiling (cal.) | measured / modelled |
|---|---:|---:|---:|---|---:|---:|
| serial (80/80, nothing in DDR) | 7.199 [7.14-7.21] | - | 0.1% | 96.0 / 0.06 | 8.10 | 0.889 |
| rows f 0.1, 8 DDR cores | 7.115 [7.09-7.12] | -1.2% | 7.4% | 89.0 / 7.1 | 7.72 | 0.921 |
| rows f 0.2, 14 DDR cores | 7.140 [7.10-7.15] | -0.8% | 15.1% | 81.5 / 14.5 | 7.70 | 0.927 |

The synthetic fill of the serial cell (3 repetitions) is 7.261 tok/s, +0.9% against the real fill; the c1 hybrid 80/80 cell gives 7.259 (the same placement, another
setting name) and the DDR cell 4.674 against 4.65 real. **No row split beats the all-HBM layout when everything fits, again** (P3: -2.3% at 4096).
The combined read rate falls as HBM cores are given to DDR (P1: 36 + 20 cores deliver 521 GB/s against 569 for 56 + 0), and the 8-14 DDR cores add less than they remove.

### 4.2 K/V spills (65536, 98304, 131072 positions, synthetic K/V, 5 repetitions) (measured)

| context | setting | layers in HBM (weights / K/V) | DDR fraction | decode tok/s median [min-max] | vs serial-w80 | DDR share of CAS bytes (modelled) | modelled ceiling (cal.) | measured / modelled |
|---:|---|---|---:|---:|---:|---|---:|---:|
| 65536 | serial-w80 (planner K/V) | 80 / 56 | - | 5.295 [5.29-5.30] | - | 11.0% (11.0%) | 6.05 | 0.876 |
| 65536 | serial (planner, all auto) | 80 / 56 | - | 5.291 [5.29-5.30] | -0.1% | 11.0% | 6.05 | 0.875 |
| 65536 | serial-kv80 | 58 / 80 | - | 4.981 [4.98-4.99] | -5.9% | 17.1% | 5.52 | 0.903 |
| 65536 | rows-auto-c20 | all / 80 | 0.266 | 5.284 [5.25-5.29] | -0.2% | 16.4% (16.6%) | 5.73 | 0.922 |
| 65536 | rows-auto-c14 | all / 80 | 0.266 | 5.112 [5.09-5.12] | -3.5% | 16.4% (16.6%) | 5.85 | 0.873 |
| 65536 | rows-kvmatch-c20 | all / 56 | 0.2 | 5.085 [5.08-5.09] | -4.0% | 23.2% (23.5%) | 5.34 | 0.953 |
| 98304 | serial-w80 | 80 / 37 | - | 4.186 [4.18-4.19] | - | 24.7% (25.1%) | 4.76 | 0.880 |
| 98304 | rows-auto-c20 | all / 80 | 0.711 | 3.395 [3.38-3.40] | -18.9% | 36.9% (37.4%) | 3.33 | 1.021 |
| 98304 | rows-kvmatch-c20 | all / 37 | 0.2 | 4.035 [4.03-4.04] | -3.6% | 35.1% (35.6%) | 4.31 | 0.937 |
| 131072 | serial-w80 | 80 / 28 | - | 3.420 [3.41-3.43] | - | 34.5% (34.9%) | 3.93 | 0.870 |
| 131072 | rows-auto-c20 | - | 0.9 | not admitted: node needs 69.3 GB of HBM, 9.1 GB over the budget | | | | |
| 131072 | rows-kvmatch-c20 | all / 28 | 0.2 | 3.307 [3.29-3.31] | -3.3% | 43.4% (44.0%) | 3.62 | 0.915 |

(`serial-w80` and `serial` are the same plan at 65536. The planner puts 56, 37 and 28 K/V layers in HBM at 65536, 98304 and 131072 positions, with all 80 weight layers.)

Reading (derived from the table and the profile):

* **No row-split cell beats the serial hybrid when K/V is what spills.** The best are equal (65536, `rows-auto-c20`, -0.2%) or 3 to 4% below. The 98304-position `rows-auto` cell is 19%
  below: keeping every K/V layer in HBM forces 71% of the weight rows to DDR (37% of the bytes against 25%), and the DDR group of 20 cores (88 GB/s per socket here) limits the step.
* **The row split moves weight rows, not K/V.** In `serial-w80` the weights already read at the HBM rate; the spilled bytes are K/V layers, read by attention at the full-socket DDR
  rate, outside the concurrent part. Rows can only shuffle weight bytes into DDR, which adds DDR bytes the serial layout does not have (`rows-kvmatch` at 98304: 35.1% DDR bytes against
  24.7%) while taking 20 of the 56 cores from the HBM stream. It trades a 3 to 4% loss.
* **The model agrees on the sign and the size.** The calibrated ceiling of the row split at its own fraction and cores is 4.31 tok/s at 98304 against 4.76 for the serial layout (-9.5%),
  measured -3.6%. With the planner's own layers (all 80 weight layers in HBM and 37 or 28 K/V layers) the model puts every row-split cell with the serial cell's K/V at or just below serial (4.31 to 4.74 against 4.76 at 98304),
  because the bytes the split moves to DDR are weights that were already read at the HBM rate. The model gains only if the weight rows moved to DDR free HBM for more K/V layers: with
  16 DDR cores and fraction 0.28 the planner admits 54 K/V layers at 98304 and 40 at 131072, which the model reads as +1.7% and +1.3% over serial; with 8 cores and fraction 0.15 (46 and 34 layers) +0.6% and +0.4% (modelled, calibrated
  ceilings of `tools/hybrid_limits.py` at commit 96a20ed, computed from the resolved `--plan-only` reports of the grid job; table in section 4.2.1). Section 4.4 gives the result.
  The earlier figure of +5 to +8% came from the library's joint-layer placement, which is not the planner's.
* **Stop rule.** Every C3/C4 cell is within 30% of its calibrated ceiling (0.87 to 1.02). The serial cells are 0.87 to 0.90 of it (a constant offset: the calibration anchor is
  the 611-position run; the 32768 and 65536 real cells give the same ratio). Before the correction of the concurrent formula (docs/plans/hybrid-limits.md, 27 November), the `rows-auto`
  cells at 98304 and `rows-kvmatch` at 98304 and 131072 read 0.74, 0.77 and 0.82 of the modelled value, which is why the formula was corrected; the old figure charged K/V at the HBM group's rate.
* **The planner's capacity was the model's error, now fixed** (derived from the `--plan-only` reports of this campaign, section 4.3). The limits model placed 80 of 80 layers in HBM at
  65536 positions (6.21 tok/s calibrated serial); the planner admits 56 K/V layers (5.29 measured, 0.85). The model had charged a K/V layer once; the planner charges three arenas per layer (K, V and
  the BF16 copy of the AMX attention), sized for the whole context. With `plan_layers` the model reproduces every layer count of this campaign (section 4.3) and the library's own serial ceilings (joint-layer placement, docs/plans/hybrid-limits.md section 6) read 0.92 to 0.96 of the measured rate;
  the tables of this page use the planner's layers of each cell and read 0.87 to 0.88 for the serial cells.

#### 4.2.1 Predictions of the grid job (modelled)

Calibrated ceilings, tok/s, of the cells of the grid job at the planner's resolved plans (weights 80 layers in HBM everywhere; `tools/hybrid_limits.py` at commit 96a20ed; the same function
writes the job's `plan/*.pred` and the "measured / cal." column of `tools/hybrid_campaign_summary.py`). The capacity model of 96a20ed changes which K/V layer counts the library predicts
for a fraction, not the throughput of a given plan: the predictions below are equal to the ones the plan job wrote before 96a20ed existed.

| context | cell | K/V layers in HBM | modelled tok/s | vs serial-w80 |
|---:|---|---:|---:|---:|
| 98304 | serial-w80 | 37 | 4.758 | - |
| 98304 | rows-kvfit c16 f 0.28 | 54 | 4.839 | +1.7% |
| 98304 | rows-kvfit c8 f 0.15 | 46 | 4.788 | +0.6% |
| 98304 | rows-kvmatch c16 f 0.28 (control) | 37 | 4.736 | -0.5% |
| 98304 | rows-kvmatch c8 f 0.15 (control) | 37 | 4.735 | -0.5% |
| 131072 | serial-w80 | 28 | 3.929 | - |
| 131072 | rows-kvfit c16 f 0.28 | 40 | 3.980 | +1.3% |
| 131072 | rows-kvfit c8 f 0.15 | 34 | 3.946 | +0.4% |
| 131072 | rows-kvmatch c16 f 0.28 (control) | 28 | 3.914 | -0.4% |
| 131072 | rows-kvmatch c8 f 0.15 (control) | 28 | 3.913 | -0.4% |

### 4.3 Planner capacity model (derived)

Each K/V layer in HBM costs three arenas of `kv_layer_bytes` (K, V and the BF16 copy), sized for the context (warm + 256 positions); weights cost the weights arena (layers, classifier and the
5.3 MB of norms) less the DDR rows of the split (per matrix `floor(fraction x rows / 32) x 32` rows); work, exchange and logits are linear in the context. The budget is 60,129,542,144 bytes per
socket. Against the planner's reports: all-HBM weights with 56, 37 and 28 K/V layers at 65536, 98304 and 131072; rows fraction 0.28 / 0.15 with 54 / 46 layers at 98304 and 40 / 34 at 131072
(the grid job's resolved values, 4 of 4); 64 weight layers and none for K/V on one socket at 32768; and the smallest fractions the planner chose for "all K/V in HBM" (0.266 at 65536, 0.711 at
98304 on two sockets, 0.643 on one), each of which fails one grid step lower. The planner charges HBM only for bytes that live in HBM: the DDR rows arena is exactly the rows moved, and the HBM
weights arena shrinks by the same amount.


### 4.4 The grid job: row splits that free HBM for more K/V layers (job 5963869, 98304 and 131072 positions, synthetic K/V, 5 repetitions) (measured)

Cells (5 settings, one job, one node): `serial-w80` (anchor); `rows-kvfit-c16-f0.28` and `rows-kvfit-c8-f0.15`, the two balance points of the limits model with the most K/V layers the planner admits with that
fraction of the weight rows in DDR (resolved in the plan gate by a search over `--plan-only`: 54 and 46 layers at 98304, 40 and 34 at 131072, the four values the capacity model predicts, section 4.3);
`rows-kvmatch-c16-f0.28` and `rows-kvmatch-c8-f0.15`, the same split at the serial cell's own K/V layers (37, 28; the null control).

At 98304 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| serial-w80 | 80/37 | - | 5 | 4.177 [4.163-4.195] | - | 24.8% (25.1%) | 105.3 / 34.7 | 4.76 | 0.878 |
| rows-kvfit-c16-f0.28 | 80/54 | 0.28, 16 | 5 | 4.149 [4.148-4.155] | -0.7% | 29.4% (29.9%) | 99.0 / 41.2 | 4.84 | 0.857 |
| rows-kvfit-c8-f0.15 | 80/46 | 0.15, 8 | 5 | 4.152 [4.120-4.154] | -0.6% | 27.3% (27.7%) | 101.8 / 38.3 | 4.79 | 0.867 |
| rows-kvmatch-c16-f0.28 | 80/37 | 0.28, 16 | 5 | 4.113 [4.112-4.114] | -1.5% | 39.2% (39.8%) | 85.1 / 54.9 | 4.74 | 0.869 |
| rows-kvmatch-c8-f0.15 | 80/37 | 0.15, 8 | 5 | 4.130 [4.129-4.137] | -1.1% | 32.5% (32.9%) | 94.6 / 45.5 | 4.73 | 0.872 |

At 131072 positions:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| serial-w80 | 80/28 | - | 5 | 3.430 [3.414-3.437] | - | 34.5% (34.9%) | 106.2 / 55.9 | 3.93 | 0.873 |
| rows-kvfit-c16-f0.28 | 80/40 | 0.28, 16 | 5 | 3.406 [3.389-3.407] | -0.7% | 38.9% (39.6%) | 99.0 / 63.1 | 3.98 | 0.856 |
| rows-kvfit-c8-f0.15 | 80/34 | 0.15, 8 | 5 | 3.396 [3.382-3.397] | -1.0% | 37.2% (37.7%) | 101.8 / 60.3 | 3.95 | 0.861 |
| rows-kvmatch-c16-f0.28 | 80/28 | 0.28, 16 | 5 | 3.361 [3.345-3.370] | -2.0% | 46.9% (47.7%) | 86.1 / 76.0 | 3.91 | 0.859 |
| rows-kvmatch-c8-f0.15 | 80/28 | 0.15, 8 | 5 | 3.374 [3.367-3.381] | -1.6% | 41.1% (41.8%) | 95.4 / 66.7 | 3.91 | 0.862 |

* **No split cell beats the anchor in any repetition** (0 of 5 in all 8 comparisons). The two balance-point cells (more K/V in HBM) are -0.7% and -0.6% at 98304, -0.7% and -1.0% at 131072; the controls
  -1.5% and -1.1%, -2.0% and -1.6% (measured). Moving weight rows to DDR in order to hold more K/V layers in HBM is better than keeping the K/V layers fixed, by about 0.8 to 1.3 percentage points, and
  still not a gain.
* **The model predicted +1.7%, +0.6% (98304) and +1.3%, +0.4% (131072) for the four balance-point cells and -0.5%, -0.4% for the controls** (section 4.2.1; modelled). The measured differences are 1.4 to 2.4 points
  below that. The measured / modelled ratio is 0.856 to 0.872 for the split cells against 0.878 and 0.873 for the anchors: the model is about 1.5% too optimistic for a split, a constant that the
  limits model does not charge (section 6).
* **DDR bytes:** the balance-point cells read 29.4% and 27.3% of their bytes from DDR at 98304 (anchor 24.8%) and 38.9% and 37.2% at 131072 (anchor 34.5%); the controls 39.2%, 32.5%, 46.9% and 41.1%.
  More K/V layers in HBM take 6 to 9 GB per token away from DDR for the 54- and 46-layer cells, which the split gives back as DDR weight bytes.



## 5. C6 and C6b: 70B on one socket (the weights do not fit)

On one socket the 70B weights (73.9 GB) cannot be in HBM (56 GiB budget): the planner's serial layout keeps 64 weight layers in HBM and 16 in DDR (K/V 9 layers in HBM at 4096; none at 32768).
The row split keeps every weight layer in HBM except the last rows of each matrix. This is the case the concurrent mode is for.

Cells (job 5962626, tree c834754, 5 repetitions interleaved; the first c6 job, 5962578, gave the real-fill check and four cells):
`serial` is the planner's layout with K/V layers auto; `serial-b` the same setting run a second time as the drift anchor; `serial-kv80` forces every K/V layer into HBM (weights spill to 59 and 28
layers); `rows-auto-c20` is the P3 shape (every K/V layer in HBM, the fraction the planner needs: 0.254 at 4096, 0.643 at 32768); `rows-kvmatch-cN` takes the serial cell's K/V layers (9 at 4096, 0 at
32768) and leaves the fraction to the planner (0.201 and 0.196); `ddr` is all-DDR.

Decode at 4096 positions (real fill):

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| serial | 64/9 | - | 5 | 4.969 [4.935-4.975] | - | 22.2% (22.2%) | 59.8 / 17.0 | 4.72 | 1.052 |
| serial-b | 64/9 | - | 5 | 4.952 [4.919-4.962] | -0.3% | 22.1% (22.2%) | 59.8 / 17.0 | 4.72 | 1.049 |
| serial-kv80 | 59/80 | - | 5 | 4.838 [4.833-4.855] | -2.6% | 24.9% (24.9%) | 57.7 / 19.1 | 4.56 | 1.061 |
| rows-auto-c20 | 80/80 | 0.254, 20 | 5 | 5.395 [5.390-5.410] | +8.6% | 24.1% (24.1%) | 58.3 / 18.5 | 4.75 | 1.137 |
| rows-kvmatch-c14 | 80/9 | 0.201, 14 | 5 | 5.483 [5.462-5.487] | +10.4% | 22.2% (22.3%) | 59.7 / 17.0 | 5.12 | 1.070 |
| rows-kvmatch-c20 | 80/9 | 0.201, 20 | 5 | 5.092 [5.069-5.103] | +2.5% | 22.2% (22.3%) | 59.7 / 17.0 | 4.42 | 1.152 |
| ddr | ddr | - | 5 | 3.173 [3.157-3.174] | -36.1% | 99.9% (100.0%) | 0.1 / 76.7 | 3.02 | 1.052 |

Decode at 32768 positions (synthetic fill; the real fill of the serial cell, 3 runs, job 5962578, is below):

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| serial | 64/0 | - | 5 | 3.460 [3.417-3.465] | - | 37.6% (37.8%) | 60.0 / 36.1 | 3.45 | 1.004 |
| serial-b | 64/0 | - | 5 | 3.462 [3.439-3.466] | +0.1% | 37.6% (37.8%) | 59.9 / 36.2 | 3.45 | 1.005 |
| serial-kv80 | 28/80 | - | 5 | 2.973 [2.965-2.976] | -14.1% | 49.4% (49.6%) | 48.6 / 47.4 | 2.92 | 1.017 |
| rows-auto-c20 | 80/80 | 0.643, 20 | 5 | 2.767 [2.758-2.770] | -20.0% | 48.7% (49.0%) | 49.3 / 46.8 | 2.41 | 1.148 |
| rows-kvmatch-c14 | 80/0 | 0.196, 14 | 5 | 3.544 [3.525-3.554] | +2.4% | 37.3% (37.5%) | 60.2 / 35.8 | 3.64 | 0.974 |
| rows-kvmatch-c20 | 80/0 | 0.196, 20 | 5 | 3.452 [3.428-3.474] | -0.2% | 37.3% (37.5%) | 60.2 / 35.8 | 3.27 | 1.057 |
| ddr | ddr | - | 5 | 2.480 [2.478-2.481] | -28.3% | 99.9% (100.0%) | 0.1 / 96.0 | 2.44 | 1.016 |

Real against synthetic fill, serial cell:

| setting | layers in HBM (weights/K/V) | DDR fraction, cores | n | decode tok/s median [min-max] | vs reference | DDR share of CAS bytes (modelled) | HBM / DDR GB per token | ceiling (cal.) | measured / ceiling |
|---|---|---|---:|---:|---:|---|---|---:|---:|
| serial, real fill (job 5962578) | 64/0 | - | 3 | 3.433 [3.430-3.434] | - | 37.6% (37.8%) | 60.0 / 36.2 | 3.45 | 0.996 |
| serial, synthetic (job 5962578) | 64/0 | - | 5 | 3.460 [3.457-3.465] | - | 37.6% (37.8%) | 60.0 / 36.1 | 3.45 | 1.004 |

* **Row split at the serial cell's own K/V layers wins when weights spill:** with 14 DDR cores +10.4% at 4096 positions (5.483 against 4.969 tok/s; 5 of 5 repetitions above the serial cell of the same
  repetition) and +2.4% at 32768 positions (3.544 against 3.460; 5 of 5). With 20 cores +2.5% (5 of 5) and -0.2% (2 of 5): the 20-core split gives more cores to the DDR group than the 20% of rows
  can use. The anchors agree: `serial-b` is -0.3% and +0.1% against `serial` (the drift of the job).
* **The gain falls from +10.4% to +2.4% as context grows** because K/V (read after the split GEMV at the full-socket rate, never concurrently) grows from 3.5% of the bytes (2.7 of 76.6 GB) at 4096 to 22% (21.5 of 95.4 GB) at 32768, and the
  weights are the only concurrent part. Profiled (one run per cell, ms per token): the 14-core split cuts the GEMV phases from 173.1 to 152.2 ms at 4096 and from 169.1 to 146.1 ms at 32768 (-21 and -23 ms, derived),
  but attention grows by 1.7 ms at 4096 (22.7 to 24.4) and by 15.6 ms at 32768 (114.4 to 130.0): the net step is 201.7 to 183.2 ms (-9.2%) and 289.2 to 282.9 ms (-2.2%) (measured, profile).
* **`rows-auto-c20` (the P3 shape):** +8.6% at 4096 (it keeps all K/V in HBM and the planner needs a fraction of 0.254) and -20.0% at 32768 (fraction 0.643: 64% of the weight rows in DDR; 49% of the bytes in DDR
  against 38% for serial; the DDR group limits the step). The P3 spill job (5935860, 4096 positions) measured +11.1% for this shape against a serial layout with K/V in HBM, a different baseline.
* **Placing K/V in HBM costs when the weights do not fit:** `serial-kv80` is -2.6% at 4096 and -14.1% at 32768 against the planner's layout, which puts the weights first.
* **Synthetic against real fill (serial, 32768):** +0.8% (3.460 against 3.433, job 5962578); the synthetic runs' slowest decode step is at most 0.42 s.
* **Against the ceilings:** serial 1.052 and 1.004 of the calibrated ceiling, the row-split cells 0.974 to 1.152 (one socket reuses the two-socket efficiencies, an estimate); the stop rule holds.

## 6. The cost of a split that the model does not charge

All eight split cells of the grid job are 1.4 to 2.4 points below their prediction relative to the anchor (measured / modelled 0.856 to 0.872 against 0.873 and 0.878; derived from the tables of section 4.4). The
campaign profiles only one cell per setting, and the grid job profiled only the serial anchor, so the phases of the split cells at 16 and 8 DDR cores were not measured. What the other profiles show
(job 5962577, rows-kvmatch with 20 DDR cores and fraction 0.2 against serial-w80, profiled, ms per token):

| context | GEMV phases (qkv, wo, w1+w3, w2, classifier), serial / split | attention, serial / split | forward, serial / split |
|---:|---|---|---|
| 65536 | 94.4 / 89.8 (the serial run has an outlier phase) | 104.8 / 103.1 | 203.3 / 197.1 |
| 98304 | 78.2 / 88.3 | 157.1 / 155.7 | 239.5 / 248.3 |
| 131072 | 78.0 / 89.0 | 210.3 / 209.9 | 292.6 / 303.8 |

* **The extra time is in the weight (GEMV) phases and not in attention:** +10 to +11 ms per token (+13 to +14% of the GEMV time) at 98304 and 131072 with 20 DDR cores and fraction 0.2, attention unchanged to 1%
  (derived from the profile). On one socket (C6b, 14 and 20 cores) the GEMV phases of the split cells are faster than the serial cell's (146 and 162 ms against 169 ms at 32768) because they read 20% fewer
  bytes from DDR, as the model says.
* **The model over-charges the 20-core split and under-charges the 16- and 8-core ones:** with the calibrated rates the model predicts a GEMV time 26% above serial for 20 cores and fraction 0.2
  (107.7 against 85.6 ms) and measures +13%; for the grid cells it predicts step times within 0.5% below and 1.7% above serial and the measurement is 1.4 to 2.4 points lower. The model's error on the split's GEMV time changes sign between 20 cores
  (too high) and 16 and 8 cores (too low, by what the grid shows), so a single multiplicative term does not fit.
* **No model term is proposed.** A constant of 3 to 4 ms per token (about 1.5% of the 240 ms step, 11 microseconds per GEMV call of 320 calls) would fit the grid, and is what a barrier between
  the two core groups would cost; the data cannot separate it from the K/V-rate offset below. Separating them needs profiled runs of the 16- and 8-core split cells (two runs, about 4 minutes).
* **The calibration offset is larger than the split cost.** Against the profiles, the model's GEMV time is 10% above the serial measurement and its K/V time about 20% below the measured attention
  (157 against 125 ms at 98304; 209 against 169 ms at 131072): the calibration anchor at 611 positions puts all short-context overhead into the weight rate (docs/plans/hybrid-limits.md section 7).
  The anchor offsets cancel in the serial cells (0.87 to 0.88) and are the reason the split cells' deficit is read against the anchors, not against 1.

## 7. Pauses, synthetic fill and the stop rule

* **Pause scan** (100 ms samples of HBM+DDR reads below 10% of the window's 90th percentile, spans of 0.5 s or more, plus holes of dropped samples): every decode window of every job after the fix is clean (c3/c4:
  106 runs; c4g: 52; C6b: 84; the prewarmed TTFT cells: 0 of 47 flagged). Flagged: the cold TTFT cells (c1 t2048c: 2 of 5; c1 t8192c: 5 of 5, idle 5.8 to 13.9 s) and the voided cells of job 5950121.
* **Synthetic fill:** every synthetic run reports a slowest decode step of at most 0.47 s (guard 1 s); synthetic against real: +0.8%, +0.8%, +0.4% (70B 2S hybrid 32768, 1S 32768, ddr 32768).
* **Stop rule (30% against the `hybrid_limits.py` ceiling):** no decode cell of the campaign trips it. Calibrated-ceiling ratios: 8B 0.91 to 1.15; 70B 2S 0.86 to 1.02; 70B 1S 0.97 to 1.15. The only values outside 30% came from the first, uncorrected concurrent formula (section 1.4) and the void cells of job 5950121.
* **The TTFT ceilings:** the prefill is 8 to 9 times the AMX-peak time warm, 59 to 61 times for AVX-512; the earlier diagnosis (docs/ttft-70b-diagnosis.md) explains the AMX gap.

## 8. Not done

* C5 (BF16 and INT8 K/V) because they do not exist in the engine; C8 (Cache mode) because no Cache-mode node is on this partition.
* Real fills above 32768 positions (synthetic fills only, and only against a 32768 real check).
* A profile of the split cells of the grid job (section 6).
* A grid beyond the two balance points at 98304 and 131072, and K/V in the FP32 layout only.
