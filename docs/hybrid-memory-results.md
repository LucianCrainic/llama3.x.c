# Hybrid HBM/DDR memory: baseline campaign results (P4)

Draft in progress: this page holds the sections finished so far (C3 and C4); C0, C1, C2, C6 and C7 are added as their jobs are verified. Companion:
[hybrid-memory-results-validation.md](hybrid-memory-results-validation.md) (jobs, trees, elapsed times, result directories). Ceilings come from
[plans/hybrid-limits.md](plans/hybrid-limits.md) (`tools/hybrid_limits.py`). Every number is labelled **measured** (a job), **derived** (arithmetic on
measured values) or **modelled** (a ceiling of the limits model).

## 0. Method in six lines

* One engine process per (cell, context, repetition) (the engine takes one `--warm-context`, so there is no in-process context ladder); repetitions are interleaved, repetition
  r starting at cell r mod n, 5 repetitions for the A/B steps. One extra `LLAMA_PROFILE=1` run per cell.
* Setup of every cell: `LLAMA_GEMV_STAGGER=1`, strict placement, explicit topology, engine defaults otherwise; the prime-number prompt; 96 generated tokens.
* Bytes per tier come from the uncore CAS counters (`perf stat -a --per-socket`, 100 ms) cut to the decode window with the report's `phase_epoch_ns`: 32 bytes per HBM CAS, 64
  per DDR CAS, summed over the sockets (**measured**). They close against the model's weights plus K/V within 1.5% (70B DDR at 32768 positions: 96.1 GB measured against 95.4 GB derived).
* Contexts at and above 65536 are filled synthetically (`LLAMA_SYNTHETIC_KV=1`, a build with the 30 s primer guard of develop 5f28871); 32768 is filled by real forward
  passes. A synthetic run is rejected by the summary tool when its slowest decode step exceeds 1 s; none did (largest 0.47 s).
* The pause scan (`tools/hybrid_campaign_summary.py`) counts idle spans of the uncore reads in each run's window: none of the 126 runs of the C3/C4 job has one.
* Placement: every run passes `check_placement.py --planned`; token traces are equal inside each (model, sockets, context, kernels) group.

## 5. C3 and C4: serial hybrid against the concurrent row split, 70B on two sockets

Question: does `LLAMA_HYBRID_SPLIT=rows` (the GEMV reads weights from both tiers at once) beat the layer-serial hybrid when the model fits (C3, 32768 real) and when K/V
spills (C3 at 65536; C4 at 65536, 98304 and 131072)? The 4096 case is P3's job 5936167 (fits: best rows cell -2.3%) and is not repeated.

Cells (settings, all in one job on one node, so every row is an A/B against its serial neighbour):
`serial` is the planner's default hybrid (`LLAMA_HYBRID_WEIGHTS_HBM_LAYERS` and K/V layers auto, packed copy in DDR); `serial-kv80` forces every K/V layer into HBM (weights
spill, 58 of 80 layers in HBM at 65536); `serial-w80` keeps all 80 weight layers in HBM and leaves K/V to the planner; `rows-fX-cN` is the row split with fraction X and N DDR cores;
`rows-auto-cN` has every K/V layer in HBM and the fraction the smallest that fits; `rows-kvmatch-c20` takes the K/V layers of `serial-w80` and the planner's fraction (0.2: everything fits).

### 5.1 Fits in HBM (32768 positions, real K/V, 5 repetitions) (measured)

| setting | decode tok/s median [min-max] | vs serial | DDR share of CAS bytes | HBM / DDR GB per token | modelled ceiling (cal.) | measured / modelled |
|---|---:|---:|---:|---|---:|---:|
| serial (80/80, nothing in DDR) | 7.199 [7.14-7.21] | - | 0.1% | 96.0 / 0.06 | 8.10 | 0.889 |
| rows f 0.1, 8 DDR cores | 7.115 [7.09-7.12] | -1.2% | 7.4% | 89.0 / 7.1 | 7.72 | 0.921 |
| rows f 0.2, 14 DDR cores | 7.140 [7.10-7.15] | -0.8% | 15.1% | 81.5 / 14.5 | 7.70 | 0.927 |

The synthetic fill of the serial cell (3 repetitions) is 7.261 tok/s, +0.9% against the real fill; the c1 hybrid 80/80 cell gives 7.259 (the same placement, another
setting name) and the DDR cell 4.674 against 4.65 real. **No row split beats the all-HBM layout when everything fits, again** (P3: -2.3% at 4096).
The combined read rate falls as HBM cores are given to DDR (P1: 36 + 20 cores deliver 521 GB/s against 569 for 56 + 0), and the 8-14 DDR cores add less than they remove.

### 5.2 K/V spills (65536, 98304, 131072 positions, synthetic K/V, 5 repetitions) (measured)

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
  measured -3.6%. The limits model's best concurrent split (the `t_concurrent` search, `docs/plans/hybrid-limits.md`) is 48 + 8 cores with fraction 0.15 at 98304 (+5.3% over serial) and 40 + 16 cores
  with fraction 0.28 at 131072 (+7.6%) (modelled). **Neither point was measured**: the campaign's row cells have 20 DDR cores and fraction 0.2 or the planner's. That grid is the open
  question; the prediction is a gain of at most 5 to 8%, of the size of the run-to-run effects of P1 (3%).
* **Stop rule.** Every C3/C4 cell is within 30% of its calibrated ceiling (0.87 to 1.02). The serial cells are 0.87 to 0.90 of it (a constant offset: the calibration anchor is
  the 611-position run; the 32768 and 65536 real cells give the same ratio). Before the correction of the concurrent formula (docs/plans/hybrid-limits.md, 27 November), the `rows-auto`
  cells at 98304 and `rows-kvmatch` at 98304 and 131072 read 0.74, 0.77 and 0.82 of the modelled value, which is why the formula was corrected; the old figure charged K/V at the HBM group's rate.
* **The planner's capacity differs from the model's.** The limits model places 80 of 80 layers in HBM at 65536 positions (6.21 tok/s calibrated serial); the engine's planner admits only 56 K/V
  layers (5.29 measured, 0.85), because the BF16 copy of K/V for AMX prefill takes another half of the K/V bytes. The library does not count it (a known simplification; HBM-only 80/80
  is not admitted at 65536 either).

### 5.3 What C3 and C4 do not show

* A grid of DDR cores (8 to 16) and fractions (0.15 to 0.30) at 98304 and 131072 positions: the model's best splits are in it.
* Real fills above 32768 positions (synthetic fills passed the slowest-step guard and equal the real fill within 1% at 32768).
* 4096 positions (P3, cited, not rerun).
