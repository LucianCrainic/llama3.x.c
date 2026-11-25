# Hybrid HBM/DDR limits model: the ceiling every campaign measurement is compared with

**Status: analysis only (25 November 2026, branch `docs/hybrid-limits` from `develop` 59f6bc6). No cluster job was run for this page.** Every
number is **measured** (job id, or a document that gives the jobs), **derived** (formula shown, computed by
[`tools/hybrid_limits.py`](../../tools/hybrid_limits.py)), **modelled** (a prediction of a model whose inputs are listed) or **estimated** (a planning
value). Tables between `hybrid-limits:begin` and `hybrid-limits:end` markers are generated: `python3 tools/hybrid_limits.py` prints them,
`python3 tools/hybrid_limits.py --update-doc docs/plans/hybrid-limits.md` rewrites them, and `tests/test_hybrid_limits.py` (part of `make test`) fails if
the page and the tool disagree. GB is 1e9 bytes, GiB is 2^30; "per socket" is one Xeon Max 9480 (56 cores, 64 GB HBM, about 512 GB DDR5, about 350 W
package limit). Placement is `LLAMA_PLACEMENT=hbm-only|ddr|hybrid` (layer-granular).

## 1. Answer

1. **Decode ceiling, 70B Q8_0 gs64, two sockets, short context.** One socket reads 36.92 GB per token (derived, section 3.1). At the best raw HBM stream
   (569.1 GB/s) that is **15.3 tok/s**, at the DDR stream (262.5 GB/s) **7.1 tok/s**. The engine delivers 430 GB/s per socket from HBM and 231 GB/s from DDR
   (derived from 11.583 and 6.216 tok/s, jobs 5935580 and 5929407): **76% and 88% of the raw rates**. With those two efficiencies the calibrated ceilings are
   11.6 and 6.2 tok/s by construction, and the model then predicts the other measured decode cells within -11% to +9% (section 6).
2. **The older calibrated model (9.4 tok/s HBM, 5.5 DDR at 8192 positions) is reproduced exactly by the tool and was a little pessimistic for HBM on the current
   tree**: it predicts 10.64 tok/s at 611 positions where 11.583 is measured (+8.9%), 10.36 where 10.709 is measured at 2177 (+3.4%), and 10.02 at 4160 where
   10.00 (stagger off) and 10.60 (stagger on) are measured. For DDR it is 2% optimistic. Its K/V rate (200 GB/s) is too slow at depth and too fast at short context
   (section 3.4).
3. **Concurrent reading never beats HBM-only when everything fits** (P1: raw reads at best -1.2%; the engine GEMV +1.4% to +3.9%, an upper bound of 0.3-0.5 tok/s;
   section 4). Concurrency only matters for what does not fit: 70B on one socket (a forced 22-63% of the bytes in DDR, section 3.3) and 70B on two sockets beyond
   about 69k positions of FP32 K/V. There the concurrent ceiling is 1.15-1.3 times the serial hybrid.
4. **TTFT: the compute ceiling is 0.656 s for 2113 tokens and 0.167 s for 547 (2 x 225 TOP/s AMX-INT8, derived), against 8.0-10.2 s and 2.1-2.7 s measured: the
   engine is 12 to 16 times above the ceiling**, and the 8B shows the same factor (9 to 15, section 5.4). So the gap is not 70B-specific. Prefill is
   compute-bound at every rate the engine reaches (the DDR packed-copy read floor, 1.3 s at 2113 tokens, is a sixth of the measured time, 0.42 s at 547 a fifth), which is why DDR and HBM TTFT differ by
   only 3%. The measured-minus-8B-scaled residual is 0.16-0.20 s **per prefill batch** in all eight 70B cells, not per token: that is the first thing to explain
   (section 5.5).
5. **Biggest levers by size, derived:** the in-situ AMX GEMM at 8-14% of peak (TTFT); engine HBM efficiency 0.76 of raw (decode); K/V at long context (70B 2S decode at
   131,072 positions: 3.8 tok/s calibrated serial, 4.5 concurrent, against 5.5 and 6.5 at peak).

## 2. Inputs

### 2.1 Measured rates (per socket)

| quantity | value | label and source |
| --- | ---: | --- |
| HBM-only read, 56 cores, 2 MiB pages | 569.1 [554.2-571.2] GB/s | measured, P1 section 3 (E1 56+0, 7 reps, job 5916195) |
| HBM-only, barrier lockstep, stripes staggered 1 MiB + 4 KiB | 578.3 [562.1-595.6] GB/s | measured, P1 section 3 (second-round jobs, see the P1 validation record) |
| DDR-only read, 56 cores | 262.5 [262.1-262.8] GB/s | measured, P1 section 3 (E1 0+56) |
| DDR-only read, 28 cores (best) | 266.4 [265.3-268.1] GB/s | measured, P1 section 3 |
| DDR5-4800, 8 channels x 8 B x 4.8 GT/s | 307.2 GB/s | derived (platform figure recalled, not in the repository); the stream reaches 85%. No HBM datasheet figure is in the repository, so none is used |
| engine decode GEMV, all-HBM, 56 workers, 4 KiB pages | 447.2 / 459.0 GB/s | measured, P1 section 5 (jobs 5923181, 5924010) |
| same, 2 MiB pages (stripe aliasing) | 411.1 GB/s | measured, P1 section 5.1 (job 5924010); the stagger recovers 447.5 |
| both sockets reading their own HBM (node) | 1127.8 [1102.1-1148.8] GB/s | measured, P1 E4 (0.99 of 2 x 569.1) |
| AMX-INT8 peak, 56 cores | 225 TOP/s | measured micro-benchmark, [hbm-roofline.md](../hbm-roofline.md) section 1 |
| oneDNN INT8 prefill shape, M = 256, local HBM | 69 TOP/s | measured, same source |
| package power, active socket | 348-355 W in every P1 window | measured, P1 section 6 |

The tier rates under concurrency are the P1 E1 table (`E1` in the tool; section 4 quotes it).

### 2.2 Engine measurements used as anchors and checks

| cell | jobs | where |
| --- | --- | --- |
| 70B gs32 and gs64, hybrid 80/80 (weights and K/V in HBM, packed copy in DDR), 547 and 2113 prompt tokens, 128 greedy tokens | 5935580 | [xengine-70b.md](../xengine-70b.md) |
| 70B gs32 DDR (all in DDR) | 5929406 | same |
| 70B gs64 DDR | 5929407 | same |
| 70B 2S gs64 hybrid, warm 4096, stagger off/on, 2 MiB and 4 KiB pages | 5934864 | [gemv-stagger.md](../gemv-stagger.md) |
| 8B gs64 and gs32 decode, 1S and 2S, workload D | 5913494 | [gs32-amx.md](../gs32-amx.md) section 6 |
| 8B DDR decode, 1S and 2S (older tree) | 5892202 | [xengine-results.md](../xengine-results.md) |
| 8B TTFT P and L, gs64 | 5910223 | [gs64-gemm.md](../gs64-gemm.md) |
| 8B TTFT and in-situ GEMM rates, gs32 and gs64 | 5913457 | [gs32-amx.md](../gs32-amx.md) sections 4 and 6 |
| 70B older-tree deep-context decode (8192 to 130,944 positions), load time | 5904621, 5904761, 5904763, 5904765, 5904620 | [hbm-ddr-research.md](../hbm-ddr-research.md), reference numbers of the campaign branch |
| P1 | 5916195, 5916213, 5923181, 5924010, 5924011, 5925935 | [concurrent-tier-bandwidth-validation.md](../concurrent-tier-bandwidth-validation.md) |

The xengine rows ran 55 threads per socket, the stagger job 56.

## 3. Decode ceilings

### 3.1 Model

Per socket and decoded token the engine reads the weights W and the K/V of the C positions in the context. Tensor parallelism splits every layer between the
sockets (heads, hidden dimension, vocabulary rows), so a socket reads half. All of W and all K/V are read once per token, so the order of the bytes does not matter, only
which tier holds them.

* Matmul parameters per layer = rows x columns summed over Wq, Wk, Wv, Wo and the three FFN matrices; the classifier is one more matrix. W = parameters x
  (1 + 4/gs) bytes: 1.0625 at gs64, 1.125 at gs32 (the int8 value plus one FP32 scale per group). Norms (5.3 MB at 70B) and the one embedding row are neglected.
* K/V bytes per position per socket = 2 (K and V) x layers x (KV heads / sockets) x head_dim x bytes per element. INT8 K/V is not in the engine
  ([hbm-capacity-study.md](hbm-capacity-study.md) R6); the table assumes one FP32 scale per (K or V, layer, KV head) row.

<!-- hybrid-limits:begin sizes -->
| model | layers x KV heads x head_dim | matmul params per layer | all layers | classifier | total read per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 70B | 80 x 8 x 128 | 855.638 M | 68.4510 G | 1.0507 G | 69.5017 G |
| 8B | 32 x 8 x 128 | 218.104 M | 6.9793 G | 0.5253 G | 7.5047 G |

| model | group | bytes per weight | sockets | GB per socket per token | GiB per socket per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 70B | gs64 | 1.0625 | 1 | 73.85 | 68.77 |
| 70B | gs64 | 1.0625 | 2 | 36.92 | 34.39 |
| 70B | gs32 | 1.1250 | 1 | 78.19 | 72.82 |
| 70B | gs32 | 1.1250 | 2 | 39.09 | 36.41 |
| 8B | gs64 | 1.0625 | 1 | 7.97 | 7.43 |
| 8B | gs64 | 1.0625 | 2 | 3.99 | 3.71 |
| 8B | gs32 | 1.1250 | 1 | 8.44 | 7.86 |
| 8B | gs32 | 1.1250 | 2 | 4.22 | 3.93 |

| model | sockets | FP32 KiB per token per socket | BF16 | INT8 incl. scales | of which INT8 scales |
| --- | ---: | ---: | ---: | ---: | ---: |
| 70B | 1 | 640.0 | 320.0 | 165.0 | 5.0 |
| 70B | 2 | 320.0 | 160.0 | 82.5 | 2.5 |
| 8B | 1 | 256.0 | 128.0 | 66.0 | 2.0 |
| 8B | 2 | 128.0 | 64.0 | 33.0 | 1.0 |
<!-- hybrid-limits:end sizes -->

Checks: 69.50 G multiply-adds per token for the 70B (the 69.5 G of the brief); 36.92 GB per socket at gs64 (the capacity study's 36.9 GB, 73.85 GB for the node);
320 / 160 / 82.5 KiB of K/V per position per socket at 70B for FP32 / BF16 / INT8 (80 KiB of values plus 2.5 KiB of scales).

Time per token, tiers read one after the other or at the same time, with `BW` the raw P1 rate of the cores streaming a tier and `eta` the engine's efficiency
against it (`eta` = 1 is the **peak** model):

```
t_hbm_only   = (W + KV(C)) / BW_hbm(56 cores)                                     all bytes must fit the HBM budget
t_ddr        = (W + KV(C)) / BW_ddr(56 cores)
t_serial     = B_hbm / BW_hbm(56) + B_ddr / BW_ddr(56)                            layer-granular hybrid, tiers read in turn
t_concurrent = min over P1 splits (c_h + c_d = 56) and weight-row splits of
               max( B_hbm / BW_hbm(c_h), B_ddr / BW_ddr(c_d) )                    rates measured while both groups run
tier time with efficiency   =  W_tier / (eta_w BW) + KV_tier / (eta_a BW)
```

Placement for the serial and concurrent columns: whole layers (the weights and the K/V of the layer together) go to HBM until 56 GiB less 0.5 GiB of work buffers
(estimated) and the classifier are used; the rest is in DDR. The planner's two independent layer knobs can do better by less than one layer of bytes. In the
concurrent model the weight rows may move between tiers (balance point), the K/V stays where the layers put it.

### 3.2 Calibration to the engine

`eta` comes from the measured anchors, one equation per tier: `1 / tok/s = W / R_w + KV(depth) / R_a`, solved for the weight rate `R_w` and divided by the raw rate.
The K/V rate `R_a` is **not** the old 200 / 95 GB/s: it is taken from the two deep-context 70B runs that read a lot of K/V (table below, 285 GB/s HBM and 240 GB/s
DDR). A single rate does not describe K/V reads, because short contexts carry a fixed per-token cost (section 3.4).

<!-- hybrid-limits:begin calibration -->
| configuration | anchor HBM tok/s | anchor DDR tok/s | depth | HBM weight stream GB/s | eta HBM (vs 569.1) | DDR weight stream GB/s | eta DDR (vs 262.5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70B-2S | 11.583 | 6.216 | 611 | 431.2 | 0.758 | 230.7 | 0.879 |
| 8B-1S | 51.070 | 29.800 | 150 | 410.1 | 0.721 | 238.8 | 0.910 |
| 8B-2S | 97.230 | 55.900 | 150 | 390.3 | 0.686 | 223.9 | 0.853 |
<!-- hybrid-limits:end calibration -->

The 70B 1S column of every table below reuses the 70B 2S efficiencies (**estimated**: there is no one-socket 70B decode measurement, because 70B does not fit one
socket's HBM and the cross-socket reduction wait differs). The 8B rows use their own anchors; the 8B DDR anchors come from the older tree (job 5892202).

### 3.3 Anchor: 70B Q8_0 gs64, two sockets, short context

<!-- hybrid-limits:begin anchor -->
Bytes per socket per token at depth 611: weights 36.92 GB, K/V 0.200 GB (FP32). Calibration: HBM weight stream 431.2 GB/s (eta 0.758), DDR weight stream 230.7 GB/s (eta 0.879); K/V read at 285 / 240 GB/s.

| ceiling, 70B gs64, 2S, depth 611 | peak tok/s (raw P1 rates) | engine-calibrated tok/s |
| --- | ---: | ---: |
| HBM only (569.1 GB/s) | 15.33 | 11.58 |
| HBM only, staggered lockstep read (578.3 GB/s) | 15.58 | 11.77 |
| DDR only, 56 cores (262.5 GB/s) | 7.07 | 6.22 |
| DDR only, 28 cores (266.4 GB/s) | 7.18 | 6.31 |
| DDR at the DDR5-4800 datasheet rate (307.2 GB/s, derived) | 8.28 | 7.27 |
| serial 40/80 layers in HBM, 40 in DDR (the pilot split) | 9.73 | 8.13 |
| concurrent, best split with no forced spill | 15.33 | 11.58 |
| concurrent with half of the weights forced to DDR (28 HBM + 28 DDR cores) | 11.90 | 10.46 |
<!-- hybrid-limits:end anchor -->

* **Peak: 15.3 tok/s from HBM, 7.1 from DDR** (the brief's about 15 and about 7). The staggered lockstep read would allow 15.6; the datasheet DDR rate 8.3. No
  measured 70B decode reaches more than 76% of the raw HBM peak or 89% of the raw DDR peak (the 8B DDR decode reaches 91%; section 6).
* **Calibrated: 11.6 and 6.2 tok/s** (anchors; the weight stream is 431 and 231 GB/s).
* **The 40/80 serial split** (the pilot placement, 8.18 tok/s measured on the older tree, [hbm-ddr-research.md](../hbm-ddr-research.md)): 9.73 peak, 8.13 calibrated. The
  calibrated figure is within 0.6% of the pilot, although the pilot ran on a different tree.
* The concurrent ceiling with no forced spill is HBM-only (15.3): in P1 no DDR cores ever helped the raw read. With half of the weights forced to DDR the best split is
  28 + 28 cores (11.90 peak, 10.46 calibrated), **22% below the all-HBM peak 15.33; read in turn the same half costs 37%** (derived: 50% of the bytes at each rate is
  `1 / (0.5/569.1 + 0.5/262.5)` = 359.3 GB/s, 9.68 tok/s at 37.12 GB per token; the 40/80 row above is the same idea at a layer split).

### 3.4 Old calibrated model against the new measurements

The capacity study's model ([hbm-capacity-study.md](hbm-capacity-study.md) section 5) used weight rates 397 (HBM) and 238 (DDR) GB/s and K/V rates 200 and 95 GB/s,
calibrated on the 8B. The tool reproduces its 9.4 and 5.5 tok/s at 8192 positions (and 6.8 at 32768). Against the current-tree measurements:

<!-- hybrid-limits:begin old_vs_new -->
| point | old model HBM | measured HBM | measured / model | old model DDR | measured DDR | measured / model | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor, 547 prompt | 10.64 | 11.58 | 1.089 | 6.36 | 6.22 | 0.977 | jobs 5935580, 5929407 |
| 2113 prompt | 10.36 | 10.71 | 1.034 | 6.15 | 6.01 | 0.978 | jobs 5935580, 5929407 |
| warm 4096 (56 threads, stagger off) | 10.02 | 10.00 | 0.998 | 5.90 | - | - | job 5934864 |
| warm 4096 (stagger on) | 10.02 | 10.60 | 1.058 | 5.90 | - | - | job 5934864 |
| 8192 real, grouped (older tree) | 9.40 | 9.75 | 1.038 | 5.45 | - | - | job 5904761 |
<!-- hybrid-limits:end old_vs_new -->

* The old HBM weight rate (397) is below the 70B engine's current 431 GB/s, so the old model is 3% to 9% below the measurement at 611 to 2177 positions. At 4160 positions it matches the
  stagger-off run (10.02 against 10.00) and is 5.5% low for the stagger-on run. Its DDR prediction is 2% high.
* The 8192-position row is older-tree data: the old model's 9.40 sits between the older tree's split (9.36) and grouped (9.75) kernels.
* K/V reads are not a constant-rate stream:

<!-- hybrid-limits:begin attention_rate -->
| K/V read rate observation (70B 2S, FP32 K/V) | GB/s per socket (or ms) | source |
| --- | ---: | ---: |
| deep, HBM: 73,728 positions, attention 84.84 ms per step (older tree, grouped) | 285 | job 5904763 |
| deep, DDR: 131,072 positions, DDR-only 2.95 tok/s less the weight stream (older tree) | 240 | job 5904765 |
| 4096 positions, HBM: profile attention 13.0 ms per token, stagger on | 103 | job 5934864 |
| 4096 positions, HBM: fixed part if K/V were read at the deep HBM rate (13.0 ms - K/V / deep rate) | 8.3 ms | derived |
| marginal 611 to 2177 positions, HBM hybrid, gs64: dKV / d(1 / tok/s) | 73 | job 5935580 |
| marginal 611 to 2177 positions, HBM hybrid, gs32: dKV / d(1 / tok/s) | 67 | job 5935580 |
| marginal 611 to 2177 positions, DDR, gs64 | 94 | job 5929407 |
| marginal 611 to 2177 positions, DDR, gs32 | 101 | job 5929406 |
<!-- hybrid-limits:end attention_rate -->

  Between 611 and 2177 positions each added K/V byte costs as much as reading it at 67-73 GB/s from HBM (94-101 GB/s from DDR); at 73,728 positions the whole attention
  reads at 285 GB/s. The 4096-position profile (13.0 ms of attention for 1.34 GB) fits a fixed cost of about 8 ms per token plus K/V at the deep rate. The weight rate
  calibrated at 611 positions therefore **includes about 8-9 ms of fixed attention time per token** (about 10% of the step): with it removed the weight stream would be
  about 480 GB/s, which is what the 4096-position profile gives for the weight phases (36.92 GB in 76.9 ms, [gemv-stagger.md](../gemv-stagger.md) section 4, derived). The
  fixed part is the first suspect when a short-context decode falls under the calibrated ceiling (sections 6 and 7).

### 3.5 Decode ceilings against context (FP32 K/V, gs64)

Columns: the four placements, each as **peak** (raw P1 rates, eta = 1) and **calibrated**. "HBM-only" needs everything in the 56 GiB budget (`-` otherwise);
"serial hybrid" is the layer-granular HBM-first split read in turn (equal to HBM-only whenever everything fits); "concurrent" is the best P1 core split. The calibrated
concurrent column assumes the engine's efficiencies carry over to the two core groups; where it is above HBM-only (8B 1S, up to 1.7%) that is an artefact of
`eta_DDR` > `eta_HBM` and is of the size P1 measured for the engine GEMV (+1.4 to +3.9%), not a prediction.

**70B, two sockets** (K/V per socket 1.25, 10, 20, 40 GiB at the four contexts; weights 34.4 GiB):

<!-- hybrid-limits:begin ctx_70b_2s -->
| context | GiB read per socket per token | layers in HBM | DDR byte share | HBM-only peak | DDR peak | serial hybrid peak | concurrent peak | HBM-only cal. | DDR cal. | serial hybrid cal. | concurrent cal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 35.6 | 80/80 | 0.000 | 14.87 | 6.86 | 14.87 | 14.87 | 11.07 | 6.04 | 11.07 | 11.07 |
| 32768 | 44.4 | 80/80 | 0.000 | 11.94 | 5.51 | 11.94 | 11.94 | 8.11 | 4.88 | 8.11 | 8.11 |
| 65536 | 54.4 | 80/80 | 0.000 | 9.75 | 4.50 | 9.75 | 9.75 | 6.21 | 4.01 | 6.21 | 6.21 |
| 131072 | 74.4 | 59/80 | 0.261 | - | 3.29 | 5.46 | 6.52 | - | 2.95 | 3.80 | 4.47 |
<!-- hybrid-limits:end ctx_70b_2s -->

**70B, one socket** (the spill case: 68.8 GiB of weights alone exceed the 56 GiB budget; K/V is 640 KiB per position):

<!-- hybrid-limits:begin ctx_70b_1s -->
| context | GiB read per socket per token | layers in HBM | DDR byte share | HBM-only peak | DDR peak | serial hybrid peak | concurrent peak | HBM-only cal. | DDR cal. | serial hybrid cal. | concurrent cal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 71.3 | 62/80 | 0.222 | - | 3.43 | 5.91 | 7.10 | - | 3.02 | 4.67 | 5.52 |
| 32768 | 88.8 | 49/80 | 0.383 | - | 2.75 | 4.13 | 5.28 | - | 2.44 | 3.24 | 4.01 |
| 65536 | 108.8 | 40/80 | 0.495 | - | 2.25 | 3.09 | 3.80 | - | 2.00 | 2.44 | 3.21 |
| 131072 | 148.8 | 29/80 | 0.633 | - | 1.64 | 2.05 | 2.35 | - | 1.48 | 1.66 | 1.99 |
<!-- hybrid-limits:end ctx_70b_1s -->

**8B, one socket and two sockets** (everything fits at every context: 39.4 GiB at 131072 positions on one socket):

<!-- hybrid-limits:begin ctx_8b_1s -->
| context | GiB read per socket per token | layers in HBM | DDR byte share | HBM-only peak | DDR peak | serial hybrid peak | concurrent peak | HBM-only cal. | DDR cal. | serial hybrid cal. | concurrent cal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 8.4 | 32/32 | 0.000 | 62.90 | 29.01 | 62.90 | 62.90 | 43.08 | 26.41 | 43.08 | 43.80 |
| 32768 | 15.4 | 32/32 | 0.000 | 34.36 | 15.85 | 34.36 | 34.36 | 20.16 | 14.45 | 20.16 | 20.50 |
| 65536 | 23.4 | 32/32 | 0.000 | 22.63 | 10.44 | 22.63 | 22.63 | 12.54 | 9.53 | 12.54 | 12.66 |
| 131072 | 39.4 | 32/32 | 0.000 | 13.44 | 6.20 | 13.44 | 13.44 | 7.14 | 5.66 | 7.14 | 7.14 |
<!-- hybrid-limits:end ctx_8b_1s -->

<!-- hybrid-limits:begin ctx_8b_2s -->
| context | GiB read per socket per token | layers in HBM | DDR byte share | HBM-only peak | DDR peak | serial hybrid peak | concurrent peak | HBM-only cal. | DDR cal. | serial hybrid cal. | concurrent cal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 4.2 | 32/32 | 0.000 | 125.80 | 58.03 | 125.80 | 125.80 | 82.64 | 49.89 | 82.64 | 83.66 |
| 32768 | 7.7 | 32/32 | 0.000 | 68.72 | 31.70 | 68.72 | 68.72 | 39.53 | 28.01 | 39.53 | 40.02 |
| 65536 | 11.7 | 32/32 | 0.000 | 45.25 | 20.87 | 45.25 | 45.25 | 24.76 | 18.66 | 24.76 | 24.94 |
| 131072 | 19.7 | 32/32 | 0.000 | 26.89 | 12.40 | 26.89 | 26.89 | 14.18 | 11.19 | 14.18 | 14.18 |
<!-- hybrid-limits:end ctx_8b_2s -->

Reading:

* **70B 2S is HBM-resident up to about 69k positions**; the ceiling falls from 14.9 (peak) / 11.1 (calibrated) tok/s at 4096 to 9.75 / 6.2 at 65,536 because K/V
  is 37% of the bytes there (20.0 of 54.4 GiB). At 131,072 positions 21 of 80 layers (26% of the bytes) are forced to DDR: serial 5.46 / 3.80, concurrent 6.52 / 4.47, DDR-only 3.29 / 2.95.
  The older-tree measurement at 130,944 positions (4.13, section 6) sits between the calibrated serial and concurrent ceilings.
* **70B 1S**: weights and K/V in the 56 GiB budget cover 62 of 80 layers at 4096 positions and 29 at 131,072. The serial hybrid is 5.9 / 4.7 tok/s at 4096 and
  2.05 / 1.66 at 131,072; **concurrent reading is worth +20%, +28%, +23%, +15% over the serial hybrid at 4096, 32768, 65536, 131072 positions (peak) and +18%, +24%, +32%, +20% (calibrated)** and is the only case
  in this page where the concurrent ceiling is the target for a design. Nothing was ever measured here: 70B 1S is the campaign's unmeasured cell.
* **8B** never spills at these contexts; the table is the context cost alone: 1S 62.9 / 43.1 tok/s (peak / calibrated) at 4096 down to 13.4 / 7.1 at 131,072.

### 3.6 K/V dtype

BF16 K/V halves, INT8 K/V (with scales) quarters the K/V bytes and moves the HBM-resident edge of the 70B on two sockets from 69k (FP32) to 138k (BF16) and 268k positions
(INT8 with scales; 277k without, [hbm-capacity-study.md](hbm-capacity-study.md); derived: (56 GiB - 0.5 GiB - 36.92 GB) / bytes per position). Neither is in the engine; the numbers bound what R6 could give.

<!-- hybrid-limits:begin kv_dtype -->
| context | KV dtype | KV GiB per socket | layers in HBM | HBM-only peak | serial peak | concurrent peak | serial cal. | concurrent cal. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | fp32 | 10.0 | 80/80 | 11.94 | 11.94 | 11.94 | 8.11 | 8.11 |
| 32768 | bf16 | 5.0 | 80/80 | 13.46 | 13.46 | 13.46 | 9.57 | 9.57 |
| 32768 | int8 | 2.6 | 80/80 | 14.34 | 14.34 | 14.34 | 10.49 | 10.49 |
| 65536 | fp32 | 20.0 | 80/80 | 9.75 | 9.75 | 9.75 | 6.21 | 6.21 |
| 65536 | bf16 | 10.0 | 80/80 | 11.94 | 11.94 | 11.94 | 8.11 | 8.11 |
| 65536 | int8 | 5.2 | 80/80 | 13.40 | 13.40 | 13.40 | 9.52 | 9.52 |
| 131072 | fp32 | 40.0 | 59/80 | - | 5.46 | 6.52 | 3.80 | 4.47 |
| 131072 | bf16 | 20.0 | 80/80 | 9.75 | 9.75 | 9.75 | 6.21 | 6.21 |
| 131072 | int8 | 10.3 | 80/80 | 11.86 | 11.86 | 11.86 | 8.03 | 8.03 |
<!-- hybrid-limits:end kv_dtype -->

At 131,072 positions BF16 K/V turns a spilled 5.46 (serial) into an all-HBM 9.75 tok/s peak (+79%); at 65,536 INT8 is worth +37% over FP32.

## 4. Fits but split: can placing part of the model in DDR be faster?

**Condition (concurrent model).** Move c of the 56 cores to a DDR group; `R_h(k)` and `R_d(k)` are the raw rates of k cores on each tier while both groups run. With the bytes split
so both groups finish together it pays iff `R_h(56 - c) + R_d(c) > R_h(56)` for some c. For the raw read the balance gives `B / t = R_h + R_d`, so the
condition is that **the DDR group adds more bandwidth than the HBM group loses by giving up the cores**. Per core the DDR group adds 6-10 GB/s; each core taken from HBM
costs 10-13.5 GB/s, because the remaining HBM cores also slow down (the package is at its 350 W limit).

**P1, raw reads (measured, 7 reps, 2 MiB pages).** Never:

<!-- hybrid-limits:begin fits_split -->
| HBM + DDR cores | HBM GB/s | HBM rate lost GB/s | lost per core moved | DDR GB/s | DDR per core moved | combined GB/s (P1, median of sums) | vs HBM-only 569.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 54 + 2 | 542.1 | 27.0 | 13.5 | 20.3 | 10.2 | 562.4 | -1.2% |
| 52 + 4 | 518.7 | 50.4 | 12.6 | 37.0 | 9.2 | 555.9 | -2.3% |
| 48 + 8 | 479.3 | 89.8 | 11.2 | 75.5 | 9.4 | 554.8 | -2.5% |
| 44 + 12 | 442.3 | 126.8 | 10.6 | 95.4 | 8.0 | 537.1 | -5.6% |
| 40 + 16 | 407.6 | 161.5 | 10.1 | 135.5 | 8.5 | 542.2 | -4.7% |
| 36 + 20 | 363.5 | 205.6 | 10.3 | 156.9 | 7.8 | 521.2 | -8.4% |
| 32 + 24 | 301.1 | 268.0 | 11.2 | 180.4 | 7.5 | 483.4 | -15.1% |
| 28 + 28 | 283.3 | 285.8 | 10.2 | 219.7 | 7.8 | 501.1 | -11.9% |
| 24 + 32 | 210.7 | 358.4 | 11.2 | 224.3 | 7.0 | 434.7 | -23.6% |
| 16 + 40 | 123.7 | 445.4 | 11.1 | 251.4 | 6.3 | 375.1 | -34.1% |
<!-- hybrid-limits:end fits_split -->

The best is 54 + 2 at -1.2% (ranges overlap), nothing is above 569.1 (nor above 578.3). Shared cores (E2) reach at most 561.2 (-1.4%); interleaved pages (E3, E4) show +0.8% to
+3.5% over their own region's all-HBM point, a small, partly replicated effect that stays at or below the best HBM-only figure.

**P1, the engine's decode GEMV (E5, weights only, 4 KiB pages)** is the one place where a split wins:

<!-- hybrid-limits:begin engine_split -->
| E5 GEMV, gs64, 4 KiB pages | all-HBM GB/s | best split GB/s | gain stated in P1 | tok/s-equivalent all-HBM | tok/s-equivalent split | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| node A run 1, r = 0.20 / 14 DDR workers | 447.2 | 453.4 | +1.4% | 12.11 | 12.28 | job 5923181 |
| node A run 2, r = 0.20 / 14 DDR workers | 459.0 | 469.8 | +2.3% | 12.43 | 12.72 | job 5924010 |
| node B, r = 0.20 / 14 DDR workers | 453.3 | 470.9 | +3.9% | 12.28 | 12.75 | job 5924011 |
<!-- hybrid-limits:end engine_split -->

+1.4%, +2.3%, +3.9% (gs32 +0.7% to +2.9%), with 14 DDR workers on 20% of the rows (r = 0.1-0.25 with 6-18 workers is the good region; 12 or 16 workers at r = 0.2 lose 3-5%). That is **0.2-0.5 tok/s of
the weight stream at most** (derived: 12.11 to 12.28, 12.43 to 12.72, 12.28 to 12.75 tok/s-equivalent), before attention, the classifier and reductions, which a split cannot help.
The explanation P1 gives (consistent with the data, not proven) is that the engine kernel reaches only 79% of the raw stream (447 of 569.1 GB/s) so some HBM capacity is idle, and
that the socket is power-limited, not controller-limited: package power is 348-355 W in every window including DDR-only; the core clock falls from 2.28 GHz (DDR-only) to 1.4-1.7 GHz as
soon as cores read HBM; the DDR group draws from the same budget.

**Verdict.**

* If the model fits, **HBM-only is the ceiling**: 15.3 tok/s peak at 611 positions; a split can add at most the E5 gain (about 4%, not robust; p = 0.066 after a Bonferroni
  correction for the best of 11 splits) and the layer-granular serial form pays 2.2 times the time per byte for every byte it puts in DDR (569.1 / 262.5; section 3.3).
* The page-aliasing fix is a larger and more certain lever than the split: the stagger gives +6.0% on the 70B decode (job 5934864, 5 of 5 repetitions) and +5.0% on the 8B 1S (job 5934862).
  With 2 MiB pages the best split is 11% above the aliased 56-worker all-HBM point (P1 section 5.2: 456.8-461.2 against 410.2 GB/s on node A) but within 1.3% of the best all-HBM point, i.e. it is
  mostly the aliasing.
* A split can only be interesting **when something does not fit**: then the concurrent ceiling beats the serial by 1.15-1.3 (section 3.5: 70B 2S at 131,072 positions
  +19% peak, 70B 1S +15% to +28%).
* The decision rule for the campaign: a concurrent engine path needs to beat the all-HBM engine by more than the P1 spread (3%) on a **fitting** cell to be worth building for
  that case; for a spilled cell the target is the concurrent ceiling.

## 5. TTFT ceilings

### 5.1 Operations

Prefill of P tokens does, per token and layer, the matmul multiply-adds of one decode token (68.45 G for the 70B layers; the classifier, 1.05 G, runs on the last row
only) and causal attention. Operations count 2 per multiply-add. Attention from an empty cache is `2 x 2 x (P^2 / 2) x heads x head_dim x layers`.

<!-- hybrid-limits:begin ttft_ops -->
| model | prompt tokens | GEMM TOP | attention TOP | total TOP | attention share | AMX peak 1S s | AMX peak 2S s | oneDNN M=256 rate 2S s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70B | 547 | 74.9 | 0.39 | 75.3 | 0.5% | 0.335 | 0.167 | 0.546 |
| 70B | 2048 | 280.4 | 5.50 | 285.9 | 1.9% | 1.271 | 0.635 | 2.072 |
| 70B | 2113 | 289.3 | 5.85 | 295.1 | 2.0% | 1.312 | 0.656 | 2.139 |
| 70B | 8192 | 1121.5 | 87.96 | 1209.5 | 7.3% | 5.375 | 2.688 | 8.764 |
| 8B | 2048 | 28.6 | 1.10 | 29.7 | 3.7% | 0.132 | 0.066 | 0.215 |
| 8B | 8192 | 114.3 | 17.59 | 131.9 | 13.3% | 0.586 | 0.293 | 0.956 |
<!-- hybrid-limits:end ttft_ops -->

2 x 69.5 G x 2048 = 284.7 TOP (the brief's 285 TOP counts the classifier on every row); the tool gets 280.4 TOP of layer GEMM plus 5.5 TOP of attention plus the last-row classifier =
285.9 TOP, **0.635 s at 2 x 225 TOP/s** (0.656 s for 2113 tokens, 0.167 s for 547). Attention is 2% of the operations at 2048 tokens and 7% at 8192. Peak: the 225 TOP/s micro-benchmark used
56 cores; the xengine rows used 55 (-2%, not applied).

### 5.2 Ceilings with weights or packed copies in DDR, and load time

Every prefill batch of M rows streams the whole packed copy (about the size of the weights, 36.9 GB per socket at gs64, 39.1 at gs32) once. With the xengine prefill batch of 256 that
is 9 passes for 2113 tokens and 3 for 547.

<!-- hybrid-limits:begin ttft_floor -->
| gs | prompt | batch M | batches | packed GB per socket | DDR read floor s | HBM read floor s | AMX peak s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 547 | 256 | 3 | 36.9 | 0.42 | 0.19 | 0.167 |
| 64 | 547 | 512 | 2 | 36.9 | 0.28 | 0.13 | 0.167 |
| 64 | 547 | 1024 | 1 | 36.9 | 0.14 | 0.06 | 0.167 |
| 64 | 2113 | 256 | 9 | 36.9 | 1.27 | 0.58 | 0.656 |
| 64 | 2113 | 512 | 5 | 36.9 | 0.70 | 0.32 | 0.656 |
| 64 | 2113 | 1024 | 3 | 36.9 | 0.42 | 0.19 | 0.656 |
| 32 | 547 | 256 | 3 | 39.1 | 0.45 | 0.21 | 0.167 |
| 32 | 547 | 512 | 2 | 39.1 | 0.30 | 0.14 | 0.167 |
| 32 | 547 | 1024 | 1 | 39.1 | 0.15 | 0.07 | 0.167 |
| 32 | 2113 | 256 | 9 | 39.1 | 1.34 | 0.62 | 0.656 |
| 32 | 2113 | 512 | 5 | 39.1 | 0.74 | 0.34 | 0.656 |
| 32 | 2113 | 1024 | 3 | 39.1 | 0.45 | 0.21 | 0.656 |

| gs | packed copy in | read s per batch | AMX-peak compute ms per batch row | break-even batch rows |
| --- | ---: | ---: | ---: | ---: |
| 64 | DDR | 0.141 | 0.3042 | 462 |
| 64 | HBM | 0.065 | 0.3042 | 213 |
| 32 | DDR | 0.149 | 0.3042 | 490 |
| 32 | HBM | 0.069 | 0.3042 | 226 |
<!-- hybrid-limits:end ttft_floor -->

* With the packed copy in DDR the read floor at M = 256 is **1.27 s (gs64) and 1.34 s (gs32) for 2113 tokens, twice the compute ceiling (0.656 s)**; in HBM it is 0.58 s (below it).
  The two floors cross at **462 (gs64) / 490 (gs32) batch rows from DDR and 213 / 226 from HBM** at peak compute, so at peak rates prefill with the packed copy in DDR is memory-bound
  below about 460-490 rows per batch. The engine default batch (1024, [gs64-gemm.md](../gs64-gemm.md)) is above it; the xengine rows' 256 are not.
* At the rates the engine actually reaches (8-14% of peak) compute time is 7-12 times the peak, far above the floor: **prefill is compute-bound in practice, so the hybrid
  placement (packed copy in DDR) should be close to HBM-packed**. The data cannot verify that: 70B cannot hold the packed copy in HBM, and the two measured 70B tier rows have the
  packed copy in DDR in both (the "HBM" row keeps weights and K/V in HBM, the packed copy in DDR). Their TTFT differs by 3.4% (547) and 3.0% (2113): that is what moving K/V,
  attention and activations from DDR to HBM is worth, nothing more.
* **Ceiling to use for the hybrid campaign:** `max(compute ceiling, n_batches x B_packed / BW_ddr)`: 1.27 s (gs64) and 1.34 s (gs32) at 2113 tokens with M = 256; 0.70 s at M = 512; 0.656 s at M = 1024
  (the read floor from DDR is 0.70 s at M = 512, which is above the compute ceiling, and 0.42 s at 1024).
* **Load time (separate):** the 75.0 GB checkpoint, measured on the older tree (job 5904620), plus the pack on the current tree, which was not measured:

<!-- hybrid-limits:begin load -->
| 70B Q8_0 checkpoint, 75.0 GB | load seconds (measured, job 5904620, older tree) | derived GB/s |
| --- | ---: | ---: |
| first start | 106 | 0.71 |
| later starts | 33-36 | 2.08-2.27 |
<!-- hybrid-limits:end load -->

  A lower bound for building the packed copy is one pass over 36.9-39.1 GB per socket in DDR (0.14-0.15 s at the DDR read rate; the write rate was not measured), negligible against the file read.

### 5.3 Scaling the 8B efficiency to the 70B

The 8B in-situ budget (P = 2048, job 5913457, the same profiled runs as [gs32-amx.md](../gs32-amx.md) section 4) split into the GEMM at the in-situ TOP/s of each matrix, attention, and
the rest (norms, quantizers, RoPE, SwiGLU, K/V stores, reductions and barrier waits = forward time less GEMM less attention):

<!-- hybrid-limits:begin ttft_budget -->
| gs | S | GEMM TOP/s, node, time-weighted | share of AMX peak | GEMM s | attention s | rest s (forward - GEMM - attention) | forward s | AMX peak GEMM s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 32.5 | 14.4% | 0.881 | 0.164 | 0.091 | 1.136 | 0.127 |
| 64 | 2 | 52.2 | 11.6% | 0.547 | 0.080 | 0.076 | 0.703 | 0.064 |
| 32 | 1 | 19.7 | 8.8% | 1.450 | 0.162 | 0.107 | 1.719 | 0.127 |
| 32 | 2 | 38.2 | 8.5% | 0.749 | 0.087 | 0.076 | 0.912 | 0.064 |

| gs | prompt | 70B AMX peak s | GEMM at 8B in-situ rates s | attention x 5.0 s | rest x 5.0 s | total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 547 | 0.167 | 1.43 | 0.03 | 0.10 | 1.56 |
| 64 | 2113 | 0.656 | 5.54 | 0.43 | 0.39 | 6.36 |
| 64 | 8192 | 2.688 | 21.48 | 6.40 | 1.52 | 29.39 |
| 32 | 547 | 0.167 | 1.96 | 0.03 | 0.10 | 2.10 |
| 32 | 2113 | 0.656 | 7.58 | 0.46 | 0.39 | 8.44 |
| 32 | 8192 | 2.688 | 29.40 | 6.96 | 1.52 | 37.88 |
<!-- hybrid-limits:end ttft_budget -->

The second table applies the 8B in-situ rate of each matrix type to the 70B operation counts (assumption: the same efficiency at 70B shapes), multiplies attention and the rest
by 5.0 (heads x head_dim x layers = 655,360 against 131,072; dim x layers the same ratio) and keeps attention quadratic in P. At 8192 prompt tokens attention at 70B scales to 6.4 s of
29.4 s (22%), against 0.43 s of 6.4 s at 2113 (7%): **attention is a first-order TTFT term only for long prompts**.

### 5.4 Gap table: measured TTFT against the ceilings

<!-- hybrid-limits:begin ttft_gap -->
| cell | S | prompt | measured s | AMX peak ceiling s | measured / peak | peak or DDR packed read floor at M=256, s | measured / that | 8B-scaled prediction s | measured / scaled | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70B hybrid 80/80 (packed in DDR), gs64 | 2 | 547 | 2.124 | 0.167 | 12.7 | 0.422 | 5.0 | 1.56 | 1.36 | job 5935580 |
| 70B hybrid 80/80 (packed in DDR), gs64 | 2 | 2113 | 8.009 | 0.656 | 12.2 | 1.266 | 6.3 | 6.36 | 1.26 | job 5935580 |
| 70B hybrid 80/80 (packed in DDR), gs32 | 2 | 547 | 2.598 | 0.167 | 15.5 | 0.447 | 5.8 | 2.10 | 1.24 | job 5935580 |
| 70B hybrid 80/80 (packed in DDR), gs32 | 2 | 2113 | 9.917 | 0.656 | 15.1 | 1.340 | 7.4 | 8.44 | 1.18 | job 5935580 |
| 70B DDR, gs64 | 2 | 547 | 2.154 | 0.167 | 12.9 | 0.422 | 5.1 | 1.56 | 1.38 | job 5929407 |
| 70B DDR, gs64 | 2 | 2113 | 8.065 | 0.656 | 12.3 | 1.266 | 6.4 | 6.36 | 1.27 | job 5929407 |
| 70B DDR, gs32 | 2 | 547 | 2.689 | 0.167 | 16.1 | 0.447 | 6.0 | 2.10 | 1.28 | job 5929406 |
| 70B DDR, gs32 | 2 | 2113 | 10.222 | 0.656 | 15.6 | 1.340 | 7.6 | 8.44 | 1.21 | job 5929406 |
| 8B HBM, gs64 (P 2048) | 1 | 2048 | 1.170 | 0.132 | 8.9 | - | - | - | - | job 5910223 |
| 8B HBM, gs64 (P 2048) | 2 | 2048 | 0.690 | 0.066 | 10.5 | - | - | - | - | job 5910223 |
| 8B HBM, gs64 (L 8192) | 1 | 8192 | 6.079 | 0.586 | 10.4 | - | - | - | - | job 5910223 |
| 8B HBM, gs64 (L 8192) | 2 | 8192 | 3.494 | 0.293 | 11.9 | - | - | - | - | job 5910223 |
| 8B HBM, gs32 (P 2048) | 1 | 2048 | 1.763 | 0.132 | 13.4 | - | - | - | - | job 5913457 |
| 8B HBM, gs32 (P 2048) | 2 | 2048 | 0.988 | 0.066 | 15.0 | - | - | - | - | job 5913457 |
<!-- hybrid-limits:end ttft_gap -->

* 70B measured / peak ceiling: **12.2 to 12.9 (gs64) and 15.1 to 16.1 (gs32)**. The 8B shows 8.9 to 11.9 (gs64) and 13.4 to 15.0 (gs32) against its own peak: the same order, so the gap is
  the engine's AMX efficiency, not something that appears at 70B.
* Measured / 8B-scaled prediction: 1.18 to 1.38; the prediction is a lower bound the 70B does not reach.
* gs32 against gs64 on the 70B: 1.24 (hybrid, 2113 tokens), 1.27 (DDR): the 8B in-situ ratio is 1.41 at 2S, so the 70B gs32 penalty is smaller than the model says.

### 5.5 TTFT diagnosis: candidate causes the data can distinguish

The residual (measured less the 8B-scaled prediction):

<!-- hybrid-limits:begin ttft_residual -->
| cell (2S) | prompt | batches at M=256 | measured s | 8B-scaled s | residual s | residual per batch s | residual per prompt token ms | packed read per batch from DDR s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70B hybrid 80/80 (packed in DDR), gs64 | 547 | 3 | 2.124 | 1.564 | 0.560 | 0.187 | 1.024 | 0.141 |
| 70B hybrid 80/80 (packed in DDR), gs64 | 2113 | 9 | 8.009 | 6.356 | 1.653 | 0.184 | 0.782 | 0.141 |
| 70B hybrid 80/80 (packed in DDR), gs32 | 547 | 3 | 2.598 | 2.096 | 0.502 | 0.167 | 0.919 | 0.149 |
| 70B hybrid 80/80 (packed in DDR), gs32 | 2113 | 9 | 9.917 | 8.438 | 1.479 | 0.164 | 0.700 | 0.149 |
| 70B DDR, gs64 | 547 | 3 | 2.154 | 1.564 | 0.590 | 0.197 | 1.079 | 0.141 |
| 70B DDR, gs64 | 2113 | 9 | 8.065 | 6.356 | 1.709 | 0.190 | 0.809 | 0.141 |
| 70B DDR, gs32 | 547 | 3 | 2.689 | 2.096 | 0.593 | 0.198 | 1.085 | 0.149 |
| 70B DDR, gs32 | 2113 | 9 | 10.222 | 8.438 | 1.784 | 0.198 | 0.844 | 0.149 |
<!-- hybrid-limits:end ttft_residual -->

**The residual does not scale with the prompt: per prompt token it moves from 1.02 to 0.78 ms (-24%) between 547 and 2113 tokens; per prefill batch it is 0.187 and 0.184 s (-2%)
(gs64 hybrid), 0.167 and 0.164 (gs32 hybrid), 0.19-0.20 (DDR rows).** It is about 1.3 times the time of one packed-copy pass from DDR (0.141 s), the same in the "HBM" and the DDR
rows (the packed copy is in DDR in both). This is derived, from two prompt lengths and two group sizes; it is a hypothesis generator, not a result.

| candidate | what the data say now | what would separate it |
| --- | --- | --- |
| **GEMM efficiency** (per-group store and rescale, operands beyond L2, [gs64-gemm.md](../gs64-gemm.md) section 1) | the 8B in-situ rate is 8.5-14% of peak; at 8B rates the 70B GEMM alone is 5.5 s of the 8.0 s (gs64, 2113). The largest term | `tests/gemm_bench.c`/`amx_bench.c` at the 70B shard shapes (N = 14336 or 28672, K = 8192 or 14336) at M = 256 and 1024: the kernel rate with no engine around it |
| **Per-batch cost** (packed-copy pass from DDR not overlapped with compute, per-batch fixed work, partial last batch: 35 and 65 rows) | the residual is per batch, 0.16-0.20 s, 1.2-1.4 packed-copy passes; equal for hybrid and DDR rows | (a) same 70B request at `LLAMA_PREFILL_BATCH` = 256, 512, 1024: a per-batch cost halves with the batch count; (b) on the 8B (fits), `LLAMA_HYBRID_PACKED` = `hbm` against `ddr` at batch 256: a read-bound per-batch cost shrinks when the read gets 2.2 times faster (569.1 / 262.5), a fixed per-batch cost does not; (c) DDR CAS bytes during the prefill (`perf`, x 64 B): 9 x 36.9 GB = 332 GB per socket expected |
| **Batch 256 against the 1024 the 8B was tuned at** | xengine rows ran 256 ([xengine-70b.md](../xengine-70b.md)); the 8B budget is at the engine default; at 256 the weights are passed 9 times for 2113 tokens against 3 times at 1024 | the same sweep as (a), plus the in-situ TOP/s per matrix in the profile |
| **Attention** | 0.43 s (5%) at 2113 tokens, 0.03 s at 547; 6.4 s (22%) at 8192 (scaled). Not the residual at these lengths | profile phases at 70B; an 8192-token cell |
| **Non-AMX phases and barriers** (norms, quantizers, RoPE, SwiGLU, K/V stores, Wo/W2 reductions) | 11% of the 8B forward (0.076 s of 0.703, 2S); scaled x 5 it is 0.39 s (5%); Wo at 2S moves least with the GEMM order (reduction-bound, [gs64-gemm.md](../gs64-gemm.md) 3.3) | the profile's per-phase times and barrier-wait counters at 70B, 2S against 1S (a one-socket 70B prefill cannot hold the model: use the 8B 1S/2S difference scaled) |
| **Clock and power under AMX** | package power 339-366 W per active socket in the 8B cross-engine runs (prefill and decode, [xengine-results.md](../xengine-results.md)); the 225 TOP/s peak is a micro-benchmark | core clock during prefill from `cycles/ref-cycles` (P1 method) against the micro-benchmark's |
| **Slow mode** ([bimodality.md](../bimodality.md)) | the three-repeat ranges of the 70B rows are within 0.5% ([2.686-2.693] at 547), so no slow mode is visible | repeat counts in the next job |
| **gs32 epilogue** (two tile stores per 64 K) | explains the gs32 / gs64 gap on the 8B (1.41 at 2S); the 70B ratio is 1.24-1.27 | in-situ rates per matrix at gs32 and gs64 on the 70B shapes |

Smallest set of measurements that separates most of these: one 70B 2S job at P = 2113 with the phase profile on and `perf` DDR/HBM CAS counters, at batch 256 and 1024 (two cells, gs64,
hybrid 80/80), and one 8B 2S pair with `LLAMA_HYBRID_PACKED` = `hbm` and `ddr` at batch 256.

## 6. Gap table: measured decode against the ceilings

Each row: measured tok/s, the bytes per socket it delivers (weights plus K/V at the mean depth, times tok/s), the peak ceiling, the calibrated ceiling, and the ratios. Depth is
prompt + 64 (mean over the 127 steps) for the xengine rows, 4096 + 64 for the stagger rows, 150 for workload D, the quoted position count for the older-tree rows.
**The calibrated ratio is 1.000 at the anchor rows by construction** (70B 547, DDR 547, 8B D), and the two deep-context rate derivations make the 73,728 (HBM) and 131,072 (DDR) rows
close by construction too; the other rows are checks.

<!-- hybrid-limits:begin decode_gap -->
| cell | tier | depth | measured tok/s | delivered GB/s per socket | peak ceiling tok/s | measured / peak | calibrated ceiling tok/s | measured / calibrated | source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 70B hybrid 80/80, gs64, 547 prompt (anchor) | hbm | 611 | 11.58 | 430 | 15.33 | 0.756 | 11.58 | 1.000 | job 5935580 |
| 70B hybrid 80/80, gs64, 2113 prompt | hbm | 2177 | 10.71 | 403 | 15.12 | 0.708 | 11.35 | 0.944 | job 5935580 |
| 70B hybrid 80/80, gs32, 547 prompt | hbm | 611 | 10.97 | 431 | 14.48 | 0.757 | 10.94 | 1.002 | job 5935580 |
| 70B hybrid 80/80, gs32, 2113 prompt | hbm | 2177 | 10.12 | 403 | 14.30 | 0.708 | 10.73 | 0.943 | job 5935580 |
| 70B DDR, gs64, 547 prompt (anchor) | ddr | 611 | 6.22 | 231 | 7.07 | 0.879 | 6.22 | 1.000 | job 5929407 |
| 70B DDR, gs64, 2113 prompt | ddr | 2177 | 6.01 | 226 | 6.97 | 0.862 | 6.13 | 0.980 | job 5929407 |
| 70B DDR, gs32, 547 prompt | ddr | 611 | 5.93 | 233 | 6.68 | 0.887 | 5.87 | 1.009 | job 5929406 |
| 70B DDR, gs32, 2113 prompt | ddr | 2177 | 5.75 | 229 | 6.59 | 0.873 | 5.80 | 0.992 | job 5929406 |
| 70B hybrid, gs64, warm 4096, stagger off (56 threads) | hbm | 4160 | 10.00 | 383 | 14.86 | 0.673 | 11.06 | 0.904 | job 5934864 |
| 70B hybrid, gs64, warm 4096, stagger on | hbm | 4160 | 10.60 | 406 | 14.86 | 0.713 | 11.06 | 0.958 | job 5934864 |
| 70B hybrid, gs64, warm 4096, 4 KiB pages | hbm | 4160 | 10.49 | 402 | 14.86 | 0.706 | 11.06 | 0.948 | job 5934864 |
| 70B weight-first, 8192 real, grouped (older tree) | hybrid | 8192 | 9.75 | 386 | 14.37 | 0.679 | 10.52 | 0.927 | job 5904761 |
| 70B weight-first, 73,728 real, grouped (older tree) | hybrid | 73728 | 5.72 | 349 | 9.05 | 0.632 | 5.79 | 0.988 | job 5904763 |
| 70B weight-first, 130,944 primed, grouped (older tree) | hybrid | 131072 | 4.13 | 330 | 5.46 | 0.756 | 3.80 | 1.086 | job 5904765 |
| 70B DDR only, 130,944 primed, grouped (older tree) | ddr | 131072 | 2.95 | 236 | 3.29 | 0.898 | 2.95 | 1.000 | job 5904765 |
| 8B 1S, gs64, D (86 + 128) | hbm | 150 | 51.07 | 409 | 71.02 | 0.719 | 51.07 | 1.000 | job 5913494 |
| 8B 2S, gs64, D | hbm | 150 | 97.23 | 390 | 142.04 | 0.685 | 97.23 | 1.000 | job 5913494 |
| 8B 1S, gs32, D | hbm | 150 | 48.04 | 407 | 67.09 | 0.716 | 48.25 | 0.996 | job 5913494 |
| 8B 2S, gs32, D | hbm | 150 | 91.16 | 387 | 134.19 | 0.679 | 91.86 | 0.992 | job 5913494 |
| 8B 1S DDR, gs64, D (older tree) | ddr | 150 | 29.80 | 239 | 32.76 | 0.910 | 29.80 | 1.000 | job 5892202 |
| 8B 2S DDR, gs64, D (older tree) | ddr | 150 | 55.90 | 224 | 65.52 | 0.853 | 55.90 | 1.000 | job 5892202 |
| 8B 1S, gs64, warm 4096, stagger off | hbm | 4160 | 42.21 | 383 | 62.79 | 0.672 | 42.97 | 0.982 | job 5934862 |
| 8B 1S, gs64, warm 4096, stagger on | hbm | 4160 | 44.31 | 402 | 62.79 | 0.706 | 42.97 | 1.031 | job 5934862 |
| 8B 2S, gs64, warm 4096, stagger off | hbm | 4160 | 73.53 | 333 | 125.57 | 0.586 | 82.44 | 0.892 | job 5934863 |
| 8B 2S, gs64, warm 4096, stagger on | hbm | 4160 | 74.95 | 340 | 125.57 | 0.597 | 82.44 | 0.909 | job 5934863 |

| P1 E5 (weights only, no attention, no classifier) | GB/s | against raw 569.1 | tok/s-equivalent at 36.92 GB | source |
| --- | ---: | ---: | ---: | ---: |
| E5 GEMV, 56 workers, 4 KiB pages, run 1 / run 2 (all-HBM) | 447.2 / 459.0 | 0.786 | 12.11 | jobs 5923181, 5924010 |
| E5 GEMV, 56 workers, 2 MiB pages, aliased | 411.1 | 0.722 | 11.13 | job 5924010 |
| E5 GEMV, 56 workers, 2 MiB, 4 KiB stagger | 447.5 | 0.786 | 12.12 | job 5924010 |
| E5 GEMV best row split, node A run 2 (r = 0.20, 14 DDR workers) | 469.8 | 0.826 | 12.72 | job 5924010 |
| E5 GEMV best row split, node B (r = 0.20, 14 DDR workers) | 470.9 | 0.827 | 12.75 | job 5924011 |
<!-- hybrid-limits:end decode_gap -->

* **Short context (anchors and gs32, 547 and 2113 prompt tokens):** 0.71-0.76 of the raw HBM peak, 0.86-0.89 of the raw DDR peak. gs32 reaches the same bytes per second as gs64 (431 against
  430 GB/s), so its lower tok/s is the 1.125 against 1.0625 bytes per weight, not a slower kernel (the calibrated ceiling with the gs64 efficiencies predicts the gs32 rows within +0.9% and -5.7%).
* **2113 prompt tokens** (calibrated 11.35): the measurement is 5.6% below. That is the fixed attention cost of section 3.4 (K/V at 67-73 GB/s marginal).
* **Warm 4096:** stagger off 0.904, stagger on 0.958, 4 KiB pages 0.948 of the calibrated ceiling; the 56-thread aliased run is 9.6% below, which is what the stagger repairs (the stagger-on run is 4.2%
  below). The 8B 2S warm-4096 rows are 9-11% below (not explained here).
* **Older-tree deep rows** (grouped kernel): 0.99 (73,728), 1.09 (130,944 weight-first, an out-of-sample check) and 1.00 (DDR only). At 8192 the older tree measured 9.75 (grouped), 0.93 of calibrated.
* **E5 GEMV (weights only):** 447-459 GB/s is 0.79-0.81 of the raw HBM peak; the engine's whole decode (430 GB/s, attention and classifier included) reaches 0.96 of the 4 KiB-page GEMV rate and
  exceeds the 2 MiB aliased GEMV (411 GB/s); consistent with the xengine rows' 55 workers, which avoid the 56-worker alias (P1: 53-55 workers 437-446 GB/s).
* **Hybrid 40/80** (older tree, 8.18 tok/s): ceiling 8.13 calibrated, 9.73 peak (section 3.3).

## 7. Uncertainties and what to measure first

**What this page does not know.**

1. The calibrated model puts all short-context overhead in the weight rate; the two K/V rates come from older-tree grouped-kernel runs at 73,728 and 131,072 positions. The mid range (8192 to
   65,536) on the current tree is unmeasured; the old model's 8192 match is on the older tree.
2. 70B on one socket was never measured (any placement). Its calibrated column reuses the two-socket efficiencies, which include the cross-socket reduction wait.
3. The concurrent ceilings use P1's raw per-tier rates under concurrency; the engine's own kernels reached +1.4 to +3.9% for the GEMV only (4 KiB pages), and no engine implements a concurrent decode.
4. Layer granularity is approximated by joint layers; INT8 K/V scales are an assumption; BF16 and INT8 K/V do not exist in the engine.
5. TTFT scaling to the 70B assumes the 8B in-situ rate per matrix type holds at 70B shapes; the batch differs (256 against the default); the 55-against-56-thread peak is not applied.
6. HBM datasheet bandwidth is not in the repository and was not used; the DDR5-4800 figure is from memory of the platform.
7. P1's power-cap reading is consistent with the data and unproven (no frequency or power control with this account).
8. The per-batch residual is a pattern in eight cells that come from two prompt lengths, not a measurement of a mechanism.

**Measure first (one job each; none was run for this page).**

1. **70B 2S TTFT, P = 2113, gs64, hybrid 80/80**, phase profile and `perf` CAS counters, `LLAMA_PREFILL_BATCH` = 256 and 1024 (same node, 3 repeats): separates the per-batch cost from GEMM
   efficiency and the packed-copy read (section 5.5); one 8B 2S pair with the packed copy in HBM and DDR (cheap, fits).
2. **70B 1S hybrid decode** (weights 62 of 80 layers in HBM, `LLAMA_HYBRID_WEIGHTS_HBM_LAYERS` = 62, rest in DDR) at 4096 positions: the model says 5.9 peak / 4.7 calibrated serial; it is the case the
   campaign exists for and has no number.
3. **70B 2S decode ladder on the current tree** with the stagger on at 8192, 32768, 65536 (all-HBM, FP32 K/V, strict placement): tests the K/V rate model where the old and new calibrations disagree
   most, and gives the context cost of the weight stream.

## 8. Reproduction

```
python3 tools/hybrid_limits.py                 # every table of this page
python3 tools/hybrid_limits.py anchor ctx_70b_2s
python3 tools/hybrid_limits.py --check-doc docs/plans/hybrid-limits.md
python3 tests/test_hybrid_limits.py            # also run by make test
```

The tool holds the constants with their sources (`E1`, `HBM_STREAM`, `DDR_STREAM`, `ANCHOR_*`, `DECODE_CELLS`, `TTFT_CELLS`, `INSITU_8B`), so changing a measurement is one edit and a regenerate.
