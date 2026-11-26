# Concurrent HBM and DDR streaming of the decode weights: validation record

Companion of [concurrent-tiers.md](concurrent-tiers.md). Jobs, commits, medians and gates. No host names or cluster paths are given:
a "results directory" is the per-job directory the job script creates (named `p3-20261125-<name>`), "the tree" a staged copy of the
tracked files of the commit named (a `COMMIT` file holds the full hash). Every job ran `scripts/concurrent-tiers.sbatch`
exclusively on one Flat node of the HBM partition (the same node for every job). Labels: **measured** (a job), **derived**
(arithmetic on measured values), **modelled** (a model, not run).

## 1. Result in four lines

* **Fits in HBM (8B 1S, 8B 2S, 70B 2S):** no row split beats the all-HBM reference. The best cell of each grid is -1.1%, -1.2% and
  -2.3% (measured; sections 4.1 to 4.3). The P1 point (r = 0.2, 14 cores) is -2.6%, -2.7% and -3.0%.
* **Spill, 70B on one socket (weights do not fit):** the row split is +10.1% (14 DDR cores) and +11.1% (20 cores) over the serial
  layer hybrid, 5 of 5 repetitions each (measured; section 4.4).
* **Spill by the KV cache, 70B 2S, 64K warm context, synthetic K/V:** +5.5% over its own serial control (same KV placement),
  +1.7% over the planner's default (inside the run spread), -5.1% / -8.5% with 14 cores (measured; section 4.5).
* **Correctness:** logits sha256 and token traces are equal to the unsplit run in every check (section 3); `make test` rc 0 locally
  and on the cluster.

## 2. Jobs and commits

Commits of the branch (from `develop` 59f6bc6): a313020 feature and tests; e51f02e job script and summary tool; 58cbbc7 test fix
(division by zero on x86 in the test, not the engine); 357d58f script fix (the engine stops at `step_limit`); 40f9537 and 0cbfe45
script changes (the `serial-kv80` control cell, a note that lists must not go through `--export`); the commit of this document.
The engine code (`src/`, `include/`) of every benchmark job is that of a313020 (no engine change after it).

| job | stage | tree | elapsed | result |
|---|---|---|---|---|
| 5935734 | check | e51f02e | 0:46:46 | logits and traces passed; `make test` rc 2 from the test division by zero (fixed in 58cbbc7); state FAILED |
| 5935771 | check | 357d58f | 1:03:41 | `make test` rc 0; logits 1 distinct digest in each of 8B 1S/2S x decode/batched prefill; traces 8B 1S, 8B 2S, 70B 2S: 3 each, 0 differ |
| 5935769 | bench 8B 1S | 58cbbc7 | 0:00:31 | failed by script error (the run check rejected the engine's `step_limit` stop); no data |
| 5935827 | bench 8B 1S fits | 357d58f | 0:20:11 | 18 cells x 5; 108 traces, 0 differ |
| 5936152 | bench 8B 2S fits | 357d58f | 0:22:58 | 18 cells x 5; 108 traces, 0 differ |
| 5936167 | bench 70B 2S fits | 357d58f | 2:08:47 | 17 cells x 5; 102 traces, 0 differ |
| 5935860 | bench 70B 1S spill | 357d58f | 0:41:41 | 6 cells x 5; 36 traces, 0 differ |
| 5949316 | bench 70B 2S long | a09c63f | 0:05:48 | superseded: `sbatch --export` splits its argument on commas, so only the `serial` cell ran (a09c63f was reverted in 40f9537) |
| 5949331 | bench 70B 2S long | 0cbfe45 | 0:28:44 | 4 cells x 5; 24 traces, 0 differ; every placement check PASS |

Node time (derived, sum of the elapsed times): about 6.0 node-hours of the 12 allowed, one job at a time on one node, nothing
cancelled. Results directories: `p3-20261125-check`, `-check2`, `-fits-8b-1s` (job 5935769, empty), `-fits-8b-1s-b`, `-fits-8b-2s`,
`-fits-70b-2s`, `-spill-70b-1s`, `-spill-70b-2s-long` (job 5949316), `-spill-70b-2s-long2` (job 5949331).

Setup of every benchmark cell: `LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384
LLAMA_KV_VERIFY=1 LLAMA_GEMV_STAGGER=1`, explicit topology, engine defaults otherwise (AMX prefill, the VNNI decode GEMV, 2 MiB
pages as the system chooses), one socket under `numactl --cpunodebind=0`. Every `LLAMA_*` variable of the caller is cleared before a
cell. Command: `--benchmark --chat-template --no-text -t 0 -n 0` with the prime-number prompt of the earlier campaigns, warm context
4096 (8B and 70B fits and 1S spill) with generation length 256 (8B) or 96 (70B) forward passes, and context 98304, warm 65536,
generation 64 with `LLAMA_SYNTHETIC_KV=1` (a `SYNTHETIC_KV=1` build, prime 1024) for the long-context cells. Five repetitions
interleaved (repetition k starts at cell k mod the number of cells), then one `LLAMA_PROFILE=1` run per cell. The uncore counters
(`perf stat -a --per-socket -I 100`, 32 bytes per HBM CAS, 64 per DDR CAS) are cut to the decode window with the report's
`phase_epoch_ns` (`tools/hbm_ddr_results.py`); GB per token below are decode-window reads summed over the sockets.
Every run passes `tools/check_placement.py --planned --allow-sampled` (inventory equals the plan, every arena on its node, the DDR
weights arena included). Weight cells: `hbm-only` is `LLAMA_PLACEMENT=hbm-only`; `hyb-all-hbm` is the 80/80 (32/32 for 8B) hybrid
with the classifier in HBM and the packed copy in DDR; `rows-r<f>-c<n>` adds `LLAMA_HYBRID_SPLIT=rows` with fraction f and n DDR
cores.

## 3. Bitwise checks (jobs 5935734 and 5935771)

* **Unit tests** (`make test`, local and job 5935771): `test_gemv_rows` (`tests/kernels.c`): for gs 32 and 64, W8A8 and W8A32,
  d in {32, 64, 100, 257}, every split in {0, 32, d/2, d-32, d}, stagger on and off, `k_matmul_q8_rows`, `_add_rows`, `_qkv_rows`,
  `_swiglu_rows` and the per-member entry for 1 to 9 members and 0 to 4 DDR members are `memcmp`-equal to the unsplit kernels, every
  row reported done once; the shard tests (`tests/shards.c`) rebuild every row of every split shard from the file, check the arena
  arithmetic against an independent recomputation and the two-piece packing against packing the whole; `tests/test_tp.py` runs the
  decode and batched prefill probes (scalar and AMX tile model, gs 64, gs 32, per-channel, one and two sockets, fractions 0.25 to
  0.9 and `auto`) bitwise against hbm-only, the planner arithmetic and the knob refusals. A deliberately wrong row offset made them fail.
* **Logits (measured, job 5935771; also 5935734):** `probe-int8` prints the raw float32 logits of a fixed 97-token input; the sha256
  was identical across `hbm-only`, `hyb-all-hbm`, `rows-r0.1-c20`, `rows-r0.2-c14`, `rows-r0.3-c8` for 8B one socket and two sockets,
  in the per-position (decode) mode and in the batched prefill mode with all-position logits (4 sets, 1 distinct digest each).
* **Token traces (measured):** job 5935771: 8B 1S and 2S (`hbm-only`, `rows-r0.2-c14`, `rows-r0.2-c20`) and 70B 2S
  (`hyb-all-hbm`, same two row splits): 3 traces each, 0 differ. Every benchmark job compares all its traces (all cells, all
  repetitions, profiled runs): 108, 108, 102, 36 and 24 traces, 0 differ (the serial and row-split cells of the spill jobs included).
* **Gates:** local `make test` rc 0 (macOS, software AVX-512 and AMX models); cluster `make test` rc 0 (job 5935771, under
  `numactl --cpunodebind=0 --membind=0` with the engine knobs unset).

## 4. Throughput (measured; decode tokens per second, median [min-max] of 5)

### 4.1 8B one socket, fits (job 5935827; reference `hbm-only` 44.05 [44.01-44.11])

| cell | tok/s | vs hbm-only | HBM / DDR GB per token |
|---|---|---|---|
| hyb-all-hbm | 43.99 | -0.1% | 9.0 / 0.0 |
| ddr | 24.98 | -43.3% | 0.0 / 9.1 |
| rows r0.15 c8 (best) | 43.55 | -1.1% | 7.9 / 1.1 |
| rows r0.2 c14 (P1 point) | 42.90 | -2.6% | 7.6 / 1.5 |

Grid, change against hbm-only (rows: fraction; columns: DDR cores 8 / 14 / 20): 0.1: -1.9 / -8.9 / -16.7; 0.15: -1.1 / -4.8 / -12.8;
0.2: -12.2 / -2.6 / -9.6; 0.25: -22.7 / -2.7 / -6.0; 0.3: -29.9 / -8.7 / -4.5 (percent).

### 4.2 8B two sockets, fits (job 5936152; reference `hbm-only` 74.28 [73.37-76.91], `hyb-all-hbm` 74.40)

Best cell `rows-r0.15-c8` 73.49 (-1.1% vs hbm-only, -1.2% vs `hyb-all-hbm`); `rows-r0.2-c14` 72.25 (-2.7%); `ddr` 44.75 (-39.8%).
Grid: 0.1: -2.2 / -9.3 / -15.8; 0.15: -1.1 / -7.2 / -13.9; 0.2: -10.6 / -2.7 / -10.6; 0.25: -20.0 / -3.0 / -6.1; 0.3: -26.3 / -7.3 / -2.7.

### 4.3 70B two sockets, fits (job 5936167; reference `hyb-all-hbm` 10.51 [10.44-10.56] tok/s, 76.8 GB per token from HBM)

`ddr` 5.88 (-44.0%). Best cell `rows-r0.1-c8` 10.27 (-2.3%); `rows-r0.2-c14` 10.20 (-3.0%). Grid: 0.1: -2.3 / -8.6 / -15.5;
0.15: -3.3 / -5.5 / -11.9; 0.2: -16.0 / -3.0 / -9.0; 0.25: -26.4 / -5.0 / -6.3; 0.3: -33.9 / -11.7 / -5.2. The bytes show the
split is real: `rows-r0.2-c14` reads 62.2 GB from HBM and 14.5 GB from DDR per token (18.9% DDR share), the total 76.7 GB equal to the
reference's.

### 4.4 70B one socket, weights spill (job 5935860; context 4352, warm 4096; reference `serial`)

`serial` is the layer hybrid with `LLAMA_HYBRID_WEIGHTS_HBM_LAYERS` auto (59 of 80 weight layers in HBM) and every KV layer in HBM.
The rows cells use the auto fraction (0.254, the smallest that fits the 56 GiB budget; all 80 layers, classifier in HBM).

| cell | tok/s | vs serial | faster in | HBM / DDR GB per token | DDR share |
|---|---|---|---|---|---|
| serial | 4.83 [4.81-4.84] | - | - | 57.6 / 19.2 | 25.0% |
| rows-auto-c14 | 5.32 [5.27-5.33] | +10.1% | 5 of 5 | 58.2 / 18.5 | 24.1% |
| rows-auto-c20 | 5.37 [5.01-5.39] | +11.1% | 5 of 5 | 58.2 / 18.5 | 24.1% |
| rows-auto-c28 | 4.61 [4.60-4.63] | -4.7% | 0 of 5 | 58.3 / 18.5 | 24.1% |
| rows-f0.35-c20 | 5.33 [5.33-5.34] | +10.3% | 5 of 5 | 51.4 / 25.4 | 33.1% |
| ddr | 3.17 [3.17-3.17] | -34.4% | 0 of 5 | 0.1 / 76.7 | 99.9% |

The rows cells move almost the same bytes as the serial one (the DDR share is the same 24 to 25%); the gain is time: both tiers
stream during every GEMV instead of one after the other.

**Against the bandwidth model (modelled vs measured).** The modelled throughput of the P1 study for this shard
(docs/concurrent-tier-bandwidth.md: serial reads each tier's bytes at that tier's rate, concurrent both at once)
is 4.67 serial and 5.52 concurrent; measured 4.834 (+3.5% against the model) and 5.370 (-2.7% against the model). The concurrent
split reaches 97% of the model's concurrent rate and recovers +11.1% of the +18% the model gives.

### 4.5 70B two sockets, spill by the K/V cache (job 5949331; context 98304, warm 65536, synthetic K/V; reference `serial`)

`serial` is the planner's default hybrid (weights 80 layers in HBM, K/V 37 layers in HBM, the rest in DDR). `serial-kv80` is the
control: K/V of all 80 layers in HBM, so the weights spill (23 layers in HBM) and are read serially. The rows cells keep K/V in HBM
and spill weights by the auto fraction (0.708).

| cell | tok/s median [5 repetitions in order] | vs serial | vs serial-kv80 |
|---|---|---|---|
| serial | 2.458 [2.458 2.512 2.400 2.510 2.446] | - | +3.8% |
| serial-kv80 | 2.369 [2.536 2.401 2.369 2.346 2.311] | -3.6% | - |
| rows-auto-c14 | 2.249 [2.452 2.249 2.166 2.554 2.139] | -8.5% | -5.1% |
| rows-auto-c20 | 2.500 [2.359 2.416 2.500 2.829 2.635] | +1.7% | +5.5% |

Reading: against its own serial control (the same K/V placement) concurrency gains about 5% (c20); against the planner's
default, which spills the K/V instead of the weights, it gains +1.7%, inside the spread of the runs (2.14 to 2.83), and with 14
cores it loses. At this context attention is about 35 to 40% of the token (profile: 128 to 155 ms of 338 to 473 ms) and is not changed.
The decode window of this job (64 tokens) is short, and the first attempt (job 5949316, `serial` only) showed a 33% spread
between repetitions (2.40, 2.60, 3.19) with the same plan and pages on every run; the cause of that drift was not found.

## 5. Where the optimum sits against P1

P1 (E5, r = 0.2, 14 of 56 workers on the DDR rows) gave +1.4 to +3.9% on the engine-shaped GEMV alone. In the whole engine, with
the same kernel, fraction and core count, the measured effect is -2.6% (8B 1S), -2.7% (8B 2S) and -3.0% (70B 2S): the GEMV gain does
not survive the rest of the token (attention, the classifier, the reductions and the Q/K/V and Wo/W2 joins; a likely cause, not tested here, is that the DDR-serving
members finish later than the HBM ones). The optimum of the fits grids is at the low end of both parameters (r 0.1 to 0.15 with 8
cores, -1.1 to -2.3%), the direction in which the split moves the least. With a spill the fraction is forced by the budget (0.254
on 70B 1S) and the best core count is 14 to 20 of 56 (c20 +11.1%, c14 +10.1%); 28 cores lose (-4.7%).

## 6. When to use it

Opt-in only, `LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_SPLIT=rows` with `LLAMA_HYBRID_DDR_FRACTION=auto`, and only when part of the
weights would otherwise be read from DDR serially (a Q8 model that does not fit one socket's HBM budget, as 70B on one socket:
+10 to +11% measured). Where the model fits in HBM do not split: every measured fit cell was at or below the all-HBM rate, so the
auto balance of 0.2 for the fits case (a constant from P1) is not a recommendation and should not be used for speed. With 14 to
20 DDR cores the split was never worse than serial in the 1S spill; use 14 as the default, 20 as measured best. For a spill caused
by the K/V cache the gain is about 5% over the same-placement control and none over the planner's default: keep the default.

## 7. Limits

One node, one model family, short decode windows for the long-context cells, a synthetic K/V fill (decode biased up to about 3% low;
synthetic is compared only with synthetic). The core count and the fraction were scanned on a coarse grid. Prefill with a row-split
Wo/W2 uses the barrier reduction and was not measured. The auto fraction constant 0.2 is the P1 value, not a tuned one. The
cause of the repetition drift of the long-context `serial` cell is unknown.
