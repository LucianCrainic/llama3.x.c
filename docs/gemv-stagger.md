# Staggered Q8 decode GEMV (`LLAMA_GEMV_STAGGER`)

Branch `perf/gemv-stagger`. Jobs, commits, files and the gates are in [gemv-stagger-validation.md](gemv-stagger-validation.md). Numbers are labelled measured, derived or
modelled. Decode speeds are the report's `throughput_tokens_per_second.decode` (tok/s) over 128 (70B) or 256 (8B) generated tokens after a 4096-token warm context
(`--warm-context 4096`), gs64 W8A8, median [min-max] of 5 interleaved repetitions.

## 1. Answer

* **The 2 MiB stripe-aliasing loss of [concurrent-tier-bandwidth.md](concurrent-tier-bandwidth.md) section 5.1 exists inside the engine's decode** (measured). On the
  benchmark node the system's transparent huge pages are `[always]`, so the engine's default weights are already on 2 MiB pages (the report's `thp_bytes` is the whole weight
  region with `LLAMA_THP` unset; this is the finding of [numa-placement-validation.md](numa-placement-validation.md) again). With the weights forced onto 4 KiB pages
  (`LLAMA_THP=never`, added here) 8B gs64 decode on one socket is 4.0% faster than the default (43.88 against 42.21 tok/s), and the W1/W3 phase is 10% shorter.
* **`LLAMA_GEMV_STAGGER=1` recovers it without changing a bit of the output** (measured, bitwise on 8B one and two sockets, 70B token traces). On 2 MiB pages: 8B one socket
  +5.0%, 8B two sockets +1.9%, 70B two sockets +6.0%. On 4 KiB pages it does nothing (-0.2%, -0.5%), as the aliasing explanation predicts.
* The knob is opt-in (default `0`). The recommendation is in section 7.

## 2. Mechanism and what was tested

The static team partition (`q8_team_static`) gives worker w the rows `[d*w/workers, d*(w+1)/workers)` of every segment. Where a worker's share is a multiple of 2 MiB (or
the starts fall into a few classes modulo 2 MiB), all workers stream, in barrier lockstep, addresses that are equal modulo 2 MiB. On 2 MiB pages the physical addresses
inherit that relation; the loss in the microbenchmark (8-10% for the GEMV, 27-34% for a raw lockstep read) disappears when the workers are shifted against each other by
4 KiB or more and does not occur with 4 KiB pages. The mechanism (the HBM channel and bank interleave) is inferred from where the loss appears and disappears; it is not
proven here either. Three engine observations fit it:

1. THP `[always]`: default (2 MiB) 42.21 [41.99-42.41] tok/s against 4 KiB pages (`LLAMA_THP=never`) 43.88 [43.32-44.05] on 8B one socket.
2. The knob on, 2 MiB pages: 44.31 [43.97-44.40]; on, 4 KiB pages: 43.81 [43.66-43.85], that is, no change where there is no aliasing.
3. Per matrix (profile, one run per cell): W1/W3 8.98 ms to 8.01 ms and W2 4.34 ms to 3.96 ms per token on one socket; Wq/Wk/Wv, Wo, attention and the classifier do not move.

### Which matrices qualify (derived)

Per socket shard, row stride = bytes of one weight row (gs-independent: the int8 values), share = rows x stride / workers. "Start classes" is the number of distinct
`row0 x stride mod 2 MiB` over the workers (56 or 28 workers); with 1 class every worker is aliased, with k classes each class of 56/k workers is. Script: the arithmetic
of `q8_team_static`, no measurement.

| model, sockets | matrix (rows x stride) | share | start classes of 56 workers | of 28 workers |
|---|---|---|---|---|
| 8B, 1S | Wq, Wo 4096 x 4096 | 293 KiB | 7 | - |
| 8B, 1S | Wk, Wv 1024 x 4096 | 73 KiB | 28 | - |
| 8B, 1S | **W1, W3 14336 x 4096** | 1 MiB | **2** | - |
| 8B, 1S | W2 4096 x 14336 | 1 MiB | 14 | - |
| 8B, 2S | Wq 2048 x 4096, Wo 4096 x 2048 | 146 KiB | 14 | 7 |
| 8B, 2S | Wk, Wv 512 x 4096 | 37 KiB | 56 | 28 |
| 8B, 2S | **W1, W3 7168 x 4096** | 512 KiB | **4** | **2** |
| 8B, 2S | W2 4096 x 7168 | 512 KiB | 28 | 14 |
| 70B, 2S | Wq, Wo 4096 x 8192 / 8192 x 4096 | 585 KiB | 7 | 7 |
| 70B, 2S | Wk, Wv 512 x 8192 | 73 KiB | 28 | 14 |
| 70B, 2S | **W1, W3 14336 x 8192** | **2 MiB** | **1** | **1** |
| 70B, 2S | **W2 8192 x 14336** | 2 MiB | 7 | 7 |
| every model | classifier (vocabulary / sockets rows) | 4.5-9 MiB | 56 | 28 |

So the loss is expected in W1/W3 (the largest weights, about 55% of the weight bytes of a layer, derived) at every size, and in W2 (and Wq/Wo) of the 70B; the classifier and the KV-head
matrices are not aliased.

## 3. The change

`LLAMA_GEMV_STAGGER=0|1` (default 0; anything else is a fatal error), read by `kernels_init`; reported as `build.gemv_stagger` in the JSON report; test hook
`k_gemv_stagger()` in `testing.h`. In `q8_team_static`, member w of a team starts its share at block `w mod blocks` of `BLOCK` (4) rows, counted from the first row of the
share, runs to the end, and wraps to the beginning: simultaneous members then read addresses that differ by w blocks (32 KiB per member at the 8192-byte rows of the 70B
shard, 16 KiB at the 4096-byte rows of the 8B) on top of their share offsets, which is the 4 KiB to 64 KiB stagger of the microbenchmark.

Options considered:

* (a) a skewed row partition (slice starts offset, same rows per output). The shares must still tile [0, d), so shifting every start by a different amount makes the shares uneven (some grow while others shrink), and the barrier waits for the largest. Rejected.
* **(b) a per-worker rotation of the start block inside its own share: chosen.** Same blocks (the same four-row groups counted from the share's first row), the same
  per-row arithmetic and the same output slots, only the order of the blocks changes, so the outputs and the reduction order are bitwise the same. Every worker keeps the
  same row count and bytes. About 15 lines in one function. The rows-done callback of the fused Wo/W2 reduction is called once with the whole share after its last block, as
  before, so nothing in the exchange sees a different order.
* (c) padding each matrix's per-worker block in the arena by w x 4 KiB at load time: changes the weight layout, the loader, the shard sizes, the placement plan's byte
  accounting and the packed copies; the largest change and the one most likely to break a gate. Rejected.

Costs: none measured at 4 KiB pages (section 4); the extra work is one compare per block.

## 4. Results (measured)

Interleaved repetitions (each repetition runs the four cells in a rotated order), one profiled run per cell afterwards. THP `off` is the engine default (2 MiB pages on this
node), `never` is `LLAMA_THP=never` (4 KiB).

| config (gs64, W8A8, 4096 warm) | page size | knob off, tok/s median [min-max] | knob on, tok/s median [min-max] | delta | on faster in |
|---|---|---|---|---|---|
| 8B, 1 socket, HBM | 2 MiB (default) | 42.21 [41.99-42.41] | 44.31 [43.97-44.40] | **+5.0%** | 5 of 5 |
| 8B, 1 socket, HBM | 4 KiB (`never`) | 43.88 [43.32-44.05] | 43.81 [43.66-43.85] | -0.2% | 1 of 5 |
| 8B, 2 sockets, HBM | 2 MiB (default) | 73.53 [72.26-74.87] | 74.95 [74.45-76.15] | **+1.9%** | 5 of 5 |
| 8B, 2 sockets, HBM | 4 KiB (`never`) | 74.81 [74.12-74.88] | 74.43 [74.30-75.31] | -0.5% | 2 of 5 |
| 70B, 2 sockets, hybrid all-HBM | 2 MiB (default) | 10.00 [9.98-10.02] | 10.60 [10.55-10.62] | **+6.0%** | 5 of 5 |
| 70B, 2 sockets, hybrid all-HBM | 4 KiB (`never`) | 10.49 [10.47-10.51] | 10.53 [10.48-10.54] | +0.4% | 5 of 5 |

Page size alone (knob off, 4 KiB against the 2 MiB default): 8B one socket +4.0%, 8B two sockets +1.7%, 70B +4.9%. The knob on 2 MiB pages reaches or passes the 4 KiB figure
in every configuration (44.31 against 43.88, 74.95 against 74.81, 10.60 against 10.49). An earlier run of the 8B one-socket cells with `LLAMA_THP` unset and `on` (both 2 MiB
pages on this node; job 5933868) gave 42.26 [41.75-42.28] and 42.23 [42.00-42.39] without the knob and 44.31 [44.09-44.41] and 44.26 [44.13-44.49] with it (+4.8% in both).

Decode profile, ms per generated token, one profiled run per cell (2 MiB default pages, knob off against on; the 4 KiB runs are in the validation record):

| phase | 8B 1S off | 8B 1S on | 8B 2S off | 8B 2S on | 70B 2S off | 70B 2S on |
|---|---|---|---|---|---|---|
| qkv | 1.823 | 1.828 | 1.095 | 1.165 | 7.933 | 7.715 |
| wo | 1.351 | 1.346 | 0.995 | 1.061 | 7.000 | 7.061 |
| w1_w3_swiglu | 8.978 | 8.011 | 4.232 | 4.091 | 42.064 | 40.334 |
| w2 | 4.338 | 3.959 | 2.342 | 2.388 | 25.914 | 20.710 |
| attention | 5.027 | 5.207 | 3.366 | 3.424 | 11.984 | 13.014 |
| classifier | 0.995 | 1.019 | 0.504 | 0.515 | 1.040 | 1.050 |
| forward | 23.833 | 22.739 | 13.556 | 13.675 | 100.067 | 94.485 |

Where the shares are aliased the phase shortens: W1/W3 -10.8% and W2 -8.7% on 8B one socket (start classes 2 and 14), W2 -20.1% and W1/W3 -4.1% on the 70B (7 and 1
classes, W2 holding the 2 MiB shares). On 8B two sockets the phase differences are inside the run-to-run spread of a single profiled run (W2 and Wo include the
cross-socket reduction wait); the whole-token effect (+1.9%) is the evidence there. Attention grows by 0.2-1.0 ms per token wherever the weight stream runs faster
(8B one socket, 70B) and is the same on 4 KiB pages (70B: 13.2 ms), so on 2 MiB pages attention was running in the shadow of the slower GEMV; this is an observation,
not tested further. Per-token forward time: 8B one socket -4.6%, 70B -5.6%.

## 4b. Replication on the second Flat node and on gs32 (measured)

Job 5935737, the same script (`scripts/gemv-stagger.sbatch` STAGE=ab), the same engine code (tree 59f6bc6), the same five interleaved repetitions and the same four cells, on the
second Flat node ("node B"; section 4 is node A, the first). THP is `[always]` on both. gs32 uses the checkpoints named in the validation record (8B: the Q8_0 GGUF repacked to v3;
70B: the cross-engine `q8-gguf-v3.bin`). Decode tok/s, median [min-max] of 5; the delta is knob on against off.

| config | page size | node A (first run, gs64) off to on, delta | node B gs64: knob off | knob on | delta | on faster in |
|---|---|---|---|---|---|---|
| 8B, 1 socket | 2 MiB | 42.21 to 44.31, +5.0% | 42.43 [42.13-42.64] | 44.27 [44.18-44.39] | **+4.3%** | 5 of 5 |
| 8B, 1 socket | 4 KiB | 43.88 to 43.81, -0.2% | 44.08 [43.96-44.14] | 44.02 [43.82-44.09] | -0.1% | 1 of 5 |
| 8B, 2 sockets | 2 MiB | 73.53 to 74.95, +1.9% | 74.44 [73.19-76.05] | 75.31 [60.35-76.68] | **+1.2%** | 4 of 5 |
| 8B, 2 sockets | 4 KiB | 74.81 to 74.43, -0.5% | 75.55 [75.18-75.78] | 74.81 [73.99-75.01] | -1.0% | 0 of 5 |
| 70B, 2 sockets, hybrid 80/80 | 2 MiB | 10.00 to 10.60, +6.0% | 10.03 [10.02-10.04] | 10.62 [10.59-10.63] | **+5.9%** | 5 of 5 |
| 70B, 2 sockets, hybrid 80/80 | 4 KiB | 10.49 to 10.53, +0.4% | 10.52 [10.47-10.55] | 10.51 [10.43-10.54] | -0.0% | 2 of 5 |

| config (node B, gs32) | page size | knob off | knob on | delta | on faster in |
|---|---|---|---|---|---|
| 8B, 1 socket | 2 MiB | 39.67 [39.61-39.83] | 41.73 [41.60-41.89] | **+5.2%** | 5 of 5 |
| 8B, 1 socket | 4 KiB | 41.34 [41.12-41.56] | 41.29 [40.84-41.31] | -0.1% | 2 of 5 |
| 8B, 2 sockets | 2 MiB | 69.67 [68.96-70.82] | 70.80 [70.29-72.13] | **+1.6%** | 4 of 5 |
| 8B, 2 sockets | 4 KiB | 71.18 [70.89-71.35] | 71.09 [70.99-71.53] | -0.1% | 2 of 5 |
| 70B, 2 sockets, hybrid 80/80 | 2 MiB | 8.95 [8.95-9.00] | 10.03 [9.99-10.04] | **+12.0%** | 5 of 5 |
| 70B, 2 sockets, hybrid 80/80 | 4 KiB | 9.98 [9.97-10.01] | 9.97 [9.94-9.98] | -0.0% | 2 of 5 |

* **The gs64 result replicates on node B** within about one percentage point (8B one socket +4.3% against +5.0%, 8B two sockets +1.2% against +1.9%, 70B +5.9% against +6.0%), and the
  4 KiB cells again show no gain. The knob-off values on node B are within 0.5-0.8% of node A (8B one socket 42.43 against 42.21; 8B two sockets 74.44 against 73.53; 70B 10.03 against 10.00).
* **gs32 shows the same pattern with a larger 70B gain.** The VNNI pair kernel (gs32 decode) is aliased the same way: 8B one socket +5.2%, 8B two sockets +1.6%, and 70B **+12.0%** (8.95 to 10.03 tok/s). With the knob
  on, 2 MiB pages reach the 4 KiB figure (10.03 against 9.98 tok/s); the gs32 70B is the case that loses the most to the aliasing: W1/W3 53.3 ms to 43.5 ms and W2 26.4 ms to 22.2 ms per token (profile, one run each), against 42.0 to 40.3 and 25.9 to 20.7 at gs64.
* **8B two sockets, gs64, knob on at 2 MiB has one slow repetition (60.35 tok/s, repetition 2; the others 75.06-76.68).** The median and the order statistics are unaffected (4 of 5 faster), but that single run is
  20% below its neighbours; its token trace is identical to the rest and no cause was examined (the run-to-run bimodality of [bimodality.md](bimodality.md) is a candidate, not tested). On 4 KiB pages the gs64 8B two-socket knob is -1.0% (0 of 5), a
  small loss that node A did not show (-0.5%, 2 of 5); it is within about one spread and is not a finding, but it is the only cell where the knob is slower in every repetition.
* Token traces of all 144 runs (24 per cell, six cells) are identical between knob off and on (`tokens-identical.txt` in each cell). The logits-level bitwise check of section 5 of the validation record was not repeated on gs32 or node B; the traces are the check here.

## 5. Limits and uncertainties

* The page-size attribution rests on `LLAMA_THP=never` giving 4 KiB pages (the report shows `thp_bytes` 0 for every weight region) and on the knob making no difference there.
* The mechanism is still inferred. The measurements show that something that depends on 2 MiB-aligned simultaneous streams costs 4-5% of 8B one-socket decode and that
  staggering the starts removes it; the memory controller was not examined.
* Profile times are one run per cell; on two sockets the W2 and Wo phases contain the cross-socket reduction wait, so their per-matrix differences are within the noise.
* `STAGGER=1` only reorders blocks within a worker's own share; it cannot help a layout whose shares are not aliased, and it does not change the hybrid FP32 claim path
  (FP32 weights were not studied).
* The 70B placement is the hybrid layout (all weight and KV layers in HBM, packed copy in DDR), not `hbm-only`, which does not fit with the AMX packed copy.

## 6. Reproduction

`scripts/gemv-stagger.sbatch` (`STAGE=check` for the gate and the logit digests, `STAGE=ab` for the cells) and `tools/gemv_stagger_summary.py` (tables). The unit test is
`test_gemv_stagger` in `tests/kernels.c`.

## 7. Recommendation

Do not make it the default yet; propose it for the owner's decision.

* The evidence for turning it on is strong where it applies: on the engine's default page size (2 MiB on a node with THP `[always]`) it gains +5.0% (8B one socket), +1.9%
  (8B two sockets) and +6.0% (70B two sockets), 5 of 5 repetitions each, bitwise identical output, and it does nothing on 4 KiB pages (so it cannot hurt there).
* Against that: one machine type (the second node, section 4b, agrees), the mechanism is inferred, decode-only (prefill GEMM unchanged), gs64 and gs32 only (per-channel goes through a different kernel and was not measured), and the FP32 engine is untouched. The repeat on the second Flat node and on gs32 asked for earlier is section 4b: it agrees on gs64 (+4.3%, +1.2%, +5.9%) and on gs32 (+5.2%, +1.6%, +12.0%), 5 of 5 or 4 of 5 repetitions on every 2 MiB cell and no gain on 4 KiB pages. What remains for the owner is the decision itself;
  the one cell where the knob is slower in all five repetitions is the 8B two-socket gs64 4 KiB cell (-1.0%), which is not the engine default page size on these nodes. The cost to revert is one knob.
* Alternative to the knob: `LLAMA_THP=never` recovers 4.0%, 1.7% and 4.9% by itself and is already one environment variable, but it gives up what 2 MiB pages bring to the
  rest of the engine (KV, attention, prefill GEMM: not measured here), and the stagger gives more at the same page size.
