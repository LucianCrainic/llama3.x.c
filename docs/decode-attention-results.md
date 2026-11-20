# Decode attention results (phase E5)

Results of the decode attention phase of the faster attention track ([plan](plans/E-followups/README.md), section E5). The
baseline is the B7 decode campaign ([tensor-parallel.md](tensor-parallel.md)). Provenance, gates and gaps:
[decode-attention-results-validation.md](decode-attention-results-validation.md). Every number is **measured** unless labelled
**derived**.

## 1. Result in one paragraph

The GQA-group decode kernel (`LLAMA_ATTN_DECODE=group`) passes every correctness gate. It is faster than `split` beyond
variability in all 16 cells at 8192 and 32768 tokens of context, but it misses both E5 value gates. 1B decode at 32768 on one
socket gains 1.075x (FP32) and 1.103x (W8A8) against a target of 1.3x. The 1B W8A8 two-socket gain at 32768 rises from 1.25x to
1.46x against a target of 1.5x. The gains are larger on two sockets (1B up to 1.37x) and on 8B (W8A8 one socket 1.27x at 32768).
At 512 tokens all eight cells overlap with `split`. Two of them, 8B FP32 and 8B W8A8 on one socket, fall below the B7 range (by
4.6% and 0.7%). At 8192 and 32768 tokens decode attention reaches 23-51% of the 552 GB/s per-socket HBM bandwidth in the engine, so it is
**not bandwidth-bound**. The next limit is the K/V access pattern: with the same kernel, 1B (256-byte head rows) reaches about 196 GB/s
and 8B (512-byte head rows) about 281 GB/s. BF16 K/V storage (owner decision D-E2) is therefore not proposed now.
`split` stays the decode default; switching it is the owner's decision.

## 2. What ran

- **Kernel.** `group` handles one KV group (the `kv_mul` query heads that share a KV head) and one position chunk per task.
  K and V are read once per group, not once per query head. It uses no per-score horizontal reduction and pre-scales Q. It fuses
  an online softmax per 32-position tile and the PV product. On hosts without AVX-512, or for shapes it does not support, it falls
  back to `split`.
- **Check job** 5864135 (commit 9d3f72b): `make test` on the hardware. 12 per-token logit comparisons `split` against `group`,
  8 generation comparisons and 4 Hugging Face gates.
- **Bench job** 5870935 (commit 9d3f72b): the B7 matrix subset, 192 reports. Models 1B and 8B, FP32 and W8A8, contexts
  512 / 8192 / 32768, one and two sockets, `split` and `group`, three timed repeats plus one profiled run each. It also ran the
  attention microbenchmark at 1000 repetitions per cell, for 4 and 8 local KV heads.
- Both jobs ran on the same Flat-mode node as B7. Prefill used `split` in both, the default of that commit.

## 3. Decode throughput

Median [range] of three repeats, tokens per second. "Beyond variability" compares the ranges within this job. The B7 column is
the B7 campaign's `split` result for the same cell (1x56 or 2x56).

| Model | Precision | Sockets | Context | split tok/s | group tok/s | group / split | Beyond variability | B7 | group vs B7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1B | FP32 | 1 | 512 | 92.38 [90.59-96.69] | 94.08 [90.65-97.13] | 1.018 | no (overlap) | 91.33 [91.07-96.73] | within |
| 1B | FP32 | 1 | 8192 | 70.89 [70.50-71.03] | 71.39 [71.36-71.68] | 1.007 | faster | 70.49 [70.38-70.67] | above |
| 1B | FP32 | 1 | 32768 | 44.82 [44.75-45.10] | 48.19 [48.17-48.22] | 1.075 | faster | 44.84 [44.82-45.18] | above |
| 1B | FP32 | 2 | 512 | 171.89 [168.07-174.41] | 173.30 [171.62-178.34] | 1.008 | no (overlap) | 168.85 [132.03-169.66] | above |
| 1B | FP32 | 2 | 8192 | 130.57 [130.32-131.11] | 142.32 [141.23-142.34] | 1.090 | faster | 130.68 [130.08-131.08] | above |
| 1B | FP32 | 2 | 32768 | 69.99 [69.89-70.04] | 84.81 [84.53-84.93] | 1.212 | faster | 70.22 [69.94-70.23] | above |
| 1B | W8A8 | 1 | 512 | 274.95 [270.29-279.05] | 266.72 [265.00-274.89] | 0.970 | no (overlap) | 269.09 [259.53-284.52] | within |
| 1B | W8A8 | 1 | 8192 | 121.94 [121.31-122.39] | 129.13 [129.12-129.25] | 1.059 | faster | 121.82 [121.32-122.04] | above |
| 1B | W8A8 | 1 | 32768 | 62.81 [62.79-62.84] | 69.28 [68.21-71.62] | 1.103 | faster | 62.61 [62.58-62.62] | above |
| 1B | W8A8 | 2 | 512 | 468.67 [463.87-472.28] | 480.78 [467.33-487.09] | 1.026 | no (overlap) | 466.84 [436.15-469.37] | above |
| 1B | W8A8 | 2 | 8192 | 191.09 [187.22-193.71] | 262.02 [260.29-263.70] | 1.371 | faster | 193.72 [192.64-195.00] | above |
| 1B | W8A8 | 2 | 32768 | 78.38 [78.27-78.52] | 100.82 [100.75-101.11] | 1.286 | faster | 78.72 [78.54-78.83] | above |
| 8B | FP32 | 1 | 512 | 16.43 [15.75-16.48] | 15.66 [15.63-15.77] | 0.953 | no (overlap) | 16.42 [16.13-16.44] | **below** |
| 8B | FP32 | 1 | 8192 | 13.35 [13.26-13.40] | 13.66 [13.63-13.66] | 1.023 | faster | 13.29 [13.24-13.42] | above |
| 8B | FP32 | 1 | 32768 | 9.57 [9.56-9.61] | 10.89 [10.89-10.91] | 1.138 | faster | 9.66 [9.59-9.66] | above |
| 8B | FP32 | 2 | 512 | 30.50 [30.45-30.70] | 30.69 [30.69-30.70] | 1.006 | no (overlap) | 30.63 [30.59-30.64] | above |
| 8B | FP32 | 2 | 8192 | 25.20 [25.16-25.48] | 27.09 [27.07-27.14] | 1.075 | faster | 25.50 [25.35-25.58] | above |
| 8B | FP32 | 2 | 32768 | 19.23 [19.20-19.25] | 20.37 [20.34-20.38] | 1.059 | faster | 19.18 [19.17-19.28] | above |
| 8B | W8A8 | 1 | 512 | 49.44 [48.66-49.46] | 49.18 [40.83-49.24] | 0.995 | no (overlap) | 49.52 [49.52-49.73] | **below** |
| 8B | W8A8 | 1 | 8192 | 28.63 [28.50-29.31] | 30.65 [30.64-30.68] | 1.071 | faster | 28.54 [28.48-28.57] | above |
| 8B | W8A8 | 1 | 32768 | 15.62 [15.52-15.74] | 19.82 [19.81-19.90] | 1.269 | faster | 15.59 [15.59-15.64] | above |
| 8B | W8A8 | 2 | 512 | 89.58 [89.16-91.39] | 92.38 [91.10-92.39] | 1.031 | no (overlap) | 91.25 [90.08-91.35] | above |
| 8B | W8A8 | 2 | 8192 | 50.62 [50.53-50.71] | 58.91 [58.85-58.91] | 1.164 | faster | 50.57 [50.47-51.20] | above |
| 8B | W8A8 | 2 | 32768 | 29.32 [29.29-29.41] | 34.05 [33.16-34.09] | 1.161 | faster | 29.56 [29.52-29.58] | above |

The `split` results of this job reproduce B7: 21 of 24 `split` medians are within 1.5% of B7's, and the other three (all at 512)
within 2.2%.

**Two-socket gain** (two sockets over one, medians, **derived**):

| Model | Precision | 512 split / group | 8192 split / group | 32768 split / group |
| --- | --- | --- | --- | --- |
| 1B | FP32 | 1.86 / 1.84 | 1.84 / 1.99 | 1.56 / 1.76 |
| 1B | W8A8 | 1.71 / 1.80 | 1.57 / 2.03 | 1.25 / **1.46** |
| 8B | FP32 | 1.86 / 1.96 | 1.89 / 1.98 | 2.01 / 1.87 |
| 8B | W8A8 | 1.81 / 1.88 | 1.77 / 1.92 | 1.88 / 1.72 |

For 1B the two-socket gain rises at long contexts, because the two-socket runs gain most from `group`. For 8B at 32768 it falls,
because the one-socket runs gain more (1.14x and 1.27x against 1.06x and 1.16x). This is the same pattern E4 found for prefill.

## 4. Attention time and bandwidth

From the profiled run of each cell (one run, `LLAMA_PROFILE=1`). Attention time per decode step, its share of the step, and
the K/V bytes it streams per socket per step divided by that time (**derived**: 2 x local KV heads x head_dim x 4 bytes x mean
context x layers). The K/V cache is FP32 for both precisions.

| Model | Precision | Sockets | Context | split ms (share) | group ms (share) | Attention speedup | split GB/s | group GB/s | group % of 552 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1B | FP32 | 1 | 8192 | 4.40 (31%) | 4.01 (29%) | 1.10 | 124 | 136 | 25% |
| 1B | FP32 | 1 | 32768 | 12.17 (54%) | 11.03 (53%) | 1.10 | 177 | 196 | 35% |
| 1B | FP32 | 2 | 8192 | 2.61 (34%) | 1.66 (23%) | 1.57 | 105 | 165 | 30% |
| 1B | FP32 | 2 | 32768 | 9.33 (65%) | 6.74 (57%) | 1.38 | 116 | 160 | 29% |
| 1B | W8A8 | 1 | 8192 | 4.82 (58%) | 4.39 (56%) | 1.10 | 113 | 125 | 23% |
| 1B | W8A8 | 1 | 32768 | 12.39 (77%) | 10.87 (75%) | 1.14 | 174 | 199 | 36% |
| 1B | W8A8 | 2 | 8192 | 2.84 (54%) | 1.79 (46%) | 1.59 | 96 | 153 | 28% |
| 1B | W8A8 | 2 | 32768 | 10.44 (81%) | 7.57 (76%) | 1.38 | 103 | 142 | 26% |
| 8B | FP32 | 1 | 8192 | 15.61 (21%) | 16.21 (22%) | 0.96 | 140 | 135 | 24% |
| 8B | FP32 | 1 | 32768 | 42.60 (41%) | 30.73 (33%) | 1.39 | 203 | 281 | 51% |
| 8B | FP32 | 2 | 8192 | 9.67 (24%) | 7.39 (20%) | 1.31 | 113 | 148 | 27% |
| 8B | FP32 | 2 | 32768 | 22.56 (43%) | 19.53 (40%) | 1.16 | 191 | 221 | 40% |
| 8B | W8A8 | 1 | 8192 | 16.36 (46%) | 15.22 (46%) | 1.07 | 134 | 144 | 26% |
| 8B | W8A8 | 1 | 32768 | 44.16 (69%) | 31.53 (62%) | 1.40 | 195 | 274 | 50% |
| 8B | W8A8 | 2 | 8192 | 9.43 (48%) | 7.35 (43%) | 1.28 | 116 | 149 | 27% |
| 8B | W8A8 | 2 | 32768 | 23.22 (68%) | 19.17 (65%) | 1.21 | 186 | 225 | 41% |

At 512 tokens attention is 2-16% of a step, and `group` and `split` are within 0.15 ms of each other.

**Microbenchmark** (attention alone, one layer, 56 workers on HBM node 2, median of 1000 repetitions; the bench's own GB/s):

| Shape | Local KV heads | Context | split ms | group ms | group / split speed | group GB/s median (best) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1B | 4 | 8192 | 0.187 | 0.135 | 1.38 | 124 (247) |
| 1B | 4 | 32768 | 0.611 | 0.490 | 1.25 | 137 (317) |
| 1B | 8 | 8192 | 0.317 | 0.299 | 1.06 | 112 (323) |
| 1B | 8 | 32768 | 0.724 | 0.702 | 1.03 | 191 (329) |
| 8B | 4 | 8192 | 0.267 | 0.240 | 1.11 | 140 (268) |
| 8B | 4 | 32768 | 0.696 | 0.570 | 1.22 | 236 (328) |
| 8B | 8 | 8192 | 0.372 | 0.481 | **0.77** | 139 (365) |
| 8B | 8 | 32768 | 1.279 | 0.951 | 1.34 | 282 (403) |

At 512 tokens `group` is 1.17-1.38x faster in the microbenchmark, but the work is 0.03-0.07 ms per layer.

## 5. Gates

| Gate | Target | Measured | Result |
| --- | --- | --- | --- |
| Correctness | §5.3 rows marked E5, with Hugging Face | 25 of 25 PASS (check job) | pass |
| 1B decode at 32768, one socket | ≥ 1.3x beyond variability, FP32 and W8A8 | 1.075x (FP32), 1.103x (W8A8), both beyond variability | **fail** |
| 1B W8A8 two-socket gain at 32768 | from 1.26x to ≥ 1.5x | 1.25x to 1.46x | **fail** (close) |
| Attention bandwidth | reported against 552 GB/s per socket | 23-51% in the engine at 8192 and 32768; microbenchmark best 403 GB/s (73%) | reported |
| Context 512 not slower | median within the B7 range | 6 of 8 within or above; 8B FP32 one socket 15.66 (B7 16.13-16.44), 8B W8A8 one socket 49.18 (B7 49.52-49.73) | **fail** in 2 cells (see the note) |
| Decision | bandwidth-bound (≥ 80% of 552 GB/s)? | no: at most 51% | next limit listed (section 6) |

**Note on the two 512 cells.** The 8B FP32 loss lies outside attention. In the profiled runs `group`'s attention phase is
5% faster (1.57 against 1.66 ms per step), while the weight phases `qkv`, `w1_w3` and `w2`, which `group` does not change, are
about 5% slower (4.6-5.3%). The `split` repeats are bimodal: one of three (15.75 tok/s) sits at `group`'s level. Within this job the ranges
overlap. The W8A8 cell is 0.7% below B7's lower end, and its `group` lower outlier (40.83) is one repeat. Neither cell shows a
loss that can be traced to the kernel, but the gate as written is not met. A targeted rerun with more repeats would settle it
(section 7).

## 6. Decision: not bandwidth-bound; next limit

Decode attention reaches at most 51% of 552 GB/s in the engine (8B, 32768, one socket) and 26-36% for 1B. So D-E2 (BF16 K/V
storage, which halves the bytes) does not go to the owner now. The evidence for the next limit, in order:

1. **K/V access pattern (plan candidate 3).** A head's K or V row is 256 bytes (1B) or 512 bytes (8B), and consecutive
   positions of one head are `kv_dim` x 4 bytes apart (2 KiB and 4 KiB). With the same kernel and 8 local KV heads at 32768,
   1B streams at 191 GB/s and 8B at 282 GB/s (microbenchmark medians). In the engine the figures are 196-199 against 274-281 GB/s.
   Shorter contiguous runs per head give lower bandwidth, which is what a `[kv_head][position][head_dim]` layout would change. The
   plan requires a layout flag in the bench first and a measured gain of at least 1.2x on attention time before the engine change.
2. **Run-to-run spread and idle workers at 8192 (candidate 2).** At 8192 the microbenchmark's best repetition is 2-3x its median
   (for example 323 against 112 GB/s), worker busy fractions are 0.75-0.87 and the tail fraction is 0.07-0.20. At 32768 they are
   0.90-0.96 and 0.03-0.05. The chunk plan leaves work unevenly spread at 8192.
3. **8B with 8 local KV heads at 8192** is the one shape where `group` is slower than `split`, both in the microbenchmark (0.77x)
   and in the profiled engine run (0.96x on attention time), although the timed engine runs are faster overall (1.023x). This
   needs a look at the chunk count for that shape before any default change.

## 7. Recommendation and follow-ups

- `group` is faster beyond variability in all 16 long-context cells and passes every correctness gate. It is a candidate for
  the decode default, but the value gates are not met and the 512 question is open. The default stays `split`; the owner
  decides.
- Follow-ups, in order: a targeted rerun of 8B at 512 on one socket (both precisions, more repeats, `split` and `group`
  interleaved) to settle the 512 gate; the K/V layout flag in the bench (candidate 3); the chunk plan at 8192 (candidate 2 and
  the 8B 8-head shape). **Status after section 8:** the 512 rerun and the layout flag are done. The 512 gate fails in one cell
  (8B FP32, one socket, 0.953x, in the weight phases, cause not identified); the head-major layout is faster by 1.5-2.8x in the
  microbenchmark at 8192 and 32768 and is recommended for an engine implementation (8.4). The chunk plan at 8192 (candidate 2)
  and the 8B 8-head shape remain: in the engine that shape shows no attention loss (8.3), but its microbenchmark does.
- E6 (batched decode design) can use these measurements. Decode attention at long contexts is limited by the access pattern,
  not by bandwidth. Each sequence has its own K/V, so batching sequences over shared weights would not by itself shorten
  attention time per sequence (an inference, not measured).

## 8. Follow-ups: the 512 rerun and the K/V layout probe

This section settles the first two follow-ups of section 7. The decision rules below were written **before** any follow-up
job ran (commit of this section's rules precedes the jobs; the validation record names the commits and job ids). Results are
appended under "8.3" and "8.4" after the jobs.

### 8.1 Decision rules (fixed before the runs)

**512 gate and the 8B one-socket shapes.** Cells: 8B FP32 and 8B W8A8, one socket x 56 workers, context 512 (the gate) and
8192 (the one shape where the profiled attention of `group` was slower). Per cell, nine repeats of `split` and `group`
interleaved inside each repeat; odd repeats launch `split` first and even repeats `group` first (`E5_ALTERNATE=1`), so
neither kernel is always second. The 8192 cells run the same nine repeats (the requirement is five). Three profiled runs per
kernel and cell (`LLAMA_PROFILE=1`, per-step phase times from `profile_seconds.decode`). Same node, same build as the first E5
bench job; the B7 range is the reference of section 3.

- **Verdict A (the E5.5 gate as written).** The 512 gate passes if the median decode tok/s of `group` is within or above the
  B7 range of that cell (8B FP32 16.13-16.44, 8B W8A8 49.52-49.73), i.e. not below its lower end.
- **Verdict B (relative to the same job).** Otherwise it passes in-job if the median of `group` is at least the lower end (the
  minimum) of `split`'s nine repeats **and** the two ranges overlap. If B holds and A does not, the report says whether
  `split` itself reproduced B7 in the job (median of `split` within the B7 range); if it did not, the B7 reference is not
  reproduced on that day and only B counts.
- **Fail.** Neither A nor B. The report then gives the median ratio, and the cell stays a recorded loss.
- Also reported, not gating: the median ratio `group / split`; the number of repeats (of nine) in which `group` was faster
  than `split` of the same repeat (paired); and the spread (min-max) of each kernel.
- **Does the weight-phase cost follow the kernel or the process?** From the three profiled runs of each kernel, take the
  sum of the weight phases `qkv + w1_w3 + w2` per decode step (ms), `W`. Let `d` = median(`W` of `group`) - median(`W` of
  `split`) and `s` the larger of the two kernels' (max - min) over their three runs. If `|d| <= s` the difference is not
  attributable to the kernel (it is run-to-run or process variation). If `d > s` the weight phases are slower under `group`
  (follows the kernel). If `d < -s` they are faster. The attention phase gets the same test, with `s` from its own runs.
- **8B at 8192 one socket.** `group`'s attention is called slower than `split`'s only if its median profiled attention time
  exceeds `split`'s by more than `s` (same test); the token rate is judged by the rule "beyond variability" of section 3
  (ranges do not overlap), and the microbenchmark cell (8B, 8 local KV heads, 8192) is repeated in the layout job with the
  rows layout (one repeat of 1000 repetitions per kernel).

**K/V layout probe (E5.2 candidate 3).** `attention-bench --kv-layout rows|head` reads the same values in the engine's
`[position][kv_head][head_dim]` layout (`rows`) or in `[kv_head][position][head_dim]` (`head`: position stride `head_dim`
floats, head base `context x head_dim` floats). Shapes 1B (head_dim 64) and 8B (head_dim 128); local KV heads 4 and 8;
contexts 8192 and 32768 (512 reported as well); kernels `split` and `group`; 1000 repetitions per cell, 56 threads on the HBM
node 2. Per cell the speedup is `median ms (rows) / median ms (head)`; the bench's `gbs_median` and `gbs_best`, `busy_mean`
and `tail_fraction` are reported with it.

- **Cell gate.** A cell passes if the speedup is at least 1.2 (the plan's threshold for touching the engine).
- **Verdict.** The engine change (head-major KV cache) is recommended only if `group` (the kernel proposed for the default)
  passes in all four cells at 32768 (the contexts where attention dominates a step) **and** none of its 8192 cells is slower
  than 0.95. If only some cells pass, the report lists them and recommends no engine change, naming the cells as the evidence
  to extend. If no `group` cell reaches 1.2, the layout is closed as a candidate.
- If the verdict is "recommended", the section lists what the engine change touches (KV append in `src/run.c` and
  `src/runq.c`, the local KV of `src/tp.c`, A5 placement, prefill attention reading the cache) and does **not** implement it.

### 8.2 What ran

- **Code.** Branch `feature/attn-e5-followups`, build commit 961af17 (the four commits above `develop` b111e98; the decode kernels
  take a position stride and a per-head base, `k_attention` keeps the engine's values, `attention-bench --kv-layout`, the E5
  bench-stage knobs). Same HBM node as B7 and the first E5 jobs, exclusive; Flat mode; gcc `-O3 -fopenmp`.
- **512 and 8192 engine job** 5876099 (`E5_CONTEXTS="512 8192" E5_SOCKETS=1 E5_PROFILED=3 E5_ALTERNATE=1 E5_SKIP_MICRO=1
  REPEATS=9`, 8B only): 96 reports (FP32 and W8A8 x contexts 512 and 8192 x kernels x (9 timed + 3 profiled)), all completed,
  exit 0, 56 minutes.
- **Layout jobs** 5876098 (rows measured first, then head) and 5876113 (head first, then rows, an order control added after the
  first result): `attention-bench` decode, 1B and 8B shapes, 8 and 4 local KV heads, contexts 512 / 8192 / 32768, `split` and
  `group`, 1000 repetitions, 56 threads, HBM node 2, 16 launches each, all passed, 31 s each.
- `make test` on the hardware ran in job 5875780 (all ten C test suites with 0 failures at the scalar, AVX-512 and AMX levels, 69 Python tests: PASS), which
  I cancelled afterwards because its remaining time was the common one-socket Hugging Face prelude, which the follow-up jobs
  skip (`SKIP_PRELUDE_HF=1`). Job 5875772 failed at start (PYTHON not exported, 0 s).

### 8.3 The 512 gate and the 8B 8192 shape

Decode tok/s of the nine timed repeats, median [min-max]. "Paired" counts repeats in which `group` beat `split` of the same
repeat. B7 is the range of the first E5 section 3.

| Cell (8B, one socket) | split | group | group / split | Paired | B7 | A | B |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| FP32 512 | 16.42 [16.30-16.45] | 15.65 [15.56-15.72] | 0.953 | 0 of 9 | 16.13-16.44 | **fail** (below 16.13) | **fail** (below split's 16.30, ranges apart) |
| W8A8 512 | 49.36 [49.10-49.55] | 49.46 [49.08-49.66] | 1.002 | 6 of 9 | 49.52-49.73 | fail by 0.12% (`split` itself is 0.3% below the range today) | **pass** |
| FP32 8192 | 13.29 [13.20-13.38] | 13.63 [13.62-13.66] | 1.026 | 9 of 9 | 13.24-13.42 | above | faster beyond variability (ranges apart) |
| W8A8 8192 | 28.72 [28.20-29.40] | 30.60 [30.51-30.67] | 1.065 | 9 of 9 | 28.48-28.57 | above | faster beyond variability |

**Verdict by the rules of 8.1.** 8B FP32 at 512 on one socket **fails**, and the loss is reproducible, not bimodal: nine of nine
repeats favour `split`, 4.7% apart, with narrow ranges (the first job's bimodal `split` repeat did not recur: its nine repeats
span 16.30-16.45). 8B W8A8 at 512 **passes in-job** (verdict B): the medians differ by 0.2%, and verdict A is missed
only because `split` itself is 0.3% below the B7 range on this day, so the B7 reference is not reproduced for that cell. The E5.5
gate "context 512 not slower" therefore stays failed in one cell (8B FP32, one socket), where the first job had two.

**Profiled runs** (three per kernel, ms per decode step, median [min-max]; `W` = qkv + w1_w3 + w2, `s` = the larger run-to-run
spread of the two kernels):

| Cell | Kernel | step | qkv | attention | w1_w3 | w2 | W = qkv + w1_w3 + w2 | d(W) vs split, spread s | W verdict | attention verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8B FP32 512 | split | 60.95 [60.80-60.98] | 6.54 [6.45-6.57] | 1.66 [1.66-1.68] | 30.43 [30.28-30.51] | 14.01 [14.01-14.02] | 50.96 [50.83-51.02] |  | |  |
|  | group | 64.02 [63.84-64.02] | 7.03 [7.02-7.05] | 1.58 [1.55-1.59] | 31.87 [31.79-31.87] | 14.74 [14.67-14.79] | 53.64 [53.49-53.71] | +2.68, 0.22 | group slower | group faster |
| 8B FP32 8192 | split | 75.70 [74.99-76.00] | 6.45 [6.41-6.46] | 15.87 [15.68-16.15] | 30.71 [30.48-30.76] | 14.28 [14.16-14.36] | 51.46 [51.05-51.58] |  | |  |
|  | group | 73.32 [73.31-73.41] | 6.35 [6.33-6.43] | 15.93 [15.88-16.11] | 29.59 [29.57-29.64] | 13.62 [13.62-13.73] | 49.64 [49.54-49.69] | -1.82, 0.53 | group faster | not attributable to the kernel |
| 8B W8A8 512 | split | 20.24 [20.17-20.29] | 1.90 [1.89-1.94] | 1.71 [1.70-1.72] | 8.82 [8.81-8.88] | 4.30 [4.30-4.32] | 15.07 [15.01-15.10] |  | |  |
|  | group | 20.25 [20.20-20.28] | 1.92 [1.92-1.93] | 1.66 [1.65-1.67] | 8.88 [8.85-8.88] | 4.33 [4.32-4.34] | 15.12 [15.09-15.15] | +0.06, 0.10 | not attributable to the kernel | group faster |
| 8B W8A8 8192 | split | 34.99 [34.66-35.16] | 1.96 [1.95-2.07] | 16.11 [15.81-16.16] | 8.79 [8.78-8.81] | 4.39 [4.39-4.41] | 15.13 [15.12-15.29] |  | |  |
|  | group | 32.67 [32.61-32.75] | 1.83 [1.81-1.84] | 15.16 [15.10-15.18] | 8.31 [8.30-8.34] | 4.17 [4.15-4.17] | 14.31 [14.26-14.34] | -0.83, 0.16 | group faster | group faster |

**Does the weight-phase cost follow the kernel or the process?** The data support "the kernel setting", not a one-off process
effect, but they do not say why. At 8B FP32 and 512 tokens `group` is 5.0% slower per step (64.02 against 60.95 ms) although its
attention phase is 5% shorter (1.58 against 1.66 ms): every weight phase, `qkv` (+7.5%), `w1_w3` (+4.7%), `w2` (+5.2%), and
also `wo` (+4.6%) and the classifier (+8%) which the table does not list, is slower, by more than ten times the run-to-run spread
(`d` = +2.68 ms against `s` = 0.22 ms), in all nine timed repeats whichever kernel was launched first, and in the first job's
profiled run. At W8A8 and 512 tokens there is no such effect (`d` = +0.06 ms, inside `s`). At 8192 tokens the sign reverses:
the weight phases are *faster* under `group` (FP32 -1.8 ms, W8A8 -0.8 ms, beyond `s`), and the attention phase is the same
(FP32: 15.93 against 15.87 ms, inside `s` = 0.47) or 6% shorter (W8A8: 15.16 against 16.11 ms). I did not identify the
mechanism. Phases that `group` does not run (weights, norms) change by 3-7% with the attention kernel, in either direction
depending on precision and context. Candidates, none tested: cache or prefetcher state left by the attention tasks (the group
kernel keeps up to 16 query heads' accumulators and software-prefetches V rows), the effect on the HBM page state, and a
frequency or power state change of the 56 cores.

**The 8B 8-head shape at 8192.** In the engine, `group`'s attention phase is **not slower** than `split`'s: FP32 15.93 against
15.87 ms (difference inside the spread; the first job's 0.96x was a single run), W8A8 15.16 against 16.11 ms. The token-rate gains
(1.026x and 1.065x, nine of nine repeats, ranges apart) come mostly from the weight phases for FP32 (-1.8 ms of the -2.4 ms step
difference) and from both phases for W8A8. The microbenchmark still shows the slowdown, reproducibly: 8B, 8 local KV heads, 8192,
rows layout, `split` 0.371 ms and `group` 0.480 ms (0.77x; the first job had 0.372 and 0.481 ms, and the layout job with the
reverse order 0.370 and 0.479 ms). The engine does not show it. With the head-major layout `group` is the faster kernel there
(0.170 against 0.204 ms, 1.20x).

### 8.4 The K/V layout probe (E5.2 candidate 3)

Attention time of one layer, microbenchmark, median of 1000 repetitions; speed = `rows ms / head ms`, so a value above 1
means head-major is faster; bold marks the gate (at least 1.2). The two speed columns are the two launch orders (rows layout
first, then head; head first, then rows). GB/s is the bench's own figure (unique K/V bytes over the time), median and, in
brackets, the best repetition; busy and tail are the workers' busy and idle-after-last-task fractions of the traced pass.

| Shape | Local KV heads | Context | Kernel | rows ms | head ms | head/rows speed (rows first) | (head first) | rows GB/s med (best) | head GB/s med (best) | busy rows / head | tail rows / head |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 8b | 4 | 512 | split | 0.048 | 0.035 | **1.37** | **1.61** | 44 (61) | 60 (67) | 0.69 / 0.66 | 0.16 / 0.16 |
| 8b | 4 | 512 | group | 0.039 | 0.033 | 1.17 | 1.15 | 54 (69) | 63 (77) | 0.60 / 0.54 | 0.22 / 0.24 |
| 8b | 4 | 8192 | split | 0.264 | 0.118 | **2.23** | **2.24** | 127 (282) | 283 (336) | 0.78 / 0.75 | 0.19 / 0.21 |
| 8b | 4 | 8192 | group | 0.238 | 0.102 | **2.34** | **2.37** | 141 (315) | 330 (403) | 0.80 / 0.73 | 0.17 / 0.21 |
| 8b | 4 | 32768 | split | 0.650 | 0.405 | **1.61** | **1.49** | 206 (301) | 331 (407) | 0.93 / 0.90 | 0.05 / 0.06 |
| 8b | 4 | 32768 | group | 0.579 | 0.337 | **1.72** | **1.57** | 232 (349) | 399 (531) | 0.92 / 0.90 | 0.05 / 0.05 |
| 8b | 8 | 512 | split | 0.080 | 0.034 | **2.39** | **2.31** | 52 (108) | 125 (134) | 0.72 / 0.71 | 0.23 / 0.13 |
| 8b | 8 | 512 | group | 0.061 | 0.030 | **2.07** | **1.96** | 69 (116) | 142 (157) | 0.71 / 0.64 | 0.19 / 0.18 |
| 8b | 8 | 8192 | split | 0.371 | 0.204 | **1.82** | **1.85** | 181 (290) | 329 (397) | 0.86 / 0.87 | 0.07 / 0.08 |
| 8b | 8 | 8192 | group | 0.480 | 0.170 | **2.83** | **2.66** | 140 (369) | 395 (506) | 0.87 / 0.84 | 0.09 / 0.09 |
| 8b | 8 | 32768 | split | 1.141 | 0.760 | **1.50** | **1.60** | 235 (290) | 353 (414) | 0.96 / 0.95 | 0.03 / 0.02 |
| 8b | 8 | 32768 | group | 0.955 | 0.633 | **1.51** | **1.51** | 281 (386) | 424 (508) | 0.94 / 0.94 | 0.03 / 0.03 |
| 1b | 4 | 512 | split | 0.038 | 0.034 | 1.10 | 1.04 | 28 (33) | 31 (39) | 0.67 / 0.61 | 0.13 / 0.18 |
| 1b | 4 | 512 | group | 0.035 | 0.032 | 1.07 | 1.02 | 30 (35) | 32 (38) | 0.45 / 0.43 | 0.28 / 0.29 |
| 1b | 4 | 8192 | split | 0.186 | 0.076 | **2.45** | **2.42** | 90 (230) | 220 (255) | 0.78 / 0.70 | 0.14 / 0.22 |
| 1b | 4 | 8192 | group | 0.136 | 0.066 | **2.06** | **2.06** | 123 (270) | 253 (312) | 0.78 / 0.67 | 0.16 / 0.22 |
| 1b | 4 | 32768 | split | 0.614 | 0.243 | **2.52** | **2.52** | 109 (280) | 276 (343) | 0.94 / 0.90 | 0.04 / 0.05 |
| 1b | 4 | 32768 | group | 0.468 | 0.199 | **2.35** | **2.44** | 143 (349) | 336 (427) | 0.93 / 0.86 | 0.03 / 0.05 |
| 1b | 8 | 512 | split | 0.054 | 0.032 | **1.68** | 1.14 | 39 (72) | 65 (77) | 0.72 / 0.65 | 0.20 / 0.16 |
| 1b | 8 | 512 | group | 0.035 | 0.028 | **1.23** | 1.15 | 60 (77) | 74 (87) | 0.59 / 0.58 | 0.20 / 0.20 |
| 1b | 8 | 8192 | split | 0.316 | 0.125 | **2.53** | **2.62** | 106 (261) | 269 (316) | 0.87 / 0.83 | 0.08 / 0.09 |
| 1b | 8 | 8192 | group | 0.299 | 0.105 | **2.86** | **2.78** | 112 (326) | 320 (392) | 0.85 / 0.81 | 0.09 / 0.09 |
| 1b | 8 | 32768 | split | 0.712 | 0.457 | **1.56** | **1.56** | 189 (253) | 294 (371) | 0.95 / 0.94 | 0.03 / 0.02 |
| 1b | 8 | 32768 | group | 0.703 | 0.364 | **1.93** | **1.75** | 191 (314) | 368 (480) | 0.91 / 0.91 | 0.05 / 0.02 |

**Gate (at least 1.2x on attention time).** Passed in 45 of 48 cells of the main order and 43 of 48 of the control order;
every exception is at 512 (the optional context): 8B kv4 512 `group` (1.17 and 1.15), 1B kv4 512 `split` and `group` (1.10 and
1.07; 1.04 and 1.02), and in the control order also 1B kv8 512 `split` and `group` (1.14, 1.15). Every cell at 8192 passes (1.82-2.86 in both orders) and every cell at 32768 passes (1.49-2.52 for `split`,
1.51-2.44 for `group`). The lower of the two orders is used below. The 512 cells are noisy (0.03-0.08 ms per layer, the 1B kv8
rows value moved from 0.054 to 0.040 ms between orders), so they are neither evidence for nor against.

**Verdict by the rule of 8.1: the engine change is recommended.** `group` passes in all four 32768 cells (8B kv4 1.57, 8B kv8
1.51, 1B kv4 2.35, 1B kv8 1.75, lower of the two orders) and no `group` 8192 cell is slower than 0.95 (the lowest is 2.06). The
bandwidth figures go with it. At 32768 the median of `group` is 336-424 GB/s in the head-major layout (1B kv4 336, 1B kv8
368, 8B kv4 399, 8B kv8 424) against 143-281 GB/s in the rows layout (143, 191, 232, 281), and its best repetition reaches 531
GB/s (96% of 552, 8B kv4). The gap between 1B and 8B at 32768 narrows (1B kv8 over 8B kv8: 87% of the bandwidth against 68% in
the rows layout), which supports the section 6 explanation (short contiguous runs per head under a `kv_dim` stride). The gains
do not single out a page-sized stride: 8B with 8 local heads (stride 4096 bytes, one page) gains 2.66-2.83 for `group` at 8192
and 1B with 8 local heads (stride 2048 bytes) 2.78-2.86. I did not isolate the mechanism (prefetcher streams, TLB reach, cache
set aliasing).

**Derived effect on decode** (the microbenchmark speed applied to the profiled attention time of the first E5 job, nothing
else changing; this is an **upper-end estimate**, not a measurement, because the engine's attention phase also includes
effects the microbenchmark does not have, such as 32 layers sharing the caches and the K/V append):

| Model | Precision | Sockets | Context | group tok/s (E5) | attention ms | micro speedup (lower of two orders) | derived step ms | derived tok/s | vs group | vs split |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | 1 | 8192 | 71.39 | 4.01 | 2.78 | 11.44 | 87.4 | 1.22 | 1.23 |
| 1B | FP32 | 1 | 32768 | 48.19 | 11.03 | 1.75 | 16.03 | 62.4 | 1.29 | 1.39 |
| 1B | FP32 | 2 | 8192 | 142.32 | 1.66 | 2.06 | 6.17 | 162.0 | 1.14 | 1.24 |
| 1B | FP32 | 2 | 32768 | 84.81 | 6.74 | 2.35 | 7.92 | 126.2 | 1.49 | 1.80 |
| 1B | W8A8 | 1 | 8192 | 129.13 | 4.39 | 2.78 | 4.93 | 202.8 | 1.57 | 1.66 |
| 1B | W8A8 | 1 | 32768 | 69.28 | 10.87 | 1.75 | 9.78 | 102.2 | 1.48 | 1.63 |
| 1B | W8A8 | 2 | 8192 | 262.02 | 1.79 | 2.06 | 2.90 | 345.2 | 1.32 | 1.81 |
| 1B | W8A8 | 2 | 32768 | 100.82 | 7.57 | 2.35 | 5.57 | 179.4 | 1.78 | 2.29 |
| 8B | FP32 | 1 | 8192 | 13.66 | 16.21 | 2.66 | 63.08 | 15.9 | 1.16 | 1.19 |
| 8B | FP32 | 1 | 32768 | 10.89 | 30.73 | 1.51 | 81.51 | 12.3 | 1.13 | 1.28 |
| 8B | FP32 | 2 | 8192 | 27.09 | 7.39 | 2.34 | 32.68 | 30.6 | 1.13 | 1.21 |
| 8B | FP32 | 2 | 32768 | 20.37 | 19.53 | 1.57 | 41.99 | 23.8 | 1.17 | 1.24 |
| 8B | W8A8 | 1 | 8192 | 30.65 | 15.22 | 2.66 | 23.12 | 43.3 | 1.41 | 1.51 |
| 8B | W8A8 | 1 | 32768 | 19.82 | 31.53 | 1.51 | 39.86 | 25.1 | 1.27 | 1.61 |
| 8B | W8A8 | 2 | 8192 | 58.91 | 7.35 | 2.34 | 12.77 | 78.3 | 1.33 | 1.55 |
| 8B | W8A8 | 2 | 32768 | 34.05 | 19.17 | 1.57 | 22.40 | 44.6 | 1.31 | 1.52 |

If these held, 1B at 32768 on one socket would reach 62 tok/s (FP32, 1.39x `split`, 1.29x `group`) and 102 tok/s (W8A8, 1.63x
`split`), and the 1B W8A8 two-socket gain at 32768 would be 179 / 102 = 1.76x. The value gates of section 5 (1.3x on 1B at
32768, two-socket gain 1.5x) would be met by the layout change **if the microbenchmark's speed carries over**. This needs the
engine implementation to be measured; nothing in this section is an engine measurement.

**What the engine change touches** (not implemented; it is the next step if the owner agrees):

- KV append: `src/run.c` and `src/runq.c` write each token's K and V as one `kv_dim` row at `position * kv_dim`; head-major
  writes `n_kv_heads` pieces of `head_dim` floats at `(head * seq_len + position) * head_dim`. The cache size is unchanged;
  the per-head capacity is `seq_len` (the context length), so a head's rows are contiguous up to the context limit.
- `src/tp.c`: the local KV of the two-socket runs (4 local heads per socket) with the same indexing and the same append; the
  head split between sockets becomes a contiguous block of the cache, which also simplifies the placement of the K/V region.
- A5 placement (engine-owned HBM regions, `ROLE_KV_K` and `ROLE_KV_V`): the byte size per region is unchanged; first-touch and
  verification do not depend on the layout, but any code that assumes a row-major cache (sampled verification, the tests of
  regions) must be checked.
- Prefill attention (`k_attention_batch`, split and tiled, `attention_chunk_avx512`, `tiled_transpose_keys`) reads the same
  cache with `kv_dim` as the position stride; they already take `kv_dim` as a parameter, so they need the same stride and head
  base parameters (`k_attention_layout`'s approach), and the tests of E1/E2 would run on both layouts.
- Anything that copies or shifts the cache (context shifting, warm-context paths of `tools/compare_generation.py`, the
  batched-prefill K/V writes) must use the same indexing. Results must stay bitwise equal to the rows layout (unit tests of
  8.2 already prove it for the decode kernels).
- It would be opt-in (`LLAMA_KV_LAYOUT=head`, name not fixed) until the owner's decision table changes the default.

### 8.5 Updates to sections 1, 5 and 7

- **512 gate (section 5).** The 8B W8A8 one-socket cell no longer fails: in this job `split` itself is 0.3% below the B7 range
  and `group` equals `split` within 0.2% (verdict B). The 8B FP32 one-socket cell fails reproducibly (0.953x, 0 of 9 paired),
  with the loss in the weight phases, not in attention (8.3). The note under section 5 that "the ranges overlap within this job"
  holds for W8A8 only; for FP32 they do not.
- **8B 8192 (sections 4 and 6, item 3).** The profiled attention of `group` at 8192 on one socket is not slower than `split`'s
  (8.3); the microbenchmark's 0.77x for that shape is reproduced but does not appear in the engine.
- **Section 6, item 1 (K/V access pattern)** is confirmed by the layout probe (8.4): head-major raises the bandwidth of `group`
  to 336-424 GB/s median in all shapes at 32768, and the difference between 1B and 8B nearly disappears.
- **Decision.** The default stays `split` (owner decision). Two things are open before `group` could be proposed as the
  decode default: the 5% weight-phase loss at 8B FP32 and 512 tokens (a context-dependent choice of kernel would remove the
  attention part only if the cause is the kernel's own execution, which is not shown), and the head-major layout, which the
  data recommend implementing and measuring in the engine first.

## 9. The head-major K/V cache in the engine (E5.4)

The repository owner approved building the layout that section 8.4 recommended. It is implemented as an opt-in knob,
`LLAMA_KV_LAYOUT=rows|head` (default `rows`, the engine's layout since A5; an unknown value is refused). `head` stores a layer's
cache as `[kv_head][position][head_dim]`, each head with room for `seq_len` positions, so the cache is the same size. Every reader
and writer of the cache follows the knob: the per-token and batched K/V stores of `run.c` and `runq.c` (the projections go to a row
buffer and are stored per head after RoPE), the socket-local caches of `tp.c`, the decode kernels (the position stride and head
base of `k_attention_layout`) and the batched prefill kernels (`split`, `tiled`, the portable driver), which take the same two
strides. The code is on branch `feature/kv-head-layout`. Provenance, gates and gaps:
[decode-attention-results-validation.md](decode-attention-results-validation.md) (E5.4 block). Every number below is **measured**
unless labelled **derived**.

### 9.1 Decision rules (fixed before any cluster job)

These rules were committed before the first job was submitted; the build commit of every job is the commit that holds this
section (the validation record names it). No rule is changed after a result is seen; a cell that fails is reported as failed.

**Jobs.** One check job (C1, C2) on one Flat node. Only if every C gate passes, two bench jobs in parallel, 1B on one Flat node
and 8B on the other (one job each, each job holds both layouts, so every rows against head comparison is inside one job on one
node). Engine bench cells: contexts 512 / 8192 / 32768 (`--warm-context`), one and two sockets x 56 workers, FP32 and W8A8, decode
kernels `split` and `group`, layouts `rows` and `head`; three timed repeats in which the four (kernel, layout) variants of a cell
run once, in the reverse order in even repeats (`E5_ALTERNATE=1`), plus one profiled run (`LLAMA_PROFILE=1`) per variant and cell.
The batched prefill kernel of the engine runs (warm-up and timed prompt) is set explicitly to `LLAMA_ATTN=tiled`
(`E5_ATTN=tiled`; the campaign default `split` is overridden and recorded in `build.attention` of every report). Prefill cells
(V4): one socket, prompts of 2048 and 8192 tokens, `LLAMA_ATTN=tiled`, the rungs `avx512_fp32` (M = 128) and `amx_bf16` (M = 256)
on the FP32 checkpoint and `avx512_q8` and `amx_int8` (M = 256) on the W8A8 checkpoint, both layouts, three repeats with the
layout order alternating, 32 generated tokens. "Range" is min-max of the three timed repeats and "median" their median, as in
section 3. A run that does not produce a report counts as a failed cell.

**C1. `make test` on the hardware.** Passes if the common `make test` of the job exits 0 (all C suites at the scalar, AVX-512 and
AMX levels with 0 failures, the Python suites OK), including the new head-layout cases.

**C2. Bitwise equality of the layouts.** For 1B and 8B, FP32 and W8A8, one socket and two sockets (two with
`LLAMA_TP_CHECK=1`), with `LLAMA_KV_LAYOUT=rows` against `head` on the same binary, same staged checkpoint, same environment:
- logits of the 86-token short prompt, per token at every position, for each decode kernel `split` and `group`
  (`tools/compare_engines.py --bitwise`: max |dlogit| = 0 and argmax agreement 100% at every position);
- a warm 8192-token context: the first 8184 tokens through the tiled batched prefill (`LLAMA_ATTN=tiled`, M = 64), the last 8
  per token (`--tail 8`), for each decode kernel, `--bitwise`;
- the last row after the tiled batched prefill of the 8192 tokens, `--bitwise`;
- the AMX rungs that write K/V: AMX-BF16 on the FP32 checkpoint and AMX-INT8 on the W8A8 checkpoint, 8192 tokens (M = 256), last
  row, `--bitwise`;
- 128 greedy tokens from the warm 8192-token context and the short prompt (`tools/compare_generation.py`, tiled prefill), for each
  decode kernel: the generated token sequences must be identical;
- the Hugging Face gate (`tools/validate.py`, FP32, per token) with `LLAMA_KV_LAYOUT=head`, 1B and 8B, one socket (OpenMP) and two
  sockets (teams, `LLAMA_TP_CHECK=1`): passes if it exits 0, which is the gate of the first E5 check job.
Any failure stops the campaign: no bench job is submitted, the failure is reported, and no gate is relaxed.

**V1. Decode gain, 1B at 32768, one socket.** `(head, group)` against `(rows, split)` (the engine as shipped), FP32 and W8A8
separately. Passes if the ratio of medians is at least 1.3 **and** the ranges do not overlap (the slowest `(head, group)`
repeat is faster than the fastest `(rows, split)` repeat). This is the original E5 target.

**V2. Two-socket gain, 1B W8A8 at 32768.** Using the head layout, for each decode kernel the ratio of the two-socket median to
the one-socket median; the larger of the two kernels' ratios is the "best kernel" gain. Passes if it is at least 1.5.

**V3. 512 not slower.** For every model, precision and socket count at context 512 (8 cells) and each decode kernel separately
(16 comparisons): `head` against `rows` with the same kernel passes if the median of `head` is at least the minimum of the
`rows` repeats (so it is either faster or within the spread of `rows`). The gate passes if all 16 do; failing cells are listed.

**V4. Prefill not slower.** For each prompt (2048, 8192), rung and model, `head` against `rows` (time to first token, one socket):
the cell fails if `head` is slower beyond variability (the fastest `head` repeat is slower than the slowest `rows` repeat);
otherwise it passes. The gate covers the 8B cells of `avx512_fp32` and `amx_bf16` at both prompt lengths (4 cells); the other
rungs and 1B are reported against the same rule.

**Also reported (no gate).** `head` against `rows` for each kernel separately in every cell (median ratio and whether beyond
variability); the attention phase of the profiled run per step (`profile_seconds.decode.attention` per forward pass) and the
derived K/V bandwidth (2 x local KV heads x head_dim x 4 bytes x mean context x layers, divided by the attention time, as in
section 4), and the K/V store phase; the two-socket gains of both layouts and kernels; the time to first token of every
prefill cell; for any cell the results are compared with the first E5 job's `split` figures of section 3 only as context, since
that job ran on another day (the comparison of the gates is inside each job).

**Verdict and what it does not decide.** The layout is recommended for adoption as the default only if C1, C2, V1, V2, V3 and V4
all pass. If only some pass, the section states which, and the layout stays opt-in. Whatever the outcome, this work does not
change the default of `LLAMA_KV_LAYOUT` and does not merge: the decision table of [numa-tp-prefill-results.md](numa-tp-prefill-results.md)
and the plan files are updated by the owner.

### 9.2 What ran

All jobs ran on Flat-mode nodes of the HBM partition, exclusive, gcc `-O3 -fopenmp`. Every rows against head comparison is inside
one job on one node. Rules of 9.1 were committed before the first job (commit a35b3b4 holds them and was the build of the check
job and the two bench jobs; 2dd52c5, which only changes the campaign script and its test, was the build of the V4 job).

- **Check job 5876167** (hbm14, 2 h 54 min, exit 0): common `make test`, then `E5_KV_CHECK=1`.
- **Bench jobs** 5876582 (1B, hbm13, 1 h 14 min) and 5876584 (8B, hbm14, 5 h 16 min), run in parallel, `E5_KV_LAYOUTS="rows head"
  E5_ATTN=tiled E5_ALTERNATE=1 E5_PROFILED=1 E5_SKIP_MICRO=1 E5_PREFILL_PROMPTS="2048 8192" REPEATS=3`. Each produced all 192
  decode reports (2 precisions x 3 contexts x 2 socket counts x 4 variants x (3 timed + 1 profiled)), every one `completed`.
  **Both jobs exited 1 because their 48 prefill (V4) cells failed**: the bench stage did not write `prompt-2048.txt` and
  `prompt-8192.txt` (only the check stage did), so each of those runs stopped with `cannot open prompt file`. The decode
  reports are unaffected. A first 8B submission, job 5876583, failed after 3 s: it built in the shared staged tree at the same
  time as the 1B job and `make` lost the `config.tmp` race (`mv: cannot stat`); it ran nothing and left an empty results
  directory, so 5876584 used a new one.
- **V4 job 5876592** (hbm13, 26 min 31 s, exit 0, commit 2dd52c5): the fix (the bench stage writes the prompt files of
  `E5_PREFILL_PROMPTS`; new knob `E5_SKIP_DECODE=1` leaves out the decode runs; a dry-run test covers it) and one job holding
  the prefill cells of 1B and 8B, both layouts, three repeats alternating the layout order: 96 reports, all `completed`. The
  V4 figures below come from this job, not from the failed cells.

### 9.3 Correctness gates

| Gate | Result | Evidence |
| --- | --- | --- |
| C1 `make test` on the hardware | **PASS** (all C suites 0 failures at the scalar, AVX-512 and AMX levels; 87 Python tests OK, 1 skipped) | check job `test.log`, `phase-results.txt` |
| C2 logits, rows against head, `--bitwise` | **PASS**, 48 of 48: max \|dlogit\| = 0 and argmax 100% at every compared position. Per model and precision and socket count: the short prompt per token for `split` and `group` (86 positions), the decode path after a warm 8192-token context (8 positions, tiled prefill) for both kernels, the last row after the tiled prefill of 8192 tokens, and the AMX rung (AMX-BF16 on FP32, AMX-INT8 on W8A8) | `E5/compare-*-kv-*.json` |
| C2 generation, 128 greedy tokens from a warm 8192-token context, both decode kernels | **PASS**, 16 of 16 identical (sequences and prompts) | `E5/gen-*-rows-vs-head.json` |
| C2 Hugging Face, FP32 head layout, 1B and 8B, one and two sockets | **PASS**, 4 of 4 (exit 0; largest max \|dlogit\| seen 6.7e-5, 8B two sockets) | `correctness-*-fp32-head*.json` |

### 9.4 Decode throughput, rows against head
(tables are produced from the reports of the 1B and 8B bench jobs)

#### Decode tok/s (head against rows, per kernel)

Median [min-max] of three repeats, tok/s. Ratio = median head / median rows. "Beyond" = ranges do not overlap.

| Model | Precision | Sockets | Context | Kernel | rows | head | head / rows | Beyond variability |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 1B | FP32 | 1 | 512 | split | 91.37 [90.25-96.17] | 92.30 [91.38-97.42] | 1.010 | overlap |
| 1B | FP32 | 1 | 512 | group | 91.15 [90.88-95.66] | 93.93 [91.88-94.67] | 1.031 | overlap |
| 1B | FP32 | 1 | 8192 | split | 70.27 [69.94-70.29] | 79.99 [79.29-80.44] | 1.138 | faster |
| 1B | FP32 | 1 | 8192 | group | 70.88 [70.82-71.25] | 84.29 [84.09-84.34] | 1.189 | faster |
| 1B | FP32 | 1 | 32768 | split | 44.84 [44.73-44.88] | 53.49 [53.36-53.69] | 1.193 | faster |
| 1B | FP32 | 1 | 32768 | group | 47.66 [47.63-47.81] | 60.23 [59.79-60.98] | 1.264 | faster |
| 1B | FP32 | 2 | 512 | split | 169.72 [169.09-172.49] | 172.49 [169.37-178.22] | 1.016 | overlap |
| 1B | FP32 | 2 | 512 | group | 170.57 [169.27-170.76] | 173.16 [170.95-174.68] | 1.015 | faster |
| 1B | FP32 | 2 | 8192 | split | 129.27 [129.22-129.94] | 146.80 [146.69-147.35] | 1.136 | faster |
| 1B | FP32 | 2 | 8192 | group | 140.62 [140.58-141.16] | 150.81 [150.75-152.76] | 1.072 | faster |
| 1B | FP32 | 2 | 32768 | split | 70.66 [70.60-71.10] | 102.55 [101.34-102.97] | 1.451 | faster |
| 1B | FP32 | 2 | 32768 | group | 85.30 [53.28-85.81] | 114.61 [114.52-114.95] | 1.344 | faster |
| 1B | W8A8 | 1 | 512 | split | 260.31 [247.27-260.72] | 275.95 [258.42-278.12] | 1.060 | overlap |
| 1B | W8A8 | 1 | 512 | group | 262.42 [258.10-268.85] | 282.23 [261.23-296.04] | 1.075 | overlap |
| 1B | W8A8 | 1 | 8192 | split | 120.51 [120.26-123.22] | 181.04 [180.44-181.39] | 1.502 | faster |
| 1B | W8A8 | 1 | 8192 | group | 126.86 [122.87-128.80] | 203.94 [200.78-205.23] | 1.608 | faster |
| 1B | W8A8 | 1 | 32768 | split | 62.52 [62.36-62.66] | 85.07 [84.88-85.31] | 1.361 | faster |
| 1B | W8A8 | 1 | 32768 | group | 68.96 [68.15-69.03] | 102.89 [102.35-103.39] | 1.492 | faster |
| 1B | W8A8 | 2 | 512 | split | 461.34 [389.95-470.33] | 449.20 [412.06-450.58] | 0.974 | overlap |
| 1B | W8A8 | 2 | 512 | group | 468.96 [408.03-479.29] | 453.78 [411.30-490.68] | 0.968 | overlap |
| 1B | W8A8 | 2 | 8192 | split | 205.08 [198.90-206.40] | 327.63 [326.79-336.93] | 1.598 | faster |
| 1B | W8A8 | 2 | 8192 | group | 259.71 [258.56-262.10] | 337.07 [327.33-352.04] | 1.298 | faster |
| 1B | W8A8 | 2 | 32768 | split | 78.14 [77.83-78.23] | 158.98 [158.91-159.87] | 2.034 | faster |
| 1B | W8A8 | 2 | 32768 | group | 100.81 [100.68-101.03] | 191.41 [188.78-191.64] | 1.899 | faster |
| 8B | FP32 | 1 | 512 | split | 16.45 [16.39-16.46] | 16.60 [16.60-16.65] | 1.010 | faster |
| 8B | FP32 | 1 | 512 | group | 15.65 [15.61-15.69] | 15.60 [15.59-15.61] | 0.997 | overlap |
| 8B | FP32 | 1 | 8192 | split | 13.25 [13.25-13.34] | 14.84 [14.80-14.92] | 1.120 | faster |
| 8B | FP32 | 1 | 8192 | group | 13.69 [13.68-13.72] | 15.23 [14.92-15.25] | 1.112 | faster |
| 8B | FP32 | 1 | 32768 | split | 9.62 [9.45-9.62] | 11.41 [11.36-11.45] | 1.186 | faster |
| 8B | FP32 | 1 | 32768 | group | 10.91 [10.90-10.91] | 12.25 [12.25-12.26] | 1.123 | faster |
| 8B | FP32 | 2 | 512 | split | 30.49 [30.45-30.53] | 30.86 [30.86-30.94] | 1.012 | faster |
| 8B | FP32 | 2 | 512 | group | 30.71 [30.70-30.75] | 30.94 [30.90-30.97] | 1.008 | faster |
| 8B | FP32 | 2 | 8192 | split | 25.41 [25.37-25.53] | 28.46 [28.45-28.54] | 1.120 | faster |
| 8B | FP32 | 2 | 8192 | group | 27.17 [27.12-27.17] | 29.21 [29.12-29.28] | 1.075 | faster |
| 8B | FP32 | 2 | 32768 | split | 19.16 [19.04-19.20] | 22.11 [22.03-22.14] | 1.154 | faster |
| 8B | FP32 | 2 | 32768 | group | 20.29 [20.18-20.39] | 23.77 [23.72-23.88] | 1.171 | faster |
| 8B | W8A8 | 1 | 512 | split | 49.33 [49.28-49.41] | 50.02 [38.67-50.34] | 1.014 | overlap |
| 8B | W8A8 | 1 | 512 | group | 49.38 [49.10-49.46] | 49.68 [49.64-49.70] | 1.006 | faster |
| 8B | W8A8 | 1 | 8192 | split | 28.79 [28.61-28.88] | 37.11 [37.11-37.26] | 1.289 | faster |
| 8B | W8A8 | 1 | 8192 | group | 30.62 [30.57-30.65] | 39.98 [39.86-39.98] | 1.306 | faster |
| 8B | W8A8 | 1 | 32768 | split | 15.78 [15.77-15.78] | 21.41 [21.40-21.43] | 1.357 | faster |
| 8B | W8A8 | 1 | 32768 | group | 19.83 [19.78-19.84] | 24.22 [24.02-24.31] | 1.221 | faster |
| 8B | W8A8 | 2 | 512 | split | 90.11 [89.98-90.27] | 91.60 [91.47-93.38] | 1.017 | faster |
| 8B | W8A8 | 2 | 512 | group | 91.22 [90.68-91.46] | 92.94 [92.74-93.67] | 1.019 | faster |
| 8B | W8A8 | 2 | 8192 | split | 50.56 [50.18-51.09] | 66.66 [66.20-66.95] | 1.318 | faster |
| 8B | W8A8 | 2 | 8192 | group | 58.64 [58.63-58.86] | 72.21 [71.88-72.23] | 1.231 | faster |
| 8B | W8A8 | 2 | 32768 | split | 29.29 [29.20-29.55] | 39.09 [38.94-39.19] | 1.335 | faster |
| 8B | W8A8 | 2 | 32768 | group | 33.71 [33.47-33.78] | 44.74 [44.40-44.84] | 1.327 | faster |

#### V1 (1B, 32768, one socket): (head, group) against (rows, split)

| Precision | rows+split | head+group | ratio of medians | ranges apart | verdict |
| --- | --- | --- | ---: | --- | --- |
| FP32 | 44.84 [44.73-44.88] | 60.23 [59.79-60.98] | 1.343 | yes | PASS |
| W8A8 | 62.52 [62.36-62.66] | 102.89 [102.35-103.39] | 1.646 | yes | PASS |

#### V2 (1B W8A8, 32768): two-socket gain with the head layout

| Kernel | one socket | two sockets | s2 / s1 |
| --- | --- | --- | ---: |
| split | 85.07 [84.88-85.31] | 158.98 [158.91-159.87] | 1.869 |
| group | 102.89 [102.35-103.39] | 191.41 [188.78-191.64] | 1.860 |

Best kernel gain 1.869: PASS (rows layout for comparison: split 1.250, group 1.462).

#### Two-socket gain s2 / s1 (medians, derived)

| Model | Precision | Context | rows split | rows group | head split | head group |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | 512 | 1.86 | 1.87 | 1.87 | 1.84 |
| 1B | FP32 | 8192 | 1.84 | 1.98 | 1.84 | 1.79 |
| 1B | FP32 | 32768 | 1.58 | 1.79 | 1.92 | 1.90 |
| 1B | W8A8 | 512 | 1.77 | 1.79 | 1.63 | 1.61 |
| 1B | W8A8 | 8192 | 1.70 | 2.05 | 1.81 | 1.65 |
| 1B | W8A8 | 32768 | 1.25 | 1.46 | 1.87 | 1.86 |
| 8B | FP32 | 512 | 1.85 | 1.96 | 1.86 | 1.98 |
| 8B | FP32 | 8192 | 1.92 | 1.98 | 1.92 | 1.92 |
| 8B | FP32 | 32768 | 1.99 | 1.86 | 1.94 | 1.94 |
| 8B | W8A8 | 512 | 1.83 | 1.85 | 1.83 | 1.87 |
| 8B | W8A8 | 8192 | 1.76 | 1.91 | 1.80 | 1.81 |
| 8B | W8A8 | 32768 | 1.86 | 1.70 | 1.83 | 1.85 |

#### V3: 512 not slower (median head >= min rows)

| Model | Precision | Sockets | Kernel | rows | head | median head >= min rows | verdict |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| 1B | FP32 | 1 | split | 91.37 [90.25-96.17] | 92.30 [91.38-97.42] | 92.30 vs 90.25 | PASS |
| 1B | FP32 | 1 | group | 91.15 [90.88-95.66] | 93.93 [91.88-94.67] | 93.93 vs 90.88 | PASS |
| 1B | FP32 | 2 | split | 169.72 [169.09-172.49] | 172.49 [169.37-178.22] | 172.49 vs 169.09 | PASS |
| 1B | FP32 | 2 | group | 170.57 [169.27-170.76] | 173.16 [170.95-174.68] | 173.16 vs 169.27 | PASS |
| 1B | W8A8 | 1 | split | 260.31 [247.27-260.72] | 275.95 [258.42-278.12] | 275.95 vs 247.27 | PASS |
| 1B | W8A8 | 1 | group | 262.42 [258.10-268.85] | 282.23 [261.23-296.04] | 282.23 vs 258.10 | PASS |
| 1B | W8A8 | 2 | split | 461.34 [389.95-470.33] | 449.20 [412.06-450.58] | 449.20 vs 389.95 | PASS |
| 1B | W8A8 | 2 | group | 468.96 [408.03-479.29] | 453.78 [411.30-490.68] | 453.78 vs 408.03 | PASS |
| 8B | FP32 | 1 | split | 16.45 [16.39-16.46] | 16.60 [16.60-16.65] | 16.60 vs 16.39 | PASS |
| 8B | FP32 | 1 | group | 15.65 [15.61-15.69] | 15.60 [15.59-15.61] | 15.60 vs 15.61 | FAIL |
| 8B | FP32 | 2 | split | 30.49 [30.45-30.53] | 30.86 [30.86-30.94] | 30.86 vs 30.45 | PASS |
| 8B | FP32 | 2 | group | 30.71 [30.70-30.75] | 30.94 [30.90-30.97] | 30.94 vs 30.70 | PASS |
| 8B | W8A8 | 1 | split | 49.33 [49.28-49.41] | 50.02 [38.67-50.34] | 50.02 vs 49.28 | PASS |
| 8B | W8A8 | 1 | group | 49.38 [49.10-49.46] | 49.68 [49.64-49.70] | 49.68 vs 49.10 | PASS |
| 8B | W8A8 | 2 | split | 90.11 [89.98-90.27] | 91.60 [91.47-93.38] | 91.60 vs 89.98 | PASS |
| 8B | W8A8 | 2 | group | 91.22 [90.68-91.46] | 92.94 [92.74-93.67] | 92.94 vs 90.68 | PASS |

V3: FAIL [('8b', 'fp32', 1, 'group')]

#### Attention phase and derived K/V bandwidth (one profiled run per variant)

Attention ms per decode step, K/V store ms per step, and bandwidth = 2 x local KV heads x head_dim x 4 bytes x mean context x layers / attention time (derived; mean context = warm context + 86 + 64). % of 552 GB/s per socket.

| Model | Precision | Sockets | Context | Kernel | rows attn ms | head attn ms | attn rows / head | rows GB/s | head GB/s | head % of 552 | head kv_store ms |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | 1 | 8192 | split | 4.40 | 2.43 | 1.81 | 124 | 225 | 41% | 0.007 |
| 1B | FP32 | 1 | 8192 | group | 4.02 | 1.97 | 2.04 | 136 | 278 | 50% | 0.007 |
| 1B | FP32 | 1 | 32768 | split | 12.11 | 8.48 | 1.43 | 178 | 254 | 46% | 0.008 |
| 1B | FP32 | 1 | 32768 | group | 10.99 | 6.47 | 1.70 | 196 | 333 | 60% | 0.008 |
| 1B | FP32 | 2 | 8192 | split | 2.59 | 1.48 | 1.75 | 106 | 185 | 33% | 0.004 |
| 1B | FP32 | 2 | 8192 | group | 1.63 | 1.28 | 1.27 | 168 | 214 | 39% | 0.004 |
| 1B | FP32 | 2 | 32768 | split | 9.23 | 4.43 | 2.08 | 117 | 243 | 44% | 0.004 |
| 1B | FP32 | 2 | 32768 | group | 6.66 | 3.50 | 1.91 | 162 | 309 | 56% | 0.005 |
| 1B | W8A8 | 1 | 8192 | split | 4.86 | 2.39 | 2.04 | 112 | 229 | 41% | 0.006 |
| 1B | W8A8 | 1 | 8192 | group | 4.38 | 1.85 | 2.37 | 125 | 295 | 54% | 0.006 |
| 1B | W8A8 | 1 | 32768 | split | 12.41 | 8.35 | 1.49 | 174 | 258 | 47% | 0.007 |
| 1B | W8A8 | 1 | 32768 | group | 10.92 | 6.41 | 1.70 | 198 | 337 | 61% | 0.008 |
| 1B | W8A8 | 2 | 8192 | split | 2.81 | 1.24 | 2.26 | 97 | 220 | 40% | 0.004 |
| 1B | W8A8 | 2 | 8192 | group | 1.81 | 1.18 | 1.53 | 151 | 232 | 42% | 0.004 |
| 1B | W8A8 | 2 | 32768 | split | 10.43 | 4.43 | 2.35 | 103 | 244 | 44% | 0.005 |
| 1B | W8A8 | 2 | 32768 | group | 7.60 | 3.34 | 2.28 | 142 | 323 | 59% | 0.004 |
| 8B | FP32 | 1 | 8192 | split | 15.77 | 8.53 | 1.85 | 139 | 256 | 46% | 0.032 |
| 8B | FP32 | 1 | 8192 | group | 15.86 | 6.64 | 2.39 | 138 | 330 | 60% | 0.032 |
| 8B | FP32 | 1 | 32768 | split | 44.10 | 27.94 | 1.58 | 196 | 309 | 56% | 0.034 |
| 8B | FP32 | 1 | 32768 | group | 30.87 | 22.44 | 1.38 | 280 | 385 | 70% | 0.034 |
| 8B | FP32 | 2 | 8192 | split | 10.05 | 5.22 | 1.93 | 109 | 209 | 38% | 0.014 |
| 8B | FP32 | 2 | 8192 | group | 7.31 | 4.17 | 1.75 | 150 | 262 | 48% | 0.014 |
| 8B | FP32 | 2 | 32768 | split | 22.62 | 15.45 | 1.46 | 191 | 279 | 51% | 0.015 |
| 8B | FP32 | 2 | 32768 | group | 19.46 | 12.23 | 1.59 | 222 | 353 | 64% | 0.016 |
| 8B | W8A8 | 1 | 8192 | split | 16.09 | 8.20 | 1.96 | 136 | 267 | 48% | 0.031 |
| 8B | W8A8 | 1 | 8192 | group | 15.09 | 6.54 | 2.31 | 145 | 334 | 61% | 0.032 |
| 8B | W8A8 | 1 | 32768 | split | 44.12 | 27.87 | 1.58 | 196 | 310 | 56% | 0.033 |
| 8B | W8A8 | 1 | 32768 | group | 31.24 | 22.51 | 1.39 | 276 | 383 | 69% | 0.034 |
| 8B | W8A8 | 2 | 8192 | split | 9.48 | 4.89 | 1.94 | 115 | 223 | 40% | 0.015 |
| 8B | W8A8 | 2 | 8192 | group | 7.44 | 4.06 | 1.84 | 147 | 270 | 49% | 0.012 |
| 8B | W8A8 | 2 | 32768 | split | 23.50 | 15.28 | 1.54 | 184 | 282 | 51% | 0.020 |
| 8B | W8A8 | 2 | 32768 | group | 19.40 | 12.24 | 1.58 | 222 | 352 | 64% | 0.016 |

#### V4: time to first token, one socket, tiled prefill (seconds, median [min-max] of three)

| Model | Rung | Prompt | rows | head | head / rows | verdict |
| --- | --- | ---: | --- | --- | ---: | --- |
| 1B | avx512_fp32 | 2048 | 1.66 [1.63-2.23] | 1.64 [1.60-1.66] | 0.987 | overlap: pass |
| 1B | avx512_fp32 | 8192 | 8.48 [8.47-8.50] | 8.23 [8.18-8.25] | 0.970 | faster beyond variability |
| 1B | amx_bf16 | 2048 | 0.50 [0.50-0.51] | 0.51 [0.51-0.51] | 1.016 | head slower beyond variability: FAIL |
| 1B | amx_bf16 | 8192 | 3.76 [3.76-3.79] | 3.50 [3.49-3.50] | 0.929 | faster beyond variability |
| 1B | avx512_q8 | 2048 | 1.53 [1.46-1.54] | 1.49 [1.48-1.55] | 0.974 | overlap: pass |
| 1B | avx512_q8 | 8192 | 7.79 [7.79-7.83] | 7.49 [7.47-7.51] | 0.962 | faster beyond variability |
| 1B | amx_int8 | 2048 | 0.57 [0.56-0.58] | 0.58 [0.58-0.58] | 1.022 | head slower beyond variability: FAIL |
| 1B | amx_int8 | 8192 | 4.03 [4.01-4.04] | 3.77 [3.72-3.77] | 0.935 | faster beyond variability |
| 8B | avx512_fp32 | 2048 | 10.00 [9.92-10.01] | 9.97 [9.97-9.99] | 0.997 | overlap: pass (gate) |
| 8B | avx512_fp32 | 8192 | 47.04 [47.02-47.48] | 45.91 [45.89-45.95] | 0.976 | faster beyond variability (gate) |
| 8B | amx_bf16 | 2048 | 2.43 [2.43-2.44] | 2.41 [2.40-2.42] | 0.991 | faster beyond variability (gate) |
| 8B | amx_bf16 | 8192 | 16.19 [16.12-16.22] | 14.90 [14.90-14.95] | 0.921 | faster beyond variability (gate) |
| 8B | avx512_q8 | 2048 | 8.90 [8.79-8.91] | 8.93 [8.93-8.96] | 1.004 | head slower beyond variability: FAIL |
| 8B | avx512_q8 | 8192 | 42.78 [42.73-42.81] | 42.80 [42.78-42.81] | 1.000 | overlap: pass |
| 8B | amx_int8 | 2048 | 2.37 [2.34-2.44] | 2.34 [2.32-2.35] | 0.986 | overlap: pass |
| 8B | amx_int8 | 8192 | 16.30 [16.25-16.34] | 14.93 [14.90-14.97] | 0.916 | faster beyond variability |

V4 (8B avx512_fp32 and amx_bf16, both prompts): PASS



### 9.5 Gate verdicts

| Gate | Criterion | Measured | Verdict |
| --- | --- | --- | --- |
| C1 | `make test` on hardware | pass | **PASS** |
| C2 | bitwise, generated tokens identical, HF | 48 of 48, 16 of 16, 4 of 4 | **PASS** |
| V1 | 1B 32768 one socket, (head, group) over (rows, split) at least 1.3x beyond variability | FP32 1.343x (ranges apart), W8A8 1.646x (ranges apart) | **PASS** |
| V2 | 1B W8A8 two-socket gain at 32768, head, best kernel, at least 1.5x | 1.869x (`split`), 1.860x (`group`); with rows 1.250x and 1.462x | **PASS** |
| V3 | 512 not slower: median head at least the minimum of rows, 16 comparisons | 15 of 16 pass; **8B FP32 one socket `group`: head 15.60 [15.59-15.61] against rows 15.65 [15.61-15.69], median below the rows minimum by 0.01 tok/s (0.06%)** | **FAIL** (one comparison) |
| V4 | prefill TTFT not slower beyond variability, 8B `avx512_fp32` and `amx_bf16` at 2048 and 8192 | 4 of 4 not slower (0.997x, 0.976x, 0.991x, 0.921x of the rows time) | **PASS** |

V3 fails by the rule as written. The failing cell is the one that has failed every 512 comparison since the first E5 job:
8B FP32 one socket with `group` is 4.9% below `split` at 512 in the rows layout (15.65 against 16.45) and stays there with
the head layout (15.60); the head layout does not change that kernel effect, and `split` with head is above rows in that
cell (16.60 against 16.45). The difference between head and rows for `group` there is smaller than the rows spread. It is
reported as a failed comparison, not as a pass.

The 12 V4 cells outside the gate (1B on all four rungs, 8B on the two Q8 rungs) are faster with head beyond variability in 5
of 12 cells (all at 8192 tokens, 3-8% shorter), overlap in 4 of 12 and **slower beyond variability in 3 of 12 cells by the rule of 9.1: 1B
AMX-BF16 at 2048 (1.016x, 0.50 against 0.51 s), 1B AMX-INT8 at 2048 (1.022x, 0.57 against 0.58 s) and 8B AVX-512 Q8 at 2048
(1.004x, 8.90 against 8.93 s)**. They are reported and not gated.

### 9.6 Findings

- **Decode.** Head-major is faster than rows with the same kernel beyond variability in all 32 comparisons at 8192 and 32768
  (model x precision x sockets x context x kernel): 1B 1.07-1.61x at 8192 and 1.19-2.03x at 32768, 8B 1.08-1.32x at 8192 and
  1.12-1.36x at 32768. The gain is largest for W8A8 on 1B and two sockets (2.03x with `split`, 1.90x with `group` at 32768), where the K/V
  access dominated the step. At 512 tokens the ratios are 0.97-1.08: 7 of 16 comparisons are faster beyond variability (mostly 8B), 9
  overlap, none is slower beyond variability; the V3 failure is the one cell where the median is below the rows minimum.
- **Original E5 target.** (head, group) reaches 1.34x (FP32) and 1.65x (W8A8) of the engine as shipped on 1B at 32768 one socket
  (E5 `group` alone: 1.08x and 1.10x), and the two-socket W8A8 gain at 32768 rises from 1.25x (rows, `split`) to 1.87x.
  Both E5 value gates that `group` alone missed are met by the layout.
- **Attention phase.** In the profiled runs the attention time per step falls by 1.3-2.4x (table in 9.4); the derived K/V
  bandwidth of `group` on one socket rises from 125-280 GB/s (rows) to 278-385 GB/s (head), 50-70% of the 552 GB/s per
  socket at 8192 and 32768. The K/V store phase costs 0.004-0.034 ms per step (under 0.1% of a step), i.e. the per-head
  store of one row is not a cost. These bandwidths are **derived** from one profiled run per cell.
- **Prefill.** Time to first token is equal or lower with head at 8192 tokens in every cell (0.92-1.00x); at 2048 tokens it is
  equal within 2% (three cells 0.4-2.2% slower beyond variability, listed in 9.5). The batched K/V store through a row buffer
  does not show up.

### 9.7 Conclusion

C1, C2, V1, V2 and V4 pass; V3 fails in one of 16 comparisons (8B FP32, one socket, `group`, 512 tokens, a 0.06% difference
that is the previously known `group` effect at that cell). By the rule of 9.1 the layout is therefore **not recommended for
adoption as the default**, although the evidence is strongly in its favour everywhere else: it is bitwise identical to rows
on every path tested, it meets both original E5 value targets, and it never hurts beyond the spread except in the V3 cell and
three 2048-token prefill cells of 0.4-2.2%. It stays opt-in (`LLAMA_KV_LAYOUT=head`); the decision to make it the default
(possibly together with `LLAMA_ATTN_DECODE=group` for contexts of 8192 and above) and to waive or keep the V3 cell is the
owner's. Nothing was merged and no default was changed.
