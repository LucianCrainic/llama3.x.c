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
  the 8B 8-head shape).
- E6 (batched decode design) can use these measurements. Decode attention at long contexts is limited by the access pattern,
  not by bandwidth. Each sequence has its own K/V, so batching sequences over shared weights would not by itself shorten
  attention time per sequence (an inference, not measured).
