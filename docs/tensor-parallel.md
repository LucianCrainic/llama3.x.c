# Two-socket tensor-parallel decode

> **Correction (2026-11-21):** HBM byte figures derived from `hbm_cas_rd` counts were computed at 64 B per count; the counter counts 32 B (calibrated in job 5892172, see [xengine-results-validation.md](xengine-results-validation.md)). Affected figures below are halved and marked †.

**Status: run (2026-11-17, jobs 5839003 and 5839004, commit 2aad124, node cresco8-hbm14).** The decode campaign of phase B7 ran to
completion for the 1B and 8B models (FP32, W8A8 and W8A32) with 3 repeats: 8 of 8 correctness gates passed, and every profiled
run is free of the revisit triggers. Provenance and gates: [the validation record](tensor-parallel-validation.md). Plan:
[B7](plans/B-tensor-parallel/B7-two-socket-decode-campaign.md). Branch: `feature/tp-b7-campaign` (scripts and results documents only).
The generated tables of the whole campaign (every run, profile and probe) are in [tensor-parallel-summary-table.md](tensor-parallel-summary-table.md).

Question: does splitting one model instance across both sockets (head and FFN-channel shards, socket-local KV, two cross-socket
reductions per layer) beat the best one-socket configuration (A6) for single-request decode, and where does the time go. Judged
by the study 10.4 criteria only. Medians of 3 repeats (ranges in the tables); 56 threads per socket unless stated; contexts are
retained with `--warm-context` and 128 tokens are timed; weights on HBM nodes 2 and 3, THP off.

## Result in one paragraph

Two sockets win everywhere the matrix looked, beyond run-to-run variability. 8B FP32 decodes at 30.6 / 25.5 / 19.2 tok/s at
contexts of 512 / 8192 / 32768 against 16.4 / 13.3 / 9.7 on one socket (1.87x / 1.92x / 1.99x); 8B W8A8 at 91.3 / 50.6 / 29.6
against 49.5 / 28.5 / 15.6 (1.84x / 1.77x / 1.90x). On 1B the gain is 1.85x / 1.85x / 1.57x (FP32) and 1.73x / 1.59x / 1.26x (W8A8),
falling at long contexts where attention is 65-81% of the token. The cross-socket reductions are not a cost: the fused reduction
shows 0.000 ms and the waits at most 0.022 ms per token (under 0.1% of the token at 8B), and the largest imbalance between sockets
is 1.2 ms at 8192 and 32768 tokens (attention). No profiled run reaches a revisit trigger. Tokens per second scale with the worker
count (8B FP32 12.8 / 22.4 / 25.5 / 30.6 at 14 / 28 / 42 / 56 per socket), so 56 per socket is the best setting. The packed4 FP32
layout is bitwise identical to the rows layout and 3.2% faster at context 512 only (31.61 against 30.63 tok/s on 8B), a tie at
8192 and 32768 and none on 1B, for 14.0 GB more HBM per socket: not adopted. The 112-thread control without tensor parallelism
reproduces the old figures (8B FP32 8.24 against 8.17 tok/s).

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` in each campaign directory. Rows below: 8B, context 512, the headline
configurations (one socket, two sockets in the rows layout, two sockets in packed4, and the 112-thread control without tensor parallelism).
Every run is in [tensor-parallel-summary-table.md](tensor-parallel-summary-table.md).

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 8b-fp32-omp-file-s2-t112-p512-xctl | 3 | 2 | file | off | - | avx512 | fp32 | 512 | 512 | 0.556 [0.556-0.564] | 173.20 [173.09-176.79] | 8.24 [6.18-8.25] | - | - | 4164.0 † | 22.5 | 298434 |
| 8b-fp32-teams-hbm-s1-t56-p512 | 3 | 1 | hbm (verified) | off | - | avx512 | fp32 | 512 | 512 | 0.491 [0.491-0.496] | 196.27 [193.29-196.34] | 16.42 [16.13-16.44] | - | - | 4118.8 † | 9.2 | 407 |
| 8b-fp32-teams-hbm-s2-t56-p512 | 3 | 2 | file | off | - | avx512 | fp32 | 512 | 512 | 0.300 [0.277-0.304] | 347.07 [343.73-380.68] | 30.63 [30.59-30.64] | - | - | 4109.8 † | 50.6 | 2408 |
| 8b-fp32-teams-hbm-s2-t56-p512-xp4 | 3 | 2 | file | off | - | avx512 | fp32 | 512 | 512 | 0.290 [0.286-0.300] | 361.43 [347.17-366.92] | 31.61 [31.61-31.68] | - | - | 4142.6 † | 58.6 | 2864 |
| 8b-int8-omp-file-s2-t112-p512-xctl | 3 | 2 | file | off | - | avx512 | int8_q8_0 | 512 | 512 | 0.384 [0.365-0.548] | 269.36 [172.65-287.99] | 24.67 [24.37-24.76] | - | - | 1127.6 † | 5.2 | 81553 |
| 8b-int8-teams-hbm-s1-t56-p512 | 3 | 1 | hbm (verified) | off | - | avx512 | int8_q8_0 | 512 | 512 | 0.443 [0.442-0.486] | 219.81 [197.79-220.28] | 49.52 [49.52-49.73] | - | - | 1108.2 † | 1.5 | 30 |
| 8b-int8-teams-hbm-s2-t56-p512 | 3 | 2 | file | off | - | avx512 | int8_q8_0 | 512 | 512 | 0.262 [0.255-0.265] | 400.42 [395.68-415.41] | 91.25 [90.08-91.35] | - | - | 1099.7 † | 10.0 | 725 |

## Decode tok/s by retained context

`tools/campaign_tables.py` writes this table (median [range] over the repeats). Gain is two sockets x 56 over one socket x 56;
the last column compares the packed4 copy of the FP32 shards (`LLAMA_FP32_LAYOUT=packed4`) with the row layout. The worker-count
sweep (2x14, 2x28, 2x42) and the control were run at context 512 only. Single repeats are low outliers in three cells (1B FP32 2x56 at
512: 132.0; 1B FP32 2x56p4 at 32768: 48.4; 8B FP32 control: 6.18); the medians are unaffected.

| Model | Precision | Context | 1x56 | 2x14 | 2x28 | 2x42 | 2x56 | 2x56p4 | control | Gain 2x56 / 1x56 | Beyond variability | packed4 / rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1b | fp32 | 512 | 91.33 [91.07-96.73] | 79.06 [78.98-79.32] | 139.43 [139.05-141.37] | 156.02 [155.92-157.86] | 168.85 [132.03-169.66] | 168.89 [167.69-172.11] | 45.25 [41.04-45.47] | 1.85 | yes | 1.00 |
| 1b | fp32 | 8192 | 70.49 [70.38-70.67] | - | - | - | 130.68 [130.08-131.08] | 128.42 [128.27-129.18] | - | 1.85 | yes | 0.98 |
| 1b | fp32 | 32768 | 44.84 [44.82-45.18] | - | - | - | 70.22 [69.94-70.23] | 69.37 [48.41-69.49] | - | 1.57 | yes | 0.99 |
| 1b | int8 | 512 | 269.09 [259.53-284.52] | 245.71 [245.62-246.17] | 414.54 [413.96-414.75] | 469.57 [453.60-479.94] | 466.84 [436.15-469.37] | - | 108.94 [108.31-109.96] | 1.73 | yes | - |
| 1b | int8 | 8192 | 121.82 [121.32-122.04] | - | - | - | 193.72 [192.64-195.00] | - | - | 1.59 | yes | - |
| 1b | int8 | 32768 | 62.61 [62.58-62.62] | - | - | - | 78.72 [78.54-78.83] | - | - | 1.26 | yes | - |
| 8b | fp32 | 512 | 16.42 [16.13-16.44] | 12.78 [12.78-12.79] | 22.43 [22.32-22.46] | 25.53 [25.52-25.61] | 30.63 [30.59-30.64] | 31.61 [31.61-31.68] | 8.24 [6.18-8.25] | 1.87 | yes | 1.03 |
| 8b | fp32 | 8192 | 13.29 [13.24-13.42] | - | - | - | 25.50 [25.35-25.58] | 25.20 [25.14-25.57] | - | 1.92 | yes | 0.99 |
| 8b | fp32 | 32768 | 9.66 [9.59-9.66] | - | - | - | 19.18 [19.17-19.28] | 19.30 [19.28-19.44] | - | 1.99 | yes | 1.01 |
| 8b | int8 | 512 | 49.52 [49.52-49.73] | 38.80 [38.79-38.83] | 68.46 [68.41-68.51] | 82.30 [81.57-83.27] | 91.25 [90.08-91.35] | - | 24.67 [24.37-24.76] | 1.84 | yes | - |
| 8b | int8 | 8192 | 28.54 [28.48-28.57] | - | - | - | 50.57 [50.47-51.20] | - | - | 1.77 | yes | - |
| 8b | int8 | 32768 | 15.59 [15.59-15.64] | - | - | - | 29.56 [29.52-29.58] | - | - | 1.90 | yes | - |
| 1b | w8a32 | 512 | - | - | - | - | 370.57 [366.17-370.88] | - | - | - | - | - |
| 1b | w8a32 | 8192 | - | - | - | - | 195.84 [195.25-197.05] | - | - | - | - | - |
| 8b | w8a32 | 512 | - | - | - | - | 80.66 [80.65-81.58] | - | - | - | - | - |
| 8b | w8a32 | 8192 | - | - | - | - | 47.39 [47.36-47.58] | - | - | - | - | - |

W8A32 (two sockets only; its warm-up is per token) has no one-socket counterpart in this campaign. At the 2304-token retained
context of C7, one socket reached 37.4 tok/s on 8B W8A32, against 47.4 for two sockets at 8192 tokens.

Worker counts at context 512 (tok/s, 8B): FP32 12.78 / 22.43 / 25.53 / 30.63 and W8A8 38.80 / 68.46 / 82.30 / 91.25 for 14 / 28 / 42 / 56
workers per socket; two sockets with 28 workers each (56 threads in all) already beat one socket with 56 (22.4 against 16.4 FP32,
68.5 against 49.5 W8A8). On 1B W8A8 the 42-worker setting equals 56 (469.6 [453.6-479.9] against 466.8 [436.2-469.4]); nothing else
saturates.

## Per-socket phase times (profiled runs, ms per token)

wo and ffn include their reduction; reduce and wait are its components. Imbalance = |socket 0 - socket 1|. Rows for the 2 x 56
runs at the three contexts (the 14/28/42 and packed4 profiles are in the generated table).

| Run | Socket | qkv | attention | wo | ffn | classifier | reduce | wait | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1b-fp32-teams-hbm-s2-t56-p32768-xprof | 0 | 0.419 | 9.251 | 0.458 | 2.981 | 0.941 | 0.000 | 0.009 | 14.142 |
| 1b-fp32-teams-hbm-s2-t56-p32768-xprof | 1 | 0.411 | 9.113 | 0.596 | 2.985 | 0.908 | 0.000 | 0.008 | 14.098 |
| 1b-fp32-teams-hbm-s2-t56-p32768-xprof | imbalance | 0.008 | 0.138 | 0.138 | 0.004 | 0.033 | 0.000 | 0.001 | 0.044 |
| 1b-fp32-teams-hbm-s2-t56-p512-xprof | 0 | 0.448 | 0.390 | 0.408 | 3.349 | 1.059 | 0.000 | 0.008 | 5.748 |
| 1b-fp32-teams-hbm-s2-t56-p512-xprof | 1 | 0.445 | 0.372 | 0.431 | 3.338 | 0.964 | 0.000 | 0.008 | 5.637 |
| 1b-fp32-teams-hbm-s2-t56-p512-xprof | imbalance | 0.003 | 0.018 | 0.023 | 0.011 | 0.095 | 0.000 | 0.000 | 0.111 |
| 1b-fp32-teams-hbm-s2-t56-p8192-xprof | 0 | 0.422 | 2.582 | 0.441 | 3.158 | 0.914 | 0.000 | 0.008 | 7.614 |
| 1b-fp32-teams-hbm-s2-t56-p8192-xprof | 1 | 0.408 | 2.515 | 0.516 | 3.153 | 0.899 | 0.000 | 0.011 | 7.583 |
| 1b-fp32-teams-hbm-s2-t56-p8192-xprof | imbalance | 0.014 | 0.067 | 0.075 | 0.005 | 0.015 | 0.000 | 0.003 | 0.031 |
| 1b-int8-teams-hbm-s2-t56-p32768-xprof | 0 | 0.471 | 10.359 | 0.349 | 1.090 | 0.212 | 0.000 | 0.009 | 12.684 |
| 1b-int8-teams-hbm-s2-t56-p32768-xprof | 1 | 0.470 | 10.192 | 0.517 | 1.094 | 0.210 | 0.000 | 0.008 | 12.681 |
| 1b-int8-teams-hbm-s2-t56-p32768-xprof | imbalance | 0.001 | 0.167 | 0.168 | 0.004 | 0.002 | 0.000 | 0.001 | 0.003 |
| 1b-int8-teams-hbm-s2-t56-p512-xprof | 0 | 0.203 | 0.384 | 0.234 | 1.015 | 0.249 | 0.000 | 0.008 | 2.277 |
| 1b-int8-teams-hbm-s2-t56-p512-xprof | 1 | 0.180 | 0.343 | 0.304 | 1.031 | 0.228 | 0.000 | 0.009 | 2.256 |
| 1b-int8-teams-hbm-s2-t56-p512-xprof | imbalance | 0.023 | 0.041 | 0.070 | 0.016 | 0.021 | 0.000 | 0.001 | 0.021 |
| 1b-int8-teams-hbm-s2-t56-p8192-xprof | 0 | 0.385 | 2.880 | 0.376 | 1.189 | 0.239 | 0.000 | 0.008 | 5.263 |
| 1b-int8-teams-hbm-s2-t56-p8192-xprof | 1 | 0.401 | 2.883 | 0.358 | 1.181 | 0.238 | 0.000 | 0.008 | 5.249 |
| 1b-int8-teams-hbm-s2-t56-p8192-xprof | imbalance | 0.016 | 0.003 | 0.018 | 0.008 | 0.001 | 0.000 | 0.000 | 0.014 |
| 8b-fp32-teams-hbm-s2-t56-p32768-xprof | 0 | 3.178 | 22.621 | 2.525 | 21.427 | 1.737 | 0.000 | 0.019 | 51.889 |
| 8b-fp32-teams-hbm-s2-t56-p32768-xprof | 1 | 3.100 | 21.636 | 3.596 | 21.259 | 1.758 | 0.000 | 0.019 | 51.779 |
| 8b-fp32-teams-hbm-s2-t56-p32768-xprof | imbalance | 0.078 | 0.985 | 1.071 | 0.168 | 0.021 | 0.000 | 0.000 | 0.110 |
| 8b-fp32-teams-hbm-s2-t56-p512-xprof | 0 | 3.479 | 1.133 | 2.542 | 22.864 | 2.109 | 0.000 | 0.018 | 32.533 |
| 8b-fp32-teams-hbm-s2-t56-p512-xprof | 1 | 3.359 | 1.145 | 2.602 | 22.808 | 2.083 | 0.000 | 0.022 | 32.422 |
| 8b-fp32-teams-hbm-s2-t56-p512-xprof | imbalance | 0.120 | 0.012 | 0.060 | 0.056 | 0.026 | 0.000 | 0.004 | 0.111 |
| 8b-fp32-teams-hbm-s2-t56-p8192-xprof | 0 | 3.174 | 9.786 | 2.517 | 21.627 | 1.761 | 0.000 | 0.020 | 39.273 |
| 8b-fp32-teams-hbm-s2-t56-p8192-xprof | 1 | 3.033 | 8.801 | 3.594 | 21.555 | 1.702 | 0.000 | 0.016 | 39.107 |
| 8b-fp32-teams-hbm-s2-t56-p8192-xprof | imbalance | 0.141 | 0.985 | 1.077 | 0.072 | 0.059 | 0.000 | 0.004 | 0.166 |
| 8b-int8-teams-hbm-s2-t56-p32768-xprof | 0 | 1.129 | 23.020 | 1.502 | 6.551 | 0.502 | 0.000 | 0.018 | 33.641 |
| 8b-int8-teams-hbm-s2-t56-p32768-xprof | 1 | 1.086 | 21.890 | 2.625 | 6.529 | 0.491 | 0.000 | 0.020 | 33.497 |
| 8b-int8-teams-hbm-s2-t56-p32768-xprof | imbalance | 0.043 | 1.130 | 1.123 | 0.022 | 0.011 | 0.000 | 0.002 | 0.144 |
| 8b-int8-teams-hbm-s2-t56-p512-xprof | 0 | 1.086 | 1.121 | 0.957 | 6.493 | 0.553 | 0.000 | 0.018 | 10.940 |
| 8b-int8-teams-hbm-s2-t56-p512-xprof | 1 | 1.064 | 1.060 | 1.031 | 6.494 | 0.525 | 0.000 | 0.019 | 10.860 |
| 8b-int8-teams-hbm-s2-t56-p512-xprof | imbalance | 0.022 | 0.061 | 0.074 | 0.001 | 0.028 | 0.000 | 0.001 | 0.080 |
| 8b-int8-teams-hbm-s2-t56-p8192-xprof | 0 | 1.127 | 9.464 | 1.096 | 6.471 | 0.511 | 0.000 | 0.016 | 19.603 |
| 8b-int8-teams-hbm-s2-t56-p8192-xprof | 1 | 1.036 | 8.385 | 2.269 | 6.509 | 0.479 | 0.000 | 0.018 | 19.495 |
| 8b-int8-teams-hbm-s2-t56-p8192-xprof | imbalance | 0.091 | 1.079 | 1.173 | 0.038 | 0.032 | 0.000 | 0.002 | 0.108 |

Reading: the fused reduction is accounted inside `wo` and `ffn`; its separate `reduce` is 0.000 and the waits are 0.008-0.022 ms.
At short context the two sockets finish within 0.08-0.12 ms of each other. At 8192 and 32768 tokens socket 1 spends about 1 ms more
in `wo` than socket 0 (it waits there for the slower attention of socket 0, which is 0.9-1.1 ms slower): 2-3% of the token at 8B.

## Modeled against measured

Modeled = the sum over the compute phases of the slower socket plus the exposed exchange; ceiling = checkpoint bytes over the two
sockets' stream bandwidth (2 x 600 GB/s, `tests/stream_bench.c`; 26.77 ms for 8B FP32, the 27.19 ms of study 8.3 is the same bound with
the older byte count).

| Run | Measured | Modeled | Modeled/measured | Bandwidth ceiling | Ceiling/measured | Revisit triggers |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1b-fp32-teams-hbm-s2-t14-p512-xprof | 12.68 | 12.45 | 98% | 4.12 | 32% | none |
| 1b-fp32-teams-hbm-s2-t28-p512-xprof | 7.17 | 6.96 | 97% | 4.12 | 57% | none |
| 1b-fp32-teams-hbm-s2-t42-p512-xprof | 6.27 | 5.96 | 95% | 4.12 | 66% | none |
| 1b-fp32-teams-hbm-s2-t56-p32768-xp4prof | 14.45 | 13.97 | 97% | 4.12 | 29% | none |
| 1b-fp32-teams-hbm-s2-t56-p32768-xprof | 14.21 | 13.74 | 97% | 4.12 | 29% | none |
| 1b-fp32-teams-hbm-s2-t56-p512-xp4prof | 5.83 | 5.39 | 92% | 4.12 | 71% | none |
| 1b-fp32-teams-hbm-s2-t56-p512-xprof | 5.80 | 5.40 | 93% | 4.12 | 71% | none |
| 1b-fp32-teams-hbm-s2-t56-p8192-xp4prof | 7.86 | 7.27 | 92% | 4.12 | 52% | none |
| 1b-fp32-teams-hbm-s2-t56-p8192-xprof | 7.70 | 7.21 | 94% | 4.12 | 54% | none |
| 1b-int8-teams-hbm-s2-t14-p512-xprof | 4.11 | 3.94 | 96% | 1.09 | 27% | none |
| 1b-int8-teams-hbm-s2-t28-p512-xprof | 2.52 | 2.34 | 93% | 1.09 | 43% | none |
| 1b-int8-teams-hbm-s2-t42-p512-xprof | 2.13 | 1.88 | 88% | 1.09 | 51% | none |
| 1b-int8-teams-hbm-s2-t56-p32768-xprof | 12.75 | 12.43 | 98% | 1.09 | 9% | none |
| 1b-int8-teams-hbm-s2-t56-p512-xprof | 2.33 | 2.01 | 86% | 1.09 | 47% | none |
| 1b-int8-teams-hbm-s2-t56-p8192-xprof | 5.32 | 4.88 | 92% | 1.09 | 21% | none |
| 8b-fp32-teams-hbm-s2-t14-p512-xprof | 78.33 | 77.18 | 99% | 26.77 | 34% | none |
| 8b-fp32-teams-hbm-s2-t28-p512-xprof | 44.66 | 43.90 | 98% | 26.77 | 60% | none |
| 8b-fp32-teams-hbm-s2-t42-p512-xprof | 39.23 | 37.72 | 96% | 26.77 | 68% | none |
| 8b-fp32-teams-hbm-s2-t56-p32768-xp4prof | 51.86 | 49.99 | 96% | 26.77 | 52% | none |
| 8b-fp32-teams-hbm-s2-t56-p32768-xprof | 52.23 | 50.46 | 97% | 26.77 | 51% | none |
| 8b-fp32-teams-hbm-s2-t56-p512-xp4prof | 31.69 | 30.52 | 96% | 26.77 | 84% | none |
| 8b-fp32-teams-hbm-s2-t56-p512-xprof | 32.78 | 31.79 | 97% | 26.77 | 82% | none |
| 8b-fp32-teams-hbm-s2-t56-p8192-xp4prof | 39.64 | 38.04 | 96% | 26.77 | 68% | none |
| 8b-fp32-teams-hbm-s2-t56-p8192-xprof | 39.58 | 37.94 | 96% | 26.77 | 68% | none |
| 8b-int8-teams-hbm-s2-t14-p512-xprof | 25.88 | 25.08 | 97% | 7.11 | 27% | none |
| 8b-int8-teams-hbm-s2-t28-p512-xprof | 14.72 | 14.12 | 96% | 7.11 | 48% | none |
| 8b-int8-teams-hbm-s2-t42-p512-xprof | 12.17 | 11.40 | 94% | 7.11 | 58% | none |
| 8b-int8-teams-hbm-s2-t56-p32768-xprof | 33.92 | 32.26 | 95% | 7.11 | 21% | none |
| 8b-int8-teams-hbm-s2-t56-p512-xprof | 11.09 | 10.36 | 93% | 7.11 | 64% | none |
| 8b-int8-teams-hbm-s2-t56-p8192-xprof | 19.80 | 18.75 | 95% | 7.11 | 36% | none |

The model explains 86-99% of every measured token (86-88% only for the 1B W8A8 runs at 512, whose token is 2.1-2.3 ms), so the gap to the ceiling is in
the phases, not in unmodeled waiting. At 8B FP32 and context 512 the measured 31.7-32.8 ms (packed4 / rows) is 82-84% of the
bandwidth ceiling; the largest phase is the FFN at 22 ms of 32.5. W8A8 reaches 64% of its ceiling at 512 (11.1 against 7.1 ms), 36% at 8192 and
21% at 32768, where attention takes 48% and 68% of the token. The remaining gap is the streaming efficiency of each GEMV phase below
the stream benchmark and the non-streaming work (classifier, attention, embedding); it was not split further.

## Crossover (study 10.4)

| Model | Precision | Context | Gain | Two sockets win |
| --- | --- | ---: | ---: | --- |
| 1b | fp32 | 512 | 1.85 | yes |
| 1b | fp32 | 8192 | 1.85 | yes |
| 1b | fp32 | 32768 | 1.57 | yes |
| 1b | int8 | 512 | 1.73 | yes |
| 1b | int8 | 8192 | 1.59 | yes |
| 1b | int8 | 32768 | 1.26 | yes |
| 8b | fp32 | 512 | 1.87 | yes |
| 8b | fp32 | 8192 | 1.92 | yes |
| 8b | fp32 | 32768 | 1.99 | yes |
| 8b | int8 | 512 | 1.84 | yes |
| 8b | int8 | 8192 | 1.77 | yes |
| 8b | int8 | 32768 | 1.90 | yes |

Two sockets win in all 12 cells, beyond variability. Where synchronization dominates (reductions plus waits above 20% of the token,
or peer wait above 10%): nowhere; the largest reduce-plus-wait of any of the 30 profiled decode runs is 0.022 ms of 2.1 to 52 ms.
Where attention dominates (above 50% of the token, long context): 8B W8A8 at 32768 (68%), 1B at 32768 (FP32 65%, W8A8 81%) and 1B
W8A8 at 8192 (54%); it is the largest single phase on 8B FP32 at 32768 (43%) and 8B W8A8 at 8192 (48%). The gain falls where attention is large on the small model (1B W8A8 at 32768: 1.26x), because attention is the
part that scales least with a second socket there; the cause of the weaker scaling was not isolated by this run. Capacity-only gains
(a context that fits only with both sockets' HBM): not tested; every context here fits one socket (8B KV at 32768 is 4.3 GB).

## Memory traffic and placement

Whole-run counters (the 512-token warm-up plus 128 decoded tokens), 8B, `perf_uncore=1`.

| Configuration | HBM read GB | DDR read GB | UPI rx data flits (M) | Weights, packed4, KV verified |
| --- | ---: | ---: | ---: | --- |
| 8B FP32, 1 x 56 | 4118.8 † | 9.2 | 407 | weights, KV `full` on node 2; work region unverified (10-40 MB) |
| 8B FP32, 2 x 56 rows | 4109.8 † | 50.6 | 2408 | weights, KV, work `full` on nodes 2 and 3 |
| 8B FP32, 2 x 56 packed4 | 4142.6 † | 58.6 | 2864 | additionally the packed4 copy (14.0 GB per socket) `full` |
| 8B FP32, 112 threads, no tensor parallelism (control) | 4164.0 † | 22.5 | 298434 | not applicable |
| 8B W8A8, 1 x 56 | 1108.2 † | 1.5 | 30 | as above |
| 8B W8A8, 2 x 56 | 1099.7 † | 10.0 | 725 | as above |
| 8B W8A8, 112 threads, control | 1127.6 † | 5.2 | 81553 | not applicable |

Two sockets with tensor parallelism send 0.8% (FP32) and 0.9% (W8A8) of the UPI data flits of the interleaved 112-thread control:
no weights and no attention data cross the link, only the activation exchange. The embedding table (2.1 GB) is on DDR node 0 in every
configuration. The one-socket long-context runs carry the label `hbm (unverified)` only because the small work region is unverified
there.

## Exit criteria

- [x] Two-socket decode beats the best one-socket configuration beyond run variability, with every numerical and placement gate passing, per precision, model and context: 12 of 12 cells (1.26x to 1.99x), 8 of 8 gates, placement `full`.
- [x] The crossover table records where two sockets win, where synchronization dominates and where attention dominates: wins in all cells; synchronization nowhere; attention above 50% in 4 cells (above).
- [x] Revisit triggers evaluated (reductions plus waits > 20%, peer wait > 10%, UPI carrying attention or weight traffic): none reached in 30 profiled runs (reduce plus wait at most 0.022 ms per token; UPI flits 0.8-0.9% of the control).
- [x] packed4 against rows decided: bitwise equal on the AVX-512 kernels (`b7-packed4-vs-rows-2s`: max |d| 0.0 on 53 positions, 1B and 8B), faster beyond the range only at 8B context 512 (+3.2%); **not adopted** as a default (stays an option, `LLAMA_FP32_LAYOUT=packed4`).
- [x] The 112-thread control without tensor parallelism reproduces the old figures: 8B FP32 8.24 [6.18-8.25] against 8.17 tok/s (interleaved on HBM 2,3) and 6.42 (one HBM node); 8B Q8 24.67 against 26.16.
- [x] Modeled against measured table explains the gap to the bandwidth ceiling (27.19 ms/token for 8B FP32 on two sockets, study 8.3): the model reproduces 96-97% of the measured token; the measured token is 82-84% of the ceiling at context 512.

## Decisions

| Question | Decision | Evidence |
| --- | --- | --- |
| `LLAMA_SOCKETS=2` as the decode choice | recommended for decode wherever both sockets are free; the default stays 1 (D2 decides) | 12 of 12 cells win, 1.26x to 1.99x |
| Worker count | 56 per socket (all cores) | monotonic 14 < 28 < 42 < 56 on 8B; 1B W8A8 42 = 56 |
| `LLAMA_FP32_LAYOUT=packed4` | not adopted as a default | +3.2% at one of six 8B and 1B FP32 contexts, 14.0 GB more HBM per socket |
| `LLAMA_TP_PARTITION=rows` | not built | reductions plus waits are below 0.1% of the token |
| `LLAMA_THP` | off | A6: neutral |

## Follow-ups for a later plan

- Attention at long context, mainly on 1B (1B W8A8 at 32768 tokens: attention 81% of the token and a 1.26x gain); the 8B profile shows the same shape from 8192 tokens (48%).
- Overlapping the next phase's weight prefetch with the wait for the peer is moot at decode (waits are 0.02 ms); the FFN streaming efficiency (22 ms of 32.5) is the lever for FP32.
- Asymmetric FFN shards for the socket imbalance (attention imbalance 0.9-1.1 ms at long context).
