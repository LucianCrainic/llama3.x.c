# Prefill attention results (phases E2-E4)

Results of the faster prefill attention track ([plan](plans/E-followups/README.md)). The baseline and the bound come from
E0 ([attention-profile.md](attention-profile.md)). Provenance, gates and gaps:
[attention-results-validation.md](attention-results-validation.md). Every number is **measured** unless labelled otherwise.

## 1. E2: FP32 register-tiled fused attention (`LLAMA_ATTN=tiled`)

### 1.1 Result in one paragraph

`tiled` meets every E2 value gate on one socket. 8B AMX-BF16 prefill at 8192 tokens goes from 30.68 s to 18.52 s
time to first token (**1.66x**, ranges far apart). The engine's profiled attention time falls from 22.63 s to
10.74 s (2.11x), and attention drops from 74% to 58% of the forward. That beats the 12.3 s that E0 derived for a 1.5x TTFT. The kernel bench
gives 1.90x at the gate cell (8B, 8192, M 256, 56 threads; gate 1.8x). Every 2048-token cell is faster, not just
"not slower". The check job passed every correctness gate: FP32 max |Δlogit| ≤ 1.4e-4 against the per-token reference
at 86, 2048 and 8192 tokens, argmax 100%, two sockets included. The kernel still runs at only 22% of FP32 peak. The
counters say the remaining limit is instruction overhead outside the FMAs, not cache traffic (section 1.4).

### 1.2 Engine time to first token, one socket, 56 threads

Checkpoint staged on HBM node 2, three repeats, median [min-max] in seconds. "Attention" is the `LLAMA_PROFILE=1` repeat
(`profile_seconds.prefill.attention`) and "share" is attention / total inference of that repeat.

| model | rung (M) | prompt | split TTFT | tiled TTFT | speedup | attention split → tiled | share split → tiled |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8b | AMX-BF16 (256) | 2048 | 3.309 [3.291-3.314] | 2.618 [2.617-2.627] | 1.26x | 1.34 → 0.70 s (1.91x) | 40% → 27% |
| 8b | AMX-BF16 (256) | 8192 | 30.683 [30.670-30.702] | 18.523 [18.416-18.575] | **1.66x** | 22.63 → 10.74 s (2.11x) | 74% → 58% |
| 8b | AVX-512 FP32 (64) | 2048 | 11.463 [11.349-11.471] | 10.849 [10.776-10.898] | 1.06x | 1.43 → 0.90 s (1.60x) | 12% → 8% |
| 8b | AVX-512 FP32 (64) | 8192 | 62.888 [62.887-62.914] | 52.686 [52.581-54.639] | 1.19x | 23.81 → 13.81 s (1.72x) | 38% → 26% |
| 1b | AMX-BF16 (256) | 2048 | 0.768 [0.768-0.773] | 0.588 [0.587-0.589] | 1.31x | 0.33 → 0.19 s (1.79x) | 44% → 32% |
| 1b | AMX-BF16 (256) | 8192 | 7.156 [7.131-7.162] | 4.804 [4.794-4.811] | 1.49x | 5.47 → 3.13 s (1.75x) | 76% → 66% |
| 1b | AVX-512 FP32 (64) | 2048 | 2.173 [2.141-2.591] | 2.004 [1.987-2.328] | 1.08x (ranges overlap) | 0.39 → 0.26 s (1.49x) | 18% → 13% |
| 1b | AVX-512 FP32 (64) | 8192 | 12.836 [12.830-12.850] | 10.940 [10.913-10.951] | 1.17x | 5.92 → 4.17 s (1.42x) | 46% → 38% |

The time outside attention is unchanged within about 3% in every row, which is what an attention-only change should do. The
8B split figure at 8192 (30.68 s) reproduces C7's 30.6 s.

The engine gains more than the bench (2.11x against 1.90x on the 8B AMX-BF16 8192 cell). That agrees with E0's
hypothesis that the bench under-reports `split` because its K/V stays in the L3. `tiled` streams each K/V tile once per
16 query vectors, so it depends less on where the K/V lives.

### 1.3 Kernel bench (one layer, whole prompt)

The full table is in the job's `E2/bench-summary.md`. Headline cells at 56 threads, ms per layer, median [min-max]:

| shape | prompt | M | split | tiled | speedup | tiled GFLOP/s (% of 8800) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8b | 8192 | 256 | 536.76 [536.71-546.09] | 283.00 [274.19-285.63] | **1.90x** | 1943 (22%) |
| 8b | 8192 | 64 | 534.34 [534.03-545.50] | 279.09 [276.19-279.41] | 1.91x | 1970 (22%) |
| 8b | 2048 | 64 | 28.94 [28.90-29.11] | 16.24 [16.10-16.27] | 1.78x | 2117 (24%) |
| 8b | 512 | 256 | 2.85 [2.83-2.86] | 2.71 [2.64-2.80] | 1.05x | 795 (9%) |
| 1b | 8192 | 256 | 277.04 [276.91-282.02] | 161.29 [146.21-164.06] | 1.72x | 1704 (19%) |
| 1b | 512 | 256 | 2.19 [2.19-2.20] | 2.26 [2.22-2.30] | 0.97x | 477 (5%) |

- With fewer threads the gain is larger: 2.1-2.6x at 14 and 28 threads on 8B at 2048 and 8192 tokens. At 56 threads, part
  of the time goes to per-task fixed costs.
- Half the KV heads (the two-socket local shape, `kv4`) gains 1.72-1.98x on 8B at 8192.
- The one cell where `tiled` is slower is 1B, 512 tokens, M 256, 56 threads (0.97x, 2.2 ms per layer). The work there is
  too small to amortise the task set-up. It is not a gate cell.

### 1.4 Counters (8B, 8192, M 256, 56 threads, socket 0, whole launch)

| kernel | IPC | instructions / FLOP | l1d.replacement / KFLOP | l2_rqsts.miss / KFLOP | L3-miss loads / KFLOP |
| --- | ---: | ---: | ---: | ---: | ---: |
| split | 1.96 | 0.207 | 32.10 | 6.14 | 0.000 |
| tiled | 2.15 | 0.125 | 4.39 | 4.39 | 0.000 |

L1 replacements fall 7.3x, L2 misses fall 1.4x and instructions per FLOP fall 1.65x. **Derived:** IPC divided by
instructions per FLOP gives about 17 FLOP per core-cycle, about 27% of the 64 FLOP/cycle that two 16-lane FMA ports can
deliver. The FMA ports are therefore not the limit, and neither is L1 or L2 traffic. Most of the issued instructions are
the non-FMA work around the FMAs: transposes, loads, max and exp of the online softmax, and the rescale.

### 1.5 Tiled knobs (bench, 8B, 56 threads)

| prompt, M | queries 16, longest-first (default) | queries 16, natural order | queries 8 | queries 4 |
| --- | ---: | ---: | ---: | ---: |
| 8192, 256 | 264.45 [263.76-294.11] | **242.07 [241.46-277.70]** | 327-373 | 614-713 |
| 2048, 64 | 18.03 [16.74-21.33] | **17.54 [15.53-19.60]** | 20-23 | 35-42 |

- 16 query vectors per task is clearly best, so the default stays.
- Natural task order beats longest-first by 8% at 8192. The ranges overlap at their top ends, so this is not yet beyond
  variability. E0 found the tail under 1% at 8192, so ordering has little to fix there, and the longest-first sort seems
  to cost locality. Task order does not change any numbers, because each task writes its own partial.
- The sweep cell and the matrix cell for the same default configuration differ (264 against 283 ms). The same job ran
  them at different times, which shows the run-to-run spread at 56 threads.

**Rerun with `natural` as the default (job 5864122, commit 5a354ea, five repeats, M 16 cells added).** Bench, 8B, queries 16,
ms per layer, median [min-max]:

| prompt, M | longest-first | natural |
| --- | ---: | ---: |
| 512, 16 | 4.12 [4.06-4.16] | **3.68 [3.64-3.71]** |
| 2048, 16 | **18.73 [18.25-20.61]** | 20.69 [19.13-25.81] |
| 2048, 64 | 16.89 [16.22-21.07] | **15.63 [14.93-18.19]** |
| 8192, 256 | 266.05 [262.94-291.87] | **261.18 [241.64-279.76]** |

- Natural wins 3 of the 4 cells. At 512 tokens and M 16 the ranges are apart, which is the cell longest-first was meant
  to help.
- The one loss, 2048 tokens at M 16, lies within overlapping, noisy ranges.
- The default stays `natural`.
- The gate cell is 2.03x in this job.
- Engine TTFT with the new default, five repeats:
  - 8B AMX-BF16 at 8192 tokens: **17.75 [17.70-17.78] s against split's 30.74 [30.70-30.75] s (1.73x)**, against 18.52
    [18.42-18.57] s with longest-first in job 5864099.
  - 1B AMX-BF16 at 8192 tokens: 4.38 against 7.15 s (1.63x).
  - Every 2048-token rung is again faster than split (1.06-1.37x).

### 1.6 Decode

Decode is untouched by construction (`LLAMA_ATTN_DECODE` keeps `split`). The decode cell ran with defaults:
- **Configuration:** 8B FP32, one socket, from a retained context of 8192 tokens, 128 generated tokens, checkpoint staged
  on HBM node 2.
- **Result:** 13.06 [12.98-13.09] tok/s.
- **D2 reference:** the D2 cell with `hbm` placement measured 13.31 [13.30-13.46].
- **Assessment:** about 2% lower, with a different placement, and no E2 code runs in decode. It is recorded as consistent,
  not as a regression, and E4's decode-after-prefill cells compare `split` and `tiled` under identical settings.

### 1.7 E2.8 decision input

| E2.8 row | Observed |
| --- | --- |
| Kernel ≥ 3.0x and ≥ 50% of FP32 peak | no: 1.90x, 22% |
| 1.8x-3.0x, FMA ports busy (compute-bound) | no: about 27% FMA utilisation (derived) |
| 1.8x-3.0x, L1/L2 traffic or tail limits | traffic no (L1 7.3x lower); tail no at 8192 (E0: under 1%); the limit is non-FMA instruction overhead |
| < 1.8x | no |

No row matches exactly. The nearest is "tune E2 first": the FMA ports are idle about 73% of the time, so AMX's faster
units would not be the binding constraint. The E4 TTFT target (≥ 1.5x at 8192 vs C7) is already met by `tiled` alone
(1.66x). The decision is recorded in the plan (§4 E2 row and E2.8).
