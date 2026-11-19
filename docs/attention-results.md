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

## 2. E4: prefill attention campaign (`split` against `tiled`, 2026-11-19)

### 2.1 Result in one paragraph

`tiled` is faster than `split` in every one of the 32 timed prefill cells, and the E4 value gate is met. 8B AMX-BF16 prefill at 8192
tokens on one socket goes from 25.06 s to 16.18 s time to first token (**1.55x**, ranges far apart). Against C7's 30.6 s that is 1.89x
(**derived**), and the gate limit was 20.4 s. The gain is largest on the AMX rungs at 8192 tokens (1.38-1.58x) and small on the AVX-512
FP32 rung (1.09-1.23x), where the GEMMs dominate. At 2048 tokens every cell is faster (1.04-1.32x). Decode after prefill is
unchanged within noise. Placement is verified in all 256 reports. The two-socket gain at 8192 tokens, above 2x with `split` on 8B,
is below 2x with `tiled` (at most 1.95x). The 30 logit comparisons of the check job pass. The 12 generation (determinism)
comparisons did not run in that job because of a script bug; the rerun (job 5864239) found all 12 identical. Provenance, gates and gaps:
[attention-results-validation.md](attention-results-validation.md).

The campaign ran from one commit (5a354ea) with D2's engine-owned HBM placement (`LLAMA_PLACEMENT=hbm`), three timed repeats and one
`LLAMA_PROFILE=1` repeat per cell, and a 32-token decode after each prefill. `split` is faster here than in section 1 (25.06 s against
30.68 s for the 8B AMX-BF16 8192 cell) because section 1 staged the checkpoint as files; `tiled` is also faster (16.18 s against 17.75 s
in the E2 rerun). The split baseline of this campaign is therefore the harder one, and the gate against C7's file-staged 30.6 s is met
with margin.

### 2.2 E4.3 gates

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| Value | 8B, 8192, one socket, AMX-BF16, best variant ≥ 1.5x vs C7 (30.6 s → ≤ 20.4 s) | `tiled` 16.18 [16.16-16.18] s against `split` 25.06 [25.05-25.12] s in the same job = 1.55x; 1.89x against C7 (derived) | pass |
| Value, other rungs (informational) | none in the plan | 8B AMX-INT8 1.50x; 8B AVX-512 FP32 1.16x (M 64) and 1.18x (M 128) | recorded |
| 2048 not slower | variant median ≥ lower end of the `split` range, every rung | 16 of 16 cells faster, 1.04-1.32x | pass |
| Decode unchanged | 32-token decode within the `split` range | median ratio `tiled`/`split` 0.997 over 32 cells (0.970-1.026), 15 faster and 17 slower; ranges overlap in 29 of 32 | pass, with the note below |
| Placement | verification passes in every report | 256 of 256 reports, 0 mismatched pages | pass |
| Two sockets | gates pass; gain at 8192 recorded; item 2a concluded | see 2.5 | pass |
| Attention share | `-xprof` at 2048 and 8192 per variant | 2.3 and 2.4 | recorded |
| Correctness (check job) | C3 and D1 logit gates, `LLAMA_TP_CHECK=1` on two sockets | 30 of 30 PASS (FP32 max 1.4e-4, BF16 max 0.77 at 86 tokens, W8A8 max 0.27; argmax 100%) | pass |
| Determinism (check job) | generation comparisons identical | check job: not run (script bug, fixed in b318378); rerun 5864239: 12 of 12 identical over 128 tokens | pass |

**Decode note.** The decode gate of the job's own summary was first marked FAIL (20 of 32 cells) by a rule that was too strict, and
was then corrected to range overlap (`tools/summarize_e4.py`, commit da6ce2b). The corrected summary still lists one cell below
`split` without range overlap: 8B AMX-BF16, 8192 tokens, two sockets, 25.5 [25.3-25.6] against 25.8 [25.6-25.8] tok/s (1.2% lower).
With three repeats per side and 32 cells, about 1.6 such cells are expected by chance (**estimated**). Decode runs identical code for
both variants, because `LLAMA_ATTN_DECODE` keeps `split`. The gate is treated as passed. It is not a clean pass by the summarizer's
letter, which still prints 31 of 32.

**Placement note.** Weights and KV are on HBM node 2 (one socket) and nodes 2 and 3 (two sockets) in every report. One-socket work
regions are not verified, by design. The one-socket K-cache regions report a few missing (untouched) pages in every report; D2's
reports (2026-11-17) show the same, so it predates E4.

### 2.3 Time to first token at 8192 tokens

Median [min-max] of three repeats, seconds. "Share" is attention over forward time of the profiled repeat. M is the batch size. All
cells have a 32-token decode behind them.

| model | rung (M) | sockets | split TTFT | tiled TTFT | speedup | share split → tiled |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8b | AMX-BF16 (256) | 1 | 25.057 [25.047-25.120] | 16.182 [16.163-16.182] | **1.55x** | 69% → 55% |
| 8b | AMX-BF16 (256) | 2 | 11.774 [11.771-11.777] | 8.308 [8.304-8.336] | 1.42x | 60% → 48% |
| 8b | AMX-INT8 (256) | 1 | 24.626 [24.491-24.727] | 16.370 [16.356-16.383] | 1.50x | 70% → 56% |
| 8b | AMX-INT8 (256) | 2 | 11.721 [11.694-11.722] | 8.504 [8.488-8.538] | 1.38x | 61% → 48% |
| 8b | AVX-512 FP32 (64) | 1 | 53.491 [53.467-53.573] | 46.151 [46.143-46.189] | 1.16x | 35% → 26% |
| 8b | AVX-512 FP32 (64) | 2 | 27.198 [27.190-27.211] | 24.901 [24.893-24.981] | 1.09x | 28% → 24% |
| 8b | AVX-512 FP32 (128) | 1 | 55.610 [55.543-55.767] | 46.934 [46.924-47.568] | 1.18x | 32% → 21% |
| 8b | AVX-512 FP32 (128) | 2 | 27.743 [27.721-27.747] | 24.226 [24.220-24.254] | 1.15x | 27% → 18% |
| 1b | AMX-BF16 (256) | 1 | 5.992 [5.950-5.999] | 3.788 [3.780-4.069] | 1.58x | 75% → 61% |
| 1b | AMX-BF16 (256) | 2 | 3.201 [3.193-3.217] | 2.106 [2.085-2.119] | 1.52x | 68% → 54% |
| 1b | AMX-INT8 (256) | 1 | 6.165 [6.155-6.274] | 4.050 [4.016-4.057] | 1.52x | 72% → 58% |
| 1b | AMX-INT8 (256) | 2 | 3.314 [3.307-3.345] | 2.282 [2.267-2.298] | 1.45x | 65% → 49% |
| 1b | AVX-512 FP32 (64) | 1 | 10.976 [10.933-10.985] | 9.384 [9.370-9.412] | 1.17x | 44% → 36% |
| 1b | AVX-512 FP32 (64) | 2 | 6.159 [6.098-6.175] | 5.181 [5.174-5.185] | 1.19x | 39% → 31% |
| 1b | AVX-512 FP32 (128) | 1 | 10.435 [10.433-10.442] | 8.480 [8.429-8.498] | 1.23x | 44% → 33% |
| 1b | AVX-512 FP32 (128) | 2 | 5.747 [5.728-5.751] | 4.681 [4.679-4.683] | 1.23x | 40% → 27% |

The profiled repeat of the gate cell gives attention 17.43 s → 9.06 s (1.92x) out of a forward of 25.20 s → 16.48 s, in line
with the kernel bench (section 1.3). On two sockets the same cell gives 7.01 s → 3.88 s. The AMX rungs gain most because their
GEMMs are already fast, so attention is the larger share. On the FP32 rung attention is only 27-44% of the `split` forward (2048-token share
10-18%), so removing part of it moves TTFT by 1.09-1.23x at most.

### 2.4 Time to first token at 2048 tokens

| rung | 8b, 1 socket | 8b, 2 sockets | 1b, 1 socket | 1b, 2 sockets | share split → tiled (8b, 1 socket) |
| --- | ---: | ---: | ---: | ---: | --- |
| AMX-BF16 (256) | 1.21x (2.937 → 2.423 s) | 1.18x | 1.32x | 1.23x | 34% → 23% |
| AMX-INT8 (256) | 1.18x (2.839 → 2.401 s) | 1.16x | 1.27x | 1.19x | 35% → 24% |
| AVX-512 FP32 (64) | 1.05x (10.000 → 9.493 s) | 1.04x | 1.08x | 1.07x | 11% → 8% |
| AVX-512 FP32 (128) | 1.05x (10.504 → 9.974 s) | 1.05x | 1.09x | 1.08x | 11% → 7% |

All 16 cells are faster than `split`. The full ranges and the 1B shares are in the E4 summary (`tools/summarize_e4.py bench`, run
again on the job's reports after the gate fix in da6ce2b).

### 2.5 Item 2a: does the two-socket gain above 2x persist?

Two-socket TTFT gain = one-socket median over two-socket median, 8192 tokens.

| model | rung (M) | split | tiled |
| --- | --- | ---: | ---: |
| 8b | AMX-BF16 (256) | 2.13x | 1.95x |
| 8b | AMX-INT8 (256) | 2.10x | 1.93x |
| 8b | AVX-512 FP32 (64) | 1.97x | 1.85x |
| 8b | AVX-512 FP32 (128) | 2.00x | 1.94x |
| 1b | AMX-BF16 (256) | 1.87x | 1.80x |
| 1b | AMX-INT8 (256) | 1.86x | 1.77x |
| 1b | AVX-512 FP32 (64) | 1.78x | 1.81x |
| 1b | AVX-512 FP32 (128) | 1.82x | 1.81x |

**Conclusion.** The gain above 2x does not persist. With `split` the 8B cells at 8192 tokens reach 2.00-2.13x (D2: 2.05-2.15x). With
`tiled` none exceeds 1.95x, and every 8B cell drops by 0.06-0.18. The 1B cells, already below 2x, move by at most 0.09. This
matches E0's H6 ([attention-profile.md](attention-profile.md)): the surplus above 2x belongs to attention at long prompts, and it
shrinks once attention is fast. This campaign has no cache counters, so the mechanism (L2/L3 fit of the halved K/V) stays unconfirmed.
At 2048 tokens the gain barely moves (8B AMX-BF16 1.83x → 1.79x). The two-socket path is still the faster choice everywhere: 8B
AMX-BF16 at 8192 tokens takes 8.31 s with `tiled` against 11.77 s with `split`, and 16.18 s on one socket.

### 2.6 Recommendation for `LLAMA_ATTN`

| Rung | Recommended | Gain at 8192 / 2048 tokens (8B, one socket) |
| --- | --- | --- |
| AMX-BF16 | `tiled` | 1.55x / 1.21x |
| AMX-INT8 (Q8) | `tiled` | 1.50x / 1.18x |
| AVX-512 FP32 | `tiled` | 1.16-1.18x / 1.05x |

`tiled` was faster on every rung tested, with no cell slower and decode unchanged. The gain is small on FP32, where the GEMMs
dominate, but it costs nothing there. `amx` (E3) was not built, so it was not tested. The plan (E4.4) left the default to the owner. After the
determinism comparison (job 5864239) passed, the owner switched the code default to `tiled` (2026-11-19). `LLAMA_ATTN=split` keeps
the earlier kernel; runs meant to match earlier campaigns (B, C and D series, E2 and E4 baselines) should set it, because `tiled`
agrees with `split` within the gates but not bit for bit. Hosts without AVX-512 run `split` under either setting. The decision
table entry is in [numa-tp-prefill-results.md](numa-tp-prefill-results.md) section 7.
