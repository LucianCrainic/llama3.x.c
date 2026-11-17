# B7. Two-socket decode campaign and decision

## 1. Goal and status

Measure whether the two-socket split improves single-request decode latency over the
**optimized one-socket baseline** (A6), and explain where the time goes. The split is
judged only by the study §10.4 criteria.

Status: **run on 17 November 2026 (jobs 5839003 and 5839004, commit 2aad124)**: two sockets beat one socket in all 12 cells of the 1B and 8B
decode matrix (FP32 and W8A8, contexts 512, 8192 and 32768): 8B FP32 1.87x / 1.92x / 1.99x, 8B W8A8 1.84x / 1.77x / 1.90x, 1B 1.26x to 1.85x;
8 of 8 gates passed; no revisit trigger reached; 56 workers per socket is the best setting; packed4 (bitwise equal) is not adopted (+3.2% at one context).
Results: [`docs/tensor-parallel.md`](../../tensor-parallel.md). The investigation of 8 November 2026 (leader-pinning bug, first measurements) is in Findings. Branch:
`feature/tp-b7-campaign` (scripts and results docs only).

Results document skeleton (empty until the run): [`docs/tensor-parallel.md`](../../tensor-parallel.md) and its [validation record](../../tensor-parallel-validation.md).

## 2. Depends on / unblocks

- Depends on: B6, A6 (baseline and THP default) and F3.
- Unblocks: D2.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| A6 matrix and baseline | A6 | One-socket `anon-hbm` best configuration per model and precision |
| Old two-socket runs | `scripts/cresco8-kernels.sbatch:166-168` | Measured "112 threads without TP" (6.42 and 8.17 tok/s FP32 8B); kept as a control |
| Library, summary | F3 | — |
| Study matrix | study §10.3 | Normative experiment list |

## 4. Changes

- A `PHASE=B7` matrix in `scripts/cresco8-tp.sbatch`.
- An optional all-output-row comparator, `LLAMA_TP_PARTITION=rows`, implemented here
  only if the time budget allows (study §6.4).
  - All matrices, including Wo and W2, are split by output rows. This requires an
    all-gather of attention/FFN input vectors instead of a sum.
  - It reuses existing full-row kernels and the same placement.
- Results docs: `docs/tensor-parallel.md` and `docs/tensor-parallel-validation.md`.

### 4.1 Run matrix

All runs use `LLAMA_BACKEND=teams`, HBM placement, the THP setting chosen by A6, and
3 interleaved repeats.

| Dimension | Values |
| --- | --- |
| Configurations | one socket ×56 (A6 best); two sockets 2 × {14, 28, 42, 56}; control: 112 threads, no TP, interleaved HBM (old style) |
| Models | 1B, 8B |
| Precision | FP32, Q8 W8A8; W8A32 at 2 × 56 only |
| Context | Retained-context starts at 512, 8192 and 32768 tokens (prefill not timed for decode), then 128 decode tokens |
| Profiling | One extra repeat per configuration with `LLAMA_PROFILE=1`, kept separate from the timed medians |
| Counters | perf HBM/DDR CAS reads and UPI flits, per socket, counted once per node (study §10.3) |
| Comparator | `LLAMA_TP_PARTITION=rows` at 2 × 56, 8B FP32/Q8, context 512 (if implemented) |

Long-context decode is measured from a comparable retained context, never averaged
with an empty-cache run (study §10.3). That requires a `--warm-context N` benchmark
option (prefill N tokens untimed, then time decode). It is added here, in `app.c`
`generate` (**implemented**, 8 November 2026: `warm_up` in `src/app.c`, documented in `docs/prefill.md`, tested by
`test_warm_context_is_untimed_and_uncounted`).

**Estimated** duration is 4–6 hours, so it may need to be split into two jobs by
model. It needs the user's go-ahead to submit.

### 4.1b Decision L8 and the packed4 decode layout (2026-11-08)

The `LLAMA_TP_PARTITION=rows` comparator is **not built** (only if B7 shows reductions plus waits above 20%). The repacked FP32
layout of the findings (§6, "Not done") **is**: `LLAMA_FP32_LAYOUT=packed4` (default `rows`) gives each socket a second copy of
its FP32 shards in which the four rows of a group form one sequential stream (for each block of 16 columns: 16 floats of row
0, 1, 2, 3; the last block zero-padded), in a `ROLE_PACKED` region on the socket's node, built at load on the socket's team
(`k_pack4_many`, `CMD_REPACK`). Decode streams it (`k_matmul_qkv_p4`, `k_matmul_swiglu_p4`, `k_matmul_p4_worker`, with
row-flag callbacks, claiming and the fused reduction unchanged); the batched GEMM and the BF16 / int8 packing keep reading the
row-layout shards, so the layout costs one more copy of the matrices per socket (8B FP32: about 16 GiB of 56, counted by the
HBM budget check) and applies to FP32 two-socket runs only. Every row's arithmetic is that of the four-row AVX-512 kernel
(`dots_p4_avx512`: two accumulators, 32 columns a step, the masked tail), so the result is **bitwise equal** to the row layout
whenever the row layout also forms whole four-row groups (claiming schedules); the scalar fallbacks agree to the kernel
budget. Matrices whose shard has a row count that is not a multiple of 4 keep the row layout. Software prefetch runs 4 KiB
ahead along the stream (the `pf4096` point of `tests/stream_bench.c`). **Measured: nothing yet** — the benchmark figures of §6
(612 against 529 GB/s) are the expectation. B7 runs the two-socket FP32 decode from both layouts (tag `p4`, with its own
profiled run) and gates the bitwise equality on the AVX-512 kernels first (`b7-packed4-vs-rows-2s`).

### 4.2 Script (`phase_B7`, 2026-11-08)

`PHASE=B7` in `scripts/cresco8-tp.sbatch`, `DRY_RUN=1` verified (96 launches per model, tested by
`test_campaign_scripts_dry_run_and_their_labels_parse`). It runs the gates first (one socket against two with the batched
warm-up path, FP32 strict, W8A8 statistical, W8A32 strict; a failure stops the benchmark), then per model and precision with
the checkpoint staged on DDR node 0 and engine-owned HBM placement (`B7_THP`, default off, to be set from A6):

- decode of 128 tokens from `--warm-context` 512, 8192 and 32768 (batched warm-up, `B7_BATCH` = 64) for one socket x 56 and
  two sockets x 56; the worker-count sweep (2 x 14, 28, 42) and the control (112 OpenMP threads without tensor parallelism on
  a copy interleaved over HBM 2 and 3) at 512 only, which trims the grid of §4.1 to the time budget; W8A32 at 2 x 56 for 512
  and 8192;
- REPEATS interleaved repeats, one `LLAMA_PROFILE=1` run per configuration and context (log level info, so the per-socket
  `tp decode socket` lines reach the `.stderr` files), perf counters on the first repeat;
- no `LLAMA_TP_PARTITION=rows` comparator (decision L8: only if reductions plus waits exceed 20%).

Submit with `--time=06:00:00` or one model per job. `GATES_ONLY=1` stops after the correctness gates (the packed4-against-rows bitwise gate
included), which is how the new hardware paths are checked before the campaign.

`tools/summarize_campaign.py` (with `tools/campaign_tables.py`) turns the campaign directory into `summary.md`: the main table
gains a *Warm ctx* column, followed by decode tok/s by retained context for every configuration with the two-socket gain and
whether it is beyond variability; the per-socket phase table of the profiled runs (qkv, attention, wo, ffn, classifier,
reduce, wait, and the imbalance row |s0 - s1|); the modeled-against-measured table (sum of the slower socket per compute
phase plus the larger socket's exchange cost, and the bandwidth ceiling from the checkpoint size over 2 x 600 GB/s); and the
revisit triggers of §7 (reductions plus waits above 20%, peer wait above 10%) with the runs where attention dominates.

## 5. Local tests

- A script dry run.
- `--warm-context` unit test on the fixture: timed decode excludes the warm prefill,
  and the report records `warm_context`.

## 6. Cluster run

As in §4.1. Outputs:

- `summary.md` with medians and ranges.
- A per-phase table for each socket: qkv, attention, wo, ffn, classifier, reduce,
  barrier_wait and imbalance = |s0 − s1|.
- **Modeled** vs **measured** time: the study §8.3 model ("sum of the per-layer max
  plus exchange") compared with the observed time.

### Findings of the investigation (2026-11-08)

Measured on cresco8-hbm14 (Flat mode), 86-token prompt then 128 generated tokens, `LLAMA_BACKEND=teams`, 56 workers per
socket, weights on HBM nodes 2 (and 3), `LLAMA_PROFILE=1`, no job-wide binding for two sockets. Jobs 5703348, 5703355,
5703366, 5703470, 5703473, 5703475, 5703488 and 5703489 (script `scripts/cresco8-decode-diag.sbatch`, summarizer
`tools/summarize_decode_diag.py`, streaming benchmark `tests/stream_bench.c`).

**1. Cause of the slowdown (a bug, not the design).** `team_create` pinned its caller to the first CPU of each team in
turn, so after `pool_start` the main thread (the leader of team 0) sat on socket 1's first CPU, and the socket-1
coordinator thread inherited that one-CPU mask. Both leaders ran on CPU 56. `/proc` samples of the old build show CPU 56
claimed by two threads; the fixed build never does. Fix: `team_adopt_leader` (`src/workers.c`) puts the main thread back on
team 0 at the end of `pool_start` and pins the socket-1 coordinator to team 1's first CPU; setup warns if a coordinator is
not on its leader CPU. The exchange microbenchmark of B4 had pinned its leaders separately, which is why it did not show it.

**2. Decode, one socket against two** (median tok/s, 5 interleaved repeats, final tree, job 5703488 tree `repo-decodefix7`):

| Model, precision | One socket | Two sockets, before the fix | Two sockets, now | Now / one socket |
| --- | ---: | ---: | ---: | ---: |
| 8B FP32 | 16.2 | 11.3 | 30.2 | 1.86 |
| 8B Q8 (W8A8) | 49.1 | 32.4 | 86.6 | 1.76 |
| 1B FP32 | 91.0 | 63.7 | 160.7 | 1.77 |
| 1B Q8 (W8A8) | 276 | 140 | 383 | 1.39 |

8B FP32 on two sockets streams 30.2 tok/s × 32.1 GB = 969 GB/s of weights (one socket: 521). The 1B Q8 case, where
synchronization is a larger share, scales least.

**3. The memory ceiling.** `tests/stream_bench.c` reads memory bound to each socket's HBM with one pinned team per socket:
about 600 GB/s per socket (605 on one socket, 590 / 614 with both reading at once, aggregate 1.2 TB/s) for a contiguous
stream with 4 KiB software prefetch; 574 with 1 KiB prefetch; 434-474 without prefetch. So ~600 GB/s per socket is
reachable by a pure stream; the engine's FP32 GEMV kernels reach 470-490 GB/s per socket on the 8B shard shapes
(aggregate 967-1005 GB/s on two sockets), 78-82% of it.

**4. Where the rest goes.**
- *Per-matrix dispatch and imbalance.* The kernel's arithmetic over the same row layout reaches 518-529 GB/s when one team
  job covers 4 GiB, and 408 GB/s when issued one 112 MiB matrix per job with a static row split, as the engine does. Letting
  members claim rows (the `LLAMA_GEMV_SCHED` modes below) recovers 25%: 511 GB/s (dynamic), 558 GB/s on a packed layout.
- *The reduction.* With the per-socket timers (`LLAMA_PROFILE=1`, log line `tp decode socket S ...`), 8B FP32 before the
  scheduling change: Wo and W2 together took 16.4 ms of a 43.3 ms token, of which the exchange was straggler wait 2.8 ms
  (waiting for the slowest member of the socket), peer wait 2.6-3.1 ms, add 0.1 ms, closing barrier 1.1 ms; the other
  projections scale ≈ 2×. That is about 15% of the token in reductions plus waits (the revisit trigger in §7 is 20%); it
  was not re-measured after the scheduling change.
- *Fusing the partial projection and the reduction into one team job* (`LLAMA_TP_FUSED`, default 1; bitwise equal to the
  two-job form) is worth +1% to +4% (8B FP32 26.0 against 25.7; 8B Q8 87.3 against 84.2), so the extra team dispatch was
  not the main cost.

**5. Row scheduling** (`LLAMA_GEMV_SCHED=static|dynamic|hybrid|auto`, jobs 5703473 and 5703475). Dynamic claiming of 8-32
rows from a shared counter, and a hybrid (a contiguous three quarters per member, the rest claimed), give 8B FP32 on two
sockets 29.5-29.8 tok/s against 26.0 static (+14%); one socket +2-3%. They hurt everything else: Q8 loses 6% (8B, hybrid) to
25% (1B), whose kernels prefetch along long contiguous runs, and 1B FP32 loses 4-6%. `auto` (default) is hybrid for FP32
jobs that stream at least 96 MiB (the 8B W1/W3, W2 and classifier jobs) and static otherwise and for Q8. A claim starts on a
multiple of four rows, so every row sits in the same four-row group whatever the team size or timing; logits agree with the
static split within the existing budgets.

**6. Not done, in order of expected value.**
- Repack the FP32 weights at load so that four rows form one sequential stream (`mini-gemv-packed4`): 612 GB/s on one
  socket in the benchmark (the raw stream is 601), against 529 for the engine's row layout. Per-matrix with claiming:
  558 against 511 GB/s (+9%); two sockets 1088 against 958 GB/s aggregate (+14%). A change of `shards_load`, the shard
  descriptors and the GEMV kernels.
- Q8 on two sockets is below the FP32 bandwidth (8B: ≈ 740 GB/s of weights): instruction-bound (§8), not changed here.
- Straggler and peer wait (≈ 5-6 ms per 8B FP32 token before the scheduling change) were measured and largely removed in
  the Wo/W2 investigation below (barriers replaced by row flags, join by flags); an all-column or rows-only partition
  (`LLAMA_TP_PARTITION=rows`) is still the comparator of §4.
- Everything in §4.1 beyond short-context decode: contexts of 512, 8192 and 32768 tokens, 14/28/42 workers, W8A32,
  perf counters (HBM/DDR/UPI), the control with 112 threads and no TP, the modeled-against-measured table.

### Findings of the Wo/W2 investigation (2026-11-08, second part)

Question: why do the projections that end in the cross-socket reduction (Wo, W2) reach so much less effective weight
bandwidth than W1/W3 (8B Q8 before: Wo 403, W2 714, W1/W3 972 GB/s), and how much of it can be recovered. Jobs on
cresco8-hbm14 (86-token prompt, 128 tokens, 56 workers per socket, `SAMPLE=0`): 5703717 (measurement, baseline tree),
5703719 (reduction protocol and join A/B), 5703967 (dispatch fan-out, row scheduling, socket swap), 5703968 (checks),
5703969 (final interleaved A/B). Tools added: `LLAMA_TP_TRACE=FILE` (one CSV record per Wo/W2 reduction and socket with
the time at which every step of every member happened; `tools/summarize_reduce_diag.py`), `tests/reduce_bench.c`
(`reduce-bench`: synthetic Wo, W1/W3, W2 layers with the engine's kernels, scheduling and exchange, the reduction replaced
by the real thing, by local barriers only, by nothing, or the reduction alone, on one or two sockets), the `REDUCE=1` stage
and the `*-base`, `2s-RJ`, `2s-fanN`, `*-trace` configurations of `scripts/cresco8-decode-diag.sbatch`.

**1. Where the time went (8B Q8, per Wo/W2 reduction, baseline).** Mean microseconds after the leader dispatched the team
job, from the trace (4064 reductions per kind and socket, socket 0; the sockets agree within 1-3 us):

| Step | Wo | W2 |
| --- | ---: | ---: |
| last member enters the job (dispatch) | 1.7 | 2.1 |
| projection: first / mean / last member done | 18.7 / 20.7 / 22.7 | 61.2 / 64.6 / 67.1 |
| first team barrier (last member in, all out) | +3.5 | +2.9 |
| wait for the other socket's partial (leader) | +1.2 | +3.8 |
| the add (73 floats per member, half of it read across UPI) | +4.5 | +3.9 |
| second team barrier | +2.9 | +3.3 |
| join of the team job (55 increments of one counter) | +4.6 | +4.5 |
| total / ideal stream time of the shard at 600 GB/s | 39.5 / 14 | 85.7 / 49 |

So a reduction costs 17-19 us beyond the slowest member's projection, and the same on 1B Q8 (Wo 26.7 us for 9.1 us of
projection, W2 37.9 for 21.4) and on 8B FP32 (Wo 91 for 71, W2 256-268 for 238). The 64 reductions of an 8B token cost
about 1.2 ms, **10% of the Q8 token** (the earlier estimate of 3 ms / 25% was too high); on 1B Q8 the same fixed cost is
**29% of the token** (profile: Wo 0.435 ms for 0.107 ms of projection, W2 0.585 for 0.301). The Wo shard of 8B Q8 is
8 MiB per socket, 14 us of streaming, so any fixed cost of 17 us more than doubles it; that is why Wo (403 GB/s) lost more
than W2 (714). Per-split of the profile line `tp decode socket S`: `x_close` (barrier 2) 0.55-0.59 ms and
`x_straggler` (barrier 1) 0.30-0.33 ms per 8B Q8 token, `x_peer` 0.10-0.34, `x_add` 0.05-0.09.

**2. What it is not.**
- *Stragglers inside a socket.* Last minus mean member: 2-4 us (Q8), 3-4 us (FP32 Wo), 11 us (FP32 W2 with hybrid claiming).
  The member that finishes last is spread evenly over all 56 (no slow core); reductions more than 2.5 times the median are
  0.1-1.6% and cost under 5 us on average.
- *Skew between the sockets.* Mean |difference| of the last member's finish: 4-10 us (Q8, Wo and W2 of 1B and 8B), up to
  24 us in 8B FP32 W2. In FP32 one socket is 5-8% slower than the other **per process**: the same `reduce-bench` run with
  teams swapped (`--swap 1`) gives 833/711 us per layer in one process, 746/746 in another, 783/789 swapped, i.e. the
  placement of that process's pages, not a slow socket or role. It shows as `peer wait` on the faster socket (8B FP32 W2 in the baseline trace:
  20 us on the faster socket, 5 on the slower one) and is not removed here.
- *Row scheduling.* Claiming rows makes the small projections slower (reduce-bench, Wo: Q8 26 us static, 40 hybrid, 41
  dynamic; FP32 73, 82, 88); W2 FP32 gains from hybrid (275 static, 254 hybrid), which `auto` already does. Unchanged.
- *The Q8 activation-slice quantization* is 3 us per call and 4 calls per layer (0.40 ms per 8B token, 3.5%; half of it
  before Wo/W2), serial on the leader. W2's long rows (n = 7168) cost nothing extra: W2 reaches 98% of W1/W3 in FP32 once the
  reduction is cheap. Spin/yield (`SPIN_LIMIT`) never triggers between jobs (the idle gaps are tens of us).
- *Partial buffers in the owner's HBM:* the add reads the peer's partial across UPI, 1.7-4.5 us; only that remains of it.

**3. What was changed** (all switchable, all bitwise equal to the old forms: `tests/test_tp.py`, `tests/workers.c`):
- *Flag reduction* (`LLAMA_TP_REDUCE=flags`, default; `barrier` is the old form). The kernels report every finished row
  range (`RowsDoneFn`), which sets one generation flag per output row (`exchange_rows_done`); a member waits only for the
  flags of the rows it adds on both sockets (`exchange_reduce_add_flags`) and adds them. There is no team barrier any more:
  the first one is replaced by the producers' flags, the second by the join of the team job, after which the coordinator
  publishes `consumed` (`exchange_reduce_finish`). Claiming works unchanged because the flags are per row. Saves 6-7 us per
  reduction.
- *Join by per-worker flags* (`LLAMA_TEAM_JOIN=flags`, default; `counter` is the old form). The leader of `team_run` scans
  one flag line per member instead of waiting for 55 increments of one line (4.6 us when the members finish together). It
  helps every team job, about 200 per token, and one-socket decode too (reduce-bench `none`: Wo 24 -> 19 us).
- *Two-level dispatch* (`LLAMA_TEAM_FANOUT=8`, default; 0 = flat). Workers 8, 16, ... relay the generation to their group so
  that 55 spinners do not queue on the leader's line. A small, consistent gain (job 5703967, 8B Q8 93.6 flat, 94.6-95.3 with
  fan-out 4/8/16; 8B FP32 30.7 -> 31.1; 1B Q8 496 -> 501-506).

**4. Result** (final tree, job 5703969, 5 interleaved repeats, median decode tok/s; `-base` = the build before these
changes: two-barrier reduction, counter join, flat dispatch):

| Model, precision | 1 socket base | 1 socket now | 2 sockets base | 2 sockets now | Gain | Now / 1 socket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B FP32 | 16.23 | 16.27 | 30.17 | 31.05 | +2.9% | 1.91 |
| 8B Q8 (W8A8) | 50.16 | 50.84 | 87.34 | 95.54 | +9.4% | 1.88 |
| 1B FP32 | 91.3 | 92.4 | 162.0 | 170.7 | +5.4% | 1.85 |
| 1B Q8 (W8A8) | 270.9 | 288.7 | 402.9 | 467.3 | +16% | 1.62 |

Effective weight bandwidth of Wo and W2 (all sockets, from the profile's `wo` and `w2`; W1/W3 for reference), base -> now:

| Model, precision | Wo GB/s | W2 GB/s | W1/W3 GB/s |
| --- | ---: | ---: | ---: |
| 8B Q8 | 415 -> 633 | 722 -> 863 | 971 / 959 |
| 8B FP32 | 724 -> 859 | 904 -> 973 | 999 / 988 |
| 1B Q8 | 158 -> 309 | 453 -> 704 | 1125 / 1068 |
| 1B FP32 | 440 -> 602 | 718 -> 804 | 1011 / 993 |

Per reduction the cost beyond the last member's projection fell from 17-19 us to 4.5-9 us (8B Q8 Wo 16.8 -> 4.5, W2 18.6 ->
6.5; 1B Q8 Wo 17.6 -> 5.2, W2 16.5 -> 4.8; traces of jobs 5703717 and 5703967), the rest being the add across UPI (2-5 us),
the dispatch (3-5 us in the engine) and the skew between the sockets. Numerics: one- and two-socket logits agree
(`compare_engines`, FP32 and W8A32 1e-3, W8A8 statistical gate) for 1B and 8B on AVX-512, `LLAMA_TP_CHECK=1` finds the two
residual copies bitwise equal, `make test` passes (job 5703968, `STAGE=check PHASE='B5 B6' HF_GATE=0`).

**5. What is left, in order of value.** Wo of 8B Q8 still reaches 633 GB/s against 959 for W1/W3 and Wo of 1B Q8 309: a
job of 8 MiB (1 MiB on 1B) per socket with 5-9 us of unavoidable synchronization (dispatch, the cross-UPI add, the join) and
about 4 us of Q8 kernel start-up. Further steps would be: overlapping the next phase's weight prefetch with the wait for
the peer (the workers idle 5-15 us per reduction while HBM is idle); quantizing the activation slice in parallel inside the
producing job; asymmetric FFN shards for the per-process socket imbalance (FP32); the repacked FP32 layout above.

## 7. Exit criteria (study §10.4)

**Improvement.** Two-socket decode counts as an improvement only if it beats the best
one-socket configuration beyond run variability, with every numerical and placement
gate passing. Report the result per precision, model and context.

**The crossover table records:**
- where two sockets win;
- where synchronization dominates (likely 1B Q8);
- where attention dominates (long context).

**Revisit triggers.** The design must be revisited if any of the following holds
(study §10.4):
- reductions plus waits exceed about 20% of token time;
- one socket waits on the other more than about 10% of the time;
- UPI counters show attention or weight traffic crossing sockets.

**Capacity.** A capacity-only gain, such as a context that fits only with two
sockets' HBM, is reported separately.

## 8. Risks and open questions

- **Bandwidth stops being the limit.** The ceiling modelled from **measured** stream
  rates is 27.19 ms per token for FP32 8B on two sockets (study §8.3). The kernels
  currently reach 81% of the one-socket bandwidth; if the two-socket kernels do not,
  compute or synchronization becomes the limit.
- **Interference.** The flat node is shared with other campaigns. Every result records
  the exclusivity and job ID.

## 9. Not in this phase

- Prefill on two sockets (D1, D2).
- Multi-request decode.
- KV tiering.
