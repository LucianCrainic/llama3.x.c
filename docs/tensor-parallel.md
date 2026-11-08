# Two-socket tensor-parallel decode

**Status: not run.** The campaign of phase B7 is written and dry-run verified but has not been submitted; every table below is
an empty skeleton for the numbers the run produces. Nothing in this document is a measurement. Provenance and gates go in
[the validation record](tensor-parallel-validation.md). Plan: [B7](plans/B-tensor-parallel/B7-two-socket-decode-campaign.md). Branch: `feature/tp-b7-campaign`.

Question: does splitting one model instance across both sockets (head and FFN-channel shards, socket-local KV, two cross-socket
reductions per layer) beat the best one-socket configuration (A6) for single-request decode, and where does the time go. The
investigation of 8 November 2026 (jobs 5703348 to 5703489, 5703717 to 5703969, in B7's *Findings*) fixed a leader-pinning bug and
measured 8B FP32 decode at 31.05 tok/s on two sockets against 16.27 on one (1.91x), 8B Q8 95.54 against 50.84 (1.88x); this
campaign measures the rest of the matrix with retained contexts (`--warm-context`), worker counts, W8A32, the control, the packed4
layout and perf counters. Judged by the study 10.4 criteria only.

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` (paste it here once the run is done); the sections after it are
the curated views.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| — | | | | | | | | | | | | | | | | | |

## Decode tok/s by retained context

`tools/campaign_tables.py` writes this table (median [range] over the repeats). Gain is two sockets x 56 over one socket x 56;
the last column compares the packed4 copy of the FP32 shards (`LLAMA_FP32_LAYOUT=packed4`) with the row layout.

| Model | Precision | Context | 1x56 | 2x14 | 2x28 | 2x42 | 2x56 | 2x56p4 | control | Gain 2x56 / 1x56 | Beyond variability | packed4 / rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1B | FP32 | 512 | — | — | — | — | — | — | — | — | — | — |
| 8B | FP32 | 512 / 8192 / 32768 | — | — | — | — | — | — | — | — | — | — |
| 1B | W8A8 | 512 / 8192 / 32768 | — | — | — | — | — | | — | — | — | |
| 8B | W8A8 | 512 / 8192 / 32768 | — | — | — | — | — | | — | — | — | |
| 8B | W8A32 (2x56 only) | 512 / 8192 | | | | | — | | | | | |

## Per-socket phase times (profiled runs, ms per token)

wo and ffn include their reduction; reduce and wait are its components. Imbalance = |socket 0 - socket 1|.

| Run | Socket | qkv | attention | wo | ffn | classifier | reduce | wait | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| — | 0 | — | — | — | — | — | — | — | — |
| — | 1 | — | — | — | — | — | — | — | — |
| — | imbalance | — | — | — | — | — | — | — | — |

## Modeled against measured

| Run | Measured ms/token | Modeled | Modeled/measured | Bandwidth ceiling (2 x 600 GB/s) | Ceiling/measured | Revisit triggers |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | — | — | — | — | — |

## Crossover (study 10.4)

| Model | Precision | Context | Gain | Two sockets win |
| --- | --- | ---: | ---: | --- |
| — | — | — | — | — |

Where synchronization dominates (reductions plus waits above 20% of the token, or peer wait above 10%): —.
Where attention dominates (long context): —. Capacity-only gains (a context that fits only with both sockets' HBM): —.

## Memory traffic and placement

| Configuration | HBM read GB | DDR read GB | UPI rx data flits (M) | Weights, packed4, KV verified |
| --- | ---: | ---: | ---: | --- |
| — | — | — | — | — |


## Exit criteria

Each is ticked only with the observed value written next to it.

- [ ] Two-socket decode beats the best one-socket configuration beyond run variability, with every numerical and placement gate passing, per precision, model and context: —
- [ ] The crossover table records where two sockets win, where synchronization dominates and where attention dominates: —
- [ ] Revisit triggers evaluated (reductions plus waits > 20%, peer wait > 10%, UPI carrying attention or weight traffic): —
- [ ] packed4 against rows decided: bitwise equal on the AVX-512 kernels (`b7-packed4-vs-rows-2s`) and faster beyond the range, or recorded as not adopted: —
- [ ] The 112-thread control without tensor parallelism reproduces the old figures (6.42 and 8.17 tok/s FP32 8B): —
- [ ] Modeled against measured table explains the gap to the bandwidth ceiling (27.19 ms/token for 8B FP32 on two sockets, study 8.3): —

## Follow-ups for a later plan

- Overlapping the next phase's weight prefetch with the wait for the peer; quantizing the activation slice inside the producing job; asymmetric FFN shards for the per-process socket imbalance (B7 findings, "What is left").
- `LLAMA_TP_PARTITION=rows`, only if reductions plus waits exceed 20%.
