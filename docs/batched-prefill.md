# Batched prefill on one socket

**Status: not run.** The campaign of phase C7 is written and dry-run verified but has not been submitted; every table below is
an empty skeleton for the numbers the run produces. Nothing in this document is a measurement. Provenance and gates go in
[the validation record](batched-prefill-validation.md). Plan: [C7](plans/C-batched-prefill/C7-one-socket-prefill-campaign.md). Branch: `feature/prefill-c7-campaign`.

Question: how much faster is prefill, and how much shorter the time to first token, on one socket for each rung of the ladder
(per-token baseline, AVX-512 FP32 GEMM, AMX-BF16, AVX-512 VNNI Q8, AMX-INT8 W8A8, W8A32), by model, prompt length and microbatch
size, with decode measured separately to show it is unaffected, and which defaults (`LLAMA_PREFILL_BATCH`, the kernel per
precision) the evidence supports. Earlier **measured** points (C2 to C6 checks): 8B at a 2048-token prompt reaches 570 tok/s
(AMX-BF16) and 563 tok/s (AMX-INT8) against 174 for the FP32 GEMM and 16.5 per token.

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` (paste it here once the run is done); the sections after it are
the curated views.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| — | | | | | | | | | | | | | | | | | |

## Prefill by rung, microbatch and prompt

| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | — | — | — | 1 | — | — | — | — |

## Thread counts (14 / 28 / 56, best M)

| Model | Rung | 14 | 28 | 56 |
| --- | --- | ---: | ---: | ---: |
| — | — | — | — | — |

## Achieved against the modeled ceiling

Modeled per-socket ceilings for 8B (`docs/hbm-roofline.md` section 4): about 370 / 1,800 / 4,600 tok/s for AVX FP32 / AMX-BF16 / AMX-INT8.

| Rung | Achieved TFLOP/s (or TOPS) | Prefill tok/s | Modeled ceiling tok/s | Achieved / ceiling |
| --- | ---: | ---: | ---: | ---: |
| — | — | — | — | — |

## Where the prefill time goes (profiled repeat, `C7/phase-shares.txt`)

| Rung | Prompt | GEMM | Attention | Conversion / rescale / packing | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| — | 2048 | — | — | — | — |
| — | 8192 | — | — | — | — |

At 8192 tokens attention may dominate whatever the GEMM rung; report the share and do not claim a GEMM speedup there: —.

## Decode after each rung's prefill (128 tokens from a 2048-token retained context)

| Model | Precision | Rung | Decode tok/s | One-socket reference (A6 or `docs/kernels.md`) | Within range |
| --- | --- | --- | ---: | ---: | --- |
| — | — | — | — | — | — |

## Chosen defaults

| Precision | `LLAMA_PREFILL_BATCH` | Prefill kernel | Why |
| --- | ---: | --- | --- |
| FP32 | — | — | — |
| W8A8 | — | — | — |

Whether `auto` may select AMX for prefill: —. AMX-BF16 never becomes a silent default for FP32 checkpoints (user decision).


## Exit criteria

Each is ticked only with the observed value written next to it.

- [ ] Every rung passes its gate (C2, C3, C5, C6 `compare_engines.py` on 8B at 2048 tokens, at the chosen M): —
- [ ] Prefill speedups are reported with ranges: —
- [ ] Decode is within its one-socket range (A6 or `docs/kernels.md`) after every rung: —
- [ ] Defaults are chosen and justified (M per precision; whether `auto` may select AMX): —
- [ ] The BF16 provisional gate of C5 is confirmed or revised from the Hugging Face numbers: —

## Follow-ups for a later plan

- Two-socket prefill (D1, D2). Multi-request batching.
