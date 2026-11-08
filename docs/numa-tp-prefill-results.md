# NUMA placement, two-socket tensor parallelism and batched prefill: final results

**Status: not run.** The campaign of phase D2 is written and dry-run verified but has not been submitted; every table below is
an empty skeleton for the numbers the run produces. Nothing in this document is a measurement. Provenance and gates go in
[the validation record](numa-tp-prefill-results-validation.md). Plan: [D2](plans/D-integration/D2-final-campaign-and-decision.md). Branch: `feature/integ-d2-final-campaign`.

Closes the plan (milestone M5): one controlled campaign over the whole feature set, from one commit and one node state, and a
decision for each feature: a default, an option, or a documented capacity result. Prefill and decode are always measured and
reported separately. Inputs: A6, B7, C7 and D1 (their results documents), the study (`numa-tensor-parallel-study.md`
sections 8.3, 10.3 and 10.4) and the roofline (`hbm-roofline.md` sections 3 and 4).

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` (paste it here once the run is done); the sections after it are
the curated views.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| — | | | | | | | | | | | | | | | | | |

## 1. Prefill

Per precision x sockets x prompt cell; speedup over the one-socket per-token baseline. `D2_M1` and `D2_M2` give M per rung (C7 for one socket, D1 for two): —.

| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| — | — | 86 / 2048 / 8192 | — | — | 1 / 2 | — | — | — | — |

## 2. Decode

From retained contexts of 512 and 8192 tokens (128 tokens), compared with `docs/kernels.md` and A6.

| Model | Precision | Context | 1x56 | 2x56 | Gain 2x56 / 1x56 | Beyond variability |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| — | — | — | — | — | — | — |

## 3. Crossover tables (study 10.4)

Where two sockets win for decode and for prefill, by precision, model and context or prompt; where synchronization dominates; where attention dominates.

| Model | Precision | Context | Gain | Two sockets win |
| --- | --- | ---: | ---: | --- |
| — | — | — | — | — |

## 4. Placement evidence

Region tables and verification coverage (weights, packed shards, KV per socket), THP backing: —.

## 5. Numerical evidence

Every gate's observed error. The BF16 provisional gate is confirmed or revised: —.

| Comparison | Gate | Observed | Result |
| --- | --- | --- | --- |
| FP32 two sockets against one, batched | max |dlogit| <= 1e-3 | — | — |
| W8A8 and AMX-INT8 two sockets against one | statistical gate | — | — |
| BF16 against FP32 / Hugging Face | provisional, fixed in C5 | — | — |
| `anon-hbm` against staged | bitwise | — | — |

## 6. Modeled against measured

The roofline (`hbm-roofline.md` sections 3 and 4) and the study 8.3 models against the measured results, with the gap explained: —.

## 7. Decisions

| Feature | Becomes default when | Evidence | Decision |
| --- | --- | --- | --- |
| Engine-owned placement (`LLAMA_PLACEMENT=hbm`) | not slower than staged HBM, and verified | — | — |
| Two sockets (`LLAMA_SOCKETS=2`) | lower latency than the best one-socket configuration beyond variability, per model / precision / context | — | — |
| Batched prefill (`LLAMA_PREFILL_BATCH=M`) | faster TTFT than per-token beyond variability, gates passing | — | — |
| AMX-INT8 for Q8 prefill | faster than AVX-512 Q8 batched, gate passing | — | — |
| AMX-BF16 | never a silent default for FP32 checkpoints (user decision); recommended when faster within its gate | — | — |
| `LLAMA_FP32_LAYOUT=packed4` | bitwise equal and faster beyond the range (B7) | — | — |


## Exit criteria

Each is ticked only with the observed value written next to it.

- [ ] Every row of the decision table has a decision backed by D2 evidence: —
- [ ] The README milestones show M1 to M5 done: —
- [ ] `docs/kernels.md` "Not in this change" updated: —

## Follow-ups for a later plan

BF16 checkpoints or KV, KV tiering across HBM and DDR, multi-request batching, exchange/compute overlap and AMX attention are candidates for a later plan: —
