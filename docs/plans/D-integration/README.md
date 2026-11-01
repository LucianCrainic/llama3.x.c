# Track D: integration, two-socket batched prefill and the final decision

## Goal

Combine the two-socket split (track B) with batched prefill and AMX (track C). The
result is one model instance that prefills microbatches on both sockets: each socket
works on its own heads, FFN channels and KV, and the two M×4096 reductions per layer
combine the partial results. D1 then measures prefill and decode separately, and D2
produces the final decision for all three features.

Prefill does not inherit decode's scaling result. Its reduction payload grows with M,
and it is compute-bound rather than bandwidth-bound (study §8.3, §10.4).

## Phases

| ID | Phase | Output | Gate |
| --- | --- | --- | --- |
| D1 | [Two-socket batched prefill](D1-two-socket-batched-prefill.md) | `LLAMA_SOCKETS=2` with `LLAMA_PREFILL_BATCH=M` for every ladder rung | Agreement with the one-socket batched reference; prefill and decode measured separately |
| D2 | [Final campaign and decision](D2-final-campaign-and-decision.md) | Crossover report across precision, model, context, M and sockets | Study §10.4 decision per feature |

## Track exit gate (milestones M4 and M5)

- **M4.** Two-socket batched prefill passes its numerical gates for FP32, BF16 and
  INT8, against the one-socket batched reference at the same rung.
- **M5.** `docs/numa-tp-prefill-results.md` and its validation companion are written.
  Defaults are chosen. The [plans README](../README.md) milestone table is complete.
