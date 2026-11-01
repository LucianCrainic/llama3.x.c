# Track B: splitting the model and KV cache across both sockets

## Goal

Run one model instance as two socket-owned shards, which the study calls paired
tensor parallelism (study §6.1–6.2):

- **Heads.** Each socket owns half of the query heads and half of the KV heads.
- **FFN.** Each socket owns half of the FFN intermediate channels.
- **Classifier.** Each socket owns half of the classifier's vocabulary rows.
- **Memory.** Each socket's weights and KV live in its own HBM (track A).
- **Workers.** Each socket runs on its own persistent team (F2).
- **Reductions.** Two FP32 vector reductions per layer combine the partial outputs,
  after Wo and after W2.

For Llama 3.1 8B, the split is:

| Socket | Query heads | KV heads | FFN channels | Vocabulary rows |
| --- | --- | --- | --- | --- |
| 0 | 0–15 | 0–3 | 0–7167 | 0–64127 |
| 1 | 16–31 | 4–7 | 7168–14335 | 64128–128255 |

Track A is placement only. This track changes what each socket computes.

## Phases

| ID | Phase | Output | Gate |
| --- | --- | --- | --- |
| B1 | [Shard descriptors and packing](B1-shard-descriptors-and-packing.md) | Per-socket shards: row slices and packed column slices, with Q8 scales | Reconstruction matches every value and scale |
| B2 | [Partial-output kernels](B2-partial-output-kernels.md) | `PARTIAL` epilogue; column-slice GEMV for FP32, W8A8 and W8A32 | Kernel budget vs a double reference |
| B3 | [Socket-local KV and attention](B3-socket-local-kv-and-attention.md) | Per-socket KV with local heads; per-team attention queues | Attention on a local-head shard matches the full computation |
| B4 | [Cross-socket reduction](B4-cross-socket-reduction.md) | Generation-tagged ready/consumed protocol | TSAN-clean; stress test; measured latency |
| B5 | [Two-socket FP32 forward](B5-two-socket-fp32-forward.md) | `LLAMA_SOCKETS=2` end to end for FP32 | Logits agree with one socket within 1e-3; argmax 100% |
| B6 | [Two-socket Q8](B6-two-socket-q8.md) | W8A8 and W8A32 with group-aligned partitions | Agreement with the one-socket Q8 path |
| B7 | [Two-socket decode campaign](B7-two-socket-decode-campaign.md) | Scaling, breakdown and decision | Study §10.4 criteria |

## Track exit gate (milestone M3, TP half)

- `LLAMA_SOCKETS=2` passes all correctness gates for FP32 and Q8.
- B7 reports whether two-socket decode beats the best one-socket configuration (from
  A6) beyond run variability, broken down by precision, model and context.
- A capacity gain without a latency gain is recorded as a different result. The study
  is explicit that it is not a failure (study §10.4).

## Local development model

The virtual topology (`LLAMA_TOPOLOGY=virtual:2xT`) runs two unpinned teams with
portable regions. Every B phase is first made correct on macOS, including the
reduction under TSAN. The cluster then has to show only placement and speed.

Every B test needs a new fixture, because the existing fixture has one KV head
(`tests/fixtures.py:7`, `CONFIG = (4, 8, 2, 2, 1, 128010, 32)`) and cannot be split:

- Config: `tp` fixture `(dim 16, hidden 32, layers 2, heads 4, kv_heads 2, vocab 128010,
  seq 64)`.
- INT8 group size 4, so the column slices (8 and 16) are aligned to groups.
- Shared and unshared variants.
