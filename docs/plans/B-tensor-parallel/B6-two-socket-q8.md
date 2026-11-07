# B6. Two-socket Q8_0 (W8A8 and W8A32)

## 1. Goal and status

Extend the two-socket forward pass to `runq` with the existing Q8_0 semantics:

- the same group size;
- the same scales;
- the same `k_quantize` activation quantization;
- W8A8 by default, with W8A32 kept through `LLAMA_Q8_ACTIVATIONS=fp32`.

The partitions stay aligned to groups (study §5.4, §6.2).

Status: **planned**. Branch: `feature/tp-b6-q8`.

## 2. Depends on / unblocks

- Depends on: B5.
- Unblocks: B7, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| Q8 `forward` | `src/runq.c:223-288` | `prepare_input` (:124) quantizes before each matmul: 4 per layer, plus the classifier |
| `k_quantize` | `src/kernels_simd.c:611` | Bitwise equal across ISAs (`tests/kernels.c:299-308`) |
| `fp32_activations` | `src/runq.c:14`, env at :299-302 | W8A32 switch |
| K/V scratch and copy | `src/runq.c:254-258` | Writes into the **local** KV after B3 |
| Q8 shards | B1 | Scales packed per row slice and per column slice |

## 4. Changes

- **QKV, W1, W3 (row shards).** Each socket quantizes the **full** normalized input
  `u_s` locally. The input is identical on both sockets, so the int8 values and scales
  are identical too (bitwise).
- **Wo, W2 (column shards).** Each socket quantizes **only its owned slice** of `a_s`
  or `h_s`. The slice starts on a group boundary (B1 validation), so its groups are
  exactly the original groups for those elements.
- **Classifier.** Quantize the full final `x_s` locally on each socket, then run the
  vocabulary-row shards.
- **W8A32.** No activation quantization. The `dots_q8f` path runs on the shards with
  per-row scale slices.
- **Q8 embedding.** A row-wise dequantization from the owner shard (A4), in DDR for an
  untied model, or the owner's HBM shard for a tied one.
- **Static `GS`.** `runq.c` keeps it file-static (`src/runq.c:11`), because both
  sockets share one model.

## 5. Local tests

These run on macOS, with `virtual:2x3` and the INT8 `tp` fixture at `gs = 4`.

- **Quantization equivalence.** Socket-local quantization of each owned slice must
  equal the corresponding groups of a full-vector quantization, **bitwise**.
- **One- vs two-socket logits.** Logits for `LLAMA_SOCKETS=1` vs `2` are compared with
  `compare_engines.py` for both W8A8 and W8A32. The tolerance is the fixture budget:
  only the order of the `p0+p1` FP32 sum differs.
- **Rejection.** A fixture with `gs = 16` and `dim/S = 8` must be rejected with the B1
  error.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B6`. **Estimated** at 60 minutes. It needs the
user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| `compare_engines.py` 1B and 8B INT8, W8A8 and W8A32, one socket vs two | max\|Δlogit\| ≤ 1e-3, argmax 100%. This is a C-vs-C gate, so it is strict even though Q8 vs HF is not. |
| `validate.py --max-abs 1.0 --rtol 0.05` (today's Q8 recording) | Recorded only. The two-socket max_abs must not exceed one-socket max_abs + 1e-3. |
| Smoke benchmark: 8B Q8, 2 × 56, with profiling | Recorded |

## 7. Exit criteria

- The Q8 two-socket path agrees with the one-socket Q8 path for both W8A8 and W8A32.
- Incompatible group sizes are rejected.
- The error vs HF is recorded next to the one-socket values (**measured** today: W8A8
  1.32/1.42, W8A32 0.76/0.64 for 1B/8B).

## 8. Risks and open questions

- **Instruction-bound Q8 decode.** The Q8 path is instruction-bound at 64% of
  bandwidth (`docs/kernels.md:127-133`). Doubling the bandwidth may not double the
  speed, and two sockets mainly double the compute here. B7 measures this rather than
  assuming it.
- **Sync overhead on 1B Q8.** 1B Q8 (222.65 tok/s **measured**, about 4.5 ms per
  token) is the case where synchronization most likely dominates (study §9).

## 9. Not in this phase

- AMX-INT8 (C6, D1).
- BF16 (C5).
- Re-quantization at any other group size.
