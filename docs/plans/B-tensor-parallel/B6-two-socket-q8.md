# B6. Two-socket Q8_0 (W8A8 and W8A32)

## 1. Goal and status

Extend the two-socket forward pass to `runq` with the existing Q8_0 semantics:

- the same group size;
- the same scales;
- the same `k_quantize` activation quantization;
- W8A8 by default, with W8A32 kept through `LLAMA_Q8_ACTIVATIONS=fp32`.

The partitions stay aligned to groups (study §5.4, §6.2).

Status: **implemented; W8A32 agreement passed, W8A8 recorded against a statistical gate, decode slower than one socket** (see Cluster results).

**Open risk (updated 2026-11-08):** B1/B4 placement and exchange and the AVX-512 Q8 kernels behind the `ld` argument (B2)
have now run on real sockets and pass. Open: the two-socket decode speed (as B5), and the W8A8 comparison, which can only
be judged against a statistical gate (see Cluster results).

Implementation notes:

- **Same module.** `forward_q8` in `src/tp.c`, reached from `runq.c` exactly as `run.c` reaches the FP32 path
  (`tp_create(..., quantized = 1, GS, fp32_activations)`); `GS` stays file-static in `runq.c` and `tp.c` gets the group
  size from the checkpoint. `LLAMA_Q8_ACTIVATIONS=fp32` selects W8A32 as before.
- **Quantization rule.** QKV, W1/W3 and the classifier quantize the full normalized input on each socket (identical on
  both). Wo and W2 quantize only the owned slice (`dim/2` of the attention output, `hidden/2` of the FFN activation),
  which starts on a group boundary, so its groups and scales are the original ones. The kernel tests assert that slice
  quantization equals the corresponding groups of a full quantization bitwise (B2).
- **Q8 embedding.** One row dequantized from the owner's vocabulary shard (tied) or the global shard (untied), as
  `runq.c` does.
- **Rejection.** The `gs = 16` fixture (`dim/S = 8`) fails with the B1 message ("group size 16 does not divide the
  per-socket slices"), before any region is reserved.
- **Tests** (`tests/test_tp.py`). `probe-int8` with `LLAMA_SOCKETS=2` against one socket, W8A8 and W8A32, shared,
  unshared and odd-vocabulary fixtures, T = 1, 3, 7, within the fixture budget (`abs 2e-6 + rel 2e-5`; **measured**
  worst case 1.05e-5 absolute on a logit of 6.8, W8A32); the batched entry point; identical greedy generation;
  `LLAMA_TP_CHECK=1` throughout; the `gs = 16` rejection.
- **Not done here.** `compare_engines.py` and `validate.py` runs on 1B and 8B (the HF numbers next to the one-socket
  values), and the 8B smoke benchmark with profiling (`cresco8-tp.sbatch PHASE=B6`, dry-run verified).

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

### Cluster results (2026-11-08)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Job 5694402 (commit `4677bd5`).

- **W8A32** (activations stay FP32, no requantization): one socket against two agrees to 8.3e-6 (1B) and 6.7e-5 (8B),
  argmax 100%, inside the strict 1e-3 budget.
- **W8A8**: max |Δlogit| 0.41 and rms 0.063 (1B), 0.58 and 0.063 (8B), argmax agreement 100% and 98.1%, with position 0
  equal to 4e-6 and a position-independent offset of 0.2-0.3 from position 2 on. The strict 1e-3 gate failed. The same
  signature appears with no second socket at all: scalar against AVX-512 per-token on 1B gives rms 0.063 and max 0.34
  (job 5703346), because W8A8 requantizes its activations and any change of FP32 order can flip a rounding decision that then
  sits in the KV cache. The W8A8 comparison now uses the statistical gate `W8A8_GATE` (max 2.0, rms 0.15, argmax >= 0.9,
  fixed from these observations, to be confirmed by a re-run); W8A32 stays strict as the evidence for the arithmetic.
- **Speed, smoke only** (one launch): decode 137.8 tok/s (1B Q8) and 33.0 tok/s (8B Q8), against about 250-300 and 44.5-50.8
  on one socket. Same finding and next step as B5.
- **Not run.** The Hugging Face deviation with `LLAMA_SOCKETS=2` (`HF_GATE=0`).

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
