# D1. Two-socket batched prefill

## 1. Goal and status

Run microbatched prefill on both sockets using the paired partition. Each socket:

- computes GEMMs on its own shards, with the AVX-512 FP32, AMX-BF16 or AMX-INT8 rung;
- appends K/V for its own heads;
- runs causal attention on its own heads.

The two `M × dim` FP32 partial sums per layer are combined with B4's protocol, and
the residual is applied once.

The result must agree numerically with the **one-socket batched reference at the same
rung**. Prefill and decode are measured separately.

Status: **planned**. Branch: `feature/integ-d1-two-socket-prefill`.

## 2. Depends on / unblocks

- Depends on:
  - B6: the two-socket forward pass with FP32 and Q8 shards.
  - C3: batched causal attention with local-head parameters.
  - C5 and C6: BF16 and INT8 packing on `Shard` descriptors.
- Unblocks: D2.

## 3. Reuse

There is no new scheduler, and no second implementation of any kernel.

| Code | From | Use |
| --- | --- | --- |
| Shards and per-socket regions | B1, A4 | The same row and column shards. C5/C6 packs per shard on the owning team. |
| `PARTIAL` epilogue GEMM | C2, B2 | Wo/W2 on column shards; QKV/W1/W3 on row shards |
| Batched causal attention | C3 with B3 local counts | Local heads, local KV stride, `pos0`/M identical on both sockets |
| Exchange | B4 | Buffers sized `M_max × dim`. The protocol is unchanged. |
| Batched API and microbatching | C1 | `engine_forward_batch`; `LOGITS_LAST` gathers only the final row's logits |
| Logit gather and token broadcast | B5 | The final row only |

## 4. Changes

### 4.1 Forward pass

One microbatch on socket s (study §6.2, extended to M rows):

```text
X_s[M×dim] = embed rows (coordinator gathers M rows, broadcasts)        # identical on both sockets
per layer:
  U_s = rmsnorm_rows(X_s)
  [Q_s | K_s | V_s] = GEMM(U_s, Wqkv_s)          # row shards, local heads only
  rope_rows(Q_s, K_s, pos0..pos0+M-1)
  append K_s, V_s rows to local KV                 # positions identical on both sockets
  A_s = causal_attention_local(Q_s, KV_s, pos0, M)
  P_s = GEMM(A_s, Wo_s)  [PARTIAL]                 # column shard → M×dim partial
  X_s += P_0 + P_1                                 # exchange 1: M×dim FP32
  U_s = rmsnorm_rows(X_s)
  H_s = swiglu(GEMM(U_s, W1_s), GEMM(U_s, W3_s))   # M × hidden/S
  R_s = GEMM(H_s, W2_s)  [PARTIAL]
  X_s += R_0 + R_1                                 # exchange 2: M×dim FP32
final row: rmsnorm → vocab-split classifier → gather → sample (B5)
```

### 4.2 Microbatch sizing

These figures are **modeled** for 8B. Payload per exchange is `M × 4096 × 4 B`:

| M | Payload per exchange | Bytes read from the peer per token (both exchanges, all layers) |
| ---: | ---: | ---: |
| 16 | 256 KiB | ≈ 1 MiB |
| 64 | 1 MiB | ≈ 1 MiB |
| 256 | 4 MiB | ≈ 1 MiB |

Per token the payload is constant. Per layer it grows with M. Per microbatch it is
`64 × M × 16 KiB`, i.e. 64 exchange phases of `M × 16 KiB`.

The exchange latency per byte (B4's microbenchmark at E = 4096 × {16, 64, 256}) is
compared with the per-layer GEMM time at M.

M is chosen subject to three conditions:
1. **Fits.** `M_max` fits the per-socket `ROLE_WORK` budget: activations, the exchange
   double buffers `2 × 2 × M × dim × 4`, and the attention workspace in waves (C3).
2. **Ridge.** M is at least the rung's ridge (B≈32 for FP32, B≈50 for BF16, from C7).
3. **Exchange cost.** The exchange is at most about 10% of the layer time (from the
   B4 measurements).

D1 records the selected M per rung. It does not assume that C7's one-socket optimum
carries over.

### 4.3 KV updates

- **Positions.** Both sockets append rows `pos0..pos0+M-1` to their own head shards
  before attention, with a team barrier on each side. No cross-socket barrier is needed
  for KV, since attention reads only local KV.
- **Context limit.** Admission of `pos0 + M` against the context is checked once, by
  the coordinator, before the microbatch starts on either socket (study §7).

### 4.4 Precision rungs

- `LLAMA_PRECISION=bf16` packs BF16 per shard on each socket. The per-socket budget is
  **modeled** at 20.97 GiB of weights for 8B (C5).
- AMX-INT8 quantizes:
  - full rows for the QKV/W1/W3 inputs;
  - the owned slice of each row for Wo/W2, on the original group boundaries (B6).
- W8A32 stays on AVX-512.

## 5. Local tests

These run on macOS with `virtual:2x3`, the `tp` fixture, and the scalar FP32 and BF16
references. AMX is skipped on macOS.

- **FP32.** `compare_engines.py`: `LLAMA_SOCKETS=1 LLAMA_PREFILL_BATCH=M` vs
  `LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=M`. With `all` logits, at M ∈ {1, 3, 7, 32} and
  prompts {1, 5, 33}, results are within the fixture tolerance.
- **Q8 and scalar BF16.** The same comparisons, within the C5 and C6 budgets on the
  same rung.
- **Exchange stress.** Exchange buffers at M_max, under TSAN, with delay injection
  (B4 tests reused at E = M × dim).
- **Residual copies.** `LLAMA_TP_CHECK=1` shows both sockets' `X_s` identical after
  every layer and microbatch.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=D1`. **Estimated** at 90 minutes. It needs
the user's go-ahead to submit.

| Run | Gate or record |
| --- | --- |
| 1B and 8B, every rung, M = 64, prompts 86 and 2048: one socket batched vs two sockets batched, `all` logits | FP32: max\|Δlogit\| ≤ 1e-3, argmax 100%. BF16 and INT8: within the C5/C6 recorded budgets relative to their one-socket rung. |
| `validate.py` FP32 with `LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=64` | HF gate as today |
| Placement | Shards, packed shards, KV and exchange buffers per socket, 100% verified |
| M sweep: 8B, prompt 2048, every rung, M ∈ {16, 32, 64, 128, 256}, profiled | **Measured** per-socket GEMM, attention, exchange and wait times; selects M per rung |
| Decode after a two-socket prefill, measured separately | Equal to B7's two-socket decode within range |

## 7. Exit criteria

- Two-socket batched prefill agrees with the one-socket batched reference at each
  rung.
- Placement is verified.
- M is selected per rung, with an exchange-cost breakdown.
- Decode after prefill is unaffected.

## 8. Risks and open questions

- **Long prompts.** Attention dominates at long prompts. On each socket it is local and
  balanced, but the context cost is quadratic regardless of the socket count.
- **Larger M grows the exchange.** For large M the two exchanges become
  bandwidth-relevant (4 MiB per exchange at M = 256). If they dominate, test pipelining
  the exchange of row block j with the GEMM of block j+1 as a follow-up. That is not
  planned here.

## 9. Not in this phase

- Multi-request prefill/decode interleaving.
- Overlapping the exchange with compute.
- More than two sockets.
