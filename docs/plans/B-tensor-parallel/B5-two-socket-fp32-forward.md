# B5. Two-socket FP32 forward pass

## 1. Goal and status

Wire B1–B4 into a complete `LLAMA_SOCKETS=2` forward pass for FP32 checkpoints. It must
cover:

- embedding broadcast;
- the per-layer paired projections and two reductions;
- socket-local attention;
- the vocabulary-split classifier and logit gather;
- one sampler, with the token broadcast.

Its logits must agree with the one-socket reference within the FP32 budget.

Status: **planned**. Branch: `feature/tp-b5-fp32-forward`.

## 2. Depends on / unblocks

- Depends on: B3 and B4. Through them, also B1, B2, A5 and F2.
- Unblocks: B6, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| FP32 `forward` | `src/run.c:146-205` | The layer order is preserved; only the projections and residual handling change |
| `k_rmsnorm`, `k_rope_table` | `include/kernels.h:17-21` | Computed locally on each socket's identical residual copy |
| Sampler | `src/sampler.c`, `app.c` `run_sequence` :199 | Runs once, on the coordinator |
| `engine_forward` | `include/runtime.h:65-70` | Signature unchanged; it returns the gathered logits |

## 4. Changes

### 4.1 Per-token flow

The coordinator is socket 0's leader. Each socket s executes the following inside one
`team_run` per layer group, with barriers instead of function returns (study §6.2):

```text
coordinator: x_row = embedding[token]           # untied: DDR (A4); tied: row from owner shard
both sockets: x_s = copy(x_row)                 # each socket's own residual copy
for each layer l:
  u_s = rmsnorm(x_s)
  q_s,k_s,v_s = QKV_s(u_s)                      # row shards; K/V written into local KV (B3)
  rope(q_s,k_s, local heads)
  a_s = attention_local(q_s, K_s, V_s)          # B3
  p_s = Wo_s · a_s          [PARTIAL]           # B2, column shard
  x_s += p_0 + p_1                              # B4 reduction 1
  u_s = rmsnorm(x_s)
  h_s = swiglu(W1_s·u_s, W3_s·u_s)              # local hidden/S
  r_s = W2_s · h_s          [PARTIAL]
  x_s += r_0 + r_1                              # B4 reduction 2
x_s = rmsnorm(x_s)
logits[vocab_s range] = Wcls_s · x_s            # each socket writes its own range
coordinator: wait for socket 1's "logits ready" flag (acquire); sample; next token
```

### 4.2 Implementation notes

- **Logit buffer.** It lives in the coordinator's region. Socket 1 writes its vocabulary
  range remotely; that is about 256 KB per token for 8B (study §6.3). The alternative,
  a local buffer that the coordinator reads, is chosen by measurement in B7. The first
  version uses the simpler remote write.
- **Token broadcast.** The next token reaches socket 1 through a generation-tagged slot.
  Socket 1 never runs its own sampler or RNG.
- **Residual copies stay identical.** Both residual copies remain bitwise identical
  (B4's fixed operand order). A debug mode (`LLAMA_TP_CHECK=1`) compares them after
  each layer and fails on any difference.
- **Selection.** `LLAMA_SOCKETS=2` requires `LLAMA_BACKEND=teams`. Placement and
  topology come from A1/A4. With `virtual:2xT` it runs anywhere, using portable
  regions.

## 5. Local tests

These run on macOS, using `virtual:2x3` and the `tp` fixture (see the
[track README](README.md)).

- **One-socket reference.** `compare_engines.py`:
  `LLAMA_SOCKETS=1 LLAMA_BACKEND=teams` vs `LLAMA_SOCKETS=2`. Logits at all positions
  are within `rel 2e-5 / abs 2e-6` (the fixture tolerance), with argmax agreement.
- **Residual copies.** `LLAMA_TP_CHECK=1` passes on every test.
- **Generation.** Greedy generation of 16 tokens is identical, and the chat multi-turn
  and stop-token tests from `test_runtime.py` pass with `LLAMA_SOCKETS=2`.
- **Configurations.** Shared and unshared classifiers, and an odd fixture vocabulary
  (B1 risk).
- **TSAN.** The full fixture run under TSAN is clean.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B5`. **Estimated** at 60 minutes. It needs the
user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| `compare_engines.py` 1B and 8B FP32: one socket (A6 config) vs two sockets (2 × 56, HBM 2 and 3), 4 `validate.py` prompts, all positions | max\|Δlogit\| ≤ 1e-3, argmax 100% |
| `validate.py --hf` with `LLAMA_SOCKETS=2` | Today's FP32 budget vs HF (1e-3) |
| Placement | Socket 0 regions on node 2 and socket 1 regions on node 3, 100% verified |
| Smoke benchmark: 8B FP32, 2 × 56, `short.txt`, 1 repeat, with `LLAMA_PROFILE=1` | Records decode tok/s and the per-phase breakdown. Not a gate. |

## 7. Exit criteria

- All correctness gates pass for 1B and 8B FP32.
- The residual copies are identical.
- Placement is verified per socket.
- A first per-phase breakdown is recorded, so that B7 can be planned on real costs.

## 8. Risks and open questions

- **Error tolerance.** The **measured** one-socket vs HF error on 8B is 1.6e-4
  (`docs/kernels.md`). The split's error is expected to be of the same order; 1e-3
  leaves margin. If it is exceeded, compare per layer with `LLAMA_TP_CHECK` and the
  per-layer dumps before changing tolerances.
- **Barrier count.** One `team_run` per layer, or per token with internal barriers,
  changes the barrier count. Start with the per-token form, with barriers at the
  reductions only.

## 9. Not in this phase

- Q8 (B6).
- Batched prefill on two sockets (D1).
- Distributed sampling.
