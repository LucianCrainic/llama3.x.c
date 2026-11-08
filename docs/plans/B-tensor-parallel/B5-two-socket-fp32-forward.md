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

Status: **implemented and validated on the cluster; two-socket decode is 1.86× one socket** (see Cluster results).

**Open risk (updated 2026-11-08):** none left from B5 itself. The conventions it sits on (A4 owned regions, B1 placement) and
the exchange (B4) pass their cluster checks, the forward agrees with one socket, and the decode slowdown seen in the first
smoke run was a thread-pinning bug (fixed; see B7). What remains is performance work, tracked in B7.

Implementation notes:

- **Module.** `include/tp.h`, `src/tp.c`, compiled into both executables (`COMMON`). `run.c` calls it when
  `LLAMA_SOCKETS=2`: `read_checkpoint` skips the owned loader, `engine_load_context` calls `tp_create`, then unmaps the
  file (the shards were read with `pread`); `engine_forward`, `engine_forward_batch` (per token, with `LOGITS_ALL`
  copies) and `engine_kv_stats` dispatch to it. The single-socket state is not allocated in that mode, and
  `prefill_gemm` is forced off (D1 owns the two-socket batch path). With `LLAMA_SOCKETS` unset or 1 no line of `tp.c`
  runs.
- **Pool.** `kernels_backend_init` builds two teams when `LLAMA_SOCKETS=2`: per-socket CPUs from the topology (virtual:
  unpinned), `LLAMA_THREADS` per team, a clear failure for a missing teams backend, an unknown value, or fewer than two
  sockets (`numa_check_placement`).
- **Coordinators.** The caller is the coordinator of socket 0; `tp.c` runs a thread as coordinator of socket 1
  (`kernels_set_thread_team`). Commands (`SETUP`, `LOAD`, `FORWARD`, `STOP`) run on both coordinators through a
  release/acquire generation channel, so even startup (carving, `shards_load`) runs on each socket's own team.
- **Per token.** Exactly the flow of §4.1, one `team_run` per kernel (the first-version form): embedding row copied into
  each socket's own residual (tied: from the owner's vocabulary shard; untied: the global shard), RoPE table computed on
  each socket, QKV on row shards into the local KV, local attention, `K_PARTIAL` Wo into `exchange_partial`'s buffer,
  `exchange_reduce_add`, FFN with the pair SwiGLU on `hidden/2` rows, `K_PARTIAL` W2, second reduction, final norm,
  vocabulary-row classifier writing each socket's range of one logits buffer (the remote-write form).
- **Regions.** Per socket: weights (`ROLE_WEIGHTS`, read-only after the load), `ROLE_KV_K`/`ROLE_KV_V`, a `ROLE_WORK`
  region carved by the owning coordinator, and the exchange region; the untied embedding in one global region. Node
  choice: `LLAMA_PLACEMENT=file` unbound, `ddr`/`hbm` the socket's node (the embedding always DDR), strict verification
  with `region_verify`. The HBM budget check of A4 is not applied yet.
- **Checks and report.** `LLAMA_TP_CHECK=1` copies each socket's residual after every layer and fails with the layer
  number if they differ in a single bit. The report says `sockets: 2` and lists both teams; only socket 0 times its
  phases (`Wo` and `W2` include their reductions); the dedicated `reduce`/`barrier_wait` phases are not in the report
  yet (`exchange_timing` exists).
- **Fixture and tests.** `tests/fixtures.py` gains `TP_CONFIG` (dim 16, hidden 32, 2 layers, 4 heads, 2 KV heads,
  vocabulary 128010, context 64), an odd-vocabulary variant, and a `gs` argument; it also fixes the unshared fixture for
  `dim != 4`. `tests/test_tp.py` (part of `make test`) runs `LLAMA_SOCKETS=2`, `virtual:2xT` with T = 1, 3, 7 against the
  one-socket teams backend for the shared, unshared and odd-vocabulary models: logits within `abs 2e-6 + rel 2e-5`
  (**measured** worst case 3.8e-6 absolute on logits up to 6.8), argmax agreement, the batched entry point with batch
  1, 3 and `LOGITS_ALL`, greedy 16-token generation identical (unshared fixture, where the argmax has a margin), the CLI
  `run` producing the same text and a completed report, `LLAMA_TP_CHECK=1` throughout, and the rejections (one KV head,
  teams backend missing, bad `LLAMA_SOCKETS`, one-socket topology, `hbm` on a virtual topology). ASAN/UBSAN and TSAN runs
  of `test_tp.py` are clean.
- **Not done here.** The multi-turn chat and stop-token tests of `test_runtime.py` with `LLAMA_SOCKETS=2`; the
  `compare_engines.py` harness (not in this tree; `test_tp.py` plays its role on the fixtures); the 1B/8B cluster gates,
  HF validation under two sockets, per-socket page verification, and the smoke benchmark with the per-phase breakdown
  (`cresco8-tp.sbatch PHASE=B5`, dry-run verified: `compare_engines.py` one socket vs two, the Hugging Face gate without
  the one-socket `numactl` binding, a profiled smoke run, and a weights-placement check of both sockets: socket 0 on node
  2, socket 1 on node 3, 100% verified; KV pages are not verified yet). The coordinator of socket 1 spins and yields between tokens, like the teams; the
  cost is part of the B7 measurement.

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

### Cluster results (2026-11-08)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Job 5694402 (commit `4677bd5`), `LLAMA_SOCKETS=2`, 56 workers per socket, weights on HBM nodes 2 and 3.

- **Agreement** (`compare_engines.py`, one socket against two, 53 positions, budget 1e-3, argmax 100%): max |Δlogit|
  1.0e-5 (1B) and 3.3e-5 (8B), argmax 100%. The smoke placement check passes on both models.
- **Bug found by the check.** For a model with a tied embedding the global region is never reserved; its zeroed record read as
  a bound region on node 0 and strict placement failed (fixed in `4677bd5`).
- **Speed.** The first smoke run showed decode slower than one socket (11.3 against 14.6-14.9 tok/s on 8B FP32). The cause
  was that both socket leaders ran on one CPU (found and fixed 2026-11-08, details and measurements in B7). With the fix,
  the fused reduction and the row scheduling of B7, 8B FP32 decodes at 30.2 tok/s on two sockets against 16.2 on one
  (1B: 160.7 against 91.0).
- **Re-check of the final tree** (job 5703488, `repo-decodefix7`): `make test` passes on the cluster with AVX-512, and
  one socket against two agrees on 1B and 8B FP32 (8B: max |Δlogit| 4.5e-5, rms 2.9e-6, argmax 100%) with the placement
  check passing.
- **Not run.** The Hugging Face gate with `LLAMA_SOCKETS=2` (`HF_GATE=0`).

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
