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

Status: **in progress** (8 November 2026). Branch: `feature/integ-d1-two-socket-prefill`. FP32 is implemented and
tested locally (L2), and so are W8A8 on the VNNI or portable Q8 GEMM (L3) and the packed AMX-BF16 and AMX-INT8 rungs (L4 of
[remaining-work](../remaining-work.md)), the latter on the software tile model only, and so are the stress and sizing tests (L5).
Open: the cluster check.

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

### 4.5 Implementation notes (FP32, L2)

- `tp_forward_batch` in `src/tp.c` (`CMD_FORWARD_BATCH`, `forward_fp32_batch`); `engine_forward_batch` calls it when the model
  is split, `count > 1` and the batched GEMM is on (`LLAMA_PREFILL_GEMM`: `auto` selects it only where AVX-512 or AMX
  runs, `on` forces the portable GEMM, `off` keeps the per-token loop). A single-row batch stays on `tp_forward`.
- Every socket holds `M_max` rows of `x`, `xb`, `hb`, `q` and `rope` in its work region, an attention workspace sized for
  `M_max` rows (`k_attention_init_rows`), and the exchange is created for `M_max × dim` elements, so `tp_create` takes the
  batch capacity (`model->max_batch`; 1 for Q8 until L3). `M_max × dim` above `INT_MAX` is refused at load.
- QKV is one three-segment `k_gemm_f32` into the socket's own heads and straight into its KV rows; Wo and W2 are
  `GEMM_PARTIAL` products over the owned K slice into the exchange buffer, followed by the two-barrier reduction
  (`exchange_reduce_add`) over `M × dim` elements. The per-row flag form of B7 has no row-completion callback in the GEMM
  and is not used here; it mixes with the barrier form across generations without change. The classifier stays per row
  (`k_matmul` on the socket's vocabulary range), for the last row only unless all-position logits are requested.
- `LLAMA_TP_CHECK=1` compares a hash of the whole `M × dim` residual of both sockets after every layer and microbatch.
- **W8A8 (L3).** The same `forward_batch` runs the Q8 GEMM (`k_gemm_q8`): the normalized rows are quantized once with
  `k_quantize_rows` for QKV and for W1/W3; Wo and W2 quantize only the socket's slice (`ql` and `hidden_l` values per
  row, which start on a group boundary), so their groups are the original groups; the partials go to the exchange as FP32.
  Embedding rows are dequantized from the owner's vocabulary shard, the classifier is the per-row `k_matmul_q8`. The
  quantized activation buffers are padded to whole 32-row blocks for the AMX kernel. Kernel reported: `amx-int8`
  with `LLAMA_KERNELS=amx` and a group size that is a multiple of 64 (packed shards, below), `avx512-q8` when AVX-512
  runs and the group size is a multiple of 64, else `portable-q8`.
- **W8A32** stays per token on two sockets, the C6 rule (`prefill_gemm` is forced off, the batch capacity is 1).
- **Packed shards (L4).** `LLAMA_PRECISION=bf16` (FP32 checkpoints) and AMX-INT8 (`LLAMA_KERNELS=amx`, W8A8, group size a
  multiple of 64) give every socket a `ROLE_PACKED` region on its own node with the VNNI copies of its seven shards per layer
  (`plan_packing`, `CMD_PACK`; `pack_bf16_job_from_shard` / `pack_s8_job_from_shard`). The main thread reserves, carves and
  later marks, protects and verifies the regions (the region registry is not thread-safe); the coordinators only run the
  jobs on their teams. The shards stay (decode reads them), the Q8 group scales are read from the shard in place.
  BF16 converts the activations into a per-socket panel (`k_gemm_bf16`); in BF16 mode even a single-row batch goes through
  the batched forward, as on one socket. All-position logits use a BF16 GEMM on a classifier range packed on first request
  (`pack_classifier`), the default last-row classifier stays FP32, exactly as C5 does.
- **Budget.** With `LLAMA_PLACEMENT=hbm`, `check_budget` counts per socket the shards, the packed copies, both KV caches, the
  work region and the exchange buffers against `LLAMA_HBM_BUDGET_GIB` (default 56) and names the packed bytes when it
  fails. It cannot run on the virtual topology (hbm placement is refused there), so it is checked on the cluster.
- **Sizing (L5).** The three conditions of §4.2 are pure functions in `src/tp.c` / `include/tp.h`: `tp_work_bytes` (what
  `tp_create` reserves; also behind its own check), `tp_exchange_bytes`, `tp_exchange_payload_bytes`,
  `tp_peer_bytes_per_token` and `tp_choose_batch`, which takes the measured exchange latency and bandwidth and per-layer
  times and returns, per candidate M, whether it fits, reaches the ridge and keeps the two exchanges under the share
  limit (modeled as `2 x (latency + payload / bandwidth)` against `max(shard stream, M x per-row time)`), and the largest
  feasible one. `tests/sizing.c` pins the §4.2 table (256 KiB / 1 MiB / 4 MiB payloads, 1 MiB per token independent of M)
  and each condition on the 8B and 1B shapes. The inputs in the test are modeled orders of magnitude; the cluster's M
  sweep supplies the real ones and selects M.
- **Exchange stress (L5).** `unit_exchange_batched` in `tests/workers.c` runs one exchange created for `256 x 4096`
  elements through a schedule of reductions of different lengths (256, 1, 3, 17, 64 rows of 4096, and single rows), the
  batch-sized ones in the two-barrier form and the decode-sized ones in the flag form, with random delays on either socket
  and with and without caller-owned regions, bitwise against a serial reference; it is clean under ThreadSanitizer.
  The residual-copy check (`LLAMA_TP_CHECK=1`) runs in every two-socket test of `tests/test_tp.py`.
- Not yet: overlap of the exchange with the next block's GEMM (§9).

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

`phase_D1` (and `phase_D2`) in `scripts/cresco8-prefill.sbatch` are written and dry-run verified (8 November 2026): D1 prints
120 launches per model plus 10 comparisons per model, about 2 hours; the comparison for each rung is one socket against two
at M = 64 with `--batch-a/--batch-b`, FP32 strict, W8A8 and AMX-INT8 under the statistical gate, W8A32 strict, BF16
recorded only until C5 has set its gate; placement is checked per rung with `tp_placement` (weights, packed shards for
the AMX rungs and the KV caches, which `tp_kv_stats` now verifies on both sockets); the sweep runs two sockets at every M
and one socket at M = 64 and 256 in the same job, then a profiled run per point and the decode after the two-socket
prefill. `GATES_ONLY=1` runs only the comparisons and the placement checks (about 30 minutes per model), without the sweep. The table
below is the plan; the script follows it.

`cresco8-prefill.sbatch STAGE=check PHASE=D1`. **Estimated** at 90 minutes. It needs
the user's go-ahead to submit.

| Run | Gate or record |
| --- | --- |
| 1B and 8B, every rung, M = 64, prompts 86 and 2048: one socket batched vs two sockets batched, `all` logits | FP32: max\|Δlogit\| ≤ 1e-3, argmax 100%. BF16 and INT8: within the C5/C6 recorded budgets relative to their one-socket rung. |
| `validate.py` FP32 with `LLAMA_SOCKETS=2 LLAMA_PREFILL_BATCH=64` | HF gate as today |
| Placement | Shards, packed shards, KV and exchange buffers per socket, 100% verified |
| M sweep: 8B, prompt 2048, every rung, M ∈ {16, 32, 64, 128, 256}, profiled | **Measured** per-socket GEMM, attention, exchange and wait times; selects M per rung |
| Decode after a two-socket prefill, measured separately | Equal to B7's two-socket decode within range |

### Cluster results: gates and placement (8 November 2026)

hbm14, Flat mode, `PHASE=D1 GATES_ONLY=1 SKIP_TEST=1 HF_GATE=0` (`make test` passed in job 5704671 on the same code). Jobs 5704683 and 5704685
(trees 8310e50 and e763e08) found two script problems and one gate that was too tight for 8B; **job 5704689 (tree cd31a67, 5 minutes, exit 0) passed
all 20 comparisons and all 8 placement checks.** One socket against two at M = 64 (max |dlogit| / rms; the first column is the 86-token prompt at every
position, the second the last row of the 2048-token prompt):

| Rung | 1B | 8B |
| --- | --- | --- |
| AVX-512 FP32 GEMM (strict 1e-3) | 8e-6 / 5e-6 | 7e-5 / 8e-6 |
| AMX-BF16 (recorded) | 0.08 / 0.05, rms 0.012 / 0.016 | 0.41 / 0.04, rms 0.020 / 0.008 |
| AVX-512 Q8 W8A8 (statistical) | rms 0.063 / 0.041 | rms 0.063 / 0.046 |
| AMX-INT8 W8A8 (statistical) | rms 0.064 / 0.041 | rms 0.136 (max 4.12 on one position) / 0.042 |
| AVX-512 W8A32 (strict 1e-3) | pass / pass | pass / pass |

The 8B AMX-INT8 deviation is the W8A8 requantization noise floor, not a defect of the two-socket path (diagnostic D3, job 5704685, 8B W8A8): **one
socket**, AMX against AVX-512, max 1.41 and rms 0.086 on 86 tokens, 7.58 and 0.115 on the last row of 2048; two sockets against one at the same AMX
kernel 4.12 / 0.136 and 8.86 / 0.112; two sockets with M = 64 against M = 16 bitwise identical, on both models and both prompts. The first provisional
bound (max 2.0, rms 0.15, calibrated on 1B) was too tight; `W8A8_GATE` is now max 16, rms 0.2, argmax >= 0.9. The 2048-token comparisons first
failed on a harness guard (`--last-only` needed a per-token side A) and, for W8A32, on a script bug (the flags lost `--batch-a/--batch-b`); both fixed
and covered by tests.

Placement (one two-socket run per batched rung, `tp_placement`): the weights of each socket are one region on its HBM node (2 and 3), fully verified, 0
pages elsewhere or missing (8B: 14 GiB per socket); the packed shards of the AMX rungs (8B BF16: 6.5 GiB per socket) are fully verified on the same
nodes; the KV caches of both sockets are verified with 0 pages on another node (the missing pages are positions not yet written); the untied embedding
is on DDR node 0, fully verified. **Not verified in that job:** the work region and the exchange buffers (about 10 MB per socket) were reserved on the socket's
node but never fully populated, so the page query reported them unverified. Changed afterwards (8 November 2026, not yet run on the cluster): the work
region is written whole by its socket at setup, the exchange region is zeroed whole before the exchange is created, both are verified like the other
regions, and `tp_placement` asserts them (`scratch`) in the D1 placement runs. BF16 packing took 0.11 s for 13 GiB (the `weight_pack` timing).

Open: the Hugging Face gate with two sockets (`HF_GATE=0` here); the M sweep, the profiled runs and the decode after prefill (the rest of `phase_D1`);
the verification of the work and exchange regions on the hardware.

### Cluster results: the M sweep, profiles and decode (9 November 2026)

Job 5704694 (tree 04e913b, hbm14, `PHASE=D1 SKIP_TEST=1 HF_GATE=0`, exit 0): 22 minutes, 242 reports, 0 failures; the gates and the eight placement checks
passed again, and the work and exchange regions are now fully verified on their sockets' nodes. Prompt of 2048 tokens, engine-owned HBM placement,
56 workers per socket, medians of 3 repeats. **Prefill tok/s, two sockets (and the ratio to one socket at the same M):**

| Model, rung | M = 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B AVX-512 FP32 GEMM | 249 | 313 | 353 (1.76x) | 350 | 327 (1.86x) |
| 8B AMX-BF16 | 512 | 704 | 880 (1.68x) | 1016 | **1092** (1.69x) |
| 8B AVX-512 Q8 W8A8 | 310 | 344 | 376 (1.82x) | 391 | 400 (1.84x) |
| 8B AMX-INT8 W8A8 | 575 | 761 | 897 (1.57x) | 1001 | **1068** (1.64x) |
| 1B AMX-BF16 | 2416 | 3174 | 3849 (1.53x) | 4297 | 4636 (1.56x) |
| 1B AMX-INT8 | 2607 | 3308 | 3651 (1.51x) | 4001 | 4341 (1.53x) |

TTFT of the 2048-token prompt, 8B: AMX-INT8 2.0 s on two sockets (M = 256) against 3.3 s on one; AMX-BF16 2.0 s against 3.3 s; AVX-512 FP32 6.0 s against 10.5 s;
the per-token one-socket baseline is about 126 s. **M selection:** the throughput still rises at M = 256 for the AMX rungs (+8% over M = 128) and flattens at
M = 64-128 for the AVX-512 rungs (FP32 falls at 256); the exchange is not what limits M (below).

**Where the time goes** (8B, socket 0, ms per token, M = 256): AMX-BF16 total 0.931 = rmsnorm 0.170 + qkv 0.115 + attention 0.233 + wo 0.060 (incl. its
reduction) + w1/w3 0.161 + w2 0.153 + rope 0.034; the exchange itself (add 0.014 + waits 0.017) is **3.3%** at M = 256, 4.4% at M = 64, 5.8% at M = 16 (AMX-INT8
2.3% / 4.7% / 6.6%; AVX-512 FP32 1.0% / 1.8% / 3.5%): under the 10% budget at every M, and the imbalance between the sockets is small. Two sockets give 1.5-1.9x one
socket, not 2x, because of the per-row operations that run serially on each socket's coordinator thread: **RMSNorm is 18% of the AMX-BF16 time per token (9% of the
AVX-512 FP32 time)** and RoPE 3-4%, none of which halves with the second socket, and attention is 21-25% of the AMX rungs' time (FP32 kernels, half the heads per
socket). Parallelizing the row loops over the team is the first lever: done afterwards (`k_rmsnorm_rows`, `k_rope_apply_rows`: one team job per call, each row exactly the
single-row kernel, bitwise the serial loop; used by the two-socket forward and by both one-socket batched forwards); its effect on the sweep is not measured yet. **Decode after a two-socket prefill** (128 tokens from 2048): FP32 30.6-30.7 tok/s
(8B) and 162-163 (1B), W8A8 83 (8B) and 392-396 (1B): FP32 equals B7's 30.2-31.05 within range; W8A8 is 13% below B7's 95.5 at an 86-token context, consistent with
the longer context and to be compared in B7 at equal contexts.

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
