# E6: batched decode design (design only)

Phase E6 of the [follow-up plan](README.md). It decides, with a model and a small experiment, whether multi-request batching
is worth a full implementation plan. **Nothing here is implemented**: the only code added is a benchmark extension, an
experiment script and the model script (`tools/e6_model.py`). The go/no-go (D-E3) belongs to the owner; this document prepares it.

Every number is labelled **measured** (a benchmark or engine run, source named), **derived** (arithmetic on measured values),
**modeled** (output of the model of section 3) or **assumed/estimated** (no measurement behind it). Provenance and gaps:
sections 8 and 9.

## 1. Result in one paragraph

The "weights are read once for N rows" premise is **confirmed for the FP32 GEMM and for AMX-INT8**: on 8B, one socket, the
streaming time of all decode GEMMs at M = 8 is 1.30x (FP32) and 1.24x (AMX-INT8) of M = 1, against 8x for eight GEMVs
(measured, section 4.1). The AVX-512 VNNI Q8 GEMM does not keep that property (2.44x at M = 8): it becomes compute-bound
from M of about 4. But attention is per request and not shared: on 8B W8A8 at 2048 tokens it is 3.6 ms per request per token
(measured, one socket) against a 17-21 ms weight-streaming step, so at N = 8 the model gives **2.2x (AVX-512) to 2.9x (AMX-INT8)
aggregate throughput but a per-request step time of 2.7x to 3.7x the single-request token**. The E6.3 gate (at least 2x
aggregate **and** per-request latency at most 1.5x) therefore **fails on the latency condition** in all eight configurations
modeled (one and two sockets, AVX-512 and AMX, `split` and `group`); the throughput condition alone is met at N = 8 (2.05-2.94x). The
failure does not depend on the model's optimistic parts: attention alone at N = 8 (29 ms, AMX, one socket) exceeds the whole latency budget left after the GEMMs
(33.4 ms allowed minus 21.5 ms of GEMM = 12 ms). Per-request latency stays within 1.5x only up to N = 2 (AVX-512) or N = 3 (AMX-INT8, 1.50x). FP32 on 8B,
which is weight-bound, is the case where batching pays most (4.6x at N = 8, latency 1.73x; 3.2x at N = 4, latency 1.27x). Two independent one-socket processes
(no new code) already give 2x the aggregate of one process, and batched two-socket tensor parallel at N = 8 is no better than that baseline (section 5.3).
The model reproduces the measured N = 1 token within -10.4% to +7.9% (16 engine cells, this job) and within +/-24% against B7's long
contexts, optimistically on attention (section 3.2).

**Gate verdict: FAIL (latency), throughput half met.** The decision itself (D-E3) is the owner's; the numbers it needs are in sections 3 to 5.

## 2. Inputs and where each comes from

| Input | Value / use | Label | Source |
| --- | --- | --- | --- |
| Time of one decode GEMM (qkv, wo, w1/w3, w2, classifier) at M = 1, 2, 4, 8, 16, 32 | median of 100 cold-weight calls, weights on HBM, 56 workers; AVX-512 FP32, AVX-512 VNNI Q8, AMX-INT8, AMX-BF16 (reference); one socket full shapes, two sockets shard shapes run at once | measured | job 5875687, `E6/gemm-*.out` (commit 6a8586e, `tests/gemm_bench.c`) |
| Per-layer decode attention at contexts 512 / 2048 / 8192 / 32768, 1B and 8B, all and half the KV heads, `split` and `group` | median of 1000 calls, one request | measured | job 5875687, `E6/attention-*.out` (`tests/attention_bench.c`; 2048 is new, E5 measured the other three) |
| Per-token remainder (embedding, norms, RoPE, KV store, quantize, sampling-side work) | engine total minus qkv, wo, ffn, classifier, attention of the profiled run at 2048 tokens; 8B W8A8 one socket: 1.26 ms | measured, charged per request (assumed linear in N) | job 5875687, `*-xe6prof-1.json` / `.stderr` |
| Measured N = 1 token at 512 and 2048 tokens | 3 repeats per cell | measured | job 5875687 engine runs `*-xe6-N.json` |
| B7 / E5 N = 1 tokens at 512 / 8192 / 32768 | used for the long-context reproduction check | measured (other node of the partition) | [tensor-parallel.md](../../tensor-parallel.md), [decode-attention-results.md](../../decode-attention-results.md) |
| Weight bytes, KV bytes per token, shapes | 8B 7.97 GB (W8A8) / 30.0 GB (FP32); KV 256 KiB (8B) and 64 KiB (1B) per token | derived | `tools/e6_model.py` from the model shapes |
| Cross-socket exchange for N rows | 2 per layer, 5 us + N x dim x 4 B at 50 GB/s | **modeled** (fixed part from B7's N = 1 waits, bandwidth from the remote-HBM read ceiling of [hbm-roofline.md](../../hbm-roofline.md)); not measured at N > 1 | `exchange_seconds` |
| HBM per socket 64 GiB, work-region and packed-copy reserve 4 GiB | capacity table | measured / **estimated** (reserve) | roofline; `fit_requests` |

The weight read time of the plan's formula (weight read + N x attention + batched GEMM) is **not added separately**: the bench
streams the weights from HBM in every call, so the measured GEMM time at M = N already contains the weight read and the compute.

## 3. Model

### 3.1 Form

```
step(N, C) = L x (qkv(N) + wo(N) + ffn(N)) + classifier(N)      measured gemm-bench time at M = N (best of GEMM and N GEMVs,
                                                                 and for W8A8 optionally AMX-INT8)
           + N x L x attention(C)                               measured, one request charged N times (upper bound, 3.4)
           + N x other                                          measured at N = 1, assumed linear
           + exchange(N)                                        modeled, two sockets only
aggregate tok/s = N / step,  per-request latency = step (one token per request per step)
```
Assumptions: all N requests are at the same context C; M between measured points is interpolated linearly; one request per row
(no speculation); sampling is greedy (the campaigns used `-t 0`). `tools/e6_model.py model` prints every table below.

### 3.2 N = 1 reproduction and its residual

Against the engine runs of **this job** (same node as the bench inputs; model step at N = 1 against the median ms per token of 3 repeats):

| Cell group | Cells | Residual range | Notes |
| --- | ---: | --- | --- |
| 8B and 1B, FP32 and W8A8, one and two sockets, 512 and 2048 tokens (AVX-512 policy) | 16 | **-10.4% to +7.9%** | worst: 8B FP32 two sockets at 512 (-10.4%), 8B W8A8 one socket at 512 (+7.9%) |
| the gate cell, 8B W8A8 one socket, 2048 tokens | 1 | **-4.5%** (AVX-512), -5.3% (AMX policy: modeled, no engine run uses AMX decode) | modeled 22.49 ms against measured 23.55 [22.96-25.29] |

Against **B7 / E5** (other node, long contexts; modeled N = 1 token from this job's bench inputs against B7's tok/s, 24 cells): residual **-7.7% to +24.2%**,
mean absolute 8.3%. The model is **optimistic at long context** (it predicts faster tokens in 18 of 24 cells; worst 8B W8A8 two sockets at 32768, +24.2%): the engine's attention phase is slower
than the microbenchmark (8B W8A8 one socket at 2048 tokens: 5.37 ms profiled against 3.63 ms in the bench, 1.48x). Per-phase comparison at 2048 tokens (bench x layers / profiled engine, ms per token, 8B W8A8 one
socket): qkv 1.72 / 2.01, wo 1.20 / 1.35, ffn 13.64 / 13.41, classifier 1.05 / 1.03, attention 3.63 / 5.37. The consequence for the gate is conservative
in the right direction: replacing the bench attention by the engine's (x1.48) makes N = 8 worse, not better (section 5.1).

### 3.3 Aggregate tok/s and per-request step time

Selected rows (full tables for every model, precision, kernel policy, socket count, context and N in Appendix A). Format: aggregate tok/s (step ms, which is also the per-request time per token).
Attention `split`, **modeled**.

8B W8A8, one socket, AMX-INT8 GEMM (needs the packed int8 copy the engine already builds for prefill):

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 46.1 (21.7) | 74.6 (26.8) | 110.2 (36.3) | 143.5 (55.8) | 171.1 (93.5) | 189.5 (168.9) |
| 2048 | 44.8 (22.3) | 71.3 (28.0) | 103.3 (38.7) | 131.9 (60.6) | 154.9 (103.3) | 169.8 (188.5) |
| 8192 | 32.6 (30.6) | 44.7 (44.7) | 55.5 (72.1) | 62.8 (127.3) | 67.6 (236.7) | 70.3 (455.2) |
| 32768 | 17.4 (57.4) | 20.4 (98.2) | 22.3 (179.1) | 23.4 (341.3) | 24.1 (664.7) | 24.4 (1311.3) |

8B W8A8, one socket, AVX-512 (VNNI Q8 GEMM or GEMV):

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 44.5 (22.5) | 68.1 (29.4) | 85.9 (46.6) | 97.4 (82.1) | 106.0 (150.9) | 112.1 (285.5) |
| 32768 | 17.4 (57.6) | 20.1 (99.5) | 21.4 (186.9) | 22.1 (362.8) | 22.5 (712.3) | 22.7 (1408.4) |

8B FP32, one socket (the GEMM is weight-bound up to N = 8):

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.2 (61.7) | 28.7 (69.7) | 52.1 (76.8) | 77.6 (103.1) | 108.2 (147.9) | 123.3 (259.6) |
| 2048 | 16.0 (62.3) | 28.2 (70.9) | 50.5 (79.2) | 74.1 (108.0) | 101.5 (157.7) | 114.6 (279.1) |

8B W8A8, two sockets, AMX-INT8 (tensor parallel, exchange modeled):

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 83.8 (11.9) | 123.1 (16.3) | 174.0 (23.0) | 222.2 (36.0) | 256.6 (62.4) | 280.4 (114.1) |

What the tables say. Aggregate throughput saturates (8B W8A8 AMX one socket: 170 tok/s at N = 32, 3.8x N = 1) because attention grows
linearly with N and with C; at 8192 and 32768 tokens the saturation is at 62-70 and 23-24 tok/s, 2.2x and 1.4x of N = 1. The per-request latency grows
immediately. 1B behaves the same way with smaller absolute times; at 2048 tokens 1B FP32 one socket gives 92 -> 310 tok/s at N = 8 (3.4x) with a step of 25.8 ms against 10.8 (2.4x).

### 3.4 The N x attention assumption (conservative on throughput, not on the gate)

The model charges N times the one-request attention. Reasons it can be too high: one team job over `(row, group, chunk)` tasks
fills 56 workers better than one request alone at short contexts and saves the per-call dispatch (section 6.2). Reasons it can be too low: the engine's
attention phase is 1.48x the bench (8B W8A8 2048, one socket), and N caches mean N x the K/V footprint with no cross-request reuse. For the gate the sign does not matter: with
attention **free** (0 ms) the AMX one-socket step at N = 8 is 21.5 ms (GEMM) + 10.1 ms (other, charged x N) = 31.6 ms against the 33.5 ms allowed; to pass with the real GEMM
and `other`, 8 requests' attention would have to take 1.9 ms in total, **0.24 ms per request per token, 15x faster than the measured 3.6 ms**. That is outside anything E5 found
(attention reaches 23-51% of the HBM bandwidth in the engine).

### 3.5 Two sockets: the exchange

At N rows each exchange moves N x dim x 4 bytes (16 KiB at N = 1, 512 KiB at N = 32 on 8B). The model charges 5 us + payload / 50 GB/s per exchange, two per layer: 0.34 ms
per step at N = 1 and 1.0 ms at N = 32 on 8B (**modeled**; at N = 1 B7 measured 0.000 ms of reduction and at most 0.022 ms of wait per token). It is 1-3% of the
modeled steps in the tables and does not change any conclusion; it must still be measured before a plan relies on it (section 9).

## 4. Experiment

### 4.1 Are the weights read once? (gemm-bench, M = 1 to 32, decode shapes)

Sum of the measured median GEMM times of all decode matrices of a step (L x (qkv + wo + w1/w3 + w2) + classifier), relative to M = 1 (**derived** from the **measured** cells).
Eight GEMVs (one per row, the no-sharing baseline) would give 8.0 at M = 8.

8B, one socket (the 1B and two-socket sets are in the job output):

| Kernel | M=1 (ms) | M=2 | M=4 | M=8 | M=16 | M=32 | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AVX-512 FP32 GEMM (best of GEMM, GEMV) | 58.3 | 1.08 | 1.08 | 1.30 | 1.60 | 2.58 | weight-bound to M = 8: **confirmed** |
| AVX-512 VNNI Q8 (best of GEMM, GEMV) | 17.6 | 1.11 | 1.53 | 2.44 | 4.13 | 7.34 | compute-bound from M of about 4: **not confirmed beyond M = 2-4** |
| AMX-INT8 GEMM (best with the VNNI cells) | 17.4 | 1.05 | 1.10 | 1.24 | 1.44 | 1.84 | weight-bound to M = 16: **confirmed** |

Other sets: 1B one socket FP32 1.37, VNNI Q8 2.77 and AMX-INT8 1.40 at M = 8; two sockets (per-socket shards, both running) FP32 1.47, VNNI Q8 2.53, AMX-INT8 1.30 at M = 8 (8B).
Single cells for the largest matrix, 8B w1/w3 (28672 x 4096, one socket, ms): FP32 GEMM 0.84 / 0.98 / 1.19 / 2.24 at M = 1 / 2 / 8 / 32 against GEMV rows 0.91 / 1.82 / 7.77 / 31.1; AMX-INT8
0.284 / 0.285 / 0.327 / 0.504; VNNI Q8 GEMM 0.308 / 0.325 / 0.719 / 1.97; AMX-BF16 (reference, not a decode precision) 0.54 / 0.49 / 0.56 / 0.60.
AMX-BF16 is flat from M = 1 to 32 on the 8B shapes (the BF16 weights are half the FP32 bytes, but a BF16 checkpoint is deferred).
The bench's medians at M = 1 are 1.5-2.5x the best repetition on the small matrices (for example qkv AMX 63 us median, 25 us best): cold-weight runs
vary; the model uses medians throughout.

### 4.2 AMX unit alone (amx-bench, operands in L2, 56 workers, K = 4096)

Per call (best of 5 trials) the time is flat for 1 to 16 rows (BF16 4.52 us, INT8 2.18-2.21 us) and rises by 1.5x at 32 rows (6.8 and 3.5 us): the tile unit works on whole 16-row tiles.
Useful throughput at 16 rows is 52 TFLOP/s (BF16) and 108 TOPS (INT8), against 112 and 225 at the roofline peak. At K = 512, 56 workers: INT8 168 TOPS at 16 rows. So the matrix unit is not the limit for N up to 16; the limit is attention.
(**measured**, `E6/amx-t*-k*.out`.)

### 4.3 KV capacity (N x context x per-token KV bytes)

FP32 K/V, **derived**: 8B 256 KiB per token (32 layers x 2 x 8 KV heads x 128 x 4 B), 1B 64 KiB per token (16 layers x 2 x 8 x 64 x 4 B). GiB for N requests at context C:

| Model | Context | 1 request | N = 8 | N = 16 | N = 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B | 512 | 0.125 | 1 | 2 | 4 |
| 8B | 2048 | 0.5 | 4 | 8 | 16 |
| 8B | 8192 | 2 | 16 | 32 | **64** |
| 8B | 32768 | 8 | **64** | **128** | **256** |
| 1B | 2048 | 0.125 | 1 | 2 | 4 |
| 1B | 8192 | 0.5 | 4 | 8 | 16 |
| 1B | 32768 | 2 | 16 | 32 | 64 |

Against 64 GiB of HBM per socket (measured node size; **estimated** 4 GiB reserve and the checkpoint, W8A8 8B 7.4 GiB, FP32 8B 28.0 GiB, 1B 1.2 / 4.6 GiB):
requests that fit on one socket / on both sockets (KV split by KV head, so both sockets' HBM counts): 8B W8A8 105 / 233 at 2048, 26 / 58 at 8192, 6 / 14 at 32768; 8B FP32 64 / 192, 16 / 48, 4 / 12; 1B W8A8 470 / 982, 117 / 245, 29 / 61.
The bold cells do not fit one socket (8B W8A8 N = 8 at 32768 does not fit even with the whole 64 GiB; 8B FP32 N = 16 at 8192 does not fit next to the weights). Each DDR node of the machine holds about 503 GiB
(measured, `flat-mode.json`), so KV tiering to DDR is possible for capacity but at DDR bandwidth (about half of HBM, [hbm-roofline.md](../../hbm-roofline.md)), which is exactly the part (attention) the model
shows to be the limit. The context at which a target N stops fitting is the practical ceiling of batching on 8B: N = 8 holds to about 27000 tokens on one socket in W8A8 (**derived**).

## 5. Gate evaluation (E6.3)

Gate: the model shows at least 2x aggregate tok/s at N = 8 on 8B W8A8 at a context of 2048 **with** per-request latency at most 1.5x the single-request token time.

### 5.1 Result (modeled, attention from the bench)

| GEMM policy | Sockets | Attention | Step N = 1 (ms) | Step N = 8 (ms) | Aggregate gain | Latency / single | Gain >= 2x | Latency <= 1.5x | Gate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| AVX-512 (VNNI Q8) | 1 | split | 22.5 | 82.1 | 2.19 | 3.65 | yes | no | **FAIL** |
| AVX-512 | 1 | group | | | 2.07 | 3.86 | yes | no | **FAIL** |
| AVX-512 | 2 | split | 11.9 | 46.5 | 2.05 | 3.90 | yes | no | **FAIL** |
| AVX-512 | 2 | group | | | 2.13 | 3.76 | yes | no | **FAIL** |
| AMX-INT8 | 1 | split | 22.3 | 60.6 | 2.94 | 2.72 | yes | no | **FAIL** |
| AMX-INT8 | 1 | group | | | 2.69 | 2.97 | yes | no | **FAIL** |
| AMX-INT8 | 2 | split | 11.9 | 36.0 | 2.65 | 3.02 | yes | no | **FAIL** |
| AMX-INT8 | 2 | group | | | 2.80 | 2.85 | yes | no | **FAIL** |

Sensitivity: attention at the engine's profiled cost instead of the bench's (x1.48 one socket, x0.98 two sockets) gives gain / latency 2.02 / 3.96 (AVX-512, 1), 2.58 / 3.10 (AMX, 1), 2.06 / 3.88 (AVX-512, 2), 2.67 / 3.00 (AMX, 2): the same verdict.
The latency ratio at N = 1 to 8, AMX one socket: 1.00, 1.26, 1.50, 1.74, 1.98, 2.23, 2.47, 2.72 (AVX-512: 1.00, 1.31, 1.69, 2.07, 2.47, 2.86, 3.26, 3.65). Composition at N = 8, AMX one socket: GEMM 21.5 ms,
attention 29.0 ms, other 10.1 ms, against 17.4 / 3.6 / 1.3 ms at N = 1: the step is 2.7x because **attention and per-request work grow 8x while the weight streaming grows 1.24x**.

**Verdict: fail (latency).** Not "unresolved": the margin is a factor of 1.8x to 2.6x on latency in every configuration, and section 3.4 shows that no plausible attention improvement closes it.

### 5.2 Where batching does pay

The weight-bound FP32 8B case: one socket at 2048 tokens, N = 4 gives 3.2x aggregate at 1.27x latency (passes both conditions), N = 8 gives 4.6x at 1.73x. W8A8 passes both conditions only at N = 2 (AVX-512: 1.53x gain, 1.31x latency) and borderline N = 3 (AMX-INT8: 1.50x latency, 2.0x+ gain).
At 512 tokens W8A8 AMX one socket N = 4: 2.4x at 1.67x (modeled); the larger the context the weaker the case, because attention scales with C and weight bytes do not.

### 5.3 Baseline that needs no new code: two one-socket processes

N requests split over two independent one-socket processes (each holds the whole checkpoint in its own HBM; no exchange). Modeled with the one-socket tables at N / 2 requests per socket, 8B W8A8, 2048 tokens:

| N total | AMX-INT8: aggregate tok/s (latency ms) | AVX-512: aggregate tok/s (latency ms) | Batched two-socket TP, same N (AMX / AVX-512) |
| ---: | --- | --- | --- |
| 2 | 89.7 (22.3) | 88.9 (22.5) | 123.1 (16.3) / 121.4 (16.5) |
| 4 | 142.6 (28.0) | 136.2 (29.4) | 174.0 (23.0) / 155.1 (25.8) |
| 8 | 206.5 (38.7) | 171.8 (46.6) | 222.2 (36.0) / 172.0 (46.5) |
| 16 | 263.9 (60.6) | 194.9 (82.1) | 256.6 (62.4) / 185.6 (86.2) |
| 32 | 309.8 (103.3) | 212.1 (150.9) | 280.4 (114.1) / 194.9 (164.2) |

Batched tensor parallel wins at N of 4 or fewer (123 against 90 tok/s at N = 2 with AMX, and lower latency), **ties at N = 8** (222 against 207 tok/s with AMX; 172 against 172 with AVX-512) and **loses from N = 16** (257 against 264; 281 against 310 at N = 32 with AMX). Two processes need no scheduler, no row map, no exchange and no new kernels; the question
D-E3 should ask is therefore not "N = 8 against N = 1" but "what does batching add over two sockets of independent single-request decode", and the model says: at most the weight-sharing within a socket (the 2.2x to 2.9x of 5.1) and only at the price of latency.

## 6. Design

The design is written so that a Track F plan can start from it if the owner decides "go". Nothing here is implemented.

### 6.1 What the engine has today

- One `Transformer` owns one KV cache laid out `[layer][position][kv_dim]` (`key_cache + l * seq_len * kv_dim`), one
  `RunState` and one position. `engine_forward(token, pos)` is the single-row decode; `engine_forward_batch(tokens, count,
  pos0, want)` runs `count` **consecutive positions of one sequence** through the GEMMs (`forward_layer_batch`), writes the
  K and V rows straight into the cache through the QKV segments' `c` pointer (`kc + pos0 * kv_dim`), and calls
  `k_attention_batch(..., pos0, count)`, whose row `i` attends to positions `0..pos0+i` of that one cache.
- `tp_forward` / `tp_forward_batch` do the same on two sockets: socket-local heads and KV, two exchanges per layer
  (`exchange_reduce_add`), exchange buffers sized `rows x dim`.
- Sampling is per process (`Sampler`, one RNG state), the CLI runs one prompt (`--prompt`, `--prompt-file`), and
  `check_budget` (two sockets) / the loader budget (one socket) count one KV cache per process.

### 6.2 The row map

Batched decode needs the one thing the batched prefill path does not have: **row i of a step belongs to request `slot[i]` at
position `pos[i]`**, and different rows have different caches. Everything else (RMSNorm, the GEMMs, SwiGLU, residuals, the
classifier) is already row-parallel. A step takes `n` rows described by `{token[i], slot[i], pos[i]}` and has these row-dependent
parts:

| Stage | Today | Batched decode |
| --- | --- | --- |
| Embedding | one row | gather `n` rows |
| RoPE table | `k_rope_table(pos)` once, rows at consecutive positions in the batch path | fill table row `i` from `pos[i]` (a 2 x head_dim row per request) |
| QKV GEMM output | `q` rows contiguous; K, V rows written into `cache[pos0 + i]` through `GemmSegment.c, ldc` | K, V into an `n x kv_dim` staging buffer; one `memcpy` of `kv_dim` floats per row and layer into `cache[slot[i]][layer][pos[i]]` (about 64 KB of copies per layer at n = 32 on 8B, **estimated** under 5 us) |
| Attention | `k_attention_batch(pos0, m)`: one cache | new `k_attention_rows(out, q, kv_of_slot, slot[], pos[], n)`, below |
| Wo, W1/W3, W2, classifier | GEMM at M = n | unchanged |
| Sampling | one `Sampler` | `n` samplers (section 6.5) |

**`k_attention_rows`.** One team job whose tasks are `(row, KV group, chunk)`; each task streams K/V of one row's cache, and the
per-row `merge_chunks` is the existing one. Each row uses exactly the chunk plan `k_attention` uses at its own position
(`k_attention_chunk_count(work, kv_heads, pos[i])`), so the result is **bitwise equal to the single-request `k_attention`** at
that position by construction, as `k_attention_batch` is. Rows do not share K/V (different caches), so none of the row-block
sharing of prefill applies: the cost is the sum of the per-request costs, which is what the model charges (N x one request). Two
reasons to write it as one job instead of calling `k_attention` N times: a call dispatches a team job and waits at a barrier (a
few microseconds per layer and request, at 2048 tokens about 5-10% of the one-request layer time, **estimated** from E5's
0.05-0.1 ms per layer), and at short contexts a single request produces too few chunks to keep 56 workers busy while N requests
together do. That second effect is why the model's N x attention is an upper bound at 512-2048 (section 3.4). The workspace is the
existing `k_attention_init_rows` with `rows = N`.

### 6.3 Request scheduler

State per **slot** (a fixed pool of `S` slots, `S` = the largest batch the build is sized for, 32 in the model): request id, state
(`free`, `prefill`, `decode`, `done`), position, token history, `Sampler`, per-slot stop condition and token budget, timestamps.

- **KV regions.** One region per slot, `[layer][seq_len_slot][kv_dim]`, so the existing per-sequence kernels address a slot by
  `key_cache_of(slot) + l * seq_len_slot * kv_dim` unchanged. Slot capacity is set at admission from the request's `max_context`
  (prompt + generation limit), not from a global `--context-length`. Regions are bound mappings whose untouched pages cost nothing
  (the A5 reports show `pages_missing`), so the budget check must count what is *admitted* (sum of `max_context x
  kv_bytes_per_token`), not the virtual reservation.
- **Admission.** A request is admitted when a slot is free and `weights + sum(admitted max_context) x kv_bytes_per_token +
  work(n) + exchange(n) + packed copies` stays within the socket budget (`check_budget` generalised from `2 * kv_bytes` to the
  sum over slots; on two sockets each socket holds half the KV heads of every request). Otherwise it waits in a FIFO queue; a
  request that can never fit is rejected with the number it needs (section 7).
- **Static against continuous batching.** *Static*: take up to `n` queued requests, prefill each one (the existing batched
  prefill, one request at a time), then decode in lockstep until every request is done; finished requests leave idle rows (the
  step still costs `step(n)` until the last one finishes). *Continuous*: the row set is rebuilt every step from the slots in
  `decode` state; a finished slot is released at once and the next queued request is prefilled and joins. Because every kernel is
  driven by the row map, continuous batching costs nothing extra in the kernels; the issue is **prefill interference**: a 2048-token
  prefill of 8B W8A8 takes about 3.3 s on one socket (C7, measured: 647 tok/s), during which no decode step runs. Recommended
  order: (1) static, to prove the kernels and the equality tests; (2) continuous with the prefill cut into chunks of 64-256 tokens
  (the existing microbatch) and **one chunk per decode step at most**, which bounds the extra latency of any step by one chunk
  (about 0.1-0.4 s at 8B, so a token budget per step, not a fixed chunk, is the real knob); (3) mixing prefill chunk rows and decode
  rows in the same GEMM (the GEMM is indifferent to what a row is, attention is not: chunk rows use `k_attention_batch` on their slot,
  decode rows `k_attention_rows`).
- **Batch size policy.** The scheduler runs `n = min(active, N_max)` rows, where `N_max` is the largest `N` whose modeled step stays
  within the latency target (section 5.1: N = 2 or 3 at W8A8, 2048 tokens for the 1.5x target); it is a run-time knob, `LLAMA_BATCH_MAX`, defaulting to 1 so that nothing changes silently.

### 6.4 Two sockets

- **Tensor parallel, as today.** Each socket runs all `n` rows through its shard and its own KV heads, with the exchange payload
  growing from `dim x 4` to `n x dim x 4` bytes per exchange (16 KiB at n = 1, 512 KiB at n = 32 on 8B). `tp_forward_batch` already
  sizes the exchange buffers for `rows x dim` and `tp_create` takes `max_batch`; the row map goes to both coordinators. The model
  charges the exchange at a fixed 5 us plus the payload at 50 GB/s per exchange (**modeled**, section 3.5); at n = 32 that is about 1.0 ms
  of an 8B two-socket step of 114-164 ms (sections 3.3, 3.5). It must be **measured** in the Track F plan's first phase (an `--rows` option of `reduce-bench`).
- **Data parallel as the baseline to beat.** Two independent one-socket processes, each with the whole checkpoint in its own HBM
  and half of the requests, need **no exchange and no new code** (two launches, `numactl` per socket). Their aggregate at `N / 2`
  requests per socket is computed in section 5.3 from the same one-socket model; it is the honest comparator for "two sockets,
  batched" and it is the better choice when per-request latency matters less than aggregate throughput.

### 6.5 Sampling and the surface

- **Sampling per request.** One `Sampler` per slot (temperature, top-p, seed, RNG state), so a request's token stream does not depend
  on its neighbours. Greedy decode (`-t 0`, what every campaign used) is an argmax over 128256 logits per row; the model assumes it
  and charges it inside `other`. Stochastic sampling with top-p sorts a candidate set per row, which is serial work per request on
  the leader thread; it is **not measured here**, and at n = 32 it must be spread over the team (one row per worker) if it exceeds
  about 5% of the step. The logits buffer is `n x vocab` floats (16 MB at n = 32), the existing `LOGITS_ALL` path.
- **Surface.** A library API first: `batch_open(model, slots)`, `batch_submit(prompt_tokens, sampling, max_new_tokens) -> id`,
  `batch_step()` (advances every active slot by one token and returns the emitted `(id, token)` pairs and the finished ids),
  `batch_close()`. The CLI driver is then thin: `--batch-prompts FILE` (one prompt per line or JSONL), `--max-batch N`,
  `--batch-mode static|continuous`, results as JSONL with per-request token ids, time to first token and per-token times, and
  `--benchmark` printing aggregate tok/s and per-request latency percentiles. An HTTP server is a later consumer of the same API
  and is **out of scope** of the first plan.

### 6.6 Tests (the Track F plan must have these before any timing)

1. **Attention rows, bitwise.** `k_attention_rows` against `k_attention` called per row, random per-row positions (including
   position 0, chunk-boundary positions and the largest context), GQA shapes 8/2/64, 32/8/64, 32/8/128, on the portable path and
   the AVX-512 emulation target; `LLAMA_TEAM` sizes 1, 3, 7.
2. **Row-map kernels.** RoPE rows from per-row positions equal `k_rope_apply` per row; the KV scatter writes `cache[slot][layer][pos]`
   and nothing else (a canary-filled cache).
3. **Per-request equality, FP32.** `N` requests (different prompts, different lengths, different positions) decoded as a batch
   against each decoded alone. Two oracles, because the FP32 GEMM and the GEMV sum in different orders on SIMD hosts
   (`include/gemm.h`): *scalar ISA*: **bitwise** at every position (the same property C1 and C2 test for batched prefill);
   *AVX-512*: bitwise against the single-request run **through the same GEMM** (decode with M = 1 through `k_gemm_f32`, a mode the
   plan must add for the oracle), because a GEMM row does not depend on M (D1 recorded microbatch 64 against 16 bitwise identical),
   and within the repository's FP32 gate (max |dlogit| <= 1e-3, argmax 100%) against the ordinary GEMV decode.
4. **Continuous batching.** Requests joining and leaving at arbitrary steps produce the same tokens as running them alone; slot reuse
   leaves no state behind (a finished slot's KV is not read by the next request).
5. **W8A8 and two sockets.** The statistical W8A8 gate of the campaigns against the single-request run (`W8A8_GATE`), and two
   sockets against one with `LLAMA_TP_CHECK=1` (identical residuals after every layer) at N = 1, 2, 7, 32.
6. **Admission.** Budget arithmetic for 1B and 8B (section 4.3) as a unit test, rejection message, FIFO order.
7. **Throughput guard.** A bench cell that fails when the measured step at N = 8 on 8B W8A8 at 2048 tokens exceeds the model by more
   than 20% (the model's own N = 1 residual, section 3.2, sets the tolerance).

## 7. For the owner: decision D-E3 (prepared, not taken)

Question: go or no-go on a Track F plan for batched decode.

- **The gate as written fails** (section 5.1): the model shows 2.05-2.94x aggregate at N = 8 on 8B W8A8 at 2048 tokens, but the per-request step is 2.7-3.9x the single-request token, against a limit of 1.5x. Cause: attention is per request and grows
  linearly with N, while the shared weight read (the only thing batching shares) is 17-21 ms of the 22 ms step at N = 1. The plan's own text records the same inference at the end of E5 (decode attention is access-pattern-bound and
  not shared between requests).
- **What the numbers support if the latency condition is relaxed.** Throughput-oriented serving where per-request latency of 2-4x is acceptable: W8A8 with AMX-INT8 gains up to 2.9x at N = 8 on one socket at 2048 tokens (modeled), 1.4x-2.2x at 32768 / 8192 tokens, so the benefit disappears
  as the context grows. Against the no-code baseline of two independent one-socket processes the extra gain of batched two-socket decode is zero at N = 8 and negative from N = 16 (section 5.3).
- **Where batching clearly pays: weight-bound precisions.** FP32 8B passes both conditions at N = 4 (3.2x, 1.27x); a W8A8 or BF16 checkpoint does not leave enough weight time to share. A future BF16 KV or a faster attention kernel (E5's follow-ups) would move the balance, but not by the
  15x the gate would need (section 3.4).
- **Suggested reading for the decision.** No-go on a general batched-decode implementation for W8A8 (the engine's main decode precision) at the 1.5x latency target; go only if the owner wants a throughput mode with a relaxed target (then the first step is
  the cheap one: run two one-socket processes, measure, and use the section 6 design only if more than the 2x of two processes is needed). A smaller option with a real model-backed gain is static batching of two to four requests per socket for FP32 checkpoints.
- **Decision-relevant measurements missing** (section 9): exchange cost at N > 1, attention over N caches in one team job, the engine's own batched decode.

## 8. Provenance

| Item | Value |
| --- | --- |
| Date | 19 November 2026 |
| Node | one Flat-mode node of the HBM partition (DDR nodes 0-1, HBM nodes 2-3, 56 cores per socket), the other node of the partition from the one that measured B7 and E5; no hostnames recorded here |
| Cluster job | 5875687, COMPLETED, exit 0, 55 minutes; `STAGE=bench PHASE=E6 SKIP_TEST=0 HF_GATE=0`; results directory `e6exp-20261119-1` (29 recorded results, all PASS; no engine run failed) |
| `make test` on the hardware | PASS in the same job (`test.log` ends `Ran 73 tests ... OK (skipped=1)`) |
| Commit under test | 6a8586e (`feature/decode-e6-design`); the analysis changes of this document's commit (`tools/e6_model.py` attention key, test fixture) are not in the job tree and do not affect any measurement |
| Script | `scripts/cresco8-e6.sbatch`, phase `E6`; local dry run `DRY_RUN=1` and `tests/test_tools.py::E6Tests` |
| Benchmarks | `build/release/gemm-bench`, `amx-bench`, `attention-bench` (AVX-512 build with AMX; GCC of the cluster) |
| Reproduce the tables | `python3 tools/e6_model.py model <OUT>/E6 --runs <OUT>` |

**Node cross-checks** (the other node measured B7 and E5; ratios this node / that node):

| Cell | This job | Earlier (other node) | Ratio |
| --- | ---: | ---: | ---: |
| attention-bench decode, 8B, 8 KV heads, 8192, `split`, median ms per layer | 0.3738 | 0.372 (E5) | **1.005** |
| engine 8B FP32, one socket, 512 tokens, tok/s (from median ms) | 16.41 | 16.42 (B7) | 1.000 |
| engine 8B W8A8, one socket, 512 | 49.34 | 49.52 | 0.996 |
| engine 8B FP32, two sockets, 512 | 30.52 | 30.63 | 0.996 |
| engine 8B W8A8, two sockets, 512 | 90.63 | 91.25 | 0.993 |
| engine 1B FP32, one / two sockets, 512 | 92.20 / 175.69 | 91.33 / 168.85 | 1.010 / **1.040** |
| engine 1B W8A8, one / two sockets, 512 | 251.27 / 463.03 | 269.09 / 466.84 | **0.934** / 0.992 |

All 8B cells agree within 1%. **Two 1B cells are off by more than 4%**: 1B W8A8 one socket at 512 (0.934, below B7's range of 259.5-284.5 tok/s; the token is 4.0 ms) and 1B FP32 two sockets
at 512 (1.040; B7's own range for this cell is 132.0-169.7, with one outlier). The 1B W8A8 one-socket cell is flagged as a **possible node difference or short-token sensitivity** (the E5 job on the other node
measured 270-279 tok/s for the same cell); it does not enter the 8B gate. No gemm-bench cell of C2/C6 was re-run (no raw bench file of the other node was available to this job), so the GEMM inputs are checked only through the engine's 8B N = 1 phases (section 3.2: ffn 13.64 / 13.41 ms).

## 9. Gaps

1. **No batched engine exists.** Everything past N = 1 is modeled from kernel benchmarks. In particular nothing measures the engine's attention over N separate caches in one job (the model charges N x one request), the row-map overheads (KV scatter, per-row RoPE table), or sampling of N rows.
2. **Exchange at N > 1 is modeled**, not measured (section 3.5); `reduce-bench` has no row-count option.
3. **The model is optimistic at long context** (up to +24% faster than B7 at 32768, section 3.2) because the microbenchmark attention is faster than the engine's phase (1.48x at 2048 one socket); the 8192 and 32768 rows of the tables should be read as upper bounds on throughput.
4. **`other` is charged linearly in N** (1.26 ms per request, 8B W8A8 one socket); RMSNorm, quantization and RoPE rows are parallel and probably scale better. It does not change the gate (section 3.4).
5. **Same-context assumption.** Mixed contexts (continuous batching) are not modeled; attention would be the sum over requests.
6. **AMX decode is modeled from the GEMM bench**: no engine run uses the AMX GEMM for decode, and AMX-INT8 for N of 1 is not what the engine does today (the GEMV). Its N = 1 residual against the GEMV engine is therefore a model-against-different-kernel check (-5.3% at the gate cell).
7. **Two-socket GEMM sets** time each socket's shards separately and take the slower; the engine's team dispatch, barriers and socket imbalance are in `other` and the per-phase residual (8B W8A8 two sockets wo 0.65 modeled / 1.13 profiled ms).
8. **Context 2048 for engine runs only**; 8192 and 32768 are checked only against B7 on the other node.
9. **Sampling and top-p** cost is not measured (greedy only).
10. **KV capacity reserve** (4 GiB) is an estimate; `check_budget` for N slots is designed here but not written.
11. **BF16 and BF16-KV** variants are out of scope (AMX-BF16 GEMM times are recorded in the job output for reference only).
12. **Two 1B cross-check cells** differ from B7 by more than 4% (section 8); no rerun was made.
13. **Gate reading.** The gate's two conditions together make latency the binding one (1.5x latency at N = 8 requires at least a 5.3x aggregate gain). I evaluated both as written; "per-request latency" is the step time, the time between two tokens of one request.

## Appendix A: generated model tables

Output of `python3 tools/e6_model.py model <OUT>/E6 --runs <OUT>` (all labelled modeled, except the N = 1 reproduction and per-phase tables, which compare modeled with measured values).

### Aggregate tok/s (and step time in ms) by N, attention `split`

**8b fp32 (avx512), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16.2 (61.7) | 28.7 (69.7) | 52.1 (76.8) | 77.6 (103.1) | 108.2 (147.9) | 123.3 (259.6) |
| 2048 | 16.0 (62.3) | 28.2 (70.9) | 50.5 (79.2) | 74.1 (108.0) | 101.5 (157.7) | 114.6 (279.1) |
| 8192 | 14.2 (70.7) | 22.8 (87.6) | 35.5 (112.6) | 45.8 (174.6) | 55.0 (291.1) | 58.6 (545.9) |
| 32768 | 10.3 (97.4) | 14.2 (141.1) | 18.2 (219.6) | 20.6 (388.7) | 22.2 (719.1) | 22.8 (1401.9) |

**8b fp32 (avx512), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 33.7 (29.7) | 49.5 (40.4) | 95.3 (42.0) | 144.9 (55.2) | 202.1 (79.2) | 245.8 (130.2) |
| 2048 | 32.6 (30.7) | 47.2 (42.4) | 87.1 (45.9) | 126.7 (63.1) | 168.5 (95.0) | 197.7 (161.8) |
| 8192 | 27.2 (36.8) | 36.7 (54.6) | 56.9 (70.3) | 71.6 (111.8) | 83.2 (192.3) | 89.8 (356.5) |
| 32768 | 21.6 (46.3) | 27.1 (73.7) | 36.8 (108.6) | 42.5 (188.4) | 46.3 (345.4) | 48.3 (662.8) |

**8b int8 (avx512), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 45.7 (21.9) | 71.1 (28.1) | 90.7 (44.1) | 103.6 (77.2) | 113.4 (141.1) | 120.3 (266.0) |
| 2048 | 44.5 (22.5) | 68.1 (29.4) | 85.9 (46.6) | 97.4 (82.1) | 106.0 (150.9) | 112.1 (285.5) |
| 8192 | 32.4 (30.8) | 43.4 (46.0) | 50.1 (79.9) | 53.8 (148.8) | 56.3 (284.3) | 57.9 (552.3) |
| 32768 | 17.4 (57.6) | 20.1 (99.5) | 21.4 (186.9) | 22.1 (362.8) | 22.5 (712.3) | 22.7 (1408.4) |

**8b int8 (avx512), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 91.4 (10.9) | 137.9 (14.5) | 183.2 (21.8) | 207.2 (38.6) | 227.3 (70.4) | 241.4 (132.6) |
| 2048 | 83.8 (11.9) | 121.4 (16.5) | 155.1 (25.8) | 172.0 (46.5) | 185.6 (86.2) | 194.9 (164.2) |
| 8192 | 55.5 (18.0) | 69.8 (28.6) | 79.8 (50.1) | 84.0 (95.2) | 87.2 (183.5) | 89.2 (358.9) |
| 32768 | 36.3 (27.6) | 41.9 (47.8) | 45.2 (88.4) | 46.6 (171.8) | 47.5 (336.7) | 48.1 (665.1) |

**8b int8 (amx), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 46.1 (21.7) | 74.6 (26.8) | 110.2 (36.3) | 143.5 (55.8) | 171.1 (93.5) | 189.5 (168.9) |
| 2048 | 44.8 (22.3) | 71.3 (28.0) | 103.3 (38.7) | 131.9 (60.6) | 154.9 (103.3) | 169.8 (188.5) |
| 8192 | 32.6 (30.6) | 44.7 (44.7) | 55.5 (72.1) | 62.8 (127.3) | 67.6 (236.7) | 70.3 (455.2) |
| 32768 | 17.4 (57.4) | 20.4 (98.2) | 22.3 (179.1) | 23.4 (341.3) | 24.1 (664.7) | 24.4 (1311.3) |

**8b int8 (amx), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 91.4 (10.9) | 140.1 (14.3) | 210.1 (19.0) | 284.8 (28.1) | 343.8 (46.5) | 387.9 (82.5) |
| 2048 | 83.8 (11.9) | 123.1 (16.3) | 174.0 (23.0) | 222.2 (36.0) | 256.6 (62.4) | 280.4 (114.1) |
| 8192 | 55.5 (18.0) | 70.4 (28.4) | 84.5 (47.3) | 94.5 (84.7) | 100.2 (159.7) | 103.6 (308.8) |
| 32768 | 36.3 (27.6) | 42.1 (47.6) | 46.7 (85.6) | 49.6 (161.2) | 51.1 (312.8) | 52.0 (615.1) |

**1b fp32 (avx512), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 99.2 (10.1) | 152.0 (13.2) | 269.0 (14.9) | 405.4 (19.7) | 515.7 (31.0) | 589.9 (54.2) |
| 2048 | 92.3 (10.8) | 136.3 (14.7) | 223.4 (17.9) | 310.1 (25.8) | 370.7 (43.2) | 407.6 (78.5) |
| 8192 | 69.4 (14.4) | 91.6 (21.8) | 124.2 (32.2) | 147.0 (54.4) | 159.4 (100.4) | 165.8 (193.0) |
| 32768 | 48.1 (20.8) | 57.9 (34.6) | 69.4 (57.7) | 76.0 (105.3) | 79.1 (202.2) | 80.7 (396.6) |

**1b fp32 (avx512), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 178.1 (5.6) | 258.0 (7.8) | 449.6 (8.9) | 662.5 (12.1) | 857.6 (18.7) | 971.4 (32.9) |
| 2048 | 166.9 (6.0) | 235.1 (8.5) | 384.4 (10.4) | 529.9 (15.1) | 647.8 (24.7) | 710.7 (45.0) |
| 8192 | 124.4 (8.0) | 158.7 (12.6) | 215.1 (18.6) | 254.1 (31.5) | 278.4 (57.5) | 289.4 (110.6) |
| 32768 | 67.0 (14.9) | 75.8 (26.4) | 86.7 (46.2) | 92.4 (86.6) | 95.4 (167.7) | 96.7 (331.0) |

**1b int8 (avx512), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 259.6 (3.9) | 337.9 (5.9) | 432.7 (9.2) | 492.4 (16.2) | 534.4 (29.9) | 552.3 (57.9) |
| 2048 | 216.9 (4.6) | 269.0 (7.4) | 325.8 (12.3) | 358.5 (22.3) | 380.3 (42.1) | 389.3 (82.2) |
| 8192 | 122.1 (8.2) | 137.1 (14.6) | 150.4 (26.6) | 157.1 (50.9) | 161.1 (99.3) | 162.7 (196.7) |
| 32768 | 68.7 (14.5) | 73.2 (27.3) | 76.9 (52.0) | 78.6 (101.8) | 79.6 (201.1) | 79.9 (400.3) |

**1b int8 (avx512), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 459.2 (2.2) | 622.8 (3.2) | 758.7 (5.3) | 903.2 (8.9) | 993.6 (16.1) | 1035.7 (30.9) |
| 2048 | 391.4 (2.6) | 504.2 (4.0) | 589.8 (6.8) | 673.5 (11.9) | 722.5 (22.1) | 744.5 (43.0) |
| 8192 | 217.2 (4.6) | 248.1 (8.1) | 267.1 (15.0) | 283.1 (28.3) | 291.4 (54.9) | 294.9 (108.5) |
| 32768 | 87.0 (11.5) | 91.6 (21.8) | 94.1 (42.5) | 96.0 (83.4) | 96.9 (165.1) | 97.3 (328.9) |

**1b int8 (amx), 1 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 259.6 (3.9) | 360.3 (5.6) | 505.8 (7.9) | 642.1 (12.5) | 736.7 (21.7) | 802.3 (39.9) |
| 2048 | 216.9 (4.6) | 283.0 (7.1) | 365.6 (10.9) | 431.8 (18.5) | 472.6 (33.9) | 498.8 (64.2) |
| 8192 | 122.1 (8.2) | 140.6 (14.2) | 158.4 (25.3) | 169.7 (47.1) | 175.6 (91.1) | 179.1 (178.6) |
| 32768 | 68.7 (14.5) | 74.2 (26.9) | 78.9 (50.7) | 81.6 (98.0) | 83.0 (192.9) | 83.7 (382.2) |

**1b int8 (amx), 2 socket(s)**

| Context | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 459.2 (2.2) | 663.1 (3.0) | 936.6 (4.3) | 1150.1 (7.0) | 1323.5 (12.1) | 1436.2 (22.3) |
| 2048 | 391.4 (2.6) | 530.3 (3.8) | 691.9 (5.8) | 801.8 (10.0) | 882.5 (18.1) | 931.2 (34.4) |
| 8192 | 217.2 (4.6) | 254.2 (7.9) | 286.3 (14.0) | 303.5 (26.4) | 314.3 (50.9) | 320.3 (99.9) |
| 32768 | 87.0 (11.5) | 92.4 (21.6) | 96.3 (41.5) | 98.2 (81.5) | 99.3 (161.1) | 99.9 (320.3) |

### N = 1 reproduction (model against the engine run of this job)

| Model | Precision | Policy | Sockets | Context | Measured ms [range] | Modeled ms | Residual |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| 8b | fp32 | avx512 | 1 | 512 | 60.93 [60.89-62.33] | 61.71 | +1.3% |
| 8b | fp32 | avx512 | 1 | 2048 | 64.39 [64.27-65.62] | 62.32 | -3.2% |
| 8b | fp32 | avx512 | 2 | 512 | 32.76 [32.75-32.80] | 29.36 | -10.4% |
| 8b | fp32 | avx512 | 2 | 2048 | 32.65 [32.65-32.83] | 30.35 | -7.1% |
| 8b | int8 | avx512 | 1 | 512 | 20.27 [20.26-20.31] | 21.88 | +7.9% |
| 8b | int8 | avx512 | 1 | 2048 | 23.55 [22.96-25.29] | 22.49 | -4.5% |
| 8b | int8 | avx512 | 2 | 512 | 11.03 [11.02-11.06] | 10.60 | -4.0% |
| 8b | int8 | avx512 | 2 | 2048 | 12.03 [12.02-12.08] | 11.59 | -3.7% |
| 8b | int8 | amx | 1 | 512 | 20.27 [20.26-20.31] | 21.69 | +7.0% |
| 8b | int8 | amx | 1 | 2048 | 23.55 [22.96-25.29] | 22.30 | -5.3% |
| 8b | int8 | amx | 2 | 512 | 11.03 [11.02-11.06] | 10.60 | -4.0% |
| 8b | int8 | amx | 2 | 2048 | 12.03 [12.02-12.08] | 11.59 | -3.7% |
| 1b | fp32 | avx512 | 1 | 512 | 10.85 [10.54-10.89] | 10.08 | -7.1% |
| 1b | fp32 | avx512 | 1 | 2048 | 11.31 [11.28-11.32] | 10.84 | -4.2% |
| 1b | fp32 | avx512 | 2 | 512 | 5.69 [5.57-5.71] | 5.45 | -4.3% |
| 1b | fp32 | avx512 | 2 | 2048 | 6.16 [6.13-6.18] | 5.83 | -5.4% |
| 1b | int8 | avx512 | 1 | 512 | 3.98 [3.91-4.02] | 3.85 | -3.2% |
| 1b | int8 | avx512 | 1 | 2048 | 4.49 [4.38-4.50] | 4.61 | +2.8% |
| 1b | int8 | avx512 | 2 | 512 | 2.16 [2.13-2.52] | 2.01 | -6.8% |
| 1b | int8 | avx512 | 2 | 2048 | 2.54 [2.38-2.56] | 2.39 | -6.0% |
| 1b | int8 | amx | 1 | 512 | 3.98 [3.91-4.02] | 3.85 | -3.2% |
| 1b | int8 | amx | 1 | 2048 | 4.49 [4.38-4.50] | 4.61 | +2.8% |
| 1b | int8 | amx | 2 | 512 | 2.16 [2.13-2.52] | 2.01 | -6.8% |
| 1b | int8 | amx | 2 | 2048 | 2.54 [2.38-2.56] | 2.39 | -6.0% |

### N = 1 per phase at 2048 tokens (bench x layers / profiled engine run, ms per token)

| Model | Precision | Policy | Sockets | qkv | wo | ffn | classifier | attention (split) |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| 8b | fp32 | avx512 | 1 | 6.60 / 6.44 | 4.34 / 4.22 | 43.46 / 44.48 | 3.90 / 3.57 | 3.63 / 4.91 |
| 8b | fp32 | avx512 | 2 | 3.07 / 3.27 | 1.99 / 2.65 | 20.63 / 22.21 | 1.91 / 1.92 | 2.44 / 2.28 |
| 8b | int8 | avx512 | 1 | 1.72 / 2.01 | 1.20 / 1.35 | 13.64 / 13.41 | 1.05 / 1.03 | 3.63 / 5.37 |
| 8b | int8 | avx512 | 2 | 0.78 / 1.04 | 0.65 / 1.13 | 6.58 / 6.38 | 0.53 / 0.52 | 2.44 / 2.39 |
| 8b | int8 | amx | 1 | 1.72 / 2.01 | 1.20 / 1.35 | 13.45 / 13.41 | 1.05 / 1.03 | 3.63 / 5.37 |
| 8b | int8 | amx | 2 | 0.78 / 1.04 | 0.65 / 1.13 | 6.58 / 6.38 | 0.53 / 0.52 | 2.44 / 2.39 |
| 1b | fp32 | avx512 | 1 | 0.81 / 0.90 | 0.49 / 0.61 | 6.11 / 6.38 | 1.77 / 2.11 | 1.54 / 1.20 |
| 1b | fp32 | avx512 | 2 | 0.39 / 0.44 | 0.25 / 0.46 | 3.25 / 3.30 | 0.94 / 0.97 | 0.96 / 0.61 |
| 1b | int8 | avx512 | 1 | 0.23 / 0.43 | 0.22 / 0.32 | 1.85 / 1.87 | 0.48 / 0.50 | 1.54 / 1.18 |
| 1b | int8 | avx512 | 2 | 0.16 / 0.23 | 0.11 / 0.39 | 0.89 / 1.08 | 0.26 / 0.24 | 0.96 / 0.64 |
| 1b | int8 | amx | 1 | 0.23 / 0.43 | 0.22 / 0.32 | 1.85 / 1.87 | 0.48 / 0.50 | 1.54 / 1.18 |
| 1b | int8 | amx | 2 | 0.16 / 0.23 | 0.11 / 0.39 | 0.89 / 1.08 | 0.26 / 0.24 | 0.96 / 0.64 |

### E6.3 gate: 8B W8A8, context 2048, N = 8

| Policy | Sockets | Attention | Aggregate gain over N = 1 | Latency / single | Throughput >= 2x | Latency <= 1.5x | Gate |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| avx512 | 1 | split | 2.19 | 3.65 | yes | no | FAIL |
| avx512 | 1 | group | 2.07 | 3.86 | yes | no | FAIL |
| avx512 | 2 | split | 2.05 | 3.90 | yes | no | FAIL |
| avx512 | 2 | group | 2.13 | 3.76 | yes | no | FAIL |
| amx | 1 | split | 2.94 | 2.72 | yes | no | FAIL |
| amx | 1 | group | 2.69 | 2.97 | yes | no | FAIL |
| amx | 2 | split | 2.65 | 3.02 | yes | no | FAIL |
| amx | 2 | group | 2.80 | 2.85 | yes | no | FAIL |
