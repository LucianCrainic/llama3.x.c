# Follow-up work after M5: brief for a specialist agent

Written 18 November 2026, after milestone M5 closed (D2, jobs 5845031 and 5845032, commit 8f3114f). This is the starting
document for an agent that takes the follow-ups listed in [`docs/numa-tp-prefill-results.md`](../../numa-tp-prefill-results.md)
and [`docs/batched-prefill.md`](../../batched-prefill.md). **Priority 1 is a faster attention kernel.** The other items are
ordered below it. Read the sections you need; every number cites the document it comes from, and each is labelled
**measured**, **modeled** or **estimated** as in [`docs/hbm-roofline.md`](../../hbm-roofline.md).

## 0. Working rules

- **Repository.** C engine for Llama 3 on Sapphire Rapids HBM (AVX-512, AMX, two sockets, NUMA-owned weights). Work from
  `develop`. One branch per item, `feature/<slug>`, conventional commits (`feat:`, `fix:`, `perf:`, `test:`, `docs:`),
  merge into `develop` when its gates pass, push `develop`. Do not rewrite history.
- **Never put hosts, gateways, credentials, absolute cluster paths or personal details in the repository.** Model paths come from
  `MODELS="NAME:ARTIFACTS:HF ..."` at submission time; the campaign layout and SSH access are in the owner's private
  configuration, not here (ask the parent session if you do not have them).
- **Cluster.** The owner's standing instruction (16 November 2026): submit, monitor and analyse jobs for this project
  without asking per job, and come back only for an important finding or a change of plan. The node (one Flat-mode HBM
  node) is exclusive, so jobs run one at a time. Build and test inside jobs, never on the login node. Use `ssh -O check`
  before using the cluster; if the master is down, stop and tell the owner (never automate the password).
- **Local first.** A cluster job exercises code that already passed locally. Everything below has a local test path:
  `make test` (macOS and Linux), `MODE=tsan`, and the emulated AMX and AVX-512 headers (`tests/amx_emul.h`,
  `tests/avx512_emul.h`, the `probe-*-amx-emul` targets). Only AVX-512 and real-AMX speed and numerics need the hardware.
- **Gates.** FP32 paths: max |Δlogit| <= 1e-3 and argmax 100% against the one-socket per-token reference
  (`tools/compare_engines.py`). BF16: max 2.0, rms 0.05, argmax >= 0.97 (`BF16_GATE` in `scripts/lib/cresco8-common.sh`,
  fixed from the K1 data). W8A8: statistical gate (rms <= 0.2, argmax >= 0.9, max <= 16; `W8A8_GATE`). Do not loosen a gate
  without data and a written reason in the phase document.
- **Records.** Each campaign gets a results document and a `-validation` companion (date, node, compiler, job id, commit,
  campaign directory), following [`docs/batched-prefill.md`](../../batched-prefill.md) and its validation record. Tick an
  exit criterion only with the observed value written next to it.
- **Defaults never change silently.** New kernels are opt-in knobs (env style, `LLAMA_*`) until a decision table says
  otherwise. BF16 is an explicit precision mode (owner decision).

## 1. State to start from

| Fact | Value | Source |
| --- | --- | --- |
| 8B prefill, 2048 tokens, one socket | AMX-BF16 640 tok/s, AMX-INT8 647, AVX-512 FP32 183, per token 17.3 | measured, C7 |
| 8B prefill, 2048 tokens, two sockets | AMX-BF16 1354 tok/s, AMX-INT8 1341 | measured, D1 |
| Attention share of the forward, 8B AMX | **41% at 2048 tokens, 74-75% at 8192** | measured, C7 phase shares |
| Attention share, 1B decode | 65-81% of a token at a 32768 context | measured, B7 |
| 8B AMX-BF16 prefill, 8192 tokens | 30.6 s, about 23 s of it attention; AMX rungs plateau near 270 tok/s | measured, C7 |
| Two-socket decode gain, 1B W8A8 at 32768 | only 1.26x (8B: 1.8-2.0x), attention 81% of the token | measured, B7 |
| Achieved against modeled ceiling | AMX-BF16 36%, AMX-INT8 14% (8B, 2048 tokens) | measured / modeled, C7 |

## 2. Priority 1: a faster attention kernel

### 2.1 Why it is the next win

At 2048 tokens attention is 41% of the 8B AMX forward and at 8192 it is 74%, so a faster GEMM rung no longer helps there
(AMX-INT8 equals AMX-BF16 at 2048). Estimated FLOP count for causal attention of an 8B prompt of n tokens is about
`2 * n^2 * dim * layers` (QK^T and PV, causal halves it): at n = 8192, dim = 4096 and 32 layers that is **about 17.6
TFLOP** (estimated). At roughly 23 s of attention on one socket this is **about 0.8 TFLOP/s achieved**, against about 10
TFLOP/s the same socket reaches in the AMX GEMMs. The headroom is in the order of 5-10x on the attention part
(**estimated**; real kernels will not reach the GEMM rate because of the softmax). At 8192 tokens that would take
prefill from about 270 to roughly 600-900 tok/s on one socket.

Decode attention is a different regime: single query rows, FP32 KV, bandwidth-bound. It is what limits long-context decode
and the two-socket gain at 32768 tokens on 1B (see 2.6).

### 2.2 What exists today

- Single-token and batched causal attention live in `src/kernels_simd.c` (batched section "Batched causal attention (C3)",
  `k_attention_batch`, `attention_batch_driver`, `batch_task`, `attention_chunk_avx512`), declared in
  `include/kernels.h` with the `AttentionWork` workspace (`k_attention_init_rows`, `LLAMA_ATTN_WORKSPACE_MIB`, default 32).
- Design: split-K over position chunks of 32-256, per (KV group, chunk, block of 4 query rows) tasks claimed from an atomic
  queue, a numerically stable merge per head (`merge_chunks`). A task reads one K/V chunk for a block of `BATCH_ROW_BLOCK = 4`
  rows so the chunk stays in L2. Each row uses exactly the chunk plan `k_attention` uses at its position, so the batched
  result is **bitwise equal** to per-token attention; the scores scratch is `BATCH_MAX_GROUP * BATCH_MAX_CHUNK` floats on the
  stack. Limits: `head_dim <= 256`, GQA group <= 16, m > 1.
- The kernel is FP32 AVX-512 on FP32 K/V rows laid out `[position][kv_dim]`. It does not use AMX, and each K/V element is
  used for at most 4 query rows (the row block), which keeps the arithmetic intensity low. In the AMX rungs the surrounding
  GEMMs use AMX; attention is the one stage that did not move.
- Callers: `src/run.c`, `src/runq.c` (one socket) and `src/tp.c` (two sockets, local heads: `SPROFILE(PHASE_ATTENTION, ...)`).
  Phase timing is available through the profile build (`PHASE_ATTENTION`, `tools/summarize_profile.py`, the `-xprof` runs).
- AMX kernels and packing are isolated in `src/gemm.c` (the AMX target attribute stays there), with tests in `tests/amx.c`,
  `tests/gemm.c`, `tests/bf16.c` and the software tile model.

### 2.3 Design options (the agent decides, with measurements)

1. **A larger register/row block FP32 kernel first (small, bitwise-compatible).** Raise the effective query block (8 or 16
   rows per K/V chunk), tile QK^T so each loaded K row feeds several queries held in registers, and fuse the online softmax with
   the PV accumulation to avoid the scores round trip. Keep the split-K merge so results stay bitwise or within 1e-5 of the
   current kernel. This is the cheap step and tells how much of the gap is memory traffic and instruction mix.
2. **A fused AMX-BF16 flash-style kernel (the large win).** Per (KV group, query tile of 16 rows x GQA heads): QK^T with BF16
   tiles (K^T as the B operand, packed per chunk), online softmax in FP32 registers (running max and sum per row), convert P to
   BF16, accumulate PV with AMX, FP32 accumulators merged across chunks as today. Needs BF16 copies of Q (per microbatch),
   and K/V either converted per chunk on the fly or kept in a BF16 shadow of the cache for the prompt (memory: half of the FP32
   KV, per layer and socket; count it in the per-socket budget `check_budget` and `tp_choose_batch`). Opt-in under
   `LLAMA_PRECISION=bf16` or a new `LLAMA_ATTN=amx` knob; it is not bitwise and is judged by the BF16 gate plus perplexity-style
   end-to-end agreement, never against the 1e-3 FP32 budget.
3. **AMX-INT8 attention is out of scope** until BF16 shows where the time goes (softmax and scales make it unattractive).
4. **KV-aware scheduling.** Causal work is triangular: later query rows have more positions. Check the balance of the
   `(group, chunk, block)` task queue, the chunk plan for short rows, and the NUMA locality of K/V reads on two sockets
   (each socket owns its local heads and KV, so reads are already local; verify with the placement counters).

Order: measure first (2.4), then option 1, then option 2 if the profile says attention is compute-bound rather than
bandwidth-bound.

### 2.4 First task: a standalone microbenchmark

Before changing anything, add `tests/attention_bench.c` in the style of `tests/gemm_bench.c` and `tests/amx_bench.c`:
`k_attention_batch` for 8B and 1B shapes (heads 32 / KV heads 8, head_dim 128 and 64), prompt sizes 512, 2048, 8192, M in
{16, 64, 256}, 14 / 28 / 56 threads, reporting ms per layer, achieved GFLOP/s and K/V GB/s. On the hardware this separates
"limited by the instruction mix" from "limited by L2/HBM traffic", which decides between options 1 and 2. Expected shape
(**estimated**): if it reaches only about 1 TFLOP/s while K/V traffic is far below 500 GB/s, it is compute/instruction bound
and AMX pays off.

### 2.5 Correctness and tests

- Unit: new tests next to the C3 ones (`tests/prefill.c`, `tests/kernels.c`): bitwise equality of option 1 with the current
  `k_attention_batch` where the chunk plan is kept; for option 2 a double-precision reference with a BF16 error budget
  (as `tests/bf16.c` does for the GEMM). Cover M = 1, 3, 7, 32, prompts at chunk boundaries (31/32/33, 255/256/257 positions),
  GQA group sizes 1, 4 (1B/8B use 4), `head_dim` 64 and 128, `pos0 > 0` (warm context and microbatches after the first).
- Engine: `tools/compare_engines.py` one socket batched against per-token for FP32 (budget 1e-3), `LLAMA_PRECISION=bf16`
  against the BF16 gate; two sockets with `LLAMA_TOPOLOGY=virtual:2x3`; `LLAMA_TP_CHECK=1` for identical exchange state.
- Sanitizers: `MODE=sanitize` and `MODE=tsan` runs of the new tests (the kernel runs inside the team barrier protocol).
- Hardware: `STAGE=check` of `cresco8-prefill.sbatch` (C3 and D1 gates), then the C7 and D1 sweeps limited to 2048 and 8192
  tokens, `GATES_ONLY=1`, `SKIP_TEST=1` where the full `make test` already passed. Report prefill tok/s, TTFT, the attention
  share (`-xprof`) and the gate numbers against the C7/D1 tables in [`docs/batched-prefill.md`](../../batched-prefill.md) and
  [`docs/numa-tp-prefill-results.md`](../../numa-tp-prefill-results.md).

### 2.6 Decode attention (second part of the same item)

Single-row decode attention is bandwidth-bound on FP32 KV. At a 32768-token context 1B decode spends 65-81% of the token
in attention, and two sockets gain only 1.26x (1B W8A8). Candidates: BF16 or FP16 KV storage (halves the bytes, needs a
precision-mode decision and the dequantizing read in `k_attention`), better split-K scheduling for few KV heads per socket
(each socket has 4 local KV heads on 1B/8B with two sockets, so `attention_chunks` decides the parallelism), and checking
that the 32768 case is not limited by chunk plan or merge cost. Measure with the B7 matrix (`PHASE=B7`, contexts 8192 and
32768, profiled runs).

### 2.7 Done when

- A microbenchmark and a profile show where attention time goes today (committed as a document).
- The new kernel is opt-in, passes its gates, and 8B prefill at 8192 tokens gains at least 1.5x end to end on one socket
  (target: well above that if AMX attention lands), with 2048 tokens not slower, decode unchanged, placement unaffected.
- A results document and validation record, the D2 decision table updated for the new knob, and the roofline's attention
  paragraph updated with the measured figure.

## 3. Other follow-ups, in priority order

2. **Explain and verify two results that look odd** (cheap, counter-based). (a) Two-socket prefill gain of 2.05-2.15x on 8B at
   8192 tokens, above the 2x of the hardware: look at the UPI, HBM and DDR counters of D2 (`perf_uncore`), per-socket cache
   effects (each socket attends over half the heads, so half the K/V working set) and whether the one-socket baseline suffers
   from the chunk plan or the 32 MiB workspace waves (`LLAMA_ATTN_WORKSPACE_MIB`). (b) Three two-socket 1B W8A8 runs from an
   8192-token context stopped on a stop token after 29 of 128 tokens in every repeat: add a determinism comparison of the
   generated text across kernels and sockets to the gates and decide whether it is the W8A8 noise floor or a bug.
3. **Batched decode / multi-request batching.** The first change that raises throughput rather than latency: several
   requests per step share each weight read, so decode gets closer to the compute roofline. Needs a request scheduler, per-request
   KV, and the batched kernels already used by prefill. Design phase first; it is the largest item.
4. **Exchange and compute overlap on two sockets.** Overlap the next phase's weight prefetch with the wait for the peer;
   quantize the activation slice inside the producing job (B7 follow-ups). Reductions plus waits are at most 0.022 ms per
   token in B7, so the expected gain is small; do it only if a profile shows peer wait above 10%.
5. **Asymmetric FFN shards** for the per-process socket imbalance (B7 follow-ups). Measure the imbalance first
   (`tools/campaign_tables.py` per-socket phase table).
6. **BF16 checkpoints and BF16 KV storage; KV tiering HBM -> DDR.** Needs a format decision and the owner's agreement:
   BF16 is an explicit precision mode and FP32 execution never changes silently. KV tiering matters once contexts exceed the
   HBM budget; the capacity result is in the D2 document.
7. **Housekeeping.** K3, the F3 acceptance comparison of the kernels script against `cresco8-placement.sbatch`, was skipped
   (nothing depends on it). Low priority.

## 4. How to report

Return, in under 300 words: what changed, the measurements with ranges (three repeats), gate results against the numbers
above, anything the owner must decide, and the exact commands or job ids that reproduce the result. Put the long form in the
results document, not in the report.
