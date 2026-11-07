# C1. Batched forward API and microbatched prefill

## 1. Goal and status

Introduce a batched forward entry point and drive prompt processing through it in
microbatches of M tokens. Logits are computed only where the API needs them, i.e. the
final prompt position.

The first implementation is a **reference** that calls today's single-token kernels
once per row. Its logits at the final position must be **bitwise** equal to the
per-token path, so all later kernel phases have an exact oracle.

Status: **implemented; cluster check passed** (see Cluster results).

Implementation notes:

- The reference path is layer-major (all rows of a microbatch through one layer before the
  next) and calls the existing GEMV/attention kernels row by row. It lives in `run.c` and
  `runq.c` behind `engine_forward_batch`; `src/prefill.c` holds `LLAMA_PREFILL_BATCH` /
  `LLAMA_PREFILL_LOGITS` parsing and input validation. User documentation is
  [`docs/prefill.md`](../../prefill.md) and local evidence
  [`docs/prefill-c1-validation.md`](../../prefill-c1-validation.md) (developed on
  `codex/prefill-c1-batch-api`, merged here).
- Report schema v2: `prefill = {microbatches, tokens, seconds, logits_rows}` and
  `runtime.prefill_batch` / `runtime.prefill_logits` are now filled.
- `tools/compare_engines.py` gained `--batch-a/--batch-b M` (runs the probe's `batch` mode with
  `LLAMA_PREFILL_LOGITS=all`) and `--bitwise` (the C1 gate).
- Even at M = 1 the unused prompt classifiers are skipped, so `LLAMA_PREFILL_LOGITS=all`
  is the per-token timing baseline.

## 2. Depends on / unblocks

- Depends on: F1 (descriptors). F2 is not required yet: the reference path uses the
  existing kernels.
- Unblocks: C2, C3, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `run_sequence` | `src/app.c:199-262` | Prefill loop at :209-223; decode loop after it |
| `engine_forward` | `include/runtime.h:65-70`; `src/run.c:146`, `src/runq.c:223` | One token, one position, full logits |
| Metrics | `include/runtime.h:116-125` (`prefill_s`, `ttft_s`, `forward_passes`) | Timing semantics documented in `docs/phase1.md` |
| Probe | `tests/probe.c` | Logits output for C-vs-C comparison |
| Classifier cost | `docs/hbm-roofline.md` §4 | 21% of 1B and 7% of 8B per-token weights. Skipping it is the first, trivial gain. |

## 4. Changes

### 4.1 API

```c
/* Proposed, include/runtime.h. */
/* Processes tokens[0..m) at positions pos0..pos0+m-1. Returns logits for the last row
 * when want_logits == LOGITS_LAST, for all rows (m x vocab) when LOGITS_ALL, else NULL. */
typedef enum { LOGITS_NONE, LOGITS_LAST, LOGITS_ALL } LogitsWanted;
float *engine_forward_batch(Model *m, const int *tokens, int count, int pos0, LogitsWanted want);
int engine_max_batch(const Model *m);
```

### 4.2 Run state

- **Batched buffers.** `RunState` gains batched activations, each a row-major
  `M × dim` (or `M × hidden`) buffer: `X`, `XB`, `Q`, `HB`, `HB2`, and Q8 `XQ`/`HQ`
  with per-row scales.
- **Batched logits.** `LOGITS` is allocated `vocab × M` only when `LLAMA_PREFILL_LOGITS=all`.
- **`M_max`.** The maximum batch is `LLAMA_PREFILL_BATCH`. The buffers come from the
  socket's `ROLE_WORK` region (A5), or from `memory_alloc` before A5.
- **Activation sizing.** `M_max × hidden × 4` bytes is 7.3 MB at M = 128 for 8B
  (**modeled**). The KV, not the activations, dominates memory.

### 4.3 Reference implementation

For each layer, for each row i, call today's kernels on row i. RoPE and attention
for row i use position `pos0 + i`, and K/V row i is written before attention of
row i. Rows are processed in order, so causality is exact. This repeats the per-token
order exactly, and therefore gives bitwise equality.

### 4.4 `app.c` changes

- **Microbatching.** `run_sequence` cuts the prompt into microbatches of
  `min(M, remaining, steps - position)`. Only the final microbatch requests
  `LOGITS_LAST`. The step-limit and echo behaviour of :209-223 are preserved exactly.
- **Metrics.**
  - `prefill_s` times each microbatch call.
  - `processed_tokens` is incremented by `count`.
  - `forward_passes` is incremented by `count`, to keep the meaning "tokens forwarded".
  - A new `prefill.microbatches` counter is added.
  - TTFT semantics are unchanged.
- **Chat mode.** The EOT forward (`src/app.c:318`) stays a single-token call.

### 4.5 Knobs

- `LLAMA_PREFILL_BATCH` (default 1, so behaviour is unchanged).
- `LLAMA_PREFILL_LOGITS=last|all`. `all` is diagnostic: it feeds `compare_engines.py`
  at every position.

## 5. Local tests

These run on macOS, for FP32 and INT8, with shared and unshared fixtures.

- **Bitwise equality.** For M ∈ {1, 2, 3, 7, 32} and prompt lengths {1, 5, 31, 32, 33},
  the final logits are bitwise equal to `LLAMA_PREFILL_BATCH=1`. With
  `LLAMA_PREFILL_LOGITS=all`, every position is bitwise equal.
- **`test_runtime.py`.** Step limits inside a microbatch, stop tokens, echo,
  chat multi-turn and JSON accounting invariants (`tests/test_runtime.py:110-127`) all
  hold with M = 7.
- **Skipped classifier.** With `last`, the classifier runs once per microbatch
  sequence. This is checked with a test counter.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C1`. **Estimated** at 30 minutes. It needs
the user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| `compare_engines.py` 1B and 8B FP32/INT8, M = 1 vs M = 32, `all` logits | Bitwise identical |
| Prefill speed, 8B FP32, 86-token and 2048-token prompts | **Measured** gain from the skipped classifier only (expected about 7% on 8B per the **modeled** share) |

### Cluster results (2026-11-07)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, `STAGE=check`, `HF_GATE=0`. Two jobs:
5693938 (commit `fec5f44`, `m2-prefill-20261107-1`) and the re-run 5693952 (commit `3f94f34`, `m2-prefill-20261107-2`,
5 min), results under `~/PFS/lpor`.

- **First run, a test-matrix flaw.** All eight FP32 bitwise comparisons failed (max |Δlogit| 1.1e-5 on 1B, 1.4e-4 on 8B,
  argmax 100%, the same for M = 7 and M = 32 and already at position 0). On AVX-512 hosts the default batched FP32 path is
  the C2 GEMM (`LLAMA_PREFILL_GEMM=auto`), which sums in another order than the GEMV, so a bitwise check of that path is
  wrong. Its agreement is C2's budget gate (passed, see C2). The Q8 comparisons were bitwise identical in that run.
- **Re-run, the reference path.** With `LLAMA_PREFILL_GEMM=off` on the batched side for FP32, all 18 comparisons are
  **bitwise identical** (max |Δlogit| 0.0 at all 53 positions): 1B and 8B, FP32 and Q8_0, `omp` and `teams`, M = 7 and
  M = 32, plus Q8 W8A32 at M = 32. The matrix now pins this setting (`phase_C1`).
- **Skipped classifier** (8B FP32, 2048-token prompt, per-token M = 1, median of 3): prefill 16.5 tok/s with `last`
  against 15.4 with `all` (+7.1%, the **modeled** ~7%); 1B 111.8 against 89.9 (+24%, the classifier is a larger share of
  a small model).

## 7. Exit criteria

- The batched API exists, and microbatched prefill is bitwise equal to per-token.
- Prompt-position logits are skipped by default and available on request.
- Accounting and limits are unchanged.

## 8. Risks and open questions

- **`forward_passes` meaning.** It changes its basis if counted per call. Keep it per
  token, and add `prefill.microbatches` for the call count, so existing summaries stay
  comparable.

## 9. Not in this phase

- Any GEMM (C2).
- Batched attention kernels (C3). The reference runs attention row by row.
