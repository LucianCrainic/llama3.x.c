# C7. One-socket prefill campaign

## 1. Goal and status

Measure prefill throughput and TTFT on one socket for each rung of the ladder, by
model, prompt length and microbatch size, with decode measured **separately** to show
no regression. Then choose defaults (`LLAMA_PREFILL_BATCH` and the prefill kernel per
precision) from the evidence.

Status: **planned**. Branch: `feature/prefill-c7-campaign` (scripts and results docs
only).

## 2. Depends on / unblocks

- Depends on: C6 (all rungs) and F3. A6 is optional: if it is done, C7 runs on
  `anon-hbm` placement; otherwise on staged `/dev/shm` HBM, and records which.
- Unblocks: D2, and the default settings for D1.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Library, summary | F3 | `make_prompt` lengths 86/512/2048/8192 |
| Workload flags | `scripts/cresco8-kernels.sbatch:116-136` | Same seed, temperature and chat template |
| llama.cpp reference | `docs/llamacpp-comparison.md` (batched ubatch 512, **measured** in Cache mode) | Context only, since the memory mode differs |
| Prefill roofline | `docs/hbm-roofline.md` §4 | **Modeled** per-socket ceilings |

## 4. Changes

- A `PHASE=C7` matrix in `scripts/cresco8-prefill.sbatch`.
- Results docs: `docs/batched-prefill.md` and `docs/batched-prefill-validation.md`.
- Updated defaults in a follow-up commit, if the evidence supports them.

### 4.1 Run matrix

All runs are on socket 0, with `LLAMA_BACKEND=teams`, HBM, and 3 interleaved repeats.

| Dimension | Values |
| --- | --- |
| Rungs | `per-token` (M = 1, the baseline); `avx512-fp32`; `amx-bf16` (`LLAMA_PRECISION=bf16`); `avx512-q8` (W8A8); `amx-int8` (W8A8); `avx512-q8-w8a32` |
| Models | 1B, 8B |
| Prompt lengths | 86 (`short.txt`), 512, 2048, 8192 |
| M | 1, 16, 32, 64, 128, 256 |
| Threads | 56, plus 14 and 28 for the best M per rung |
| Generation | 32 tokens, so decode time is measured but not dominant |
| Separate decode runs | 128 tokens after an untimed `--warm-context 2048` (B7 option), per rung's prefill setting, to confirm decode is unaffected by mode, AMX frequency effects or workspace growth |
| Profiling | One extra repeat with `LLAMA_PROFILE=1`: GEMM vs attention vs conversion/rescale/packing time |

The grid is trimmed to stay under about 5 hours (**estimated**):

- The full M sweep runs only at prompt 2048.
- The other prompt lengths use M = 1 and the best two M per rung.

It needs the user's go-ahead to submit.

### 4.2 Correctness gate (before benchmarks)

The C2, C3, C5 and C6 `compare_engines.py` gates are rerun on 8B at prompt 2048, for
each rung, at the chosen M values.

## 5. Local tests

None beyond C1–C6. Run the script dry run.

## 6. Cluster run

As in §4.1. `summary.md` reports, per rung, model, prompt and M:

- prefill tok/s and TTFT (median and range);
- the speedup over per-token;
- achieved TFLOP/s against the **modeled** ceiling (`docs/hbm-roofline.md` §4: 8B
  ~370 / ~1,800 / ~4,600 tok/s for AVX FP32 / AMX-BF16 / AMX-INT8);
- the attention share;
- decode tok/s from the separate runs.

## 7. Exit criteria

- Every rung passes its gate.
- Prefill speedups are reported with ranges.
- Decode is within its one-socket range (A6 or `docs/kernels.md`).
- Defaults are chosen and justified: M per precision, and whether `auto` may select
  AMX for prefill. AMX-BF16 never becomes a silent default for FP32 checkpoints,
  because BF16 is an explicit mode.

## 8. Risks and open questions

- **Attention dominates long prompts.** At 8192 tokens, attention (C3) may dominate
  regardless of the GEMM rung. Report the share, and do not claim a GEMM speedup there.

## 9. Not in this phase

- Two sockets (D1, D2).
- Multi-request batching.
