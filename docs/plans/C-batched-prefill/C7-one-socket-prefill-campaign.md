# C7. One-socket prefill campaign

## 1. Goal and status

Measure prefill throughput and TTFT on one socket for each rung of the ladder, by
model, prompt length and microbatch size, with decode measured **separately** to show
no regression. Then choose defaults (`LLAMA_PREFILL_BATCH` and the prefill kernel per
precision) from the evidence.

Status: **script written, dry-run verified; nothing run** (2026-11-07). Branch: `feature/prefill-c7-campaign` (scripts
and results docs only). `phase_C7` in `scripts/cresco8-prefill.sbatch` implements §4.1 and §4.2 (pass structure in the
comment above `c7_best`); `DRY_RUN=1` prints 245 launches per model (with the dry run's placeholder best sizes) and 16 correctness gates. The results documents
`docs/batched-prefill.md` and `docs/batched-prefill-validation.md` and the choice of defaults wait for the run; it needs
C4-C6 to have passed on the hardware first.

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

None beyond C1–C6. Run the script dry run:

```text
DRY_RUN=1 SLURM_JOB_ID=1 ROOT=$PWD OUT=/tmp/c7-dry PYTHON=python3 PHASE="C4 C5 C6 C7" \
  MODELS="m1b:/a/m1b:/hf/m1b m8b:/a/m8b:/hf/m8b" bash scripts/cresco8-prefill.sbatch
```

(done 2026-11-07: exit 0, every run label parses with `tools/summarize_campaign.py`).

## 6. Cluster run

As in §4.1. `summary.md` reports, per rung, model, prompt and M:

- prefill tok/s and TTFT (median and range);
- the speedup over per-token;
- achieved TFLOP/s against the **modeled** ceiling (`docs/hbm-roofline.md` §4: 8B
  ~370 / ~1,800 / ~4,600 tok/s for AVX FP32 / AMX-BF16 / AMX-INT8);
- the attention share;
- decode tok/s from the separate runs.

### Implementation notes (script)

- **Rungs** are the `c_rung` environments: `per_token` and `per_token_q8` (`LLAMA_PREFILL_BATCH=1`, AVX-512 kernels),
  `avx512_fp32`, `amx_bf16`, `avx512_q8`, `amx_int8`, `avx512_q8_w8a32`. Run labels carry the rung in `-k...` and the
  precision field is `fp32`, `bf16`, `int8` or `w8a32`, which `tools/summarize_campaign.py` already parses.
- **Passes** (per model and precision, checkpoints staged on HBM node 2, one at a time): gates; M sweep at 2048 tokens (M ∈
  {16, 32, 64, 128, 256} for the batched rungs, 1 for the baselines, {1, 64} for W8A32), repeats interleaved; best two M per
  rung read back from the sweep reports (`c7_best`); prompts 86 / 512 / 8192 at M = 1 and the best two; 14 and 28 threads at
  the best M; decode runs of 128 tokens; one `LLAMA_PROFILE=1` repeat at 2048 and 8192 tokens, summarized in
  `C7/phase-shares.txt`.
- **Differences from §4.1**, all to keep to the time budget or because a piece does not exist yet: the per-token baselines run
  once (not 3 times) at 8192 tokens and on 14 and 28 threads; the decode runs prefill 2048 tokens in the same process (the
  report times prefill and decode apart) because `--warm-context` (B7) does not exist; at 2048 tokens the gates compare the
  last position only (1 GB of logits per side otherwise), and the AMX-INT8 gate there is against per-token, not against the
  batched AVX-512 path (`compare_engines.py --last-only` needs A to be per-token). BF16 comparisons are recorded
  (`expect any`) until C5 has set its provisional gate.
- **Estimate.** About 5 hours for 1B and 8B, dominated by the per-token baselines (8B FP32 about 85 minutes of it): submit
  with `--time=06:00:00`; the script header asks for 04:00:00.

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
