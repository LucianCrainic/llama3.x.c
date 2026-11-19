# Prefill attention results (E2-E4): validation record

**Status: E2 run (2026-11-18); E4 run (2026-11-19), determinism rerun pending.** Provenance, gates and gaps. Results: [attention-results.md](attention-results.md). Evidence file names are
relative to each job's results directory.

## E2 provenance

| Item | Check job | Bench job |
| --- | --- | --- |
| Date (UTC) | 2026-11-18 | 2026-11-18 |
| Slurm job ID | 5863571 (`STAGE=check PHASE="E1 E2"`, `REPEATS=3`, `BASELINE_ROOT` = the 8f3114f tree) | 5864099 (`STAGE=bench PHASE=E2`, `REPEATS=3`, `SKIP_TEST=1`, `SKIP_PRELUDE_HF=1`, `D2_M1="avx512_fp32=64 amx_bf16=256"`) |
| Commit | 0b26adb (`develop`) | 0b26adb (`develop`; the later bf175de changes only `phase_E0`) |
| Node, memory mode | one node of the HBM partition, exclusive; Flat mode; 2 x Xeon Max 9480, 56 cores per socket | same |
| Compiler, flags | gcc, `-O3 -fopenmp -std=c11 -pthread` (`build-config-repo-0b26adb.txt`) | same |
| Script | `scripts/cresco8-prefill.sbatch`, `phase_E2` (`e2_check`); summary `tools/summarize_e2.py checks` → `E2/summary.md` | `phase_E2` (`e2_bench`); `tools/summarize_attention_bench.py --compare`, `tools/summarize_e2.py ttft` → `E2/bench-summary.md`, `E2/ttft.csv` |

A first bench submission (5864098) was cancelled about a minute after it started because `MODELS` was empty, which would have
skipped the engine runs. It produced no results that are used here.

## E2 gates

| Gate | Criterion | Observed | Result | Evidence (check job unless noted) |
| --- | --- | --- | --- | --- |
| `make test` on hardware | exit 0 | every C check 0 failures; tiled unit errors 8.7e-7 vs double, 9.2e-7 vs split; Python suites OK | pass | `phase-results.txt`, `test.log` |
| FP32 tiled vs per-token | max \|Δ\| ≤ 1e-3, argmax 100% | 1B: 1.62e-5 / 5.72e-6 / 1.31e-5 at 86 / 2048 / 8192; 8B: 1.37e-4 / 1.91e-5 / 9.54e-6; argmax 100% | pass | `E2/summary.md`, `compare-*-e2-tiled-m64-*.json` |
| FP32 tiled, queries 4 | same | 1B 1.31e-5, 8B 9.54e-6 at 8192 | pass | `compare-*-e2-tiled-q4-*.json` |
| BF16 tiled vs split | `BF16_GATE` (max 2, rms 0.05, argmax ≥ 97% at 86) | max 0.084 / 0.043 (1B), 0.768 / 0.042 (8B) at 86 / 2048; argmax 100% | pass | `compare-*-e2-bf16-tiled-vs-split-*.json` |
| W8A8 tiled vs split | `W8A8_GATE` (max 16, rms 0.2) | max 0.205 (1B), 0.278 (8B) at 2048; argmax 100% | pass | `compare-*-int8-e2-w8a8-*.json` |
| Two sockets | 8B FP32 2048, `LLAMA_TP_CHECK=1`, FP32 gate | 1.34e-5, argmax 100% | pass | `compare-8b-fp32-e2-s2-tiled-m64-prompt2048.json` |
| Default unchanged | `split` bitwise vs 8f3114f (8B FP32 2048: M 64, per-token, two sockets) | bitwise, max 0 | pass | `compare-8b-fp32-e1-*.json` |
| Run completeness (bench) | every launch exit 0 | 28 of 28 bench launches PASS; 64 engine TTFT reports (48 timed, 16 profiled) and 3 decode reports present | pass | bench job `phase-results.txt` |
| Value: kernel | bench 8B 8192 M 256 56 threads ≥ 1.8x | 1.90x | pass | bench job `E2/bench-summary.md` |
| Value: engine TTFT | 8B AMX-BF16 8192 one socket ≥ 1.5x, beyond variability | 1.66x; split [30.670-30.702] vs tiled [18.416-18.575] s | pass | bench job `E2/ttft.csv` |
| Value: 2048 not slower | tiled median ≥ split lower end, every rung | every tiled median is below the split minimum (1B FP32 at 2048: ranges overlap at the top, median 2.004 vs split minimum 2.141 s) | pass | bench job `E2/ttft.csv` |
| Decode untouched | one decode cell within its range | 13.06 [12.98-13.09] tok/s vs D2 13.31 [13.30-13.46] (`hbm` placement); this run staged the file on node 2 | recorded: about 2% lower under a different placement; no E2 code runs in decode | bench job `8b-fp32-*-xdec-*.json` |

## Gaps

1. **Decode reference.** The decode cell was run with file staging, not the D2 `hbm` placement, so it is not a like-for-like comparison. E4 runs
   decode after prefill for `split` and `tiled` under identical settings.
2. **Bench and engine attention differ in absolute terms.** This carries over from E0. The value gate on TTFT uses the engine, and the bench
   is used for relative comparison only.
3. **The task-order knob** was re-measured with `natural` as the default in job 5864122 (commit 5a354ea, `make test` on the hardware
   passed, 41 of 41 launches PASS, five repeats). Natural is faster in 3 of the 4 sweep cells; the exception is 2048 tokens at M 16, with
   overlapping ranges. See [attention-results.md](attention-results.md) §1.5.

## E4 provenance

| Item | Check job | Bench job | Determinism rerun |
| --- | --- | --- | --- |
| Date (UTC) | 2026-11-18 (start 23:55) | 2026-11-19 | queued, no result yet |
| Slurm job ID | 5864123 (`STAGE=check PHASE=E4`) | 5864134 (`STAGE=bench PHASE=E4`, `REPEATS=3`, `SKIP_TEST=1`) | 5864239 |
| Commit | 5a354ea (`develop`) | 5a354ea (`develop`) | b318378 (fixes the generation-comparison output directory) |
| Node, memory mode | one node of the HBM partition, exclusive; Flat mode; 2 x Xeon Max 9480, 56 cores per socket | same | same |
| Placement | engine-owned HBM (`LLAMA_PLACEMENT=hbm`), as D2 | same | same |
| Script | `scripts/cresco8-prefill.sbatch`, `phase_E4`; summary `tools/summarize_e4.py` | same; 256 reports (2 models x 16 cells x `split`/`tiled` x 3 timed repeats plus one `-xprof`) | same script, generation comparisons only |
| Result | 30 of 30 logit comparisons PASS; 12 generation comparisons did not run | all E4.3 gates evaluated | pending |

`make test` passed on the hardware for commit 5a354ea in job 5864122 (the E2 rerun, section E2 gap 3). The summarizer of the bench job
was corrected after the job in commit da6ce2b, and the numbers in [attention-results.md](attention-results.md) section 2 come from the
corrected summarizer run on the job's 256 reports.

## E4 gates

| Gate | Criterion | Observed | Result | Evidence (job directory) |
| --- | --- | --- | --- | --- |
| Logits, FP32 | tiled and split vs per-token, max \|Δ\| ≤ 1e-3, argmax 100%; 86 and 2048 tokens, M 64 and 128, two sockets with `LLAMA_TP_CHECK=1` | 1B max 1.62e-5; 8B max 1.37e-4 (tiled, 86 tokens); argmax 100% | pass | check job `E4/summary.md`, `compare-*-e4-*.json` |
| Logits, BF16 | tiled vs split, `BF16_GATE`; two sockets | max 0.084 (1B), 0.768 (8B) at 86 tokens; 0.043 and 0.042 at 2048; argmax 100% | pass | same |
| Logits, W8A8 | tiled vs split, `W8A8_GATE`; two sockets | max 0.215-0.268; argmax 100% | pass | same |
| Determinism | generation comparisons identical | 12 comparisons did not run (wrote to a missing `E1/` directory) | **pending** (rerun 5864239, commit b318378) | none yet |
| Run completeness (bench) | every launch produced a report | 256 reports | pass | bench job `E4/campaign-summary.md` |
| Value | 8B AMX-BF16 8192 one socket ≥ 1.5x vs C7 (≤ 20.4 s), beyond variability | 16.182 [16.163-16.182] s; split in the same job 25.057 [25.047-25.120] s (1.55x) | pass | bench job `E4/campaign-summary.md`; `8b-bf16-teams-hbm-s1-t56-m256-p8192-kamx_bf16-xattn{split,tiled}-{1,2,3}.json` |
| 2048 not slower | tiled median ≥ lower end of the split range, 16 cells | 16 of 16 | pass | same |
| Decode unchanged | tiled decode range overlaps split's, or median ≥ split's | 31 of 32 by that rule; ratio 0.997 (0.970-1.026), ranges overlap in 29 of 32; the exception, 8B AMX-BF16 8192 two sockets, is 25.5 against 25.8 tok/s (1.2%) | pass, with the exception explained by chance (about 1.6 of 32 cells expected) and identical decode code | `*-xattn{split,tiled}-*.json` (`throughput_tokens_per_second`) |
| Placement | region table: no mismatched page, regions verified | 256 of 256 reports, 0 mismatched pages; weights and KV on node 2 (one socket), 2 and 3 (two sockets) | pass | `regions` in every report |
| Attention share | `-xprof` at 2048 and 8192 per variant | recorded for every cell | recorded | `*-xprofsplit-1.json`, `*-xproftiled-1.json` (`profile_seconds`) |
| Item 2a | two-socket gain at 8192 recorded and concluded | 8B 2.00-2.13x (split) to 1.85-1.95x (tiled) | recorded | bench job `E4/campaign-summary.md` |

## E4 gaps

1. **Generation (determinism) comparisons.** The 12 comparisons of the check job did not run, because the script wrote into an `E1/`
   directory that did not exist (fixed in b318378). Job 5864239 reruns them and is queued. Until it passes, `tiled` has logit-level
   correctness evidence only, and the `LLAMA_ATTN` default should not be flipped.
2. **Summarizer gates fixed after the job.** The job's own summary marked decode FAIL (20 of 32 cells) and placement FAIL. The decode
   rule was too strict for three-repeat noise on identical decode code. The placement rule read the placement label instead of the region table. Both were corrected in da6ce2b and the summary was
   regenerated from the same reports. Nothing was remeasured. The corrected decode rule (range overlap, or median not lower)
   still lists one cell, covered in the decode note of [attention-results.md](attention-results.md) section 2.2.
3. **One-socket regions.** The one-socket work regions are not verified, by design. The K-cache regions of one-socket runs report a few
   missing (untouched) pages in every report, as D2's reports (2026-11-17) do.
4. **Baseline differs from section 1.** This campaign uses engine-owned HBM placement, so its `split` baseline (25.06 s) is faster than
   C7's file-staged 30.6 s and than job 5864099's 30.68 s. The gate against C7 is stated from C7's figure; the `split` and `tiled`
   comparison is within the job.
5. **No cache counters.** Item 2a is concluded from timings only; the mechanism of the two-socket surplus stays unconfirmed.
6. **`amx` attention (E3) not tested**, because E3 was postponed by the owner.
