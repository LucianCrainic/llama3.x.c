# Prefill attention results (E2-E4): validation record

**Status: E2 run (2026-11-18).** Provenance, gates and gaps. Results: [attention-results.md](attention-results.md). Evidence file names are
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
