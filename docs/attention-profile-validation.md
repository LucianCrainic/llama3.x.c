# Attention profile (E0): validation record

> **Correction (2026-11-21):** HBM byte figures derived from `hbm_cas_rd` counts were computed at 64 B per count; the counter counts 32 B (calibrated in job 5892172, see [xengine-results-validation.md](xengine-results-validation.md)). Affected figures below are halved and marked †.

**Status: run (2026-11-18).** Provenance, gates and gaps of phase E0. Results: [attention-profile.md](attention-profile.md).
Evidence file names are relative to the job's results directory (`E0/` holds the phase's files; `phase-results.txt`, `environment.txt` and
`flat-mode.json` sit beside it).

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-18 |
| Node, memory mode | one node of the HBM partition, exclusive; Flat mode (`flat-mode.json`: DDR nodes 0 and 1, HBM nodes 2 and 3); 2 x Xeon Max 9480, 56 cores per socket, 225 MiB L3 over two sockets (`environment.txt`) |
| Slurm job ID | 5860637 (one job with both phases, `STAGE=check PHASE="E0 E1"`; E1 is recorded in its own results document) |
| Commit | 397aa10 (`develop`, E0 and E1 code merged) |
| Compiler, flags | gcc 11.4.1 20231218 (Red Hat 11.4.1-3), `-O3 -fopenmp -std=c11 -pthread` (`build` of `engine-profile.json`) |
| Script and stage | `scripts/cresco8-prefill.sbatch`, `phase_E0`; bench `tests/attention_bench.c`; tables by `tools/summarize_attention_bench.py` (`E0/summary.md`, `E0/counters.csv`) |
| Bench environment | `LLAMA_BACKEND=teams`, `LLAMA_THREADS` 14 / 28 / 56, `LLAMA_KERNELS=avx512`, `OMP_PROC_BIND=false`, run under `numactl --cpunodebind=0`, K/V on HBM node 2 (`--node 2`); `LLAMA_ATTN_ROW_BLOCK` 1 / 2 / 4 / 8 / 16 in the sweep, 4 otherwise |
| Matrix | prefill: 8b and 1b, prompts 512 / 2048 / 8192, M 16 / 64 / 256, 14 / 28 / 56 threads, 3 repetitions; row-block sweep; half-KV-head (`--kv-heads-local 4`) runs at 56 threads for 2048 and 8192 with M 64 and 256; decode: 8b and 1b, context 8192 / 32768, 8 and 4 KV heads, 200 calls, `--cold-mib` 1024 |
| Counters | `perf stat -a --per-socket` with per-core events (cycles, instructions, `l1d.replacement`, `l2_rqsts.miss`, `mem_load_retired.l3_miss`) and the uncore events of the campaign (`ddr_cas_rd`, `hbm_cas_rd`, UPI data flits) |
| Engine run | 8B FP32 checkpoint with AMX-BF16 prefill GEMMs, one socket, 56 threads, 8192-token prompt file (8226 tokens with the chat template), M 256, `LLAMA_PROFILE=1`, generation length 1 (`engine-profile.json`) |
| Repeats | 3 per prefill cell (median and range reported); decode 200 calls per cell; counters one launch per cell; the profiled engine run once |

## Gates

| Gate | Criterion | Observed | Result | Evidence |
| --- | --- | --- | --- | --- |
| Correctness: row block bitwise (engine) | `compare_engines.py --bitwise`, 8B FP32, 2048 tokens, M 64, last row, `LLAMA_ATTN_ROW_BLOCK` 4 against 16 | bitwise true, max_abs 0.0, argmax agreement 1.0 | pass | `compare-8b-fp32-e0-rowblock-4-vs-16-m64-prompt2048.json` (`.out`, `.err` empty) |
| Correctness: row block bitwise (unit) | `test_attention_batch` with `LLAMA_ATTN_ROW_BLOCK` 1, 4, 16 | all C tests passed on the hardware (job log of `make test`; the log is not part of the copied results) | pass (carried from the job log) | job 5860637 log |
| `make test` as a whole | exit 0 | exit 2: one Python test, `tests/test_tools.py`, failed. A test-isolation bug: the job's campaign variable `BASELINE_ROOT` leaked into `test_e1_dry_run_matrix`. Fixed in a8123c0 (already on `develop` after the job's commit 397aa10, not re-run on hardware) | fail (test isolation, not a code or measurement fault) | `phase-results.txt` (`FAIL make-test exit=2`) |
| Run completeness | every bench and engine launch exits 0 | 18 launches recorded PASS (the 8b and 1b prefill runs at 14 / 28 / 56 threads, the five row-block runs, both half-KV prefill runs, the four decode runs, the engine bitwise compare) and 1 FAIL (1b decode counters); the prefill counters run records nothing on success and its three output files exist | pass, with that exception (see Gaps 1) | `phase-results.txt` |
| Value: cross-check | bench `ms per layer x 32` within 20% of the profiled engine attention time, 8B / 8192 / M 256 | bench 540.48 ms x 32 = 17.30 s; engine attention 22.42 s (forward 30.39 s); ratio **0.77** (0.78 after scaling by (8226/8192)^2) | **fail** | `E0/summary.md` (cross-check), `engine-profile.json`, `bench-prefill.jsonl` |
| Decision table filled | every row of E0.6 has a verdict | H1 confirmed; H2 not isolated; H3 1.24x, below 1.3x; H4 tail only at small prompts; H5 partly (counters missing); H6 confirmed in effect | pass | [attention-profile.md](attention-profile.md) section 4 |

A missed value gate is a result: the bench does not represent the engine in absolute terms. The cause that is proposed (K/V resident in the L3 in the
bench, evicted by the weight stream in the engine) is a hypothesis; the bench has no way yet to test it. The consequence, recorded in the results
document, is that E2's value gate uses the engine's profiled attention time and the bench is used for relative comparison only.

## Numbers checked against the files

The tables of the results document are the output of `tools/summarize_attention_bench.py` on the job's files. Recomputed from the raw records for this
record: the cross-check (17.30 s against 22.42 s), the row-block ratios against rows per task 4 (0.70 / 0.82 / 1.00 / 1.15 / 1.24), the H6 ratios, the thread
scaling ratios, and the per-FLOP counter figures (instructions 5.57e11 over 5 passes of 549.8 GFLOP = 0.203 per FLOP; l1d.replacement 31.79, l2 miss
6.06 and l3-miss loads 0.0001 per KFLOP). They agree with `E0/summary.md`. Counters cover the whole launch, so they include start-up, which is
small against five passes of 0.54 s (about 0.4 GB of HBM reads per pass (†) against 70 GB streamed, and 7.7e7 cycles of start-up on the idle socket).

## Deviations from the plan

- Decode counters absent (Gaps 1). The plan's counters row asked for the 1b / 32768 decode cell.
- E0.5 planned 1.5 h; the E0 part of the job was a few minutes of bench time (the bench files are time-stamped within 4 minutes).
- The profiled engine run is a 1-token generation with the chat template (8226 tokens), not exactly 8192; the effect on the ratio is 0.01.
- The row-block sweep, the 56-thread matrix and the engine run used different processes; they agree where they overlap (8b / 8192 / M 256, row block 4: 538.5 ms in the sweep
  and 540.5 ms in the matrix).

## Known gaps

1. **1b / 32768 decode counters failed.** `counters-decode-1b-32768.err` holds the bench's usage line and the launch exited with an argument error: the
   phase passed `--reps 2000` and the bench accepts at most `MAX_REPS` = 1024. The `perf.csv` has only start-up counts (4.5e6 instructions) and
   `counters-decode-1b-32768.jsonl` is empty. Fix: `--reps 1000` in `phase_E0` (script change, no bench change). H5's "instruction-bound" half is open until then.
2. **Cross-check failed**, the bench is warm: no cold prefill mode yet (`--cold-mib` exists for decode only). Proposed follow-up, not implemented.
3. **H2 not isolated.** The experiment "cut the K chunk to 64 positions in the bench" was not run (no knob), only the L1D replacement rate is available.
4. **H6 mechanism open.** No per-socket L3 counters from a D2-shaped two-socket run were collected; the one-socket counters show L3 misses of about 0 already at
   the full 64 MiB K/V, so a cache-fit explanation of the superlinear 2.08-2.35x at 8192 is not supported by the counters taken.
5. **Counters are one launch of one prefill cell**, no repeats; instruction mix (which instructions) was not measured.
6. **H5 chunk-count and head_dim experiments** (time against chunk count, time against head_dim at equal bytes) were not run; the decode chunk plan is the
   engine's (32 chunks at 8192, 128 at 32768).
7. **Thread counts 14 and 28** were run for the bench only, without engine comparison. The engine cross-check is the 56-thread, one-socket case.
8. **Sanitizer modes** (`MODE=sanitize`, `MODE=tsan`) are local-loop gates of the phase and are not part of this job.
9. The K/V and Q values of the bench are synthetic (a fixed pseudo-random generator), not those of a model; softmax work is value independent here, but cache and
   branch behaviour could differ slightly from a real prompt.
