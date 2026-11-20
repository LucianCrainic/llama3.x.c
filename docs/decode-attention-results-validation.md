# Decode attention results (E5): validation record

**Status: E5 run (2026-11-19).** Provenance, gates and gaps. Results: [decode-attention-results.md](decode-attention-results.md).
Evidence file names are relative to each job's results directory.

## Provenance

| Item | Check job | Bench job |
| --- | --- | --- |
| Date (UTC) | 2026-11-19 (02:10-04:28) | 2026-11-19 (05:03-11:56) |
| Slurm job ID | 5864135 (`STAGE=check PHASE=E5`, `REPEATS=3`, `SKIP_TEST=0`) | 5870935 (`STAGE=bench PHASE=E5`, `REPEATS=3`, `SKIP_TEST=1`) |
| Commit | 9d3f72b (`develop`) | 9d3f72b (`develop`) |
| Node, memory mode | one node of the HBM partition, exclusive, the node of B7; Flat mode (`flat-mode.json`: DDR nodes 0 and 1, HBM nodes 2 and 3); 2 x Xeon Max 9480, 56 cores per socket | same |
| Compiler, flags | gcc, `-O3 -fopenmp -std=c11 -pthread` (`build-config-repo-9d3f72b.txt`) | same |
| Placement | one socket: engine-owned HBM (`hbm`); two sockets: file staging per socket, as B7 | same |
| Prefill attention | `split` (the default of 9d3f72b) | same |
| Script | `scripts/cresco8-tp.sbatch`, `phase_E5` (check) | `phase_E5` (bench); `tools/summarize_campaign.py` → `E5/summary.md`; microbenchmark `E5/bench-decode.jsonl` |
| Result | 25 of 25 PASS | 192 reports (2 models x 2 precisions x 3 contexts x 2 socket counts x 2 kernels x 4 runs); 8 of 8 microbenchmark launches PASS |

The tables in the results document were computed from the bench job's reports (`throughput_tokens_per_second.decode`,
`profile_seconds.decode`) and from `E5/bench-decode.jsonl`. The B7 reference ranges are those of
[tensor-parallel.md](tensor-parallel.md).

## Gates

| Gate | Criterion | Observed | Result | Evidence |
| --- | --- | --- | --- | --- |
| `make test` on hardware | exit 0 | every C check 0 failures; Python suites OK | pass | check job `test.log` |
| Unit | M = 1 against the double reference and `split` (1e-5), across chunk boundaries and at 32767 | in `make test` (`tests/kernels.c`) | pass | check job `test.log` |
| FP32 per-token | `split` against `group`, max \|Δ\| ≤ 1e-3, argmax 100% | 1B 9.5e-6 (53 tokens), 2.6e-5 (2048); 8B 4.5e-5 (53), 5.0e-4 (2048); two-socket `group` against one-socket `split` 9.5e-6 (1B), 7.1e-5 (8B); argmax 100% | pass | check job `E5/compare-*-fp32-e5-*.json` |
| W8A8 per-token | `W8A8_GATE` (max 16, rms 0.2) | max 0.38-8.75, rms 0.061-0.124; argmax 96-100% | pass | check job `E5/compare-*-int8-e5-w8a8-*.json` |
| Hugging Face | `tools/validate.py`, FP32, 1B and 8B, one and two sockets, `LLAMA_ATTN_DECODE=group` | max \|Δ\| 2.2e-5 (1B), 1.2e-4 (8B); generated tokens match | pass | check job `correctness-*-group*.json` |
| Determinism (recorded, not gated) | 128 greedy tokens from a warm 8192-token context, `split` against `group` | FP32 4 of 4 identical; 8B W8A8 2 of 2 identical; 1B W8A8 diverges at token 34 (one socket, top-2 margin 0.119) and token 4 (two sockets, margin 0.883) | recorded | check job `E5/gen-*.json` |
| Run completeness | every launch produced a report | 192 reports, 8 microbenchmark cells x 2 kernels | pass | bench job |
| Value: 1B 32768 one socket | ≥ 1.3x beyond variability | FP32 1.075x, W8A8 1.103x (both beyond variability) | **fail** | bench job `1b-*-s1-*-p32768-xattn{split,group}-{1,2,3}.json` |
| Value: 1B W8A8 two-socket gain at 32768 | ≥ 1.5x | 1.46x (from 1.25x with `split`) | **fail** | same, `-s2-` |
| Value: bandwidth | reported against 552 GB/s per socket | 23-51% (engine, 8192 and 32768); microbenchmark best 403 GB/s | reported | `*-xattn*prof-1.json`, `E5/bench-decode.jsonl` |
| Value: 512 not slower | median within the B7 range | 6 of 8 within or above; 8B FP32 one socket and 8B W8A8 one socket below | **fail** (2 cells) | `*-p512-xattngroup-*.json` |
| Decision | bandwidth-bound? | no; next limit listed | recorded | results section 6 |

The two 1B W8A8 generation divergences are smaller than the largest per-token W8A8 logit difference between `split` and `group`
in the same job (1.09 at 2048 tokens), and both are top-2 swaps at a near tie. W8A8 requantizes activations, so a small change in
the attention output can move a logit by about that much. The FP32 and 8B W8A8 comparisons are identical over all 128 tokens.

The bench job also ran the one-socket Hugging Face gate of every precision with the default kernels (`split`); W8A8 records its
known deviation (exit 1, not gated) as in B7, and FP32 and W8A32 pass.

## Gaps

1. **Value gates missed.** The plan's 1.3x and 1.5x targets assumed decode attention was instruction-bound (E0's H5). The
   measured kernel speedup on 1B at one socket is 1.10-1.14x on attention time, and attention reaches 35-36% of the bandwidth.
   So the remaining limit is neither instructions nor bandwidth; results section 6 lists it.
2. **512 cells.** The two cells below the B7 range lose their time in weight phases that `group` does not run, and `split`'s own
   repeats in one of them are bimodal. With three repeats this is not settled; a targeted rerun is the first follow-up.
3. **One profiled run per cell.** The attention times, shares and bandwidths of results section 4 come from a single profiled
   run each. The 8B one-socket 8192 cell (attention 0.96x in the profile, token rate 1.023x in the timed runs) is the one place
   where they disagree in direction.
4. **Bandwidth is derived** from the bytes the kernel must read (2 x local KV heads x head_dim x 4 bytes x mean context x layers)
   and the profiled attention time. No cache or memory counters split attention traffic from the rest of the step.
5. **Two-socket placement** is file staging per socket, as in B7, not engine-owned HBM, so the one-socket and two-socket rows
   differ in placement as they did in B7.
6. **Prefill used `split`.** Commit 9d3f72b predates the `tiled` prefill default. Decode does not run the prefill kernel, so
   the decode figures are not affected; time to first token in these reports reflects `split`.

## Follow-ups (2026-11-19): 512 rerun, 8B 8192, K/V layout probe

**Status: run.** Results: [decode-attention-results.md](decode-attention-results.md) section 8; the decision rules of its 8.1 were
committed before any of these jobs ran (commit 961af17 is the rules commit and the build commit of all of them).

### Provenance

| Item | Hardware `make test` | 512 / 8192 engine job | Layout probe (main) | Layout probe (control order) |
| --- | --- | --- | --- | --- |
| Slurm job ID | 5875780 (`STAGE=bench PHASE=E5 SKIP_TEST=0`, cancelled after `make test`) | 5876099 | 5876098 | 5876113 |
| Date (UTC) | 2026-11-19 11:32-12:25 (tests PASS 12:17) | 2026-11-19 12:26-13:23 (56 min 31 s) | 2026-11-19 12:26 (31 s) | 2026-11-19 13:24 (31 s) |
| Commit | 961af17 (`feature/attn-e5-followups`) | same | same | same |
| Node, memory mode | one node of the HBM partition, exclusive, the node of B7 and the first E5 jobs; Flat mode; 2 x Xeon Max 9480 | same | same | same |
| Compiler, flags | gcc, `-O3 -fopenmp -std=c11 -pthread` | same | same | same |
| Settings | common `make test`, then the E5 bench stage | `E5_CONTEXTS="512 8192" E5_SOCKETS=1 E5_PROFILED=3 E5_ALTERNATE=1 E5_SKIP_MICRO=1 REPEATS=9 SKIP_TEST=1 SKIP_PRELUDE_HF=1`, 8B only (FP32 and W8A8) | `E5_SKIP_ENGINE=1 E5_LAYOUTS="rows head" E5_MICRO_CONTEXTS="512 8192 32768" SKIP_TEST=1 SKIP_PRELUDE_HF=1` | same with `E5_LAYOUTS="head rows"` |
| Result | `make test` PASS (10 C suites 0 failures, 69 Python tests OK) | 96 reports, all completed, exit 0 | 16 of 16 launches PASS (48 cells) | 16 of 16 launches PASS (48 cells) |

Job 5875772 failed in its first second (`PYTHON` not exported at submission; no results). Job 5875780 was cancelled after its
`make test` had passed, because the remaining common prelude (one-socket Hugging Face gate of every precision) would not fit in
its 2 h limit; the later jobs set `SKIP_PRELUDE_HF=1`, which the phase records in `no-prelude-hf.txt`. The tables of results
section 8 come from the 96 reports (`throughput_tokens_per_second.decode`, `profile_seconds.decode` / `forward_passes`) and from
`E5/bench-decode.jsonl` of the two layout jobs. The B7 reference ranges are those of results section 3. The decode kernels'
`LLAMA_ATTN` (prefill) setting of the engine runs is the campaign default `split`.

### Gates

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| `make test` on hardware | exit 0 with the head-layout unit tests | PASS (job 5875780); local: `make test` exit 0, `MODE=sanitize` test-kernels and test-kernels-emul 0 failures, `MODE=tsan` test-kernels-emul with 1 and 3 teams 0 failures | pass |
| Head-layout unit test | `split` and `group` on `[kv_head][position][head_dim]` against the double reference (2e-5), against the rows layout bit for bit, `group` against `split` (1e-5), NaN-poisoned rows after the position and NaN padding between heads | 0 failures at every instruction-set level and team size, positions up to 32767 | pass |
| Engine unchanged | engine callers keep stride `kv_dim` and head base `head_dim` | the E1/E2/E5 matrices (rows layout, bitwise comparisons between waves and variants) pass unchanged | pass |
| 512 not slower, 8B FP32 one socket | median within the B7 range (A) or in-job (B) | `group` 15.65 [15.56-15.72] against `split` 16.42 [16.30-16.45], B7 16.13-16.44; 0 of 9 paired | **fail** (A and B) |
| 512 not slower, 8B W8A8 one socket | same | `group` 49.46 [49.08-49.66] against `split` 49.36 [49.10-49.55], B7 49.52-49.73; 6 of 9 paired | pass by B (A missed by 0.12%; `split` itself 0.3% below the B7 range) |
| 8B 8192 one socket | recorded (rule 8.1) | token rate `group` 1.026x (FP32) and 1.065x (W8A8), 9 of 9 paired; attention phase not slower in the profile (FP32 inside the spread, W8A8 6% shorter) | recorded; the microbenchmark's 0.77x for this shape reproduced (0.371 / 0.480 ms) but not seen in the engine |
| Weight-phase cost | rule 8.1 (`d` against spread `s`) | FP32 512: `d` +2.68 ms, `s` 0.22 (follows the kernel setting); W8A8 512: +0.06 against 0.10 (not attributable); FP32 8192: -1.82 against 0.53 (group faster); W8A8 8192: -0.83 against 0.16 (group faster) | recorded; cause not identified |
| Layout cell gate | `rows ms / head ms` at least 1.2 | 45 of 48 cells (main order), 43 of 48 (control); every exception at 512 | recorded per cell |
| Layout verdict | `group` passes all four 32768 cells and none of its 8192 cells below 0.95 | lowest `group` speed at 32768 1.51 (8B kv8), lowest at 8192 2.06 | **engine change recommended** (not implemented) |

### Gaps

1. **Layout is a microbenchmark result.** The probe measures one layer on one socket's K/V with a cold-cache cycle; it does not
   include the K/V append, 32 layers sharing the caches, or the other phases. The effect on decode in the engine table of
   results 8.4 is **derived** (microbenchmark speed applied to one profiled attention time per cell), an upper-end estimate. The
   gate for the engine change is a measured gain in the engine, which needs the implementation.
2. **Two launch orders only for the layout probe, one repeat of 1000 repetitions per cell and order.** The two orders agree for
   every cell at 8192 and 32768 within 10% of the speed (1B kv8 32768 `group` 1.93 against 1.75 is the largest difference);
   the 512 cells differ more and carry no conclusion.
3. **The weight-phase effect has no identified cause.** It follows the kernel setting in 9 of 9 repeats and 3 of 3 profiled
   runs at 8B FP32 and 512 tokens, reverses sign at 8192, and is absent at W8A8 and 512 tokens. No cache, frequency or memory
   counters were collected (`PERF` was not set); candidates are listed in results 8.3.
4. **One node, one day, one model size.** The 8B cells were repeated nine times in one job; B7 and the first E5 job were
   different days, and `split` in this job is 0.3% below the B7 range for W8A8 at 512 and equal to it for FP32. No 1B engine
   cell and no two-socket engine cell was rerun.
5. **E5_ALTERNATE changes the launch order** (odd repeats `split` first, even repeats `group` first) relative to the first E5
   job, where `split` was always first. Paired comparisons are within a repeat. The first job's `split` repeat at 15.75 tok/s
   (FP32 512) did not recur.
6. **Profiled runs** are three per kernel and cell, with `LLAMA_PROFILE=1` and log level `info`; the profile adds timing calls
   to every phase, so the phase times are not the unprofiled step times (the profiled token rates agree with the timed ones
   within about 1%).

## E5.4: head-major K/V layout in the engine (2026-11-19/20)

**Status: run.** Results: [decode-attention-results.md](decode-attention-results.md) section 9. The decision rules (9.1) were
committed in a35b3b4 before any job was submitted. Branch `feature/kv-head-layout`, merged into `develop` after review as opt-in; made the engine default on 2026-11-20 by the owner's decision (V3 cell waived; see the results, section 9.7).

### Provenance

Node A is the node of B7, E4 and E5; node B is the partition's second Flat-mode node (same hardware, `flat-mode.json` in
each results directory).

| Item | Check job | 1B bench job | 8B bench job | V4 job |
| --- | --- | --- | --- | --- |
| Slurm job ID | 5876167 | 5876582 | 5876584 | 5876592 |
| State | COMPLETED, exit 0, 2 h 54 min | **FAILED**, exit 1, 1 h 14 min (decode reports complete, 48 prefill cells failed) | **FAILED**, exit 1, 5 h 16 min (same) | COMPLETED, exit 0, 26 min 31 s |
| Node (Flat mode, exclusive) | node A | node B | node A | node B |
| Commit | a35b3b4 | a35b3b4 | a35b3b4 | 2dd52c5 (only the campaign script and its test differ from a35b3b4) |
| Results directory (under the results store) | `kvlayout-check-20261119-1` | `kvlayout-bench-20261119-1b` | `kvlayout-bench-20261119-8b-2` | `kvlayout-v4-20261119-1` |
| Settings | `STAGE=check PHASE=E5 E5_KV_CHECK=1 SKIP_PRELUDE_HF=1` (common `make test` ran) | `STAGE=bench PHASE=E5 REPEATS=3 SKIP_TEST=1 SKIP_PRELUDE_HF=1 E5_KV_LAYOUTS="rows head" E5_ATTN=tiled E5_ALTERNATE=1 E5_PROFILED=1 E5_SKIP_MICRO=1 E5_PREFILL_PROMPTS="2048 8192"`, MODELS = 1B | same, MODELS = 8B | same layouts and prompts with `E5_SKIP_DECODE=1 E5_PROFILED=0`, MODELS = 1B and 8B |
| Result | 69 of 69 PASS | 192 decode reports completed; 48 prefill runs failed | 192 decode reports completed; 48 prefill runs failed | 96 prefill reports completed |

Failed jobs and cells, recorded as they happened:

- **5876582 and 5876584, V4 cells.** The 48 runs per job named `run-*-xkv{rows,head}-*` stopped with `[ERROR] cannot open prompt
  file '<OUT>/prompt-2048.txt'` (and 8192): the bench stage never wrote the prompt files (only the check stage did). Both jobs
  therefore ended with exit 1. Their decode data is valid and is what section 9.4 uses; their prefill reports are `status: error`
  and are not used. The V4 figures come from job 5876592 (commit 2dd52c5), which includes the fix (the bench stage writes the prompt
  files of `E5_PREFILL_PROMPTS`) and a dry-run test for it.
- **5876583** (the first 8B submission, node A, 3 s): `mv: cannot stat 'build/release/config.tmp'`. It started `make` in the staged
  tree `repo-a35b3b4` at the same moment as the 1B job; the Makefile's `config` rule races when two jobs build in one tree. It
  produced no data and left an empty results directory (`kvlayout-bench-20261119-8b`); 5876584 was submitted after the 1B job's
  build, as a no-op build, with a new directory. For parallel submissions from one tree: build once first, or skip the build when
  it is current.

The bench jobs ran with `LLAMA_ATTN=tiled` (`build.attention` of every decode and prefill report). The comparisons of the gates
are inside each job. The 1B and 8B decode figures come from two different nodes (node B and node A); no gate compares across them.

### Gates

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| C1 | `make test` on hardware | 0 failures in every C suite, 87 Python tests OK (job 5876167 `test.log`) | pass |
| C2 logits | rows against head, `--bitwise`, 1B/8B, FP32/W8A8, 1 and 2 sockets, decode `split` and `group`, tiled prefill, warm 8192 + short prompt, AMX rungs | 48 of 48 bitwise, max abs 0, argmax 100% | pass |
| C2 tokens | 128 greedy tokens from a warm 8192-token context identical | 16 of 16 | pass |
| C2 Hugging Face | FP32 with `LLAMA_KV_LAYOUT=head`, 1B/8B, one and two sockets | 4 of 4 exit 0 | pass |
| V1 | 1B 32768 one socket, (head, group) over (rows, split) at least 1.3x beyond variability | FP32 1.343x, W8A8 1.646x, ranges apart | pass |
| V2 | 1B W8A8 two-socket gain at 32768, head, best kernel, at least 1.5x | 1.869x (`split`), 1.860x (`group`) | pass |
| V3 | 512 not slower, 16 comparisons | 15 pass; 8B FP32 one socket `group`: head 15.60 [15.59-15.61] below rows minimum 15.61 | **fail** (1 of 16) |
| V4 | prefill TTFT not slower beyond variability, 8B `avx512_fp32` and `amx_bf16`, 2048 and 8192 | 0.997x, 0.976x, 0.991x, 0.921x | pass |
| Reported | other V4 cells | 3 of 12 slower beyond variability at 2048 tokens (1B AMX-BF16 1.016x, 1B AMX-INT8 1.022x, 8B AVX-512 Q8 1.004x) | recorded |

Local checks (macOS arm64, commit c0ca9b2 content; the final tree differs only by the later campaign-script and test commits):
`make test` exit 0 (run twice, the second on the final C and Python tree); `MODE=sanitize` `test-kernels` and `test-kernels-emul`
0 failures (run before the whitespace-only clang-format pass; the two binaries were rebuilt afterwards, not rerun); `MODE=tsan`
`test-kernels-emul` with `LLAMA_TEST_TEAMS=1,3` 0 failures and no ThreadSanitizer report. Mutation checks: breaking the head-major
store index (`k_kv_store`) failed the engine bitwise test (`test_tp.py`) and the kernel test; breaking the head base of
`k_attention_batch_kv` aborted `test-kernels-emul` on the NaN-poisoned padding. Both reverted. DRY_RUN output of B5/B6/B7, E5
(bench and check, default and with `E5_ALTERNATE=1 E5_PROFILED=2`), D2 and E4 is byte-identical to `develop` after the
change (compared with ROOT normalized).

### Gaps

1. **One profiled run per variant and cell**; attention times and bandwidths are single-run and **derived** (bytes from the
   shapes, mean context = warm context + 150).
2. **V3 fails in one cell by the rule as written** (a 0.01 tok/s difference); the cell is the known 8B FP32 one-socket `group`
   512 effect (weight-phase cost, section 8.3), which the layout does not touch.
3. **Three repeats per variant**; ranges of three values are a coarse variability measure (the 1B W8A8 two-socket 512 `split`
   rows range is 390-470 tok/s).
4. **AMX-level unit tests** ran on the hardware only inside `make test` (not locally: this host has no AMX).
5. **The V4 cells ran in a separate short job** (node B) from the decode cells of the 1B (node B) and 8B (node A) jobs; the layouts
   are compared inside each job.
6. **No placement report check** was part of the gates; `tools/check_placement.py` was changed to allow the one partly touched page
   per (layer, KV head) that the head layout leaves, and is tested, but no cluster report was run through it.
7. **First-touch effect**: with `head` the projection outputs go through a row buffer and the per-head store runs on the calling
   thread, so on the owned-placement paths the K/V pages are first written by that thread; the K/V store phase costs under
   0.04 ms per step and in all 480 completed reports of the three data jobs the KV pages were verified with none on another node (`kv.verified`,
   `kv.pages_wrong_node` 0), rows and head alike; first-touch cost itself was not compared.
