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
