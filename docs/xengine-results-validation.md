# Cross-engine comparison: validation record

**Status: run (2026-11-21).** Provenance, checks and the source of every number of [xengine-results.md](xengine-results.md). Plan:
[plans/xengine-comparison.md](plans/xengine-comparison.md). Raw data stays outside the repository in three campaign directories (small files only: JSON, CSV,
text; 9.2, 9.3 and 5.2 MB); they are named below by role and time stamp, and files by their path inside the campaign directory.

## Provenance

| Item | Value |
| --- | --- |
| Date (UTC) | 2026-11-21. Environment records start at 13:33 (node A, node B) and 12:57 (Cache node) |
| Slurm jobs | node A (Flat, HBM) 5892201, campaign directory `A-20261121-143313`; node B (Flat, DDR, plus the anchor cell) 5892202, `B-20261121-143314`; Cache node (Cache mode) 5892196, `cache-20261121-135743`. All three exited with code 0 (`exit-status.txt`) |
| Hardware | 2 x Intel Xeon Max 9480 (56 cores each, 112 CPUs, one thread per core, max 1.9 GHz in `lscpu`), RHEL 9.4, kernel 5.14.0-427.13. Flat nodes: 4 NUMA nodes, DDR nodes 0 and 1 (CPUs 0-55 and 56-111, about 500 GiB each), HBM nodes 2 and 3 (CPU-less, 64 GiB each; `flat-mode.json`). Cache node: 2 NUMA nodes with a 64 GiB memory-side cache each (`cache-mode.json`) |
| Harness commits | `6cf3b0f` (node A, node B; script `scripts/xengine.sbatch`, helpers `tools/xengine_cell.py`, `tools/xengine_vllm_client.py`); `af6b1b8` (Cache node: the same without the `LLAMACPP_2S_VARIANT` label, so its llama.cpp two-socket rows have an empty variant). The perf-clock rule of `9e810ed` is applied by the summariser to all three from the stored anchors (`tools/xengine_cell.py`, `perf_clock`) |
| llama3.x.c | engine sources of `a3402b6`: `git diff a3402b6 6cf3b0f -- src include` is empty and the Makefile differs by one line. Built in the job with gcc 11.4.1, `-O3` (`build-config-repo-*.txt`, `build-repo-*.log`); binary sha256 `40a89259...c537e2` identical in all three jobs (`environment.txt`) |
| llama.cpp | v1.0.0, build 12359, commit `b1b770d90`, gcc-toolset-14 (14.2.1; gcc-toolset-13 -O3 builds a binary that crashes in the AMX kernel, setup record), `GGML_NATIVE=ON`, OpenMP off. GGUF Q8_0 (8 540 776 128 B, 8.54 GB) converted and quantised by its own tools from Hugging Face revision `0e9e39f`, sha256 `896ad72a...ffcd6f0` |
| vLLM | 0.34.0+cpu, Apptainer image sha256 `3f5c8a7e...1f1b2` (torch 2.14.0+cpu), checkpoint RedHatAI Meta-Llama-3.1-8B-Instruct-quantized.w8a8 revision `024e24cb`; kernel `CPUInt8ScaledMMLinearKernel` (SGL AMX kernel for M > 4), bf16 LM head; server flags `--max-num-seqs 1 --no-enable-prefix-caching --max-model-len 8704 --kv-cache-memory-bytes 4G --dtype bfloat16 --seed 0`; patch sha256 `42efd365...66a6a` (the patched worker module, bound read-only into the container; used only by the `patched` configurations) |
| Tools | `perf` 5.14.0 (`perf stat -a --per-socket -I 200 -x,` on the Flat nodes, wrapper mode `perf stat -I 200 -x,` with user-mode events on the Cache node), Apptainer 1.4.0, Python stdlib only for the analysis |
| Prompts | `prompts.manifest.json`: short-86 (86 tokens), prompt-2048 (2048), prompt-8192 (8192), each with a BOS first and a stored id file used verbatim by llama3.x.c and vLLM; llama-bench times random tokens of the same length |
| Matrix | 3 repeats of every cell (repeat 1 perf group MEM, repeat 2 COMPUTE, repeat 3 none; Cache node: repeat 1 USER, repeats 2 and 3 none). Per node: llama3.x.c 1S and 2S x P, D, L; llama.cpp 1S and 2S x P, D, L, LP; vLLM one server per configuration x P, D, L; node A adds the vLLM stock + membind control (D only, 3 cells), node B the anchor cell (llama3.x.c, 1S, HBM, D, 3 cells). Cells: 63 + 63 + 60 = 186 |
| Settings | llama3.x.c: 56 threads per socket, `LLAMA_BACKEND=teams`, engine-owned placement, `numactl --cpunodebind=0` for one socket. llama.cpp: `-fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -lm none --repack 1 --cpu-strict 1`, 56 threads on one socket (`numactl --cpunodebind=0 --membind=<node>`), 64 threads (32 per socket) interleaved over the two HBM or DDR nodes on two sockets (layout `c2` of the tuning job 5892194) on the Flat nodes, 112 threads without interleave on the Cache node; 3 timed repetitions per invocation (`-r 3`) after one warm-up. vLLM: TP 1 or 2, 55 OpenMP threads per rank, the client on the spare core |
| Setup evidence | CAS calibration job 5892172; llama.cpp teardown check job 5892195; llama.cpp two-socket tuning job 5892194; vLLM and llama.cpp setup records (commands, summaries); each in the setup directory of the campaign |
| Analysis | `tools/xengine_summary.py` (commit 375d677 and later) builds `runs.csv` (264 rows, 3 campaign directories); `tools/xengine_tables.py` builds every table of the results document from it |

## Checks

All of these were recounted from the campaign directories for this record.

| Check | Criterion | Observed | Result |
| --- | --- | --- | --- |
| Cell status | every `cell.json` has `status: ok` | 63, 63 and 60 of 63, 63 and 60 | pass |
| Cell checks (`check.txt`) | every line `PASS` | 366 PASS and 0 FAIL (node A), 378 and 0 (node B), 336 and 0 (Cache node), one file per cell | pass |
| Phase results (`phase-results.txt`) | cells, sanity runs, prefix checks | 68 PASS lines and no other line (node A: 63 cells, 2 llama.cpp sanity runs, 3 vLLM prefix checks), 67 (node B: 63, 2, 2), 64 (Cache node: 60, 2, 2) | pass |
| Generated tokens equal G, EOS not hit | llama3.x.c: `generated tokens == G`; vLLM: `completion tokens == G`, `G token arrivals`; llama.cpp: `-n` of the test shape | in every cell's `check.txt` (llama3.x.c: 18, 21 and 18 cells on node A, node B and the Cache node; vLLM: 21, 18, 18; llama.cpp: the test shape of 24 cells per job) | pass |
| Prompt tokens | the engine reports the exact prompt length (`prompt tokens exact`) | 39, 39 and 36 cells (llama3.x.c and vLLM); llama.cpp reports the same `n_prompt` | pass |
| llama.cpp test shape | `test shape`, `threads`, `repetitions`, `strict CPU mask`, `weights loaded without mmap` | 24 cells per job, all PASS | pass |
| Prefix caching off (vLLM) | `vllm:prefix_cache_hits_total` = 0 after each server's runs (`prefix-check.txt`) | 0.0 for all 7 configurations (3 on node A, 2 on node B, 2 on the Cache node) | pass |
| No multiplexing | enabled/running percentage of every perf interval | the minimum over 64 706 (node A), 69 072 (node B) and 8 205 (Cache node, 9 `<not counted>` lines after process exit) intervals is 100.00%, and `min_pct_running` of every row of `runs.csv` is 100 | pass |
| Placement proof | `numa_maps.json` and `numastat.txt` per run, server workers per configuration, plus the CAS split | table below; the CAS split is section 3 of the results | pass, with the stock-vLLM control failing as intended |
| Memory mode | Flat: 4 NUMA nodes, two CPU-less 64 GiB nodes; Cache: 2 nodes, memory-side cache 68 719 476 736 B per node | `flat-mode.json`, `cache-mode.json` | pass |
| llama.cpp output sanity | `llama-completion`, 32 greedy tokens, 6 configurations | all six stdouts are byte-identical (md5 `784b0fc3...`) and coherent | pass |
| AMX proof | `EXE.AMX_BUSY` counted in each prefill window; `AMX_OPS_RETIRED.INT8` also counted | AMX_BUSY > 0 in all 34 prefill windows of repeat 2 on the Flat nodes (17 + 17) and in all 16 user-mode windows of the Cache node; AMX_OPS_RETIRED.INT8 is 0 in every window, including those with AMX busy, so it is uninformative | pass (AMX_OPS unusable) |
| Anchor | llama3.x.c, 1S, HBM, D on both Flat nodes | decode ratio A/B 1.004, prefill 0.997 | pass |

Placement evidence of repeat 1 of the D runs (`numa_maps.json`, resident pages by node, anonymous and file mappings together) and of the
vLLM server workers (`numa_maps-<pid>.json` in `vllm-config-*`; the two ranks of a two-socket server are numbered 0 and 1). The llama3.x.c rows also
carry the engine's own region verification (`check.txt`: weights, KV, packed and work regions on the expected node, 0 mismatched pages, the embedding on DDR node 0).

| Node | Run (repeat 1, D) or server worker | GiB on NUMA node 0 / 1 / 2 / 3 |
| --- | --- | ---: |
| node A | c-s1-hbm-D-r1 | 0.54 / 0.00 / 14.02 / 0.00 |
| node A | c-s2-hbm-D-r1 | 0.54 / 0.02 / 7.05 / 7.05 |
| node A | llamacpp-s1-hbm-D-r1 | 0.00 / 0.00 / 8.97 / 0.00 |
| node A | llamacpp-s2-hbm-c2-D-r1 | 0.00 / 0.00 / 4.48 / 4.48 |
| node A | vllm config s1-hbm-patched rank 0 | 0.25 / 0.00 / 18.13 / 0.00 |
| node A | vllm config s1-hbm-stock-membind rank 0 | 15.59 / 0.00 / 0.25 / 0.00 |
| node A | vllm config s2-hbm-patched rank 0 | 0.08 / 0.17 / 11.32 / 0.25 |
| node A | vllm config s2-hbm-patched rank 1 | 0.08 / 0.17 / 0.24 / 11.35 |
| node B | c-s1-ddr-D-r1 | 14.54 / 0.00 / 0.00 / 0.00 |
| node B | c-s1-hbm-D-r1 | 0.54 / 0.00 / 14.03 / 0.00 |
| node B | c-s2-ddr-D-r1 | 7.58 / 7.06 / 0.00 / 0.00 |
| node B | llamacpp-s1-ddr-D-r1 | 8.97 / 0.00 / 0.00 / 0.00 |
| node B | llamacpp-s2-ddr-c2-D-r1 | 4.49 / 4.49 / 0.00 / 0.00 |
| node B | vllm config s1-ddr rank 0 | 17.32 / 0.00 / 0.00 / 0.00 |
| node B | vllm config s2-ddr rank 0 | 11.37 / 0.48 / 0.00 / 0.00 |
| node B | vllm config s2-ddr rank 1 | 0.27 / 11.59 / 0.00 / 0.00 |
| Cache node | c-s1-cache-D-r1 | 13.13 / 0.00 / 0.00 / 0.00 |
| Cache node | c-s2-cache-D-r1 | 7.60 / 7.08 / 0.00 / 0.00 |
| Cache node | llamacpp-s1-cache-D-r1 | 8.97 / 0.00 / 0.00 / 0.00 |
| Cache node | llamacpp-s2-cache-D-r1 | 6.88 / 2.10 / 0.00 / 0.00 |
| Cache node | vllm config s1-cache rank 0 | 17.34 / 0.00 / 0.00 / 0.00 |
| Cache node | vllm config s2-cache rank 0 | 11.41 / 0.44 / 0.00 / 0.00 |
| Cache node | vllm config s2-cache rank 1 | 0.30 / 11.55 / 0.00 / 0.00 |

## Counter windows

A phase window is cut from the run's `perf.csv` with the wall-clock markers of `markers.json` and the perf clock of the run (derivation in the docstring of
`tools/xengine_cell.py`; interval edges are prorated). Marker uncertainty is about 0.1 s. `window_cover` is the covered fraction of the window; `perf_clock_skew_s` is the difference between the launch-based and the
end-aligned estimate of the perf clock (on the Flat nodes the first perf interval starts 0.10-0.25 s after the job's perf-start anchor, the start-up time of perf, which the end alignment removes; on the Cache node, in wrapper mode, perf time 0 is the launch and the
skew reflects the shutdown of perf after the child exits, up to 0.71 s for one 2048-token prefill).

| Node | rows with perf | window_cover min | rows below 0.98 | perf_clock_skew_s range | min_pct_running |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 60 | 1.000 | 0 | -0.230 to -0.107 | 100 |
| B | 60 | 1.000 | 0 | -0.252 to -0.103 | 100 |
| Cache | 28 | 0.914 | 7 | -0.233 to 0.714 | 100 |

The seven Cache-node rows below 0.98 coverage (0.914-0.978) are `c` 1S P and 2S P and D decode, and llama.cpp 1S D, L, P and 2S P, all repeat 1; only AMX-busy counts of the user-mode group are taken from them, understated by up to 9% in the worst case.

## Provenance of the numbers

Paths are relative to a campaign directory; `<label>` is the cell label (`c-s1-hbm-D-r1`, `llamacpp-s2-hbm-c2-D-r1`, `vllm-s1-hbm-patched-P-r2`, and so on). Cells are medians (min-max) of the 3 repeats (`tools/xengine_tables.py`).

| Number | Source | Field or event | Derivation | Label |
| --- | --- | --- | --- | --- |
| llama3.x.c prefill tok/s, TTFT; decode tok/s, TPOT | `runs/<label>/report.json` | `throughput_tokens_per_second.prefill` and `.decode`; `timing_seconds.time_to_first_token`, `.average_decode_step`; `prefill.tokens` | as reported; uniform TPOT = (`total_inference` - `time_to_first_token`) / `decode_steps` (127) | measured; uniform derived |
| llama.cpp prefill, decode | `runs/<label>/report.json` (first entry of `llama-bench -o json`) | `avg_ts`, `avg_ns`, `n_prompt`, `n_gen`, `n_depth`, `samples_ts` | pp: tok/s = `avg_ts`, TTFT = `n_prompt` / `avg_ts`; tg: tok/s = `avg_ts`, TPOT = 1 / `avg_ts`; each value is the mean of 3 timed repetitions inside one invocation, the table median is over the 3 invocations | measured; TTFT, TPOT derived |
| vLLM prefill, decode | `runs/<label>/report.json` (from `client.jsonl`) | `t_first_token`, `token_arrivals`, `usage_prompt_tokens`, `usage_completion_tokens`, `t_end` | TTFT = `t_first_token`; prefill tok/s = prompt tokens / TTFT; decode tok/s = (arrivals - 1) / (last - first arrival); uniform TPOT = (`t_end` - TTFT) / (arrivals - 1) | measured; rates derived |
| 8192-token prefill | L cells of llama3.x.c and vLLM (prefill phase); LP cells of llama.cpp | as above | none | measured |
| Long-context decode | L cells (decode phase) | as above | none | measured |
| 2S/1S, HBM/DDR, Cache/Flat ratios | medians of the cells above | | ratio of medians; anchor-corrected ratio = ratio / (node A over node B of the anchor cell) | derived |
| GB/s per tier and socket | `runs/<label>/perf.csv` (repeat 1) | `ddr_cas_rd` (`uncore_imc`, event 0x05 umask 0xcf), `hbm_cas_rd` (`uncore_type_14`, same encoding), `-I 200` per socket | window sum x bytes per count / window length; DDR 64 B, HBM **32 B** (calibrated, below) | measured counts, derived bytes |
| GB per token | same | | (DDR + HBM bytes of both sockets in the decode window) / decoded tokens (127; llama.cpp 128) | derived |
| % of roofline | | | GB/s / (sockets used x 552 or 262 GB/s) | derived |
| Nominal GB per token | model sizes | | 7 974 764 544 B for the Q8_0 engines (stored region), 8.03 GB for vLLM (6.979e9 int8 + 1.051e9 bf16 bytes: 32 x (2 x 4096^2 + 2 x 4096 x 1024 + 3 x 4096 x 14336) = 6.979e9 layer weights, 128256 x 4096 = 5.253e8 LM-head weights) | derived |
| KV bytes at depth 8192 | architecture | | 8192 + 64 positions x 32 layers x 2 x 1024 x element size (4, 2, 2 B) | derived |
| UPI GB per token | `perf.csv` (repeat 1) | `unc_upi_rxl_flits.all_data` (and `txl`), per socket | flits x 64/9 B (PCM convention; consistent with the 50% remote share of llama.cpp), receive summed over both sockets / decoded tokens | measured flits, derived bytes |
| AMX-busy cycles, share, IPC, GHz | `perf.csv` (repeat 2; Cache node repeat 1, user mode) | `EXE.AMX_BUSY`, `cycles`, `instructions`, `ref-cycles` | sums over sockets in the window; share = AMX_BUSY / cycles; GHz = cycles / ref-cycles x 1.9 | measured; ratios derived |
| Ideal AMX cycles per token | architecture | | 6.979e9 MACs / 1024 MACs per AMX-busy cycle = 6.82 M (assumption: 16x16x64 int8 tile product per 16 cycles) | derived |
| Power per phase | `perf.csv` (repeats 1 and 2) | `power/energy-pkg/`, `power/energy-ram/` | joules per window / window; two repeats, median | measured |
| J/token, net | same, `idle-baseline.perf.csv` | | (package + DRAM joules of both sockets) / tokens; net subtracts the idle power of the same job x window (idle: 10 s at job start) | derived |
| Perf overhead | cells of repeats 1, 2, 3 | | tok/s of the perf repeat / tok/s of repeat 3 for the same cell, summarised per tier and phase | derived |
| Placement, tier shares | `runs/<label>/numa_maps.json`, `vllm-config-*/numa_maps-*.json`, `numastat-*.txt` | `bytes_by_node` | none | measured |
| Tuning (llama.cpp two sockets) | setup record job 5892194 | `tune2s/summary.txt` | none | measured |

Bytes per `hbm_cas_rd` count. The summariser's default is 32 B, set by `--calibrate` on the decode of llama3.x.c, one socket, D (check job 5892172: DDR at 64 B per count 7.86 GB per token, HBM at 32 B 7.83 GB, at 64 B 15.66 GB, against 7.97 GB of weights). In this campaign the same
test, `tools/xengine_summary.py ... --calibrate`, gives (repeat 1 of the D decode, one socket): HBM 15.418 / 15.609 GB per token at 64 B (x 1.935 / 1.959 nominal) and 7.709 / 7.805 GB at 32 B (0.968 / 0.980), DDR 7.850 GB (0.987 / 0.986): the 32 B choice stands. The three engines also agree on 32 B with
the DDR tier for the same model: HBM over DDR GB per token 0.982 (llama3.x.c), 0.995 (llama.cpp) and 0.991 (vLLM) on one socket. The UPI factor (64/9 B per data flit) is a documented assumption; the llama.cpp interleave gives it an independent check (4.44 GB per token of UPI data for 8.82 GB read, 50.3%, where 50% is the geometric expectation of interleaving over two nodes).

## Spot checks recomputed from the raw files

The first five rows are computed directly from the report files with a separate script. The rows on counters and energy marked (re-integrated) were recomputed from `perf.csv` and `markers.json` with an independent re-implementation of the window integration (prorated intervals, the same marker rule); the others are values of the `runs.csv` row:

| Quantity | Raw values (repeats 1, 2, 3) | Median | Table |
| --- | --- | --- | --- |
| llama3.x.c, 1S, HBM, D decode tok/s (node A) | 52.5836, 51.8287, 51.8727 (`throughput_tokens_per_second.decode`) | 51.87 | 51.9 (headline) |
| llama3.x.c, 1S, HBM, P TTFT s | 2.21395, 2.26645, 2.25844 (`time_to_first_token`) | 2.258 | 2.258 |
| llama.cpp, 1S, HBM, D decode tok/s | 40.5767, 40.5236, 40.8681 (`avg_ts`; `n_gen` 128, `n_depth` 86, 3 samples each) | 40.58 | 40.6 |
| vLLM patched, 1S, HBM, D decode tok/s | 40.5147, 40.3171, 40.6250 ((128 - 1) / (last - first arrival)); `usage_completion_tokens` 128, prompt tokens 86 | 40.51 | 40.5 |
| vLLM patched, 1S, HBM, P TTFT s | 0.61145, 0.63281, 0.61779 (`t_first_token`); prompt tokens 2048, completion 1 | 0.618 | 0.618 |
| llama3.x.c, 1S, HBM, D, repeat 1 | 979.9 GB read in a 2.4473 s window (HBM counts x 32 B on socket 0: 978.8 GB; DDR 0.7 GB), 127 decoded tokens | 7.716 GB per token, 400.0 GB/s (re-integrated: 7.7158) | 7.72, 400 |
| vLLM stock + membind, D, repeat 1 | DDR 1009.7 GB, HBM 0.76 GB on socket 0 in a 5.1325 s window | 197.0 GB/s read, 99.9% DDR | 196.8 DDR, 0.2 HBM |
| llama.cpp, 2S, HBM, D, repeat 1 | HBM 563.1 + 563.5 GB, UPI receive 314.2 + 254.4 GB in a 5.3529 s window, 128 tokens | 4.44 GB per token of UPI data (re-integrated: 4.4418) | 4.442 |
| llama3.x.c, 1S, HBM, P, repeat 2 | AMX_BUSY 1.3487e10 cycles of 2.6206e11 unhalted core cycles in a 2.2059 s window | 5.15% | 5.1% (median of the one run) |
| llama3.x.c, 1S, HBM, D, repeat 1 | package 359.4 W and 161.3 W, DRAM 22.8 W and 20.8 W, 1380.8 J in the window, 127 tokens | 10.87 J/token (re-integrated: 10.872; 4.78 net) | 10.96 and 4.82 (median of repeats 1 and 2) |

## Independent verification

A separate verifier re-derived the results from the raw per-run files of the three campaign directories with its own scripts (not `runs.csv`, not these tables). Result: **verified**, with four wording corrections (applied).
- All 18 headline cells, both metrics (36 values): maximum deviation 0.17% (llama.cpp 1S DDR decode), none above 1%.
- About 18 counter-derived cells re-integrated from `perf.csv` and the anchors (bandwidth, GB per token, UPI, AMX share, J per token of the llama3.x.c one-socket decode): all reproduce; the 23 placement rows reproduce to 0.01 GiB; the control cell's server log confirms vLLM re-binds to its CPUs' node.
- Exactness of all 186 runs (token counts, prompt lengths, no stop token, HTTP 200, 7 prefix-cache checks at 0.0), the check and phase-result counts, and the absence of multiplexing: identical.
- Corrections applied: "about 4 ms" consistently; basis of "3.6x" (tok/s; 3.65x and 3.71x by TTFT); the UPI factor is "consistent with", not "confirmed"; the llama.cpp 2S Cache cell and the TTFT definitions noted under the headline; job ids in the captions; the nominal-bytes comparison of vLLM reworded.
- **Not re-derived:** the CAS calibration job 5892172 and the tuning job 5892194 (read as setup records only), the idle-baseline netting, the power ranges (339-366 W), the long-context (L) and 8192-token prefill tables, and the Cache-node window coverage (0.914 minimum). The byte factors (64 B, 32 B, 64/9 B) were taken from this record.
- Follow-up job 5892206 (llama.cpp two sockets, 64, 48, 40, 32 threads): 64 stays best (see the results document, section 4).

## How to reproduce

```bash
tools/xengine_summary.py <A> <B> <Cache> -o runs.csv          # 264 rows (90 + 90 + 84)
tools/xengine_summary.py <A> <B> <Cache> --calibrate          # CAS calibration, 3 runs
tools/xengine_tables.py runs.csv                              # every table of the results document (T0 ... T11)
python3 tests/test_xengine.py                                 # 28 tests
```

`<A>`, `<B>` and `<Cache>` are the three campaign directories. The tables of the results document are the output of the third command, pasted without editing (generated with `tools/xengine_tables.py` from the commit of this record).

## Deviations from the plan and known limits

- **vLLM on HBM is a patched variant** (decision D5); stock vLLM with `numactl --membind` reached DDR (control cell). The 2S-DDR cases are stock.
- **L for llama.cpp** is `-d 8192` with untimed depth prefill, for the others an 8192-token prompt with the decode phase cut by the markers; the plan's "minus the 8192-prompt output-1 run" subtraction for vLLM is not needed because the phases are separated by the first-token time.
- **The 8192-token prefill** was kept (LP cells for llama.cpp, the prefill phase of L for the others), which the plan allowed to drop.
- **Uniform TPOT** was computed within a run; the cross-run definition needs G = 1 runs that the matrix does not contain.
- **llama.cpp's thread sweep** (plan section 5) was done in the setup (56, 48, 40, 32 threads on one socket: 41.5, 40.4, 40.3, 38.9 tok/s); 56 was used. The two-socket layout was tuned in job 5892194 (8 layouts) and chosen from decode and prefill tok/s there. The Cache node did not get the tuning (untuned two-socket layout).
- **C engine on the Cache node** runs with `LLAMA_PLACEMENT=ddr` (no HBM node exists), un-pinned memory-side caching does the rest.
- **Counter windows** are 1.3-9.0 s for decode and 0.37-33 s for prefill (8192-token prefill up to 33 s); the shortest (llama3.x.c, two sockets, D) carry about 0.1 s of edge uncertainty. The 86-token prefill rows (D prefill, 0.06-0.17 s) are indicative only and unused.
- **The Cache node has no uncore and RAPL counters** (privileged counters unavailable, decision D2); only user-mode `cycles`, `instructions` and `EXE.AMX_BUSY` of repeat 1 exist. It is a different physical node from node A and node B with no anchor cell.
- **Single repeat counters.** CAS, UPI and power per cell come from one or two runs, not three; their repeatability was not measured here. The MEM group is only in repeat 1.
- **llama3.x.c, llama.cpp and vLLM greedy texts** of the campaign were checked only for llama.cpp (the six sanity runs) and, for vLLM, by the continuation of the D warm-up request in every configuration; the C engine's output was checked in the setup job on the earlier tree, not in the campaign (`--no-text` in all benchmark cells).
- **EOS** is ignored or proven not hit in every cell (`generated tokens == G`).
- **Earlier documents with 64 B per `hbm_cas_rd`** were not edited (results document, section 3).
