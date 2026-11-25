# 70B TTFT diagnosis: validation record

Companion of [ttft-70b-diagnosis.md](ttft-70b-diagnosis.md). No host names or cluster paths: a "results directory" is the per-job directory the job script creates on the cluster's results file system.
Every figure of the main page is traced here to a job and a raw file, or marked derived with its formula.

## 1. Jobs and commits

| job | what | tree | result | elapsed |
|---|---|---|---|---|
| 5935737 | stagger replication, 6 cells (8B and 70B, gs64 and gs32), second Flat node; also the prompt text round trip | develop 59f6bc6 (staged tracked files) | all cells rc 0; the round trip failed (below) | 1:37:23 |
| 5935738 | first TTFT job (profile through the command line with the prompts as text) | | held, cancelled before it ran: the round trip of 5935737 showed 972 against 547 and 2138 against 2113 prompt ids (the chat template adds a system prefix the fixtures lack) | 0 |
| 5935864 | **TTFT diagnosis** (`scripts/ttft-diagnosis.sbatch`), second Flat node | develop 59f6bc6 + `tools/xengine70/native_profile.c` built through an extra makefile fragment | 15 runs, all rc 0 | 0:13:07 |

Node time: 1.62 h + 0.22 h = 1.84 node-hours of the 6 allowed (the gate job that ran first, 5935721, belongs to another work item). The engine sources (`src/`, `include/`, `Makefile`) are those of 59f6bc6 unchanged (`source.sha256` in each results directory);
the one added source file, `native_profile.c`, and the binaries have their sha256 in `executables.sha256` of the TTFT directory. Branch `docs/ttft-diagnosis` (from develop cb28b94) adds `scripts/ttft-diagnosis.sbatch`, `tools/ttft_diagnosis_summary.py`, `tools/xengine70/native_profile.c` and the documents; nothing is merged or pushed.

## 2. Setup

* Environment of each run (job script `engine_env`): `LLAMA_SOCKETS=2 LLAMA_THREADS=55 LLAMA_PREFILL_BATCH=B LLAMA_PLACEMENT=hybrid LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384 LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=N LLAMA_HYBRID_KV_HBM_LAYERS=N LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=hbm|ddr`
  (N = 80 for the 70B, 32 for the 8B; no packed knob for the AVX-512 run, where the engine rejects it), the process pinned to cpus 0-54 and 56-110; the caller's other `LLAMA_*` cleared. Context 8192. Prompt fixtures `tools/xengine70/prompts/{short,long}.ids.json` (547 and 2113 ids; sha256 in [xengine-70b.md](xengine-70b.md)).
* Checkpoints: the 70B and 8B gs64 `int8-v3.bin` of the campaign. Plain driver: `xengine70-q8` (`tools/xengine70/native_c.c`). Profiled driver: `native_profile.c`, one warm-up and three measured requests per prompt, 8 greedy tokens.
* perf: `perf stat -a --per-socket -I 50 -x,`; events of the plain driver: DDR and HBM CAS read and write counts, UPI data flits (`perf_ok_mem=1`); of the profiled driver: `cycles,ref-cycles,instructions,EXE.AMX_BUSY` (`perf_ok_cmp=1`). CAS bytes: 64 (DDR) and 32 (HBM). `perf-t0` holds the epoch of the counters' start.
* Window of a request: `[start_epoch_ns, start_epoch_ns + ttft_s]`; interval counts are prorated by overlap; window coverage 1.000 in every profiled window of the summary.

## 3. Raw files (in the TTFT results directory)

`runs.log` (rc of each run), `environment.txt`, `source.sha256`, `executables.sha256`, `build.log`, `plan-70b-b256.json`, `plan-70b-b1024.json`; per run `drv-LABEL.{stdout,stderr,perf.csv,perf-t0,launch,end,numastat}` (stdout: one JSON line per request) and
`prof-LABEL.{stdout,stderr,perf.csv,perf-t0,launch,end}`; `idle.perf.csv` (5 s baseline). Labels: `8b-{hbm,ddr}-b{256,1024}`, `70b-ddr-b{256,1024}`, `70b-b256-avx512` (profiled only). Tables: `tools/ttft_diagnosis_summary.py DIRECTORY` (stdlib; needs `tools/xengine_summary.py`).
The stagger job 5935737 results are in its own directory (`gs64-8b-1s` ... `gs32-70b-2s`, `rc.txt`, `prompts/roundtrip.txt`).

## 4. Per-request values (TTFT seconds, request 0 is the warm-up)

| run | prompt | requests 0-3 |
|---|---:|---|
| drv-70b-ddr-b256 | 547 | 2.150 2.115 2.098 2.106 |
| drv-70b-ddr-b256 | 2113 | 7.909 7.921 7.915 7.915 |
| drv-70b-ddr-b1024 | 547 | 1.778 1.614 1.599 1.609 |
| drv-70b-ddr-b1024 | 2113 | 5.588 5.602 5.603 5.623 |
| prof-70b-ddr-b256 | 547 | 2.183 2.103 2.116 2.099 |
| prof-70b-ddr-b256 | 2113 | 7.914 7.946 7.922 7.959 |
| prof-70b-ddr-b1024 | 547 | 1.596 1.578 1.578 **13.005** |
| prof-70b-ddr-b1024 | 2113 | 5.709 5.729 5.776 5.773 |
| prof-70b-b256-avx512 | 547 | 22.290 9.884 9.944 9.910 |
| drv-8b-hbm-b256 | 547 / 2113 | 0.261 0.250 0.228 0.232 / 0.849 0.833 0.866 0.851 |
| drv-8b-ddr-b256 | 547 / 2113 | 0.238 0.237 0.232 0.226 / 0.860 0.870 0.847 0.845 |
| drv-8b-hbm-b1024 | 547 / 2113 | 0.185 0.184 0.176 0.174 / 0.716 0.713 0.724 0.710 |
| drv-8b-ddr-b1024 | 547 / 2113 | 0.187 0.174 0.177 0.187 / 0.723 0.719 0.751 0.717 |
| prof-8b-hbm-b256 / ddr-b256 | 2113 | 0.791 0.802 0.790 0.781 / 1.018 0.872 0.887 0.878 |
| prof-8b-hbm-b1024 / ddr-b1024 | 2113 | 0.778 0.767 0.766 0.765 / 0.735 0.714 0.713 0.715 |

Median and range of the main page are over requests 1-3. The 13.005 s request is the fourth request of the short prompt in `prof-70b-ddr-b1024`; the medians of phase times over requests 1-3 are used throughout, so it enters only the range.

## 5. Derived values and formulas

* Ideal times: GEMM operations `2 x MACs per token and layer x layers x rows` = 289.3 TOP (2113 rows) and 74.9 TOP (547), attention `4 x (P^2/2) x heads x head_dim x layers` = 5.85 and 0.39 TOP ([plans/hybrid-limits.md](plans/hybrid-limits.md) section 5.1); divided by 450 TOP/s: 0.643 s and 0.013 s (2113), 0.166 s and 0.001 s (547).
  Per-matrix MACs per token and layer for the 70B: qkv 83.9 M, Wo 67.1 M, W1+W3 469.8 M, W2 234.9 M (sum 855.6 M, x 80 = 68.45 G).
* GEMM rate per matrix = matrix operations of the prompt / profiled phase seconds; GEMM share = (qkv + Wo + W1/W3 + W2 + classifier) / TTFT of the profiled run; share of peak = rate / 450.
* Measured / peak ceiling = plain-driver median TTFT / (0.656 or 0.167).
* DDR read ratio = DDR read GB per request / (batches x 72.729 GB); 72.729 GB = `packed_weights_peak` of the job 5934864 profiled report (both sockets, 36.365 GB per socket; the `ddr packed` arena of both plans is 36,364,615,680 B per socket); 8B packed copy 7.416 GB (engine log of job 5935737, 3,707,764,736 per socket).
* The two-parameter fit of main page section 4: the normal equations of `TTFT = c x batches + r x rows` over the four plain-driver medians (9, 2113, 7.915), (3, 547, 2.106), (3, 2113, 5.603), (1, 547, 1.609): c = 0.3693 s, r = 2.1437 ms; the per-phase fits use the profiled medians of main page section 3 the same way.
* Stream rate for comparison: 2 x 262.5 GB/s DDR (P1 E1 0+56 per socket, `tools/hybrid_limits.py` constant `DDR_STREAM`).
* Time series quoted in main page section 5 (DDR GB/s per 0.25-0.5 s bins of one request, both sockets): the same window method on the third request of the long (and short) prompt of `drv-70b-ddr-b256` and `drv-70b-ddr-b1024`; one-off analysis from the raw perf files, not part of the summary tool.

## 6. Checks

| check | result |
|---|---|
| perf events counted without multiplexing | `min % running` 100 in every profiled-driver window; the uncore events of the plain driver are separate PMUs |
| perf clock alignment | first-line method; the DDR rate steps at the request boundaries in the time series |
| same ids in driver and profiled driver | both read `short.ids.txt` and `long.ids.txt` generated from the fixtures; `prompt_tokens` 547 and 2113 in every request line; first generated token 8586 in every profiled request of the 2113 prompt |
| packed copy placement | `hybrid plan: 80 of 80 weight layers and 80 KV layers in HBM, classifier=hbm, packed=ddr` in the 70B runs' stderr; 8B `packed=hbm` and `packed=ddr`; both plans admitted |
| non-AMX control | `prof-70b-b256-avx512`: `EXE.AMX_BUSY` 0.00% of cycles, 8.1 TOP/s |
| AMX against profile | AMX busy share and profile GEMM share of peak: 8.95% / 9.1% (M = 256), 12.35% / 12.8% (M = 1024) at 2113 tokens |
| engine gate | not run (no engine file changed); the tree is the one whose `make test` is the gate job 5935721 |

## 7. What this record does not cover

No repetition of the job (one process per cell); one node; no gs32 TTFT; no M = 512; the cause of the W1/W3 behaviour and of the DDR re-reads is untested (main page section 6); the 13 s request and the 6-7% DDR write traffic are unexplained; the exchange component totals are quoted from the engine's log, not tabulated.
