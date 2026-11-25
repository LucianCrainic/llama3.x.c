# 70B TTFT diagnosis: where the time goes at P = 2113 and 547

Measurement only: no engine code or default was changed. Numbers are labelled measured (job id and raw file in [ttft-70b-diagnosis-validation.md](ttft-70b-diagnosis-validation.md)) or derived (formula given).
Question from [plans/hybrid-limits.md](plans/hybrid-limits.md) sections 5.4-5.5 and 7: the 70B 2S hybrid 80/80 TTFT is 12-16 times the AMX-INT8 peak ceiling, with a residual of 0.16-0.20 s per prefill batch (M = 256).
Setup of every cell: two sockets of 55 threads, `LLAMA_PLACEMENT=hybrid` with weights and K/V in HBM and the AMX packed copy in DDR (`LLAMA_HYBRID_PACKED=ddr`), gs64 checkpoint, engine defaults otherwise
(AMX-INT8 prefill, AMX attention), context 8192, develop 59f6bc6; the same node (second Flat node) for everything, one job (5935864, 13 min).

## 1. Answer

1. **The batch size explains most of the "gap".** The xengine rows used M = 256; the engine's own default is 1024. Measured, the same engine, node and prompts:

| 70B hybrid 80/80, gs64 | prompt | M = 256 TTFT s | M = 1024 TTFT s | change | AMX peak ceiling s | measured / peak at 256 | at 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 warm-up + 3 measured, median [min-max] | 547 | 2.106 [2.098-2.115] | **1.609** [1.599-1.614] | -23.6% | 0.167 | 12.6 | **9.6** |
| | 2113 | 7.915 [7.915-7.921] | **5.603** [5.602-5.623] | -29.2% | 0.656 | 12.1 | **8.5** |

2. **About 89% of the M = 256 TTFT is the AMX GEMM, at 9.1% of peak (40.8 TOP/s of 450); 6% is attention; 4% is the small phases (quantize, norms, RoPE, K/V store); the exchange and everything else is under 1%** (profiled prefill, section 3). At M = 1024 the GEMM
   is 87% of the time at 12.8% of peak (57.5 TOP/s). The GEMM efficiency (hypothesis 1 of section 5.5) is the dominant term; measured with the core counters, `EXE.AMX_BUSY` is 8.95% (M = 256) and 12.35% (M = 1024) of unhalted core cycles, the same figures as the profile (9.1%, 12.8%), so the GEMM rate and the AMX unit's busy share agree.
3. **The per-batch "fixed cost" is not a fixed cost and not spread over the phases: it is in the fused W1/W3 (SwiGLU) matrix.** Between M = 256 and M = 1024 at 2113 tokens W1/W3 goes from 4.518 s to 2.387 s (35.2 to 66.5 TOP/s) and accounts for 2.13 s of the 2.17 s difference; qkv, Wo, W2, attention and the classifier are the same within 6% (section 4).
   A two-parameter fit of TTFT to batches and rows (four points, section 4) gives 0.37 s per batch and 2.14 ms per prompt row; the per-batch part of the profile fit is 0.34 s, all in W1/W3 (the other phases: -0.01 to +0.02 s per batch).
4. **The packed copy is read from DDR 2.4 times per M = 256 batch, not once** (measured with the uncore CAS counters over the prefill windows): 1559 GB for a 2113-token request against 9 batches x 72.7 GB = 655 GB (ratio 2.38), at a steady 197-218 GB/s (37-41% of the 525 GB/s two-socket stream rate); at M = 1024 it is 348 GB against 3 x 72.7 GB (ratio 1.60) at 43 GB/s on the 1024-row batches. The 8B reads its packed copy exactly once per batch (ratio 1.00) and does not
   care where it is (8B TTFT with the packed copy in HBM and in DDR: 0.851 and 0.847 s at M = 256, 0.713 and 0.719 s at M = 1024). **The single-pass DDR read floor of the limits page (1.27 s at 2113 tokens) is therefore not the right floor for the 70B; the read traffic is 2.4 times larger,
   and it coincides with the one phase whose rate depends on the batch size.** Whether the re-reads are the cause of W1/W3's slowness or an effect of it is not separable with this data (section 6).
5. Non-AMX reference: AVX-512 prefill kernels, P = 547, M = 256: **9.91 s** (9.884-9.944; the warm-up 22.3 s), 4.7 times the AMX TTFT at the same batch and 6.2 times at M = 1024; GEMM 8.1 TOP/s (1.8% of the AMX peak), `EXE.AMX_BUSY` 0.00%.

## 2. What was run

Job 5935864 (hbm14, `scripts/ttft-diagnosis.sbatch`), one process per row, `LLAMA_*` of the caller cleared, `LLAMA_PREFILL_BATCH` equal to the driver's batch argument:

* **Plain driver** (`tools/xengine70`, `xengine70-q8`): the two prompt fixtures of [xengine-70b.md](xengine-70b.md) as exact id arrays (547 and 2113 tokens, each the whole prompt including the chat header), 1 warm-up + 3 measured requests each, 128 greedy tokens after the first; TTFT is the driver's own `ttft_s`. `perf stat -a --per-socket -I 50` counted the uncore events (DDR and HBM CAS reads and
  writes, UPI data flits) over the whole process; each request's prefill window is `[start_epoch_ns, start_epoch_ns + ttft_s]`, with the perf clock fixed by the epoch of the first interval line minus its time stamp (accuracy about 10 ms of 50 ms intervals; edge intervals are prorated).
* **Profiled driver** (`tools/xengine70/native_profile.c`, new, measurement tool only): the same requests (same ids, same call sequence) with the engine's phase profile (the `LLAMA_PROFILE` machinery, switched on with `engine_set_profiling`) reset before and read after each prefill, 8 greedy tokens; it is built in the job with an extra makefile fragment, so no engine file or the Makefile changed. perf group: `cycles, ref-cycles, instructions, EXE.AMX_BUSY`
  (no multiplexing: 100% running in every window). The profile adds two clock reads per phase call; the profiled TTFT is within 0.4% of the plain at M = 256 (7.946 against 7.915 s at 2113 tokens) and 3% at M = 1024 (5.773 against 5.603 s).
* Cells: 70B at M = 256 and 1024; 8B (2S, `LLAMA_HYBRID_*_LAYERS=32`, classifier HBM) with the packed copy in HBM and in DDR at M = 256 and 1024; 70B AVX-512 at P = 547.
* A first plan to run the profile through the command line with the prompts as text failed its own check (job 5935737, prompts/roundtrip.txt: 972 against 547 and 2138 against 2113 ids; the engine's chat template adds the system prefix that the fixtures do not have). The held job 5935738 was cancelled before it ran; nothing from the command-line route is used.
* Plan budget (`--plan-only`): `LLAMA_PREFILL_BATCH=1024` is accepted for the 70B (admitted). Per socket the work arena grows 94.5 to 223.7 MB and the exchange 25.2 to 100.7 MB (both HBM), HBM charged 41.07 to 41.28 GB of the 60.13 GB budget (+205 MB); DDR (the packed copy, 36.36 GB per socket) is unchanged.

## 3. TTFT attribution (profiled driver, 70B, medians of 3 measured requests)

All values measured except the derived columns. "small" = embedding, RoPE table, RMSNorm, quantize, RoPE, K/V store; "other" = TTFT less every phase. The Wo and W2 phases include their cross-socket reduction wait.
Peak = 2 x 225 TOP/s (docs/hbm-roofline.md; the engine runs 55 threads per socket, 2% lower, not applied). Ideal times derived: GEMM TOP / 450 TOP/s (289.3 TOP at 2113 tokens, 74.9 at 547) and attention TOP / 450 (5.85 and 0.39 TOP).

| cell | TTFT s | qkv | Wo | W1/W3 | W2 | attention | small | other | GEMM share | GEMM TOP/s | share of peak | AMX busy / cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2113, M = 256 (9 batches) | 7.946 | 0.509 | 0.630 | **4.518** | 1.434 | 0.487 | 0.349 | 0.007 | 89.3% | 40.8 | 9.1% | 8.95% |
| 2113, M = 1024 (3 batches) | 5.773 | 0.536 | 0.576 | **2.387** | 1.531 | 0.462 | 0.246 | 0.004 | 87.2% | 57.5 | 12.8% | 12.35% |
| 547, M = 256 (3 batches) | 2.103 | 0.144 | 0.173 | 1.230 | 0.401 | 0.057 | 0.096 | 0.002 | 92.7% | 38.4 | 8.5% | 8.84% |
| 547, M = 1024 (1 batch) | 1.578 | 0.124 | 0.150 | 0.804 | 0.368 | 0.053 | 0.079 | 0.001 | 91.6% | 51.8 | 11.5% | 11.65% |

(GEMM includes the last-row classifier, 0.001-0.004 s.) Rate per matrix, TOP/s (share of 450):

| cell | qkv | Wo | W1/W3 | W2 | attention (AMX) |
|---|---:|---:|---:|---:|---:|
| 2113, M = 256 | 55.7 (12.4%) | 36.0 (8.0%) | **35.2 (7.8%)** | 55.4 (12.3%) | 12.0 |
| 2113, M = 1024 | 52.9 (11.8%) | 39.4 (8.8%) | **66.5 (14.8%)** | 51.9 (11.5%) | 12.7 |
| 547, M = 256 | 51.1 | 34.0 | 33.4 | 51.2 | 6.8 |
| 547, M = 1024 | 59.4 | 39.1 | 51.2 | 55.9 | 7.4 |

Where the 2113-token, M = 256 time goes (derived from the table; ideal = at AMX peak):

| term | seconds | share of TTFT | ideal s | excess s |
|---|---:|---:|---:|---:|
| GEMM, all matrices | 7.095 | 89.3% | 0.643 | 6.452 |
| of which W1/W3 above its M = 1024 time (4.518 - 2.387) | 2.131 | 26.8% | | |
| attention | 0.487 | 6.1% | 0.013 | 0.474 |
| small phases | 0.349 | 4.4% | | |
| other (forward less all phases) | 0.007 | 0.1% | | |
| **total** | **7.946** | | 0.656 | 7.290 |

Attention is 6-8% of the 2113-token TTFT at 12-13 TOP/s (3% of peak; the scaled 8B prediction of the limits page was 0.43 s, measured 0.46-0.49). The exchange (reduction) components of the engine's profile log are of the order of 0.1 ms per token per socket in Wo and in W2 at M = 256 (inside the Wo and W2 phases, not additional; the log lines are in `prof-70b-ddr-b256.stderr`), 0.5 s at most for 2113 tokens; they are not separated in the table.

## 4. The batch effect, phase by phase

Plain-driver TTFT (batches, prompt rows): (9, 2113) 7.915 s; (3, 547) 2.106 s; (3, 2113) 5.603 s; (1, 547) 1.609 s. Least squares of `TTFT = c x batches + r x rows` (derived): **c = 0.369 s per batch, r = 2.14 ms per row**; the residuals are -0.06, +0.17, +0.03, -0.07 s, so the model is a rough description, not a law.
The same fit on the profiled phases (profiled medians of section 3):

| phase | per batch s | per row ms | worst residual s |
|---|---:|---:|---:|
| qkv | -0.003 | 0.256 | 0.013 |
| Wo | +0.009 | 0.259 | 0.003 |
| **W1/W3** | **+0.340** | 0.666 | 0.154 |
| W2 | -0.013 | 0.738 | 0.035 |
| attention | +0.001 | 0.214 | 0.065 |
| small | +0.016 | 0.094 | 0.011 |

Only W1/W3 has a per-batch term. At M = 1024 W1/W3 reaches 66.5 TOP/s (14.8%), at M = 256 it runs at 35 TOP/s: this, and no fixed per-batch work in the other phases, is the difference between the xengine rows and the engine default. The limits page's residual of 0.16-0.20 s per M = 256 batch over its 8B-scaled prediction
(computed from 8B rates measured at the engine's batch) is thus largely the W1/W3 rate at M = 256, which the 8B does not show to this degree (8B W1/W3 at 2113 tokens: 52.4-71.1 TOP/s at both batch sizes, section 5).

## 5. DDR traffic during prefill, and the 8B pair

Measured with the uncore CAS counters (DDR 64 B per CAS, HBM 32 B), plain driver, median over the 3 measured requests; both sockets summed; packed copy 72.73 GB for both sockets (70B gs64, job 5934864 `packed_weights_peak`), 7.42 GB (8B, engine log `packed 3707764736 + 3707764736 bytes`).

| run | prompt | batches | TTFT s | DDR read GB | DDR read / (batches x packed copy) | DDR GB/s | HBM read GB | HBM GB/s | DDR write GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70B M = 256 | 547 | 3 | 2.106 | 460.4 | 2.11 | 218.0 | 52.1 | 24.8 | 28.1 |
| 70B M = 256 | 2113 | 9 | 7.915 | 1558.6 | 2.38 | 196.9 | 135.8 | 17.2 | 110.8 |
| 70B M = 1024 | 547 | 1 | 1.609 | 107.7 | 1.48 | 66.9 | 48.7 | 30.2 | 9.3 |
| 70B M = 1024 | 2113 | 3 | 5.603 | 348.4 | 1.60 | 62.1 | 165.5 | 29.4 | 16.6 |
| 8B M = 256, packed in DDR | 2113 | 9 | 0.847 | 67.2 | 1.01 | 79.3 | 25.9 | 30.6 | 0.8 |
| 8B M = 256, packed in HBM | 2113 | 9 | 0.851 | 0.6 | 0.01 | 0.6 | 91.4 | 105.6 | 0.9 |
| 8B M = 1024, packed in DDR | 2113 | 3 | 0.719 | 23.8 | 1.07 | 32.5 | 37.8 | 50.3 | 1.8 |
| 8B M = 1024, packed in HBM | 2113 | 3 | 0.713 | 1.0 | 0.05 | 1.5 | 55.9 | 77.2 | 1.8 |

* The 70B rate in time (request 3, 2113 tokens): M = 256 flat at 196-206 GB/s over the whole prefill; M = 1024 flat at 42-44 GB/s while the two 1024-row batches run and 300-400 GB/s in the last 0.4 s (the 65-row batch, where the weights are streamed against almost no compute).
  The DDR peak the machine reaches for such a stream is 525 GB/s (2 x 262.5, [concurrent-tier-bandwidth.md](concurrent-tier-bandwidth.md)): the M = 256 run is at 38-41% of it, so it is neither at the DDR bandwidth limit nor idle.
* DDR write traffic is 6-7% of the read traffic in every 70B window with the same time profile and is not understood (the 8B has 1%); the packed copy is read-only after the pack. Nothing here depends on it.
* 8B profiled pairs (3 measured requests, one process each): packed in HBM 0.790 s and in DDR 0.878 s at M = 256, 0.766 s and 0.714 s at M = 1024: the sign of the difference changes with the batch, and the plain-driver pair above is 0.851/0.847 and 0.713/0.719, so a packed copy in DDR costs the 8B nothing measurable (process-to-process spread 5-10% in the profiled runs, as in [bimodality.md](bimodality.md)).
  The 8B W1/W3 rate is 52.4 (DDR) and 66.9 (HBM) TOP/s at M = 256 and 69.8 and 71.1 at M = 1024.

## 6. Hypotheses: ruled in, ruled out, open

| hypothesis | verdict | evidence |
|---|---|---|
| GEMM efficiency at the 70B shapes is the largest term | **in** | GEMM 87-93% of TTFT in all four 70B cells; 8.5-12.8% of peak; AMX busy 8.8-12.4% of cycles; at the 70B shapes the other three matrices run at the 8B in-situ rate (qkv, W2 about 12%, Wo 8%) and W1/W3 at 8% (M = 256) or 15% (M = 1024) |
| per-batch fixed cost (barriers, quantize, exchange, K/V store) | **mostly out** | qkv, Wo, W2, attention: per-batch term -0.013 to +0.009 s; small phases 0.016 s per batch (0.1 s of 7.9 s); exchange 0.1 ms per token per socket in Wo and W2; "other" 0.1% |
| per-batch cost = the packed-copy read from DDR (once per batch, 0.141 s) | **out as stated; a larger read effect is in** | reads are 2.38 passes per M = 256 batch and 1.60 per M = 1024 batch (not 1); the 8B, which reads once, shows no DDR cost; the per-batch term is 0.34 s, 2.4 times the single-pass time, and sits in W1/W3 only |
| attention share | **in, small** | 6.1% at 2113 tokens (0.49 s at 12 TOP/s), 2.7% at 547; the same at both batch sizes |
| non-AMX phases | **in, small** | small phases 4.4%; classifier 0.1% |
| AMX not engaged for part of the work | **out** | `EXE.AMX_BUSY` agrees with the profile GEMM share; AVX-512 reference at 1.8% of the AMX peak and 0% AMX busy |
| why W1/W3 and why 2.4 passes | **open** | candidates, untested: the W1 and W3 shards of one socket (about 125 MB each at gs64) exceed the L3 of 112 MB, so successive row blocks re-stream them from DDR; a 70B-only effect (the 8B shards are about 31 MB). `gemm_bench` at the 70B shard shapes (N = 14336 or 28672, K = 8192, M = 256 and 1024) with the weights in DDR and in HBM would separate the kernel from the memory path |

## 7. Uncertainties

* One node, one job; profile and CAS from one process per cell (three measured requests each). The 70B profiled M = 1024, 547-token cell has one measured request of 13.005 s among 1.578, 1.578 (the median is used; the plain driver's 1.599-1.614 s shows no such request); cause not examined.
* The profile is of a warm process and the exact fixture ids; the profiled and plain TTFT agree to 0.4% (M = 256) and 3% (M = 1024); the slower W1/W3 rate at M = 256 is in the profile and in the plain TTFT.
* Phase times are wall times of the calling thread; a phase ends at a team barrier, so waiting for the slowest worker is inside the phase (a late worker lowers the GEMM rate by that amount); the engine's per-worker busy time is not available without code changes.
* The DDR read/write counts are uncore CAS counts at 64 B; the 32-byte HBM CAS and the read amplification are calibrated by [xengine-results.md](xengine-results.md). The window edges are known to about one 50 ms interval (under 1% of a window).
* The 8B-scaled prediction and peak (225 TOP/s per socket at 56 cores) are the limits page's constants; the 55-thread engine peak is 2% lower.
* gs32 was not measured here (gs32 cells would be 15-25% slower by the limits page); the 70B packed copy in HBM does not fit, so no direct HBM-packed 70B reference exists.
* Not measured: M = 512; per-matrix DDR traffic (phases are 5-50 ms per layer, below the counter's resolution of a few intervals).

## 8. What to do next

1. Decide the batch for the 70B rows: the xengine comparison at M = 256 understates the C engine by 24-29% in TTFT (1.61 s and 5.60 s at the engine default of 1024 against 2.11 s and 7.92 s); re-run `c-q8` with `LLAMA_PREFILL_BATCH=1024` for the cross-engine tables (the other engines' own batches are their defaults, the comparison fixed 256 for all).
2. A kernel micro-benchmark at the 70B W1/W3 shard shape (M = 256, 512, 1024; weights in DDR and in HBM) to test the cache-capacity reading of section 6; if confirmed, a loop order that does not re-stream W1/W3 per M block is the lever for M = 256 (and for the 65-row tail batch).
3. Not worth pursuing from this data: the exchange, the small phases and the classifier (together under 6%), or moving the packed copy: the 8B shows no difference between HBM and DDR.
