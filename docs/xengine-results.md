# Cross-engine comparison on Xeon Max: llama3.x.c, llama.cpp and vLLM (CPU), Llama 3.1 8B INT8: results

**Status: run (2026-11-21; Flat HBM job 5892201, Flat DDR job 5892202, Cache job 5892196).** Single request, batch 1, greedy, three repeats
of every cell, 186 cells, all passed their checks. Plan: [plans/xengine-comparison.md](plans/xengine-comparison.md). Provenance, checks and the
source of every number: [the validation record](xengine-results-validation.md). Branch: `feature/xengine-comparison`. The tables below are
generated from `runs.csv` (`tools/xengine_summary.py`) by `tools/xengine_tables.py` and are not edited by hand.

Every cell is a **measured** median over the 3 repeats with (min-max) behind it, unless a heading says **derived** (computed from measured
numbers) or **modelled**. Nothing in this document is modelled except where a column says so (the ideal AMX cycle count, section 5, and
the KV bytes of section 2, both **derived** from the architecture). Labels: node A = Flat mode, everything on HBM (job 5892201); node B =
Flat mode, everything on DDR (job 5892202); Cache node = Cache mode, no binding (job 5892196). Engine pins: llama3.x.c commit a3402b6 (the
sources are identical in the harness commits 6cf3b0f and af6b1b8; only a Makefile line differs), llama.cpp v1.0.0 (b1b770d90, gcc-toolset-14,
Q8_0 GGUF, 8.54 GB), vLLM 0.34.0+cpu (Apptainer image) with the RedHatAI W8A8 checkpoint (revision 024e24cb).

## Result in one paragraph

On one socket HBM speeds up decode by 1.74x (llama3.x.c), 1.63x (llama.cpp) and 1.63x (vLLM, patched to reach HBM) over DDR on the same
engine and an equivalent node (anchor ratio 1.004), but only 1.00-1.20x for prefill, which is compute-bound. llama3.x.c decodes fastest
everywhere (51.9 tok/s on one socket and 99.1 on two, both HBM, against 40.6 / 23.9 for llama.cpp and 40.5 / 61.2 for vLLM) and scales almost
perfectly to two sockets (1.91x, 0.015 GB of UPI traffic per token, each socket reads its own shard). vLLM is 3.6x faster than llama3.x.c in
prefill by tok/s (3.65x by TTFT at one socket, 3.71x at two; TTFT 0.618 s against 2.258 s for 2048 tokens on one socket): all three engines use AMX, but llama3.x.c keeps the AMX unit busy
for 5% of its core cycles against 21% for vLLM. llama.cpp is the one engine that gets slower on two sockets in HBM (0.59x): its single
process reads interleaved pages, 4.44 GB per token (50% of all reads) cross UPI, and the 106 GB/s that crossed is about twice the
roofline of remote HBM (about 50 GB/s). Stock vLLM never reached HBM in Flat mode: with `numactl --membind` to the HBM node it decodes at
24.7 tok/s with 99.9% of its reads on DDR; a one-line patch of its worker gives 40.5. Cache mode lands between DDR and HBM for decode (75% of the
way for llama3.x.c on one socket) and gives nothing for prefill. HBM power is not visible in the DRAM RAPL domain; package power sits
at 339-366 W per active socket for every engine and tier, so energy per token follows tokens per second (decode on one socket:
10.96 J/token on HBM against 21.03 on DDR for llama3.x.c, both sockets counted).

**Note (2026-11-21, after the prefill-gap work, [prefill-gap.md](prefill-gap.md) section 7).** vLLM's prefill lead is measured at lower logit
fidelity than llama3.x.c's. A Hugging Face FP32 model with the RedHatAI checkpoint's GPTQ weights and vLLM's per-token int8 inputs, BF16 norms,
embedding and LM head (fake-quantized, **modelled**, job 5892436) is at rms 0.395 against the FP32 logits of prompt-2048 and fails the rms ≤ 0.2
gate that llama3.x.c's W8A8 is held to. The same model of llama3.x.c's Q8_0 gs=64 scheme is at 0.123, and the engine itself measures 0.137 on its batched prefill (job 5892459). The cause is the per-token input
quantization: the same GPTQ weights with FP32 inputs give 0.146. vLLM's own logits were not captured, so this accuracy comparison is modelled; the tables below compare speed only.

## Headline

Decode tok/s (workload D, 86-token prompt, 128 generated tokens) and TTFT in seconds (2048-token prompt), median of 3, **measured**. Each
cell: decode tok/s / TTFT s. Different nodes per column; engines are compared within a column only. vLLM on HBM is the **patched** variant (section 3).
Notes: (a) the llama.cpp 2S Cache cell (19.3 / 5.707) is the out-of-the-box layout of the Cache set (112 threads, no interleave, by design not tuned), not the tuned layout of the Flat cells (section 7). (b) TTFT is each engine's own definition: llama3.x.c `time_to_first_token` (includes tokenisation and the first sample); llama.cpp 2048 / the `llama-bench` pp rate; vLLM the client-side latency to the first streamed token over HTTP.

**Important: the vLLM "Flat HBM" rows use the patched vLLM, not stock.** The patched worker maps each rank's memory to the HBM node of its socket. Stock vLLM 0.34.0 never reaches HBM in Flat mode: its CPU worker rebinds memory to the NUMA node of its cores (DDR) and overrides `numactl --membind`. In the stock + membind control (job 5892201), 99.9% of decode reads came from DDR and decode ran at 24.7 tok/s, the same as the DDR row; the patched variant decodes at 40.5 tok/s (section 3).

| Engine | Sockets | Flat HBM (node A) | Flat DDR (node B) | Cache (Cache node) |
| --- | ---: | ---: | ---: | ---: |
| llama3.x.c | 1 | 51.9 / 2.258 | 29.8 / 2.526 | 46.3 / 2.523 |
| llama3.x.c | 2 | 99.1 / 1.366 | 55.9 / 1.494 | 79.2 / 1.590 |
| llama.cpp | 1 | 40.6 / 4.562 | 24.9 / 4.599 | 32.8 / 4.632 |
| llama.cpp | 2 | 23.9 / 3.814 | 29.7 / 3.189 | 19.3 / 5.707 |
| vLLM (HBM: patched) | 1 | 40.5 / 0.618 | 24.9 / 0.740 | 34.1 / 0.668 |
| vLLM (HBM: patched) | 2 | 61.2 / 0.369 | 41.2 / 0.456 | 50.2 / 0.399 |

## 1. Q1: prefill, decode and long-context decode

Prefill (workload P: 2048-token prompt, one generated token; 8192 tokens: the prefill phase of L for llama3.x.c and vLLM, workload LP
for llama.cpp). tok/s is each engine's native prefill rate (prompt tokens over its prefill time; for llama.cpp the `llama-bench` pp rate; for
vLLM prompt tokens over the time to the first streamed token); TTFT is the time to the first token (it includes tokenisation and the first
sample for llama3.x.c, so 2048 / TTFT is slightly below the native tok/s). **Measured**, jobs 5892201 (HBM), 5892202 (DDR), 5892196 (Cache).

| Configuration | Tier | TTFT 2048 s | prefill tok/s (2048) | TTFT 8192 s | prefill tok/s (8192) |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM (job 5892201) | 2.258 (2.214-2.266) | 932 (929-951) | 14.75 (14.72-14.78) | 562 (561-563) |
| llama3.x.c 1S | DDR (job 5892202) | 2.526 (2.513-2.563) | 831 (818-835) | 15.91 (15.81-15.92) | 520 (520-524) |
| llama3.x.c 1S | Cache (job 5892196) | 2.523 (2.509-2.980) | 832 (716-837) | 16.20 (16.15-16.63) | 511 (498-513) |
| llama.cpp 1S | HBM (job 5892201) | 4.562 (4.560-4.602) | 449 (445-449) | 29.82 (29.81-29.91) | 275 (274-275) |
| llama.cpp 1S | DDR (job 5892202) | 4.599 (4.579-4.615) | 445 (444-447) | 29.93 (29.42-30.87) | 274 (265-278) |
| llama.cpp 1S | Cache (job 5892196) | 4.632 (4.626-4.656) | 442 (440-443) | 29.79 (29.69-30.24) | 275 (271-276) |
| vLLM patched 1S | HBM (job 5892201) | 0.618 (0.611-0.633) | 3315 (3236-3349) | 2.73 (2.72-2.74) | 3004 (2989-3009) |
| vLLM 1S | DDR (job 5892202) | 0.740 (0.730-0.746) | 2767 (2747-2804) | 3.13 (3.11-3.13) | 2621 (2621-2635) |
| vLLM 1S | Cache (job 5892196) | 0.668 (0.664-0.683) | 3068 (2998-3087) | 2.90 (2.90-2.92) | 2823 (2806-2825) |
| llama3.x.c 2S | HBM (job 5892201) | 1.366 (1.330-1.412) | 1564 (1511-1608) | 8.10 (8.06-8.13) | 1033 (1030-1037) |
| llama3.x.c 2S | DDR (job 5892202) | 1.494 (1.481-1.512) | 1427 (1408-1439) | 8.80 (8.77-8.80) | 949 (949-954) |
| llama3.x.c 2S | Cache (job 5892196) | 1.590 (1.568-1.756) | 1337 (1206-1357) | 8.75 (8.53-8.83) | 954 (946-979) |
| llama.cpp 2S | HBM (job 5892201) | 3.814 (3.795-3.843) | 537 (533-540) | 21.45 (21.38-21.78) | 382 (376-383) |
| llama.cpp 2S | DDR (job 5892202) | 3.189 (3.162-3.194) | 642 (641-648) | 17.92 (17.91-18.20) | 457 (450-457) |
| llama.cpp 2S | Cache (job 5892196) | 5.707 (5.339-5.736) | 359 (357-384) | 31.36 (29.32-32.66) | 261 (251-279) |
| vLLM patched 2S | HBM (job 5892201) | 0.369 (0.367-0.369) | 5557 (5556-5581) | 1.38 (1.38-1.39) | 5925 (5900-5944) |
| vLLM 2S | DDR (job 5892202) | 0.456 (0.453-0.460) | 4491 (4457-4525) | 1.68 (1.67-1.68) | 4887 (4875-4896) |
| vLLM 2S | Cache (job 5892196) | 0.399 (0.393-0.402) | 5129 (5091-5216) | 1.47 (1.46-1.47) | 5591 (5579-5599) |

Decode. D: 86-token prompt, 128 generated tokens; L: 128 generated tokens at a depth of 8192 (llama3.x.c and vLLM: an 8192-token prompt,
with the decode phase cut by the markers; llama.cpp: `-d 8192`). TPOT (native) is each engine's own definition (llama3.x.c: its average
decode step; vLLM: mean gap between streamed tokens; llama.cpp: 1 / `llama-bench` tg rate). TPOT (uniform) is the within-run form of the
plan's definition, (end-to-end time - TTFT) / (generated - 1) for llama3.x.c and vLLM; for llama.cpp it equals the native value by construction
(`llama-bench` times all 128 tokens as one block). The plan's separate G = 1 runs were not part of the matrix, so the cross-run definition
(T(G) - T(1)) / (G - 1) is not available; native and uniform TPOT agree within 1-3% for llama3.x.c (the sampling and bookkeeping of each
step) and exactly for vLLM. **Measured.**

| Configuration | Tier | decode tok/s (D) | TPOT ms (native) | TPOT ms (uniform) | long-context tok/s (L) | TPOT ms (L) | L / D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | 51.9 (51.8-52.6) | 19.3 (19.0-19.3) | 19.5 (19.3-19.6) | 37.2 (37.2-37.5) | 26.9 (26.7-26.9) | 0.72 |
| llama3.x.c 1S | DDR | 29.8 (29.7-29.8) | 33.6 (33.6-33.6) | 33.8 (33.8-33.8) | 22.2 (22.1-22.3) | 45.1 (44.9-45.2) | 0.74 |
| llama3.x.c 1S | Cache | 46.3 (45.7-46.6) | 21.6 (21.5-21.9) | 21.9 (21.7-22.2) | 32.5 (32.5-32.5) | 30.8 (30.8-30.8) | 0.70 |
| llama.cpp 1S | HBM | 40.6 (40.5-40.9) | 24.6 (24.5-24.7) | 24.6 (24.5-24.7) | 29.1 (29.0-29.3) | 34.3 (34.2-34.5) | 0.72 |
| llama.cpp 1S | DDR | 24.9 (24.9-25.1) | 40.1 (39.8-40.1) | 40.1 (39.8-40.1) | 20.0 (19.7-20.8) | 50.0 (48.0-50.9) | 0.80 |
| llama.cpp 1S | Cache | 32.8 (32.8-32.9) | 30.4 (30.4-30.5) | 30.5 (30.4-30.5) | 24.7 (24.7-25.3) | 40.5 (39.6-40.6) | 0.75 |
| vLLM stock + membind 1S (D only) | HBM | 24.7 (24.7-24.8) | 40.4 (40.4-40.5) | 40.4 (40.4-40.5) | - | - | - |
| vLLM patched 1S | HBM | 40.5 (40.3-40.6) | 24.7 (24.6-24.8) | 24.7 (24.6-24.8) | 35.3 (35.2-35.3) | 28.3 (28.3-28.4) | 0.87 |
| vLLM 1S | DDR | 24.9 (24.8-25.0) | 40.1 (40.1-40.3) | 40.1 (40.1-40.3) | 22.2 (22.1-22.2) | 45.1 (45.1-45.3) | 0.89 |
| vLLM 1S | Cache | 34.1 (33.9-34.2) | 29.3 (29.3-29.5) | 29.3 (29.3-29.5) | 30.3 (30.0-30.4) | 33.0 (32.9-33.3) | 0.89 |
| llama3.x.c 2S | HBM | 99.1 (97.2-100.2) | 10.1 (10.0-10.3) | 10.4 (10.2-10.6) | 66.4 (65.6-66.4) | 15.1 (15.1-15.2) | 0.67 |
| llama3.x.c 2S | DDR | 55.9 (55.8-56.0) | 17.9 (17.9-17.9) | 18.1 (18.1-18.1) | 40.4 (40.2-40.4) | 24.8 (24.7-24.9) | 0.72 |
| llama3.x.c 2S | Cache | 79.2 (79.1-79.7) | 12.6 (12.5-12.6) | 12.9 (12.8-12.9) | 55.6 (53.9-56.8) | 18.0 (17.6-18.6) | 0.70 |
| llama.cpp 2S | HBM | 23.9 (23.8-23.9) | 41.8 (41.8-42.1) | 41.8 (41.8-42.1) | 20.1 (20.1-20.6) | 49.7 (48.5-49.7) | 0.84 |
| llama.cpp 2S | DDR | 29.7 (29.6-30.0) | 33.7 (33.4-33.8) | 33.7 (33.4-33.8) | 22.6 (22.6-24.1) | 44.3 (41.4-44.3) | 0.76 |
| llama.cpp 2S | Cache | 19.3 (16.7-19.7) | 51.9 (50.7-60.1) | 51.9 (50.7-60.1) | 15.9 (14.2-16.6) | 63.0 (60.3-70.4) | 0.82 |
| vLLM patched 2S | HBM | 61.2 (60.5-61.4) | 16.3 (16.3-16.5) | 16.3 (16.3-16.5) | 51.8 (51.6-52.0) | 19.3 (19.2-19.4) | 0.85 |
| vLLM 2S | DDR | 41.2 (41.1-41.5) | 24.2 (24.1-24.3) | 24.2 (24.1-24.3) | 36.7 (36.6-36.7) | 27.3 (27.3-27.3) | 0.89 |
| vLLM 2S | Cache | 50.2 (49.1-50.3) | 19.9 (19.9-20.4) | 19.9 (19.9-20.4) | 43.8 (42.2-43.8) | 22.8 (22.8-23.7) | 0.87 |

What the tables say. In decode llama3.x.c is the fastest engine in all six combinations of sockets and tier; in prefill the order is vLLM, llama3.x.c, llama.cpp (two sockets, HBM, 2048 tokens: 5557, 1564, 537 tok/s). Decode at depth
8192 retains 0.67-0.74 of the short-context rate for llama3.x.c (FP32 KV, the largest KV traffic), 0.72-0.84 for llama.cpp (F16) and 0.85-0.89 for vLLM (BF16); this is consistent with the KV element size, which was not isolated. The run-to-run range of the three repeats is at most 3.4% for every Flat cell except the 2048-token prefill of
llama3.x.c on two sockets in HBM (6.2%) and three llama.cpp cells in DDR (4.8-6.9%); on the Cache node it is up to 16% (section 8).

## 2. Q2: decode memory bandwidth and bytes per token

Counters of repeat 1 (jobs 5892201 and 5892202; MEM group: `ddr_cas_rd` at 64 B and `hbm_cas_rd` at 32 B per count, see section 3 for the calibration), cut to the decode
window of the run by its marker file; one run per cell, so no range. **Measured** counters, **derived** rates. Roofline: 552 GB/s per socket
for local HBM and 262 GB/s per socket for local DDR (`hbm-roofline.md`), times the number of sockets used. Nominal bytes per token: 7.97 GB
for the Q8_0 engines (the 7974764544 B of weights streamed per token: 32 layers 7.416 GB plus the LM head 0.558 GB) and 8.03 GB for
vLLM (6.98 GB of int8 layers plus the 1.05 GB bf16 LM head, **derived** from the architecture). "L - D" is the extra GB per token at depth 8192;
the last column is the KV cache that one token must read at that depth, 8192 x 32 layers x 2 x 1024 x element size (4 B FP32, 2 B F16 and BF16),
**derived**.

| Configuration | Tier | GB/s S0 / S1 (D) | GB/s total (D) | % of roofline | GB / token (D) | x nominal | GB/s total (L) | % of roofline (L) | GB / token (L) | L - D GB / token | KV read at depth 8192, derived GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | 400 / 0 | 400 | 73% | 7.72 | 0.97 | 363 | 66% | 9.88 | 2.17 | 2.16 |
| llama3.x.c 1S | DDR | 232 / 0 | 232 | 89% | 7.86 | 0.99 | 221 | 84% | 10.02 | 2.16 | 2.16 |
| llama.cpp 1S | HBM | 350 / 0 | 351 | 64% | 8.72 | 1.09 | 284 | 52% | 9.74 | 1.02 | 1.08 |
| llama.cpp 1S | DDR | 219 / 0 | 219 | 84% | 8.76 | 1.10 | 206 | 79% | 9.88 | 1.13 | 1.08 |
| vLLM stock + membind 1S | HBM | 197 / 0 | 197 | 36% | 7.96 | 0.99 | - | - | - | - | - |
| vLLM patched 1S | HBM | 319 / 0 | 319 | 58% | 7.87 | 0.98 | 313 | 57% | 8.87 | 1.00 | 1.08 |
| vLLM 1S | DDR | 198 / 0 | 198 | 76% | 7.94 | 0.99 | 199 | 76% | 8.99 | 1.05 | 1.08 |
| llama3.x.c 2S | HBM | 354 / 353 | 707 | 64% | 7.46 | 0.94 | 621 | 56% | 9.58 | 2.12 | 2.16 |
| llama3.x.c 2S | DDR | 213 / 213 | 426 | 81% | 7.71 | 0.97 | 389 | 74% | 9.77 | 2.07 | 2.16 |
| llama.cpp 2S | HBM | 105 / 105 | 211 | 19% | 8.82 | 1.11 | 205 | 19% | 9.94 | 1.12 | 1.08 |
| llama.cpp 2S | DDR | 129 / 129 | 259 | 49% | 8.81 | 1.11 | 224 | 43% | 9.93 | 1.12 | 1.08 |
| vLLM patched 2S | HBM | 234 / 234 | 468 | 42% | 7.72 | 0.96 | 454 | 41% | 8.78 | 1.05 | 1.08 |
| vLLM 2S | DDR | 161 / 161 | 321 | 61% | 7.82 | 0.97 | 327 | 62% | 8.92 | 1.10 | 1.08 |

Three observations. (1) llama3.x.c on one socket streams 400 GB/s from HBM (73% of the roofline) and 232 GB/s from DDR (89%), so its HBM
speedup (1.74x) is the bandwidth ratio (1.72x) at equal bytes per token; the 2.1x ratio of the two rooflines is not reached because HBM decode
sits at 73% of its roofline. Two sockets in HBM reach 707 GB/s (64% of 1104). (2) The other two engines use the bandwidth less well and, for
llama.cpp, read more: 8.72 GB/token (1.09x nominal) for 351 GB/s, against 7.72 GB/token for 400 GB/s. Of llama.cpp's 1.28x deficit in decode
rate on one socket (40.6 against 51.9 tok/s), 1.14x is bandwidth and 1.13x is bytes (**derived**). The extra bytes are consistent with its
AMX repack layout (its AMX model buffer is 8498 MiB = 8.91 GB for 7.97 GB of per-token weights; an interpretation, not profiled). vLLM reads
0.98x of its nominal (llama3.x.c 0.97x, below nominal as well) but streams only 319 GB/s (58% of the roofline). (3) The long-context cost matches the KV size: llama3.x.c
adds 2.17 GB per token at 8192 (derived KV 2.16 GB), llama.cpp 1.02-1.13 GB and vLLM 1.00-1.10 GB (derived 1.08 GB).

Precision of these figures: the GB/token of the same configuration read from the DDR counter at 64 B and the HBM counter at 32 B agree to
0.5-1.8% for one socket (0.982 for llama3.x.c, 0.995 for llama.cpp, 0.991 for vLLM, HBM over DDR; 3.2% for llama3.x.c on two sockets), and the decode window of llama3.x.c on two
sockets in HBM is only 1.34 s long, so its GB/s carries about 0.1 s / 1.34 s of edge uncertainty.

## 3. Q3: which tier served the reads

DDR and HBM read bandwidth summed over both sockets (jobs 5892201 and 5892202), repeat 1, for decode (D) and prefill (P). "Asked for" is the tier the configuration
requested. **Measured** counters, **derived** shares. The vLLM row "stock + membind" is the control: stock vLLM started with
`numactl --membind` to the HBM node, which vLLM ignores because it binds each worker's memory to the NUMA node of its CPUs.

| Configuration | Asked for | Phase | DDR GB/s (sum of sockets) | HBM GB/s (sum of sockets) | HBM share of reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | decode (D) | 0.3 | 400.1 | 99.9% |
| llama3.x.c 1S | HBM | prefill (P) | 0.5 | 31.1 | 98.6% |
| llama3.x.c 1S | DDR | decode (D) | 232.2 | 0.2 | 0.1% |
| llama3.x.c 1S | DDR | prefill (P) | 27.7 | 0.2 | 0.7% |
| llama.cpp 1S | HBM | decode (D) | 0.3 | 350.4 | 99.9% |
| llama.cpp 1S | HBM | prefill (P) | 0.3 | 14.1 | 98.2% |
| llama.cpp 1S | DDR | decode (D) | 218.8 | 0.2 | 0.1% |
| llama.cpp 1S | DDR | prefill (P) | 14.0 | 0.2 | 1.4% |
| vLLM stock + membind 1S | HBM | decode (D) | 196.8 | 0.2 | 0.1% |
| vLLM patched 1S | HBM | decode (D) | 0.3 | 318.7 | 99.9% |
| vLLM patched 1S | HBM | prefill (P) | 0.3 | 29.3 | 98.9% |
| vLLM 1S | DDR | decode (D) | 197.7 | 0.2 | 0.1% |
| vLLM 1S | DDR | prefill (P) | 25.7 | 0.2 | 0.8% |
| llama3.x.c 2S | HBM | decode (D) | 0.7 | 706.1 | 99.9% |
| llama3.x.c 2S | HBM | prefill (P) | 0.3 | 53.0 | 99.4% |
| llama3.x.c 2S | DDR | decode (D) | 425.9 | 0.2 | 0.0% |
| llama3.x.c 2S | DDR | prefill (P) | 51.2 | 0.2 | 0.4% |
| llama.cpp 2S | HBM | decode (D) | 0.3 | 210.5 | 99.8% |
| llama.cpp 2S | HBM | prefill (P) | 0.2 | 19.0 | 98.8% |
| llama.cpp 2S | DDR | decode (D) | 258.5 | 0.2 | 0.1% |
| llama.cpp 2S | DDR | prefill (P) | 32.2 | 0.2 | 0.6% |
| vLLM patched 2S | HBM | decode (D) | 0.6 | 467.0 | 99.9% |
| vLLM patched 2S | HBM | prefill (P) | 0.6 | 53.6 | 98.9% |
| vLLM 2S | DDR | decode (D) | 321.0 | 0.2 | 0.1% |
| vLLM 2S | DDR | prefill (P) | 52.8 | 0.2 | 0.4% |

Every engine reached the tier it was asked for, with two qualifications. (a) Stock vLLM did not: the control reads 99.9% from DDR (196.8 GB/s
of DDR against 0.2 of HBM), decodes at 24.7 tok/s (the DDR rate, 24.9) and the placement evidence of its worker agrees (15.6 GiB on DDR node 0,
0.25 GiB on HBM node 2). The patched variant (one environment variable, `VLLM_HBM_NODE_OFFSET`, read by a replaced worker module that maps each
rank's memory node to its socket's HBM node) puts 18.1 GiB on node 2 (one socket) and 11.3 GiB on node 2 and 11.35 GiB on node 3 (two ranks)
and reads 99.9% from HBM. It is **never presented as stock**. (b) llama3.x.c and llama.cpp read 98.2-99.4% from HBM in prefill and 99.9% in decode;
the rest is the embedding table (0.56 GB, always on DDR in llama3.x.c) and background traffic. The DDR runs show 0.0-1.4% HBM share, the noise floor of idle traffic.

HBM CAS calibration. `hbm_cas_rd` counts 32-byte units on this CPU; with 64 B the HBM runs would read 15.4-15.6 GB/token for a 7.97 GB model
(`--calibrate`, repeat 1 of llama3.x.c, one socket, D: HBM 1.935-1.959x nominal at 64 B, 0.968-0.980x at 32 B; DDR 0.986-0.987x at 64 B). The
other engines confirm it independently (section 2, 0.5-1.8%). **Side finding:** repository text written before this campaign that converts
`hbm_cas_rd` counts with 64 B is 2x too high on HBM bytes: at least `attention-profile.md` (section 2: 60.6 million HBM CAS reads read as 3.9 GB, 1.9 GB at 32 B) and
the "HBM read GB" columns of the earlier campaign tables (`numa-tp-prefill-results*.md`), whose derived decode GB/s (token rate x bytes) are unaffected. These documents were not
edited and the other uses were not audited.

## 4. Q4: dual-socket scaling, UPI traffic and mechanism

2S/1S ratios of the median rates (**derived**) and UPI data per decoded token (repeat 1, **measured** flits, sum of the data flits received at both sockets,
x 64/9 B per flit; the 64/9 B is a documented assumption, see below). Cache node rows have no UPI counters.

| Engine | Tier | 2S/1S decode D | 2S/1S long-ctx L | 2S/1S prefill 2048 | 2S/1S prefill 8192 | UPI GB/token 1S (D) | UPI GB/token 2S (D) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c | HBM | 1.91 | 1.78 | 1.68 | 1.84 | 0.000 | 0.015 |
| llama.cpp | HBM | 0.59 | 0.69 | 1.20 | 1.39 | 0.000 | 4.442 |
| vLLM patched | HBM | 1.51 | 1.47 | 1.68 | 1.97 | 0.001 | 0.008 |
| llama3.x.c | DDR | 1.88 | 1.82 | 1.72 | 1.82 | 0.000 | 0.016 |
| llama.cpp | DDR | 1.19 | 1.13 | 1.44 | 1.67 | 0.001 | 4.405 |
| vLLM | DDR | 1.65 | 1.65 | 1.62 | 1.86 | 0.001 | 0.006 |
| llama3.x.c | Cache | 1.71 | 1.71 | 1.61 | 1.87 | - | - |
| llama.cpp | Cache | 0.59 | 0.64 | 0.81 | 0.95 | - | - |
| vLLM | Cache | 1.47 | 1.45 | 1.67 | 1.98 | - | - |

Mechanisms, from the engines' own configuration and the placement evidence of each run:

- **llama3.x.c: tensor parallel, one shard per socket.** Each socket reads only its own half of every weight matrix from its own HBM (or DDR) node
  (`numa_maps`: 7.05 GiB on node 2 and 7.05 GiB on node 3; DDR: nodes 0 and 1); the only traffic across sockets is the reduce of activations,
  0.015 GB per token. Decode scales 1.91x (HBM), 1.88x (DDR); 8192-token prefill 1.82-1.84x, 2048-token prefill 1.68-1.72x.
- **llama.cpp: one process, no tensor parallelism.** `numactl --interleave` over the two HBM (or DDR) nodes, 64 threads (32 per socket, the best of
  eight layouts of the tuning job 5892194: 112 threads 20.8 tok/s, 96 threads 22.1, 64 threads 25.9 on HBM, against 41.6 for one socket; a follow-up with 64, 48, 40 and 32 threads, job 5892206, kept 64 best: HBM 25.6, 24.3, 25.3, 25.8 and DDR 31.4, 29.1, 28.1, 25.1 tok/s). Every second
  page is remote, so about half of all reads cross UPI: 4.44 GB per token of 8.82 (50.3%) on HBM, 4.41 of 8.81 (50.0%) on DDR. That is **106 GB/s
  of UPI data (58.7 and 47.5 GB/s received at the two sockets, 53 on average, within 6% of the roofline of remote HBM, about 50 GB/s)**, so the two sockets together deliver only
  211 GB/s from HBM, below one socket's 351 GB/s; decode is 0.59x of one socket. In DDR the remote traffic (129 GB/s, 65 on average per socket) is
  small enough relative to the local DDR bandwidth that two sockets still gain 1.19x. Prefill, which is not bandwidth-bound, gains 1.20-1.44x (2048)
  and 1.39-1.67x (8192).
- **vLLM: tensor parallel TP=2, one rank per socket,** weights split by the framework, activations all-reduced through shared memory: 0.006-0.008 GB per token
  of UPI traffic. It does not scale perfectly (1.51x in HBM, 1.65x in DDR) although the traffic is negligible: the per-token time of a rank is not halved
  (TPOT 16.3 ms on two sockets against 24.7 ms on one, HBM), which points at fixed per-step costs (scheduler, all-reduce latency, sampling) of about
  4 ms (12.4 ms would be perfect scaling); this is an inference from the numbers, not a profile. On HBM, vLLM's long-context decode scales 1.47x and its prefill 1.68x (2048) to 1.97x (8192).

UPI convention. The flit-to-byte factor (64/9) is the PCM convention and was an unverified assumption; the llama.cpp rows are consistent with it to about 1%,
assuming a 50% remote share under interleave over two nodes (8 B per flit would give 56%): the counted 4.44 and 4.41 GB per token are 50.3% and 50.0% of the 8.82 and 8.81 GB read per
token. This is a consistency check, not a proof (remote writes and snoop responses also use data flits). Receive and transmit totals are equal in every run.

## 5. Q5: AMX in prefill and decode

Repeat 2 (jobs 5892201 and 5892202; COMPUTE group: cycles, instructions, `EXE.AMX_BUSY`) on the Flat nodes; the Cache node has only user-mode per-process counters of
repeat 1. `AMX_OPS_RETIRED.INT8` reads 0 on this CPU in every run, even where AMX demonstrably runs (the setup jobs show `EXE.AMX_BUSY` of 8.3e10
cycles over a whole `llama-bench` invocation of llama.cpp with its AMX repack and 0 with `--repack 0`), so it carries no information and **`EXE.AMX_BUSY` is the evidence**. The share is
`EXE.AMX_BUSY` over unhalted core cycles, summed over all counted cores. **Measured**; GHz assumes 1.9 GHz of reference cycles.

| Configuration | Tier | AMX_BUSY cycles prefill 2048 | share of cycles | AMX_BUSY cycles decode D | share | AMX_BUSY cycles prefill 8192 | share | IPC (P) | GHz (P) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | 1.35e+10 | 5.1% | 9.50e+07 | 0.0% | 5.56e+10 | 3.2% | 2.09 | 2.14 |
| llama3.x.c 1S | DDR | 1.39e+10 | 4.7% | 3.33e+08 | 0.1% | 5.56e+10 | 2.9% | 2.13 | 2.17 |
| llama3.x.c 1S | Cache | 1.18e+10 | 4.8% | 0.00e+00 | 0.0% | 5.54e+10 | 3.2% | 1.40 | - |
| llama.cpp 1S | HBM | 2.73e+10 | 4.6% | 0.00e+00 | 0.0% | 1.11e+11 | 3.0% | 0.82 | 2.36 |
| llama.cpp 1S | DDR | 2.73e+10 | 4.6% | 0.00e+00 | 0.0% | 1.11e+11 | 2.9% | 0.82 | 2.37 |
| llama.cpp 1S | Cache | 2.63e+10 | 4.6% | 0.00e+00 | 0.0% | 1.10e+11 | 3.0% | 0.82 | - |
| vLLM stock + membind 1S | HBM | - | - | 2.01e+08 | 0.0% | - | - | - | - |
| vLLM patched 1S | HBM | 1.47e+10 | 20.8% | 3.00e+08 | 0.1% | 7.20e+10 | 23.3% | 0.52 | 2.12 |
| vLLM 1S | DDR | 1.41e+10 | 17.5% | 4.28e+08 | 0.1% | 7.28e+10 | 20.1% | 0.44 | 2.16 |
| vLLM 1S | Cache | 1.44e+10 | 20.2% | 4.23e+08 | 0.1% | 7.24e+10 | 22.2% | 0.45 | - |
| llama3.x.c 2S | HBM | 1.32e+10 | 4.0% | 7.17e+07 | 0.0% | 5.55e+10 | 2.9% | 2.02 | 2.20 |
| llama3.x.c 2S | DDR | 1.35e+10 | 3.8% | 3.51e+08 | 0.1% | 5.54e+10 | 2.6% | 2.02 | 2.22 |
| llama3.x.c 2S | Cache | 1.27e+10 | 4.2% | 0.00e+00 | 0.0% | 5.45e+10 | 3.1% | 1.25 | - |
| llama.cpp 2S | HBM | 2.70e+10 | 4.7% | 0.00e+00 | 0.0% | 1.10e+11 | 3.5% | 0.68 | 2.41 |
| llama.cpp 2S | DDR | 2.70e+10 | 5.7% | 0.00e+00 | 0.0% | 1.11e+11 | 4.2% | 0.82 | 2.39 |
| llama.cpp 2S | Cache | 2.66e+10 | 1.9% | 0.00e+00 | 0.0% | 1.11e+11 | 1.4% | 0.38 | - |
| vLLM patched 2S | HBM | 1.32e+10 | 18.3% | 6.55e+08 | 0.1% | 6.98e+10 | 22.0% | 0.53 | 2.13 |
| vLLM 2S | DDR | 1.47e+10 | 15.2% | 4.67e+08 | 0.1% | 7.26e+10 | 18.1% | 0.46 | 2.11 |
| vLLM 2S | Cache | 1.16e+10 | 17.3% | 1.82e+08 | 0.0% | 7.01e+10 | 21.5% | 0.49 | - |

Per prefilled token, the 2048-token prompt, repeat 2 (**derived** from the same counters). The ideal AMX-busy cycles per token is 6.82 M: 6.979e9 int8
multiply-accumulates of the 32 layers per token over 1024 MACs per AMX-busy cycle (a 16x16x64 tile product every 16 cycles; an **assumption** about the hardware, supported by the next
table, where llama3.x.c lands at 0.94-0.99x).

| Configuration | Tier | core cycles / token (M) | instructions / token (M) | AMX-busy cycles / token (M) | x ideal (6.82 M) | TTFT ms / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | 128.0 | 268.0 | 6.59 | 0.97 | 1.11 |
| llama3.x.c 1S | DDR | 145.1 | 309.0 | 6.77 | 0.99 | 1.23 |
| llama.cpp 1S | HBM | 291.2 | 239.5 | 13.32 | 1.95 | 2.23 |
| llama.cpp 1S | DDR | 292.4 | 239.7 | 13.34 | 1.96 | 2.24 |
| vLLM patched 1S | HBM | 34.5 | 18.0 | 7.19 | 1.05 | 0.31 |
| vLLM 1S | DDR | 39.2 | 17.3 | 6.88 | 1.01 | 0.36 |
| llama3.x.c 2S | HBM | 161.5 | 325.4 | 6.44 | 0.94 | 0.69 |
| llama3.x.c 2S | DDR | 172.1 | 348.3 | 6.58 | 0.97 | 0.73 |
| llama.cpp 2S | HBM | 281.1 | 191.6 | 13.19 | 1.93 | 1.88 |
| llama.cpp 2S | DDR | 232.6 | 190.1 | 13.19 | 1.94 | 1.56 |
| vLLM patched 2S | HBM | 35.4 | 18.8 | 6.47 | 0.95 | 0.18 |
| vLLM 2S | DDR | 47.2 | 21.8 | 7.19 | 1.06 | 0.22 |

All three engines run their prefill GEMMs on AMX, and in decode none does (0.0-0.1% of the cycles; llama.cpp 0, the others a few 1e8 cycles
of sampling and the LM head), which is the VNNI path for M of 4 or less in llama.cpp and vLLM and the AVX-512 kernels in llama3.x.c. The AMX work itself is
equal across llama3.x.c and vLLM (6.6 and 7.2 M cycles per token, 0.97x and 1.05x of ideal), is the same on one and two sockets, on HBM and DDR, and grows linearly with the
prompt (8192 tokens: 5.56e10 cycles for llama3.x.c, 4.1x the 2048-token value). llama.cpp's AMX kernel is busy for 1.95x the ideal (13.3 M cycles per token), that is it executes about twice the tile work of the minimum (padding or a smaller tile shape);
the cause was not investigated.

The difference between the engines is everything around the tiles. llama3.x.c spends 128 M core cycles per prefilled token against 34 M for vLLM and 291 M for
llama.cpp, so its AMX unit is busy 5% of the time against 21% (vLLM) and 5% (llama.cpp). Its instruction count is 268 M per token, 15x vLLM's 18 M, and
close to the 218 M group accumulations that a Q8_0 matrix product with 32-element groups needs per token (6.979e9 MACs / 32): a **derived coincidence that
suggests**, but does not prove, that the per-group scale epilogue of the Q8_0 format (the int32 tile result converted and scaled for every group of 32 inputs) is the cost,
which vLLM's per-channel format avoids with one scale at the end; llama.cpp, with the same Q8_0 group format, executes 240 M instructions per token. The growth of llama3.x.c's cycles
per token from 128 M at 2048 to 209 M at 8192 tokens (**derived** from the same counters; instructions 268 M to 419 M) points at attention, which grows with the context, as a second non-AMX cost.
Neither was profiled in this campaign.

## 6. Q6: power and energy

Average RAPL power per phase (jobs 5892201 and 5892202), repeats 1 and 2 (both carry the energy events; median of the 2), both sockets summed, and joules per token
from package plus DRAM energy of both sockets over the phase window divided by the tokens of the phase (prefill: 2048; decode: 127 steps for llama3.x.c and
vLLM, 128 for llama.cpp). "net" subtracts the idle baseline of the same job (10 s at job start). The idle baseline per socket is 140.1 / 133.5 W
(package) and 22.1 / 20.5 W (DRAM) on node A, 135.9 / 134.1 W and 21.1 / 19.1 W on node B. One-socket runs include the idle second socket
(158-193 W in the package domain while the other socket works), which is why their gross J/token is not "one socket's energy". **Measured** power, **derived** energy.
The Cache node has no RAPL events (privileged counters are unavailable there) and so no power or energy numbers.

| Configuration | Tier | P: package W (2 sockets) | P: DRAM W | P: J/token | P: net J/token | D: package W | D: DRAM W | D: J/token | D: net J/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | HBM | 533 | 44 | 0.61 | 0.28 | 521 | 44 | 10.96 | 4.82 |
| llama3.x.c 1S | DDR | 527 | 49 | 0.70 | 0.32 | 535 | 87 | 21.03 | 10.54 |
| llama.cpp 1S | HBM | 504 | 44 | 1.22 | 0.52 | 511 | 43 | 13.81 | 5.94 |
| llama.cpp 1S | DDR | 509 | 47 | 1.24 | 0.55 | 524 | 85 | 24.33 | 11.94 |
| vLLM stock + membind 1S | HBM | - | - | - | - | 542 | 83 | 25.25 | 12.47 |
| vLLM patched 1S | HBM | 549 | 44 | 0.18 | 0.08 | 534 | 44 | 14.30 | 6.48 |
| vLLM 1S | DDR | 537 | 53 | 0.21 | 0.10 | 545 | 80 | 25.15 | 12.67 |
| llama3.x.c 2S | HBM | 720 | 44 | 0.50 | 0.29 | 727 | 44 | 8.02 | 4.73 |
| llama3.x.c 2S | DDR | 725 | 57 | 0.55 | 0.33 | 714 | 127 | 15.22 | 9.60 |
| llama.cpp 2S | HBM | 692 | 44 | 1.38 | 0.79 | 695 | 44 | 30.87 | 17.65 |
| llama.cpp 2S | DDR | 694 | 57 | 1.17 | 0.69 | 695 | 99 | 27.06 | 16.49 |
| vLLM patched 2S | HBM | 684 | 44 | 0.13 | 0.07 | 732 | 45 | 12.74 | 7.55 |
| vLLM 2S | DDR | 686 | 68 | 0.17 | 0.10 | 721 | 106 | 20.09 | 12.56 |

Findings. (1) **Package power is the same everywhere.** An active socket draws 339-366 W in prefill and decode, for every engine, on HBM and on DDR,
at 56 or 112 cores (for example llama3.x.c on one socket: 359 W in HBM decode, 356 W in DDR decode, although HBM moves 400 GB/s and DDR 232 GB/s). The package
looks power-limited, so **energy per token is set by tokens per second**: decode on one socket costs 10.96 J/token on HBM and 21.03 on DDR for llama3.x.c (1.92x), 13.81 and 24.33 for llama.cpp,
14.30 and 25.15 for vLLM; the lowest figure is llama3.x.c on two sockets in HBM, 8.02 J/token (4.73 net). llama.cpp on two sockets is the exception: 30.87 J/token on HBM, worse than DDR (27.06), because
the package stays at about 695 W while the rate drops to 23.9 tok/s. In prefill vLLM needs 0.18 J/token (0.13 on two sockets), llama3.x.c 0.61 (0.50) and llama.cpp 1.22 (1.38).
(2) **HBM power is not in the DRAM domain.** The DRAM RAPL domain measures only the DDR DIMMs: in HBM runs it equals the idle value (22.8 W per socket
against 22.1 W idle, with 0.3 GB/s of DDR traffic) although the socket streams 400 GB/s from HBM, and in DDR runs it rises by about 46 W per socket at 232 GB/s (0.2 W per GB/s, about 25 pJ per bit; llama3.x.c: 68 W at 232 GB/s one socket, 64.5 and 62.1 W
at 213 GB/s each on two sockets). Whether HBM sits inside the package domain cannot be settled from these data: the package is 359 W in HBM decode and 356 W in DDR decode, a difference
of 3 W, but if the package is power-limited its power does not respond to load, so this does not decide it. The conclusion
is that **HBM energy is either inside the package value or not measured at all; it is not in the DRAM value**, so J/token comparisons of HBM against DDR use package + DRAM, which is the only
consistent choice, and the HBM advantage in J/token may be slightly overstated if HBM were not counted (not testable here).

## 7. Q7: HBM speedup over DDR, and Cache mode

HBM node A (job 5892201) over DDR node B (job 5892202), medians of tok/s, **derived**: raw ratio, and in brackets the ratio divided by the node-to-node anchor (llama3.x.c, one socket, HBM,
D, run on both nodes: decode tok/s node A / node B = 1.004, 86-token prefill 0.997, section 8). vLLM on HBM is patched, on DDR stock (the DDR case is
unaffected by the patch: stock already binds DDR). The anchor shifts no cell by more than 0.01.

| Engine and sockets | decode D | long-context L | prefill 2048 | prefill 8192 |
| --- | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | 1.74 (1.74) | 1.68 (1.67) | 1.12 (1.12) | 1.08 (1.08) |
| llama.cpp 1S | 1.63 (1.62) | 1.46 (1.45) | 1.01 (1.01) | 1.00 (1.01) |
| vLLM patched 1S (HBM patched vs DDR stock) | 1.63 (1.62) | 1.59 (1.58) | 1.20 (1.20) | 1.15 (1.15) |
| llama3.x.c 2S | 1.77 (1.77) | 1.64 (1.64) | 1.10 (1.10) | 1.09 (1.09) |
| llama.cpp 2S | 0.81 (0.80) | 0.89 (0.89) | 0.84 (0.84) | 0.84 (0.84) |
| vLLM patched 2S (HBM patched vs DDR stock) | 1.48 (1.48) | 1.41 (1.41) | 1.24 (1.24) | 1.21 (1.22) |

Anchor (llama3.x.c, 1S, HBM, D) node A / node B: decode 1.004, prefill (86 tokens) 0.997. Cells: raw ratio (anchor-corrected).

HBM speeds up decode by 1.48-1.77x (one and two sockets, all engines but llama.cpp on two sockets, where it loses: 0.81x) and long-context decode by 1.41-1.68x (llama.cpp on two sockets 0.89x), but prefill by
only 1.00-1.24x (llama.cpp on two sockets 0.84x): prefill is bound by AMX and the surrounding instructions, not by memory. The HBM-over-DDR gain is similar across engines on one socket (1.63-1.74x) and is a statement about
the engine's use of the bandwidth: llama3.x.c turns 1.72x of extra bandwidth into 1.74x, llama.cpp 1.60x (351 / 219) into 1.63x, vLLM 1.61x (319 / 198) into 1.63x. The HBM / DDR rooflines differ by
2.1x; no engine reaches it because on one socket the HBM runs reach 58-73% of their roofline and the DDR runs 76-89% of theirs.

Cache mode (Cache node, job 5892196, out of the box, no binding) against Flat DDR and Flat HBM, medians of tok/s (**derived** ratios). The Cache node is a different physical node and has no
anchor run, so the ratios carry the node-to-node difference that section 8 bounds at about 1% for the two Flat nodes but not for the Cache node.

| Engine | Metric | Flat DDR | Cache | Flat HBM | Cache / DDR | Flat HBM / Cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama3.x.c 1S | decode D tok/s | 29.8 | 46.3 | 51.9 | 1.55 | 1.12 |
| llama3.x.c 1S | long-context L tok/s | 22.2 | 32.5 | 37.2 | 1.47 | 1.14 |
| llama3.x.c 1S | prefill 2048 tok/s | 830.7 | 831.8 | 931.5 | 1.00 | 1.12 |
| llama3.x.c 1S | prefill 8192 tok/s | 520.3 | 511.1 | 561.7 | 0.98 | 1.10 |
| llama3.x.c 2S | decode D tok/s | 55.9 | 79.2 | 99.1 | 1.42 | 1.25 |
| llama3.x.c 2S | long-context L tok/s | 40.4 | 55.6 | 66.4 | 1.38 | 1.19 |
| llama3.x.c 2S | prefill 2048 tok/s | 1426.8 | 1336.7 | 1563.5 | 0.94 | 1.17 |
| llama3.x.c 2S | prefill 8192 tok/s | 949.2 | 954.3 | 1032.8 | 1.01 | 1.08 |
| llama.cpp 1S | decode D tok/s | 24.9 | 32.8 | 40.6 | 1.32 | 1.24 |
| llama.cpp 1S | long-context L tok/s | 20.0 | 24.7 | 29.1 | 1.23 | 1.18 |
| llama.cpp 1S | prefill 2048 tok/s | 445.3 | 442.1 | 448.9 | 0.99 | 1.02 |
| llama.cpp 1S | prefill 8192 tok/s | 273.7 | 275.0 | 274.7 | 1.00 | 1.00 |
| llama.cpp 2S | decode D tok/s | 29.7 | 19.3 | 23.9 | 0.65 | 1.24 |
| llama.cpp 2S | long-context L tok/s | 22.6 | 15.9 | 20.1 | 0.70 | 1.27 |
| llama.cpp 2S | prefill 2048 tok/s | 642.1 | 358.9 | 537.0 | 0.56 | 1.50 |
| llama.cpp 2S | prefill 8192 tok/s | 457.0 | 261.2 | 381.9 | 0.57 | 1.46 |
| vLLM 1S | decode D tok/s | 24.9 | 34.1 | 40.5 | 1.37 | 1.19 |
| vLLM 1S | long-context L tok/s | 22.2 | 30.3 | 35.3 | 1.36 | 1.17 |
| vLLM 1S | prefill 2048 tok/s | 2767.5 | 3067.6 | 3315.1 | 1.11 | 1.08 |
| vLLM 1S | prefill 8192 tok/s | 2621.3 | 2822.8 | 3003.6 | 1.08 | 1.06 |
| vLLM 2S | decode D tok/s | 41.2 | 50.2 | 61.2 | 1.22 | 1.22 |
| vLLM 2S | long-context L tok/s | 36.7 | 43.8 | 51.8 | 1.19 | 1.18 |
| vLLM 2S | prefill 2048 tok/s | 4491.1 | 5129.5 | 5557.0 | 1.14 | 1.08 |
| vLLM 2S | prefill 8192 tok/s | 4886.8 | 5591.2 | 5925.1 | 1.14 | 1.06 |

Cache mode recovers most of HBM's decode gain without any placement: llama3.x.c reaches 46.3 tok/s on one socket (75% of the way from DDR's 29.8 to HBM's 51.9) and 79.2 on two (54%), llama.cpp 32.8
(51% on one socket), vLLM 34.1 (59%) and 50.2 on two (45%). It gives **nothing in prefill for llama3.x.c and llama.cpp** (0.94-1.01 against DDR) and 1.08-1.14x for vLLM. llama.cpp on two sockets is slower in Cache mode than in either Flat mode
(19.3 against 23.9 and 29.7 tok/s), but this is a different layout: on the Cache node it ran the untuned configuration (112 threads, no interleave; `numa_maps` 6.9 GiB on node 0 and 2.1 GiB on node 1), not the
tuned 64-thread interleave of the Flat nodes, so that row says nothing about Cache mode itself. The 8 GB working set fits in the 64 GB per-socket HBM cache, so the Cache result is the in-cache case; the 96 GB case of `hbm-roofline.md` is not covered.

## 8. Perf overhead, the anchor, and output sanity

**Perf overhead.** Repeats 1 (MEM) and 2 (COMPUTE) run under `perf stat -a -I 200`; repeat 3 runs without perf. The ratio of the rate of the perf repeat over repeat 3 for the same cell, over
the decode (D, L) and prefill (P, L, LP) cells (**derived**; the three repeats are sequential, so a first-run effect is mixed in):

| Tier | Phase | cells | repeat 1 / repeat 3 (MEM or USER group) | repeat 2 / repeat 3 (COMPUTE group) |
| --- | ---: | ---: | ---: | ---: |
| HBM | decode | 13 | 0.999 (0.980-1.025) | 0.999 (0.992-1.012) |
| HBM | prefill | 12 | 1.001 (0.972-1.021) | 0.998 (0.939-1.009) |
| DDR | decode | 12 | 0.999 (0.936-1.042) | 0.994 (0.935-1.002) |
| DDR | prefill | 12 | 0.994 (0.978-1.021) | 0.998 (0.953-1.008) |
| Cache | decode | 12 | 0.985 (0.896-1.157) | 1.000 (0.976-1.185) |
| Cache | prefill | 12 | 0.992 (0.855-1.069) | 0.999 (0.898-1.026) |

The overhead on the Flat nodes is below measurement noise: median ratios 0.994-1.001, extremes 0.935-1.042 over 12-13 cells per row. On the Cache node, where repeat 1 is the first run of every configuration, the spread is wider (0.855-1.157 for repeat 1
over repeat 3, up to 1.185 for repeat 2 over repeat 3), pointing at a first-run effect and a noisier node rather than counter cost; the medians of the three repeats used elsewhere
do not depend on it.

**Anchor (llama3.x.c, one socket, HBM, D) on node A and node B** (**measured**, 3 repeats each, jobs 5892201 and 5892202):

| Node | Job | decode tok/s repeats 1/2/3 | median | prefill (86 tokens) tok/s repeats 1/2/3 | GB/token (repeat 1) | package W (repeat 1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| node A | 5892201 | 52.58 / 51.83 / 51.87 | 51.87 | 957 / 947 / 934 | 7.72 | 521 |
| node B | 5892202 | 51.60 / 51.66 / 52.09 | 51.66 | 947 / 950 / 952 | 7.81 | 518 |

The two nodes agree within 0.4% in decode (51.87 against 51.66 tok/s), 0.3% in prefill and 1.2% in GB/token, so the Flat HBM against Flat DDR comparison is not distorted by the node.

**Spread of the three repeats** (range / median in percent of D, L, P and 8192-token prefill rates):

| Configuration | Tier | range/median of D, L, P, 8192 (%) |
| --- | ---: | ---: |
| llama3.x.c 1S | HBM | 1.5% / 0.7% / 2.4% / 0.4% |
| llama3.x.c 1S | DDR | 0.1% / 0.7% / 2.0% / 0.7% |
| llama3.x.c 1S | Cache | 1.8% / 0.1% / 14.5% / 2.9% |
| llama.cpp 1S | HBM | 0.8% / 0.9% / 0.9% / 0.3% |
| llama.cpp 1S | DDR | 0.7% / 5.9% / 0.8% / 4.8% |
| llama.cpp 1S | Cache | 0.1% / 2.5% / 0.6% / 1.8% |
| vLLM stock + membind 1S | HBM | 0.2% |
| vLLM patched 1S | HBM | 0.8% / 0.3% / 3.4% / 0.6% |
| vLLM 1S | DDR | 0.6% / 0.4% / 2.1% / 0.5% |
| vLLM 1S | Cache | 0.7% / 1.3% / 2.9% / 0.7% |
| llama3.x.c 2S | HBM | 3.0% / 1.3% / 6.2% / 0.7% |
| llama3.x.c 2S | DDR | 0.2% / 0.6% / 2.2% / 0.5% |
| llama3.x.c 2S | Cache | 0.8% / 5.2% / 11.3% / 3.5% |
| llama.cpp 2S | HBM | 0.7% / 2.5% / 1.2% / 1.8% |
| llama.cpp 2S | DDR | 1.3% / 6.9% / 1.0% / 1.6% |
| llama.cpp 2S | Cache | 16.0% / 14.9% / 7.4% / 10.9% |
| vLLM patched 2S | HBM | 1.3% / 0.8% / 0.5% / 0.7% |
| vLLM 2S | DDR | 1.0% / 0.2% / 1.5% / 0.4% |
| vLLM 2S | Cache | 2.6% / 3.7% / 2.4% / 0.4% |

**Output sanity.** The campaign checked every cell for the exact prompt and generated token counts (all passed) and every llama.cpp configuration for a 32-token greedy text:
the six `llama-completion` runs (one and two sockets, HBM, DDR and Cache) produce byte-identical output, a coherent continuation ("The capital of France is Paris. The capital of Italy is Rome.
The capital of Spain is Madrid. ..."). The llama3.x.c and vLLM greedy texts were checked in the setup jobs (not in the campaign): llama3.x.c on a chat prompt and a raw literary prompt gives
coherent English, vLLM a coherent continuation in all five of its setup configurations, and the campaign's own vLLM warm-up on the repeated D prompt continues it (with the repetition that greedy decoding gives
on a repeated prompt) in every configuration. No engine produced garbage; the texts need not agree token for token (different quantisations and token ids), and llama.cpp times random tokens by design (plan D4).

## 9. Surprises and caveats

1. **Stock vLLM cannot reach HBM in Flat mode** (section 3): `numactl --membind` is ignored, 24.7 tok/s is the DDR rate. All "vLLM on HBM" numbers are from a one-line patched variant.
2. **llama.cpp is slower on two sockets than on one in HBM** (0.59x, section 4), the result of its single-process interleaved design against the roughly 50 GB/s remote HBM limit, even with the best of
   eight tried layouts. No tuning of threads and interleaving changed that in the tuning job, which also shows that `--numa distribute` is worse (13.0 tok/s).
3. **vLLM prefill is 3.6x faster than llama3.x.c's** (3315 against 932 tok/s on one socket, HBM, 2048 tokens; 3.6x on two sockets; 5.3x and 5.7x at 8192 tokens, 3004 against 562 and 5925 against 1033 tok/s), although llama3.x.c's AMX cycles are ideal. llama3.x.c in turn is 2.1x (one socket) and 2.9x (two sockets) faster than llama.cpp. llama3.x.c's
   prefill is dominated by non-AMX work (section 5). This is the clearest improvement target the campaign found.
4. **The HBM gain in prefill is small** (1.0-1.24x, and 0.84x for llama.cpp on two sockets), so a prefill-heavy workload sees little from HBM; decode sees 1.4-1.8x.
5. **Package power is constant at about 350 W per active socket;** energy per token follows speed (section 6). HBM power is not in the DRAM domain.
6. **The earlier repository HBM byte figures that use 64 B per `hbm_cas_rd` are 2x too high** (section 3, side finding).
7. **Cache mode is far better than the plan's "DDR-only" reading suggests:** the working set fits in the HBM cache and decode reaches 75% of the HBM gain for llama3.x.c; prefill does not benefit.
8. **Caveats of the method.** (a) llama.cpp times random tokens; the other two use identical token ids (plan D4). (b) The decode phase of llama.cpp is all 128 tokens, those of llama3.x.c and vLLM the 127 after the first;
   at 40 tok/s that is under 1%. (c) The "D" prompt has 86 tokens, not about 128. (d) The 86-token prefill (D rows, 0.06-0.17 s) is dominated by thread-team start-up and
   noisy (llama3.x.c two sockets, HBM: 851 tok/s in repeat 1 against 1566 and 1584; the same pattern, repeat 1 slowest, in DDR and Cache) and is not used for any headline. (e) Counter figures are one run per
   cell (repeat 1); phases shorter than about 1 s (llama3.x.c two-socket decode: 1.3 s) carry about 0.1 s of window-edge uncertainty. (f) Counter windows of seven Cache-node rows cover 91-98% of the phase
   (the wrapper-mode perf clock); only AMX counts are taken from them. (g) The Cache node results are timing and user-mode AMX counters only; it is a different physical node from node A and node B. (h) The DDR roofline of 262 GB/s is the
   lower end of the 262-268 GB/s of `hbm-roofline.md`. (i) vLLM on two sockets uses 110 of the 112 cores (55 per rank, the documented layout), llama.cpp on two sockets 64. (j) Cache-node llama.cpp on two sockets ran an untuned layout (section 7). (k) llama3.x.c's
   Cache-node runs set `LLAMA_PLACEMENT=ddr`, because the engine's HBM placement needs HBM nodes that Cache mode does not expose; the memory-side cache does the rest. (l) vLLM's TTFT and token times are taken at the client over HTTP streaming and include the API round trip (not quantified; the other engines use internal timers). (m) Single request, batch 1, one model; no accuracy study.

## 10. Follow-ups

1. Profile the llama3.x.c prefill: the non-AMX 95% of the cycles (the Q8_0 group epilogue and the attention) are where the 3.6x to vLLM sits. A per-channel or per-row-scale prefill path, and AMX attention, are the candidates.
2. llama.cpp on two sockets: one process per socket with a split (tensor parallel through its RPC backend or a pipeline over two instances), to test whether it can leave the UPI limit; vLLM's two-socket scaling (1.51x in HBM) would also gain from finding the fixed 4 ms (3.9 ms measured).
3. Report the vLLM node-binding behaviour upstream or keep the patch with the campaign; check it with the next vLLM release.
4. G = 1 runs for the cross-run uniform TPOT definition, if the two definitions are to be compared on more than within-run data.
5. Run an anchor cell on the Cache node and repeat Cache mode with privileged counters (CAS, RAPL), and a working set larger than the HBM cache.
6. Correct the 64 B/CAS HBM byte figures of the earlier documents.
7. SNC4 and a bigger batch, as listed out of scope in the plan.
