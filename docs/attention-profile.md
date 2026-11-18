# Attention profile (phase E0)

**Status: run (2026-11-18, job 5860637, commit 397aa10; one HBM node in Flat mode).** The attention microbenchmark
(`tests/attention_bench.c`), the row-block sweep, the half-KV-head shapes, the decode bench and two `perf stat` runs ran on one
node. **The value gate failed: the bench reproduces 77% of the profiled engine attention time (gate: within 20%)**, so the bench
is a *relative* instrument for split against tiled, and the engine's profiled attention time is the yardstick for E2's value
gate (section 3). The hardware counters answer H1: prefill attention is instruction and compute bound inside the core, at 11-13%
of the FP32 peak, not memory bound (section 2). Provenance, gates and gaps: [the validation record](attention-profile-validation.md).
Plan: [E0](plans/E-followups/README.md#e0-measure-attention-before-changing-it). Branch of the code: `feature/attn-e0-bench`.

Medians of 3 repetitions (decode: 200 calls), ranges in the source tables (`E0/summary.md` of the job). One socket, 56 threads unless
a column says otherwise; K/V in HBM node 2; AVX-512 attention (`LLAMA_KERNELS=avx512`); FP32 peak 8.8 TFLOP/s and HBM 552 GB/s per socket
([hbm-roofline.md](hbm-roofline.md)). The "% of peak" column is always against the 56-thread peak, so the 14- and 28-thread rows
read low by construction.

## 1. Result in one paragraph

Prefill attention runs at 1.0-1.2 TFLOP/s at 56 threads (11-13% of the FP32 peak) for prompts of 2048 and 8192 tokens, with IPC 1.92,
0.203 instructions per FLOP, 6.1 L2 misses and 0.000 L3-miss loads per KFLOP: the cores are busy executing instructions and the memory
system is not the limit (H1 confirmed; H2 not isolated). A row block of 16 instead of 4 gives 1.24x, under the 1.3x that would make
it a separate first step (H3), and is bitwise identical. The tail of a wave is at most 1% at 8192 tokens with M of 64 or 256, and 10-16%
at 512 tokens with M 16 (H4). Decode attention reaches 16-43% of the HBM peak and leaves workers idle at 8192 context (tail
14-30%), so decode is limited by the kernel and the chunk plan, not by HBM (H5, counters missing). With half the KV heads the layer is
2.1-2.35x faster at 8192 tokens, i.e. at least linear scaling (H6). The cross-check against the engine failed (0.77): the bench
under-estimates the engine's attention time by 23%, probably because its K/V stays in the L3 between calls (section 3).

## 2. Measurements

### 2.1 Prefill, one layer, whole prompt, 56 threads

`bench-prefill.jsonl`: the whole prompt of one layer in microbatches of M rows, as `run.c` does it. GFLOP/s uses
`4 · n_heads · head_dim · Σ rows (pos + 1)`. "Busy" is the mean share of the task phases a worker spent in tasks; "tail" is the bench's tail fraction (workers idle at the end of the task phases: the
imbalance at the end of a wave). The full matrix (14 / 28 threads, 54 cells
per model) is in `E0/summary.md`.

| shape | prompt | M | ms/layer | GFLOP/s | % of FP32 peak | streamed / unique K/V | busy mean | tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8b | 512 | 16 | 4.95 | 434 | 5% | 66 | 81% | 12% |
| 8b | 512 | 64 | 3.15 | 683 | 8% | 66 | 82% | 4% |
| 8b | 512 | 256 | 2.88 | 747 | 8% | 66 | 94% | 1% |
| 8b | 2048 | 16 | 30.94 | 1111 | 13% | 259 | 88% | 9% |
| 8b | 2048 | 64 | 29.02 | 1185 | 13% | 259 | 95% | 3% |
| 8b | 2048 | 256 | 29.05 | 1183 | 13% | 259 | 98% | 1% |
| 8b | 8192 | 16 | 547.08 | 1005 | 11% | 1031 | 95% | 4% |
| 8b | 8192 | 64 | 541.17 | 1016 | 12% | 1035 | 98% | 1% |
| 8b | 8192 | 256 | 540.48 | 1017 | 12% | 1050 | 99% | 1% |
| 1b | 512 | 16 | 3.45 | 312 | 4% | 66 | 75% | 14% |
| 1b | 512 | 64 | 2.77 | 389 | 4% | 66 | 82% | 12% |
| 1b | 512 | 256 | 2.66 | 405 | 5% | 66 | 90% | 2% |
| 1b | 2048 | 16 | 18.98 | 906 | 10% | 259 | 85% | 9% |
| 1b | 2048 | 64 | 16.24 | 1059 | 12% | 259 | 92% | 3% |
| 1b | 2048 | 256 | 16.63 | 1034 | 12% | 259 | 95% | 1% |
| 1b | 8192 | 16 | 281.92 | 975 | 11% | 1031 | 93% | 4% |
| 1b | 8192 | 64 | 279.52 | 984 | 11% | 1031 | 97% | 1% |
| 1b | 8192 | 256 | 277.57 | 990 | 11% | 1037 | 98% | 0% |

Reading: GFLOP/s is flat at about 1.0-1.2 TFLOP/s once the prompt is 2048 tokens or longer, nearly independent of M and the same for
8b and 1b (head_dim 128 and 64): the kernel runs at the same rate whatever it is fed.
M barely matters for time (8b at 8192: 547 / 541 / 540 ms for M = 16 / 64 / 256, 1.2% apart, ranges overlapping).
Each K/V byte is streamed into L2 about 1050 times at 8192 tokens (8b, M 256), and the work is 7.8 FLOP per streamed byte
(`flops_per_streamed_byte` of the job's record).

Thread scaling (medians, 8b / 1b, ms per layer):

| prompt | M | 14 threads | 28 threads | 56 threads | 14 to 56 (ideal 4.0) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 64 | 9.06 / 6.40 | 5.79 / 3.86 | 3.15 / 2.77 | 2.9x / 2.3x |
| 2048 | 64 | 109.81 / 50.07 | 51.39 / 25.82 | 29.02 / 16.24 | 3.8x / 3.1x |
| 8192 | 256 | 1531.14 / 747.62 | 829.78 / 396.43 | 540.48 / 277.57 | 2.8x / 2.7x |

Over all cells 14 to 56 threads gives 2.7-3.0x at 8192 tokens, 2.7-3.8x at 2048 and 1.9-2.9x at 512; 28 to 56 threads gives only
1.4-1.6x at 8192. Efficiency per thread falls with the thread count (about 70% at 56 threads against 14), and most at short prompts.

### 2.2 Row block (H3), 8b, 8192 tokens, M 256, 56 threads

`rowblock.jsonl`; the engine's default is 4 rows per task (`LLAMA_ATTN_ROW_BLOCK`).

| rows per task | ms/layer (min-max) | vs 4 | tasks/layer | streamed / unique K/V | busy mean | tail |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 764.09 (760.03-764.69) | 0.70x | 1,124,352 | 4096.5 | 97% | 0% |
| 2 | 657.26 (656.23-658.48) | 0.82x | 565,832 | 2066.7 | 99% | 0% |
| 4 | 538.52 (535.86-538.54) | 1.00x | 286,112 | 1050.0 | 99% | 1% |
| 8 | 467.14 (467.06-479.70) | 1.15x | 145,728 | 539.6 | 98% | 2% |
| 16 | 432.96 (412.31-459.39) | 1.24x | 76,072 | 286.4 | 96% | 3% |

The gain follows the streamed bytes (down 3.7x from 4 to 16 rows) only weakly: 3.7x fewer bytes streamed into L2 buys 1.24x in time, so
K/V streaming is a minor share of the time (consistent with H1). The 16-row gain is beyond variability (ranges do not overlap) but below the
1.3x of the gate. The results are bitwise identical: the engine comparison of row block 4 against 16 (8B FP32, 2048 tokens, M 64, last
row) gives `bitwise=True, max_abs=0.0, argmax agreement 100%` (`compare-8b-fp32-e0-rowblock-4-vs-16-m64-prompt2048.json`).

### 2.3 Half the KV heads (H6), 56 threads

`bench-prefill-kv4.jsonl` against the 8-KV-head rows of section 2.1: one socket's share of a two-socket model (KV heads and Q heads halved,
head_dim unchanged), same thread count and same prompt.

| shape | prompt | M | ms/layer, 8 KV heads | ms/layer, 4 KV heads | ratio (2.00 = half the work in half the time) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1b | 2048 | 64 | 16.24 | 14.13 | 1.15 |
| 1b | 2048 | 256 | 16.63 | 9.87 | 1.69 |
| 1b | 8192 | 64 | 279.52 | 125.30 | 2.23 |
| 1b | 8192 | 256 | 277.57 | 133.73 | 2.08 |
| 8b | 2048 | 64 | 29.02 | 18.44 | 1.57 |
| 8b | 2048 | 256 | 29.05 | 14.12 | 2.06 |
| 8b | 8192 | 64 | 541.17 | 233.52 | 2.32 |
| 8b | 8192 | 256 | 540.48 | 229.72 | 2.35 |

At 8192 tokens the ratio is 2.08-2.35, above 2: the half-size problem runs faster per unit of work (the bench's own K/V halves from 64 to 32 MiB
for 8b). At 2048 tokens it is 1.15-2.06 and not monotone in M (1b: 1.15 at M 64, 1.69 at M 256). Two
sockets therefore scale at least linearly for attention in this bench, with no cross-socket cost in the picture; whether the cause is the cache
fit of the K/V is open (section 4, H6).

### 2.4 Decode, one layer, last position (H5)

`bench-decode.jsonl`: `k_attention` at the last position of a context, 200 calls per cell, cycling through 1 GiB of K/V copies (`--cold-mib` 1024)
so that every call reads memory the previous one left cold. The bench runs the whole socket's K/V of one layer; "4 KV heads" is a two-socket half.

| shape | KV heads | context | chunks | tasks | ms/layer median (min-max) | GB/s median | % of 552 GB/s | busy mean | tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1b | 4 | 8192 | 32 | 2048 | 0.19 (0.08-0.23) | 89 | 16% | 64% | 30% |
| 1b | 4 | 32768 | 128 | 8192 | 0.64 (0.26-0.68) | 105 | 19% | 92% | 3% |
| 1b | 8 | 8192 | 32 | 4096 | 0.31 (0.12-0.38) | 109 | 20% | 75% | 19% |
| 1b | 8 | 32768 | 128 | 16384 | 0.66 (0.54-1.15) | 202 | 37% | 93% | 2% |
| 8b | 4 | 8192 | 32 | 2048 | 0.27 (0.15-0.41) | 126 | 23% | 68% | 28% |
| 8b | 4 | 32768 | 128 | 8192 | 0.56 (0.47-0.60) | 239 | 43% | 90% | 5% |
| 8b | 8 | 8192 | 32 | 4096 | 0.36 (0.24-0.88) | 187 | 34% | 73% | 14% |
| 8b | 8 | 32768 | 128 | 16384 | 1.20 (0.92-1.79) | 224 | 41% | 94% | 3% |

Every cell is below 45% of the HBM peak; none comes near 80%. At 8192 context the layer takes 0.2-0.4 ms, workers are idle 14-30% of the task
phases and the cells are probably limited by dispatch and the chunk plan (32 chunks for 56 workers). At 32768 the workers are busy (90-94%)
and the bandwidth is still 19-43% of the peak, so the kernel itself does not draw the bytes fast enough; the 1b cells with 4 KV heads (16-19%) are the
slowest per byte. The best call of a cell is up to 2.5x faster than the median at 8192 (min column), so per-call wake-up and barrier time is
part of the median at these sizes. The instruction-bound part of H5 could not be confirmed with counters (section 2.5).

### 2.5 Counters (H1, H2)

`perf stat`, per-core events summed over socket 0 (the socket the bench ran on) for the whole launch, bench start-up included
(`counters.csv`). Per pass: 549.8 GFLOP of the cell times the 5 passes a launch makes (warm-up, 3 repetitions, traced pass).

| cell | cycles | instructions | IPC | instructions / FLOP | l1d.replacement / KFLOP | l2_rqsts.miss / KFLOP | l3_miss loads / KFLOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill 8b, 8192, M 256 | 2.89e11 | 5.57e11 | 1.92 | 0.203 | 31.79 | 6.06 | 0.000 |
| decode 1b, 32768 | - | - | - | - | - | - | - |

The decode run produced no usable counters (section 5 of the validation record: the bench exited on an argument error). Socket 1 counted
only start-up (7.7e7 cycles).

- **Instruction mix.** 0.203 instructions per FLOP is 4.9 FLOP per instruction, 15% of the 32 FLOP an AVX-512 FP32 FMA retires; with IPC 1.92 the cores
  are issuing close to two instructions per cycle on average. This is consistent with H1's picture (horizontal reductions, divisions, stores and reloads
  of scores, broadcasts around the FMAs). (Which instructions was not measured: the event set has no instruction mix.)
- **Cache traffic.** 31.8 L1D replacements per KFLOP is 2.0 bytes of L2-to-L1 traffic per FLOP (64-byte lines). At 1.0-1.2 TFLOP/s that is
  about 2.0-2.4 TB/s over 56 cores, about 36-43 GB/s per core, or roughly 19-22 bytes per cycle at 1.9 GHz (**derived**; the node's
  lscpu maximum clock); a core can take 64 bytes per cycle from L2 at best. L2 misses are 6.1 per KFLOP, and l3_miss loads are
  285,000 over 2.75 TFLOP, so the L2 and L3 serve the working set. DRAM and HBM read traffic over the launch: 60.6 million CAS reads on
  socket 0's HBM (3.9 GB for 5 passes, 0.78 GB per pass, against 70 GB streamed per pass): the K/V of this bench comes from cache.

## 3. Cross-check against the engine: failed

| | value |
| --- | ---: |
| bench, 8b / 8192 tokens / M 256 / 56 threads | 540.48 ms per layer (range 540.23-554.51) |
| bench times 32 layers | 17.30 s |
| profiled engine, same configuration (8B, AMX-BF16 GEMMs, 8192-token prompt = 8226 tokens after the chat template, M 256, `LLAMA_PROFILE=1`) | attention 22.42 s of a 30.39 s prefill (73.8%) |
| ratio bench / engine | **0.77** (gate: 0.80-1.20) |

Source: `engine-profile.json` (`profile_seconds.prefill`). The engine prefill has 8226 tokens: scaling the bench by (8226/8192)² = 1.0083 for the
extra work gives 17.44 s and a ratio of 0.78. The gate is missed either way. Other engine phases of that run, in seconds: qkv 1.53, wo 0.76,
w1/w3/SwiGLU 2.59, w2 2.98, everything else under 0.1; non-attention time is 7.97 s. The attention kernel in the engine runs at
17.7 TFLOP / 22.42 s = 0.79 TFLOP/s (9% of the FP32 peak), the bench at 1.02 TFLOP/s.

**Explanation to be confirmed (hypothesis).** The bench reuses the K/V of one layer for every microbatch and every repetition: 8192 positions x 8 KV
heads x 128 x 4 B x 2 (K and V) = 64 MiB, well inside the 112.5 MiB of L3 of the socket (the node has 225 MiB over two sockets); the counters agree
(l3_miss 0.000 per KFLOP, 0.78 GB of HBM reads per pass against 70 GB streamed). In the engine the 32 layers each
have their own 64 MiB of K/V and the weights of a layer (about 436 MB at BF16 for 8B, **derived** from 218 M parameters) are streamed between two attention calls, which evicts
the K/V from L2 and L3, so the engine's attention reads its K/V from HBM. That would explain an engine that is slower than the bench
by about 30% (700 ms against 540 ms per layer); the size of the effect is not predicted by the argument, only its direction. Other candidates that the data do not exclude: the clock after AMX GEMM phases, the engine's
workspace waves with scheduling differences, and 8226 against 8192 tokens (0.8%, accounted for above). The decode bench already avoids the
effect with `--cold-mib` (default 1024); the prefill mode does not use it.

**Consequences.**

1. The E2 value gate uses the engine's profiled attention time: the E2 bench stage measures TTFT with `LLAMA_PROFILE=1`, `split` against `tiled`, on 8B/8192
   (22.42 s is the baseline; to reach TTFT 1.5x, 20.26 s, attention must fall to at most 12.3 s, a 1.8x attention speedup; for 2.0x, 15.19 s, attention at
   most 7.2 s, 3.1x; **derived** from the job's profile).
2. The bench is used only for relative comparison of split and tiled (same K/V residency for both), never as an absolute time.
3. Follow-up (proposed, not implemented): give the prefill mode a `--cold-mib N` as the decode mode has it: allocate enough K/V copies, or
   stream a flush buffer of N MiB between layers, so each pass starts with the K/V out of the caches. If the cold bench then lands within 20%
   of the engine, the hypothesis is confirmed and the bench becomes an absolute yardstick; if not, the other candidates are next. This costs one
   short bench job and should run before E2's bench stage.

## 4. Hypotheses and the E0.6 decision table

| ID | Verdict | Evidence | Consequence (E0.6 decision table) |
| --- | --- | --- | --- |
| H1 instruction-bound | **confirmed** | 11-13% of FP32 peak at 56 threads, IPC 1.92, 0.203 instr/FLOP, L3 miss 0.000/KFLOP, 6.1 L2 misses/KFLOP | E2 as planned: register tiling, no horizontal reductions, L1-sized position tiles |
| H2 L2-to-L1 bound | **not isolated** | 2.0 B/FLOP of L1D replacement, about 19-22 B/cycle/core, below 64 B/cycle: not saturated, but substantial. The chunk-size experiment (64 positions) was not run | E2 as planned (its tile design addresses it); the E2 counters will show it |
| H3 row block | **rejected as a separate step** | 16 rows: 1.24x (412-459 ms against 536-539 ms), below 1.3x; bitwise identical | not landed alone; the tiled kernel uses blocks of 16 query vectors anyway |
| H4 tail | **only for small prompts** | tail 0-1% at 8192 with M 64 or 256, 4% with M 16; at 512 tokens and M 16 it is 10-16% (56 threads: 12-14%), at 2048 and M 16 it is 4-9%; above 15% only for 1b at 512, M 16, with 14 and 28 threads (16%, 15%) | longest-first order is a small-prompt concern; E2 includes it as planned (E2.3 step 3) at low priority for 8192 |
| H5 decode | **partly** | 16-43% of 552 GB/s, tail 14-30% at 8192, 2-5% at 32768; counters failed | below 50%: E5 starts with the kernel (group kernel), not BF16 KV, and with the chunk plan at short contexts; the "instruction-bound" half needs the counter rerun |
| H6 two sockets | **confirmed in effect** | 2.08-2.35x at 8192 (8b 2.32-2.35), 1.15-2.06x at 2048 | the 2.05-2.15x of D2 is reproduced by half the work alone; record in E4 and expect the surplus to shrink once attention is fast. The cache-fit mechanism is unconfirmed (L3 misses are about 0 at full size already) |

Gate by gate (E0.6): row-block bitwise **pass**; value (cross-check) **fail**, with the explanation of section 3; the decision table is filled
above.

## 5. What E2 should do

1. Go ahead as planned: the kernel is instruction bound at 11-13% of the FP32 peak, so a register-tiled kernel without per-score reductions and
   with L1-sized position tiles has a large margin (arithmetic only: 40% of the peak would be 3.5x the current rate).
2. Measure the E2 value gate on the engine: `LLAMA_PROFILE=1` TTFT for `split` and `tiled`, 8B and 1B, 2048 and 8192, and compare the profiled attention seconds with
   22.42 s (8B, 8192). The bench only supplies relative numbers and the counters (instructions/FLOP, L1D replacement/KFLOP) of both kernels.
3. Use 16-query-vector blocks in the tiled kernel (H3: 1.24x for the split kernel from the K/V reuse alone, bitwise). Do not land row block 16 for `split`.
4. Include longest-first task order for small prompts and M 16 (H4); do not expect it to matter at 8192.
5. Add the cold prefill mode (`--cold-mib`) to the bench first, and rerun the 1b decode counters with `--reps 1000` (section 5 of the
   validation record) in the same job as the E2 bench stage.
6. Expect two-socket gains over one socket above 2x to shrink once attention is fast (H6, for E4).
