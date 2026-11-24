# gs64 GEMM efficiency (24 November 2026)

W8A8 Q8_0 gs=64 batched prefill on AMX-INT8 (8B, `int8-v3.bin`, M = 1024). Before this work the tiled kernel ran at
21–33 TOP/s in situ on one socket, against 37–60 for the per-channel format ([prefill-gap.md](prefill-gap.md) §3).

**Result: TTFT −8 to −10% at P and L on one and two sockets, bitwise.** Two changes on branch `perf/gs64-gemm`:

| Change | Knob (default first) | P 1S | P 2S | L 1S | L 2S |
| --- | --- | ---: | ---: | ---: | ---: |
| Before (pipelined, chunk order) | — | 1.302 s | 0.760 s | 6.627 s | 3.820 s |
| Tile store issued one half-step late | `LLAMA_Q8_GEMM=skewed` (`pipelined`, `chunk`) | 1.285 s | 0.756 s | 6.571 s | 3.713 s |
| + row block outer within each K-block | `LLAMA_Q8_KB_ORDER=rows` (`chunks`) | **1.170 s** | **0.690 s** | **6.079 s** | **3.494 s** |
| Change, both | | −10.2% | −9.3% | −8.3% | −8.5% |

Medians of 15 interleaved repeats, job 5910223 (hbm13, tree 9c292f6). Both changes are bitwise and on by default; the
old behaviour is `LLAMA_Q8_GEMM=pipelined LLAMA_Q8_KB_ORDER=chunks`.

The kernel change (the skewed store) gives 1–3%. The loop order gives the rest: the kernel was waiting on activation
data from LLC, not on its own instruction schedule.

## 1. Where the time went (prototype, job 5910129)

A standalone prototype of the pipelined kernel (`amx_q8_proto2.c`, not engine code) timed one 16 x 32 half-step of
64 K (65536 operations) in several variants. Cycles are derived from the measured rate at 2.0 GHz nominal.

| Variant | L2-resident, 1 thread | L2-resident, 56 threads | Beyond L2, 1 thread | Beyond L2, 56 threads |
| --- | ---: | ---: | ---: | ---: |
| v2: the engine's pipelined kernel | 135 | 135 | 158 | 293 |
| v3: full-K int32 accumulation (per-channel ceiling) | 47 | 49 | 124 | 231 |
| v5: v2 with the tile store one half-step late (skewed) | 104 | 125 | 140 | 291 |
| v6: tile products and stores only, no rescale | 89 | 92 | 125 | 249 |
| v8: 16 x 16 outputs in zmm registers over all K | 139 | 150 | 193 | 319 |
| v9: v8 over K-blocks of 8 / 16 / 32 groups | 133–142 | 145–150 | 178–196 | 316–317 |

L2-resident: A 1 MiB + B 512 KiB. Beyond L2: A 4 MiB + B 1 MiB, the shape of the engine's K-blocked pass.

- **The per-group tile stores dominate the kernel.** With no rescale at all (v6) a half-step still costs 89 cycles,
  against 47 when the int32 sums stay in tiles over all K (v3). Every 64 K the gs64 path must store two 16 x 16 int32
  tiles (2 KiB), reload them into zmm, convert and rescale, and read-modify-write 2 KiB of FP32 output.
- **A store that waits on its own products stalls.** Issuing half-step h's store after half-step h+1's tile products
  (v5) saves 31 cycles on one core when the data is in L2 (+30%), 10 cycles at 56 threads.
- **Keeping the outputs in registers does not help** (v8, v9): no spills (objdump), but the per-group
  `cvt`/`mul`/`fma` chain is the same and the tile→register path still goes through memory.
- **Beyond L2 at 56 threads every variant converges** (291–319 cycles; v3 231). The skewed store gains nothing there
  (293 → 291). Data movement, not the kernel, sets the rate, and that regime matches the engine's in-situ 21–33 TOP/s.

So the engine work had two parts: the cheap, bitwise kernel fix (v5), then where the engine's operands come from.

## 2. The changes

### 2.1 Skewed tile store (`LLAMA_Q8_GEMM=skewed`, commit 0acf64c)

`amx_q8_block_tiled`, general multi-group path. Each loop iteration h:

1. issues half-step h's tile products: C pair {0,1} or {2,3} and A tile 4 or 5 by parity, with no store;
2. stores half-step h−1's C pair into `dots[(h−1) & 1]` and rescales it into `out`.

The A tile alternates so that half-step h's load never overwrites a tile that half-step h−1's products may still read.
Tile 5 was free on this path (only the per-channel path uses it). The products, the rescale order and the `dots`
buffers are those of the pipelined kernel, so the result is bitwise the same. A `dots` buffer is reused two half-steps
later, after its rescale has finished.

### 2.2 Row block outer within each K-block (`LLAMA_Q8_KB_ORDER=rows`, commit 9c292f6)

`q8_gemm_blocks_kblocked` takes up to 64 of a worker's blocks (channel chunk of 32 x row block of 32) at a time and runs
every K-block of 2048 over all of them. The blocks used to run in list order: all 32 row blocks of chunk 0, then all
32 of chunk 1.

| Per K-block of one worker | Chunk order (before) | Row order (now) |
| --- | --- | --- |
| Activation slice of one row block (32 x 2048 B) | 64 KiB, used once, then 31 other slices (2 MiB) pass before its next use: from LLC every time | 64 KiB, used by every chunk of the piece in a row: from L2 |
| Weight slice of one chunk (32 x 2048 B; x2 for SwiGLU) | 64 KiB, reused by 32 row blocks in a row | 64 KiB per chunk, 2–3 chunks per piece, reused across the 32 row blocks: from L2 |
| Accumulators (64 blocks x 8 KiB) | 512 KiB | 512 KiB |

At M = 1024 the whole activation slice of one K-block is 1024 x 2048 B = 2 MiB, the size of the core's L2. The comment
on the K-blocked driver said that slice stayed in L2. That held at M = 256, where fix 8 introduced the K blocks; the M
sweep that followed made 1024 the default and broke it.

The row order is built once per piece (an index list sorted by row block). Each output element still adds its groups
in ascending order through the same fma, so both orders are bitwise the one-pass kernel. The order is reported in the
run report as `build.q8_kb_order`.

The pattern fits the earlier in-situ rates: W13 was the fastest gs64 matrix (32.4 TOP/s at 1S against 24–26) because
SwiGLU runs w1 and w3 against the same activation tile, so each byte of A fetched from LLC does twice the work.

## 3. Results (job 5910223, hbm13, tree 9c292f6)

### 3.1 Bitwise

The logits of all three configurations hash identically: 2048 prompt tokens with all rows, and 8192 with the last
row, on 1S and 2S. The unit tests (`tests/q8.c`) compare skewed and pipelined in rows order, and skewed in chunks
order, with the AMX chunk kernel for every GEMM kind, including K = 8192 shapes that cross K-blocks. The tp tests compare
the logits of all three modes and both orders on one and two sockets.

### 3.2 TTFT (15 interleaved repeats)

| Workload | base: median [min–max] | skewed | rows | rows − base, paired median | rows faster / slower |
| --- | --- | --- | --- | ---: | ---: |
| P 1S | 1.302 [1.290–1.460] | 1.285 [1.269–1.317] | 1.170 [1.141–1.199] | −138 ms | 15 / 0 |
| P 2S | 0.760 [0.723–0.829] | 0.756 [0.745–0.774] | 0.690 [0.669–0.702] | −74 ms | 15 / 0 |
| L 1S | 6.627 [6.595–7.254] | 6.571 [6.549–6.724] | 6.079 [6.060–6.197] | −550 ms | 15 / 0 |
| L 2S | 3.820 [3.748–4.033] | 3.713 [3.698–3.798] | 3.494 [3.467–3.563] | −287 ms | 15 / 0 |

base = `LLAMA_Q8_GEMM=pipelined LLAMA_Q8_KB_ORDER=chunks`, skewed = `skewed` + `chunks`, rows = `skewed` + `rows`
(the new default). The skewed store alone: paired medians −20 / −11 ms at P and −57 / −66 ms at L (1S / 2S).

**The slow mode in this job.** The base runs show the slow mode of [bimodality.md](bimodality.md): 2 of 15 at P 1S
(+0.14 s), 4 of 15 at L 1S (+0.55 s), 4 of 15 at L 2S (+0.25 s). The skewed and rows runs show no such outliers, except
rows at L 1S, with 5 of 15 about 0.1 s up. Fifteen repeats per arm cannot tell this apart from the job-to-job drift of
the slow fraction (3–50%), so it is an observation, not a claim. The medians above are not affected by it.

### 3.3 In-situ GEMM rates (one profiled P run per configuration)

TOP/s = 2 × tokens × 32 layers × N × K / phase seconds, as in [prefill-gap.md](prefill-gap.md) §3.

| Configuration | S | Wqkv | Wo | W13 | W2 | Attention s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1 | 24.0 | 25.1 | 32.4 | 26.3 | 0.160 |
| skewed | 1 | 23.6 | 25.1 | 32.5 | 25.4 | 0.171 |
| rows | 1 | 27.5 | 28.7 | 38.3 | 30.1 | 0.165 |
| base | 2 | 39.9 | 33.8 | 54.2 | 40.2 | 0.079 |
| skewed | 2 | 42.8 | 35.9 | 69.3 | 42.4 | 0.086 |
| rows | 2 | 55.4 | 37.3 | 80.1 | 52.5 | 0.079 |
| per-channel (5892439, for reference) | 1 / 2 | 37.0 / 76.5 | 39.6 / 23.2 | 59.8 / 117.8 | 45.2 / 74.1 | |

Rows lifts every matrix by 14–18% at 1S and Wqkv, W13 and W2 by 30–48% at 2S. Wo at 2S moves least (33.8 → 37.3): that
phase includes the two-socket reduction ([prefill-gap.md](prefill-gap.md) §10), which the GEMM order does not touch.

## 4. Gates

| Gate | Job | Result |
| --- | --- | --- |
| Local `make test` (emulated AMX) | — | pass, 0acf64c and 9c292f6 |
| Cluster `make test` + TSAN of the interleaved-reduction tp tests | 5910222 (repo-9c292f6) | `make test` rc 0, TSAN 0 race reports |
| Bitwise logits, engine | 5910223 | identical, all cells (§3.1) |

No accuracy gate is needed: the results are bitwise those of the previous default (rule D13 in
[prefill-gap.md](prefill-gap.md) §8).

## 5. Flip decisions

| Knob | Before | Now | Way back | Gate | A/B (P 1S TTFT) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `LLAMA_Q8_GEMM` | `pipelined` | `skewed` | `pipelined`, `chunk` | bitwise (5910223, unit and tp tests) | 1.302 → 1.285 s (paired −20 ms, 13 of 15 faster) | default |
| `LLAMA_Q8_KB_ORDER` | (chunk order, no knob) | `rows` | `chunks` | bitwise (5910223, unit and tp tests) | 1.285 → 1.170 s | default |

## 6. Follow-ups

1. **Still short of per-channel.** At 1S the gs64 rates are 27–38 TOP/s against 37–60. The remaining gap is the
   per-group store and rescale (§1: 89 against 47 cycles even with no data movement). A format with groups of 128
   halves the stores but changes the numerics: it needs the HF gate on both paths.
2. **Piece size.** Now that the activation slice is reused across the chunks of a piece, more chunks per piece (more
   accumulators, 8 KiB each) would reuse it further. The L2 budget is about 2 MiB; at 64 blocks the piece uses about
   0.8 MiB with SwiGLU (weights 256 KiB, accumulators 512 KiB, one activation slice 64 KiB).
3. **Wo at 2S** is bound by the reduction, not the GEMM (§3.3); see the sequence-parallel residual in
   [prefill-gap.md](prefill-gap.md) §10.
4. **The slow mode.** If later A/Bs keep showing fewer slow runs with the rows order, the LLC traffic of the old order
   belongs in [bimodality.md](bimodality.md) as a contributing cause.
