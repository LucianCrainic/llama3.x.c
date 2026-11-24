# gs32 AMX-INT8 (24 November 2026)

Group size 32 is the Q8_0 block of llama.cpp's GGUF files. A Q8_0 GGUF repacks into the engine's v3 Q8_0 format
without requantization, and the cross-engine comparison does exactly that to run both engines on the same weights.
Before this work the engine had no fast path at gs = 32. AMX-INT8 and the AVX-512 VNNI GEMM both required
`gs % 64 == 0`, so batched prefill fell back to the portable Q8 kernel.

**Result: TTFT at P drops from 60.1 s to 1.76 s on one socket (34x) and from 30.3 s to 0.99 s on two (31x).** The kernel
is the gs64 tiled kernel with half-shape tiles. gs32 stays 1.3–1.5x slower than gs64 on the same model, because it
pays the per-group store and rescale twice as often. It is bitwise equal to the chunk kernel, and it agrees with the
portable kernel up to W8A8 rounding flips (§3).

| Workload (8B, hbm13, job 5913457) | gs32 portable (before) | gs32 AMX (now) | gs64 AMX | gs32 / gs64 |
| --- | ---: | ---: | ---: | ---: |
| P 1S TTFT | 60.067 s | **1.763 s** | 1.186 s | 1.49 |
| P 2S TTFT | 30.273 s | **0.988 s** | 0.703 s | 1.41 |
| L 1S TTFT | — | **8.570 s** | 6.124 s | 1.40 |
| L 2S TTFT | — | **4.686 s** | 3.558 s | 1.32 |

Medians of 5 interleaved repeats, tree 4d13eb9. The gs32 file is the 8B Q8_0 GGUF of the cross-engine campaign repacked
to v3; the gs64 file is `int8-v3.bin`.

## 1. Design

An AMX tile is configured per register: rows (up to 16) and bytes per row (up to 64). `tdpbssd` takes K from the A
tile's bytes per row, and the B tile holds K / 4 rows of VNNI quads. The gs64 kernel uses full tiles: one 64-byte K step
per group. gs32 uses half-shape tiles on the same packed data:

| Tile | gs64 | gs32 | Where the gs32 K step reads |
| --- | --- | --- | --- |
| C 0–3 (int32 dots) | 16 x 16 | 16 x 16 | — |
| A 4–5 (activations) | 16 rows x 64 B | 16 rows x **32 B** | `+32` bytes in the same 1 KiB block, stride still 64 |
| B 6–7 (weights, VNNI quads) | 16 rows x 64 B | **8 rows** x 64 B | `+512` bytes in the same 1 KiB block |

- **No new packing.** Each 1 KiB A or B block of the gs64 layouts covers 64 K. The two halves of a gs32 pair of groups
  are its first and second 32 K. `q8_a_step` and `q8_b_step` turn K step j into a byte offset. For gs32 these are
  `(j / 2) * 1024 + (j & 1) * 32` for A and `(j / 2) * 1024 + (j & 1) * 512` for B. So the packed weights and the
  tile-order activation quantizer are unchanged, and `k % 64 == 0` is still required.
- **The configuration is per thread and lazy.** `amx_thread_configure_k(kstep)` loads the tile configuration only when
  the K step changes (`ldtilecfg` zeroes the tiles). A model has one group size, so this happens once per thread. The
  per-channel path keeps full tiles.
- **Everything around the tiles is gs64's.** The engine paths (skewed store, row-block order of the K-blocks,
  per-group rescale in ascending group order) are unchanged. Only the number of steps per group (`gs / kstep`) and the
  offsets differ.
- **Selection.** `k_gemm_q8_pick` takes AMX for `gs % 64 == 0` or `gs == 32`. `configure_q8` reports `amx-int8` at gs32
  when `dim % 128 == 0` and `hidden_dim % 128 == 0`, so that every tensor-parallel shard keeps `k % 64 == 0`. The VNNI
  rung still needs `gs % 64 == 0`. Other group sizes (16, for example) still fall back to the portable kernel; a test
  covers that case.

## 2. Gates

| Gate | Job | Result |
| --- | --- | --- |
| Local `make test` (emulated AMX) | — | pass, 4d13eb9. q8 unit test: 58,336,971 checks, 0 failures, including gs32 shapes with ragged M and N and K up to 14336 |
| Cluster `make test` + TSAN of the interleaved-reduction tp tests | 5913456 (repo-4d13eb9) | `make test` rc 0, TSAN 0 race reports |
| Tiled == chunk, bitwise, all 2048 rows of logits | 5913457 | identical, 1S and 2S |
| Greedy text, gs32 file | 5913457 | "red, blue, and yellow. These colors are called primary because they cannot be created by mixing", the same as the gs64 file |

## 3. AMX against portable: rounding flips, not a bug

The two kernels compute the same exact int32 group dots. Only the FP32 rescale differs: AMX does
`fma(dot * xs, ws, acc)` and portable does `acc += dot * ws * xs`. The results differ by about 1e-7 relative. When that
difference lands on a rounding boundary of the next layer's int8 activation quantizer, one activation moves by one step.
The logits at that position then move by 1e-3 to 2e-2, and the change reaches later positions through K and V.

The proof: with the portable kernel switched to the AMX expression (a local experiment, not committed), the two
forwards were bitwise equal on both test fixtures. The tests therefore check a flip-tolerant contract:

- the AMX paths are bitwise equal to each other and do not depend on the split;
- against the portable forward, every position keeps its argmax and the largest difference is bounded
  (`tests/q8_engine.c` 4c, `test_amx_int8_gs32_half_tiles`).

On the real model at 2048 tokens (last row, job 5913457):

| S | max abs | rms | logit rms | argmax |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.188 | 0.035 | 2.18 | same |
| 2 | 0.194 | 0.037 | 2.19 | same |

That rms is about 6x below the Hugging Face gate's 0.2. The gs32 file itself has not been run through the Hugging Face
gate; see §5.

## 4. Rates and comparison

In-situ TOP/s, one profiled P run per configuration:

| Configuration | S | Wqkv | Wo | W13 | W2 | Attention s | Forward s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gs32 AMX | 1 | 17.9 | 16.9 | 22.3 | 17.3 | 0.162 | 1.719 |
| gs64 AMX | 1 | 26.9 | 27.4 | 36.7 | 29.8 | 0.164 | 1.136 |
| gs32 AMX | 2 | 38.8 | 28.6 | 44.1 | 32.3 | 0.087 | 0.912 |
| gs64 AMX | 2 | 53.4 | 31.9 | 75.5 | 36.2 | 0.080 | 0.703 |

gs32 runs at 0.60–0.67 of the gs64 rate on one socket. Each 16 x 32 block does two tile stores, two reloads and two
rescales where gs64 does one, and the `tdpbssd` work per store halves. This is the same per-group epilogue cost that
[gs64-gemm.md](gs64-gemm.md) §1 measured, now paid every 32 K instead of every 64 K.

Against llama.cpp on the same Q8_0 weights (Flat HBM, [xengine-results.md](xengine-results.md), its `llama-bench` pp
rate), measured in each engine's own prefill tok/s:

| Prompt | llama.cpp 1S | gs32 AMX 1S | llama.cpp 2S | gs32 AMX 2S |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 448.9 | 1204 | 537.0 | 2199 |
| 8192 | 274.7 | 975 | 381.9 | 1813 |

The definitions differ ([xengine-results.md](xengine-results.md) note b), and the runs are from different jobs and
trees. The ratio (2.7–4.7x) is indicative, not a paired measurement.

## 5. Follow-ups

1. ~~**Decode at gs32 stays portable.**~~ Done: the VNNI pair kernel, §6.
2. **Hugging Face gate for the gs32 file** on both paths (batched and per token), as for every model file
   ([prefill-gap.md](prefill-gap.md)). The weights are llama.cpp's, so this measures the GGUF quantization, not the kernel.
3. **The 70B comparison** repacks its Q8_0 GGUF to gs32 in the same way (the earlier 70B comparison work). Its
   prefill runs should be rerun on this tree.
4. **Correction to [xengine-results.md](xengine-results.md) §5 and [plans/xengine-comparison.md](plans/xengine-comparison.md).**
   Both say llama3.x.c's 8B file uses 32-element groups. `int8-v3.bin` uses 64, so the "218 M group accumulations per
   token" estimate should be 109 M (6.979e9 / 64).

Scripts (not in the repository): `repack_q8_gguf.py` (a generic version of the 70B branch's `repack_q8.py`;
sha256 of the 8B output `68a6ac94…c48cbda0`, 9,034,809,600 bytes) and `gs32-ab.sbatch`.

## 6. Decode GEMV at gs32 (24 November)

`dots_q8_pair_body` in `src/gemv.c` runs W8A8 decode at gs32 when `n % 64 == 0`; other widths stay portable. One
64-byte `vpdpbusd` covers two groups: int32 lane i accumulates bytes 4i..4i+3, so lanes 0-7 hold group g's dot and
lanes 8-15 group g + 1's. The activation and weight scales are spread the same way
(`_mm512_mask_mov_ps(set1(s[g]), 0xff00, set1(s[g + 1]))`), and each lane is rescaled with the gs64 expression
`fma(cvt(dot) * xscale, wscale, acc)`. The loop is the gs64 one with one scale pair per 64 bytes instead of one scale.

Tests: `test_q8` in `tests/kernels.c` gains gs32 shapes (and n = 96, which stays portable), and `test_partial_q8`
covers gs32 with a padded K split. Swapping the two scale halves fails it by whole units. The engine test
`test_vnni_gemv_gs32_pairs` compares per-token decode on the `wide-gs32` fixture against the portable path with the
§3 flip contract: at 2S the first difference is one int8 activation in layer 1's QKV input at position 0, with
x * 127 / amax = -7.499987, one step from the tie. Gate 5913493: make test rc 0, TSAN 0 races.

A/B 5913494 (hbm14), 8B gs32 file, old tree 4d13eb9 (portable GEMV) against 9ce517c, 5 interleaved reps of decode D:

| S | Configuration | Decode tok/s | TTFT s |
| ---: | --- | ---: | ---: |
| 1 | gs32 portable GEMV | 24.57 [24.49-24.83] | 0.13 |
| 1 | gs32 VNNI pair | 48.04 [47.97-48.14] | 0.13 |
| 1 | gs64 VNNI (reference) | 51.07 [50.92-51.19] | 0.12 |
| 2 | gs32 portable GEMV | 47.79 [47.64-47.90] | 0.10 |
| 2 | gs32 VNNI pair | 91.16 [91.04-91.94] | 0.10 |
| 2 | gs64 VNNI (reference) | 97.23 [96.75-98.13] | 0.09 |

Decode at gs32 is 1.96x (1S) and 1.91x (2S) faster, and runs at 0.94 of gs64. The remaining gap is the larger scale
stream (two floats per 64 weight bytes instead of one) and the extra blend per load; decode is bandwidth bound, so
the GEMV now streams the weights at nearly the gs64 rate.

Numerics, per-token logits over the first 128 ids of the 2048 prompt: new against old has rms 0.066 (1S) and 0.063
(2S), logit rms 2.75, argmax the same at 128/128. For scale, the old tree's own 1S against 2S differs by rms 0.073
(max 2.74), also from rounding flips, so the change is within the flip noise W8A8 already has between socket counts.
Greedy text for 48 tokens is identical.
