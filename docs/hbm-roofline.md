# HBM/DDR roofline and optimization priorities

Analysis of where the Phase 2 engine loses bandwidth and throughput on the
dual-socket Xeon Max 9480 nodes, and which work has the largest expected payoff.
Every number below is labeled **measured** (from an existing cluster record),
**modeled** (arithmetic from model shapes and measured ceilings) or **estimated**
(a rough figure that still needs a profile). No new jobs were run for this
analysis.

## 1. Measured machine ceilings (flat mode, per socket)

Source: `~/PFS/lpor/hw-bench-thesis-results/flat-*` on cresco8-hbm14 (flat,
quadrant: DDR nodes 0/1, HBM nodes 2/3), campaigns `ustreams`, `uremote`,
`udual`, `ugemm`, `isa` (2026-10-11 .. 2026-10-31). Median of samples.

| Resource | 1 thread | 14 thr | 28 thr | 56 thr |
| --- | ---: | ---: | ---: | ---: |
| Local HBM read (AVX-512, 8 acc.) | 16.5 GB/s | 223 | 387 | **552** |
| Local HBM copy, NT stores | 20.5 | 269 | 446 | 616 |
| Local DDR read | 17.0 | 213 | **268** | 262 |
| Remote HBM read (s0 cores -> s1 HBM) | – | 112 | 111 | **50** |
| Remote DDR read | – | 124 | 124 | 121 |
| Both sockets, local HBM, concurrently | – | – | – | 548 + 557 ≈ **1.1 TB/s** |

| Compute (56 cores, register-resident) | Peak measured |
| --- | ---: |
| AVX-512 FP32 FMA | 8.8 TFLOP/s |
| AVX-512 BF16 (`vdpbf16ps`) | 4.2 TFLOP/s |
| AVX-512 VNNI INT8 | 35.9 TOP/s |
| AMX BF16 | 112 TFLOP/s |
| AMX INT8 | 225 TOP/s |

| oneDNN, K=4096, N=14336, weights in local HBM | BF16 | INT8 |
| --- | ---: | ---: |
| M=1 (GEMV, decode shape) | 327 GB/s | 358 GB/s |
| M=256 (prefill shape) | 27 TFLOP/s | 69 TOP/s |
| M=1, weights in local DDR | 229 GB/s | 236 GB/s |

Cache mode (same harness): an 8 GB read set reaches 560 GB/s. A 96 GB set,
larger than the 64 GB HBM cache, drops to **98 GB/s**, below flat DDR. oneDNN GEMV
over a 17 GB pool reaches 277 GB/s, compared with 327-358 GB/s in flat HBM.

What these numbers mean for planning:

- Sustained HBM read is **~550 GB/s per socket, ~1.1 TB/s per node**. The 700 GB/s
  figure is above what any kernel here reached for reads. Plan against 550.
- **HBM needs the whole socket.** It scales almost linearly up to 56 cores
  (~10 GB/s per core required). DDR saturates at about 28 cores. At 8 threads,
  HBM and DDR deliver the same bandwidth (~130 GB/s), so a test at that thread
  count cannot tell the two tiers apart.
- **Remote HBM is the worst tier on the node**: 110 GB/s, falling to 50 GB/s when
  56 cores contend for it. That is 5–11× below local HBM and below remote DDR.
  Any weight that a socket reads across UPI costs more than keeping it in DDR.
- Even oneDNN reaches only ~60–65% of HBM read bandwidth for M=1. A specialized
  GEMV with high memory-level parallelism should aim for 75–85% (≈ 420–470 GB/s).

## 2. Where the engine stands today

Phase 2 Cache Mode baselines (FP32, sequential prefill):

| Model / threads | Decode tok/s | Bytes/token | Effective BW | Per thread |
| --- | ---: | ---: | ---: | ---: |
| 3.2 1B / 8 | 5.65 | 4.94 GB | 28 GB/s | 3.5 GB/s |
| 3.1 8B / 32 | 2.75 | 30.0 GB | 83 GB/s | 2.6 GB/s |
| llama.cpp 1B / 8 | 21.3 | 4.94 GB | 105 GB/s | 13 GB/s |
| llama.cpp 8B / 32 | 4.84 | 30.0 GB | 145 GB/s | 4.5 GB/s |

**The engine is bound by compute latency, not memory.** The FP32 `matmul` inner loop
(`src/run.c`, `val += w[i*n+j] * x[j]`) is a strict floating-point reduction.
Without `-ffast-math`/`-fassociative-math` the compiler may not reorder it. The
result is one scalar dependency chain per output row. I checked this with clang
`-O3`: it emits `mulss`/`addss` in the baseline build and scalar `vfmadd...ss`
even with `-march=sapphirerapids`. It never emits `ps`/zmm. One element per
~4 cycles at ~3.5 GHz gives 3.5 GB/s per core, which matches the 1B measurement
exactly.

The INT8 loop (`src/runq.c`) does vectorize, because integer reductions can be
reordered. However, it gets only `pmovsxbw`+`pmaddwd` (128-bit in the default
x86-64 build, 256-bit with `-march=sapphirerapids`), never VNNI `vpdpbusd`. Each
group also ends in a serial float chain.

Consequence (modeled): **today the binary should run equally fast from flat
DDR and flat HBM.** 32 threads × 3.5 GB/s ≈ 112 GB/s, less than half of DDR's
262 GB/s. No placement work can show an HBM benefit until the kernel can
consume about 10 GB/s per core. This is the first experiment to run (§6, E1),
because it tests the whole analysis cheaply.

## 3. Decode roofline (single stream, memory-bound)

Bytes read per decode token = all linear weights + classifier (one embedding
row is negligible). Q8_0 uses group size 64 (1.0625 B/weight); "Q4" is a
4.5-bit reference point. All rates are tokens/s. KV traffic is excluded here
(see §5.6).

| Model | Format | GB/token | Today | 1 socket DDR (262) | 1 socket HBM (550) | Practical HBM (440) | 2 sockets HBM (1100) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | FP32 | 4.94 | 5.65 | 53 | 111 | 89 | 222 |
| 1B | BF16 | 2.47 | – | 106 | 223 | 178 | 445 |
| 1B | Q8_0 | 1.31 | – | 200 | 419 | 335 | 838 |
| 3B | FP32 | 12.85 | – | 20 | 43 | 34 | 86 |
| 3B | BF16 | 6.43 | – | 41 | 86 | 69 | 171 |
| 3B | Q8_0 | 3.41 | – | 77 | 161 | 129 | 322 |
| 8B | FP32 | 30.02 | 2.75 | 8.7 | 18.3 | 14.7 | 36.6 |
| 8B | BF16 | 15.01 | – | 17.5 | 36.6 | 29.3 | 73.3 |
| 8B | Q8_0 | 7.97 | – | 33 | 69 | 55 | 138 |
| 8B | Q4 | 4.22 | – | 62 | 130 | 104 | 261 |

The 8B FP32 gap is **6.7× to one socket's HBM roofline and 13× to the node's**.
For 1B at 8 threads it is ~20×. Decode arithmetic intensity is ~0.5 flop/byte
(FP32) to ~2 (INT8). At 550 GB/s that requires only 0.3–1.1 TFLOP/s, a small
fraction of AVX-512 alone. **AMX does not help single-stream decode.** A GEMV
fills one of AMX's 16 tile rows, and AVX-512 already has more than 8× the
compute that decode needs. For decode, the levers are the kernel's bandwidth per
core, the bytes per weight, and memory locality.

Capacity: every row above fits in HBM (64 GiB per socket). 8B FP32 is 30 GB
on one socket, or 15 GB per socket when split. Only the KV cache at long
context forces DDR decisions (§5.6).

## 4. Prefill roofline (batched, compute-bound)

Prefill currently runs the decode path once per prompt token. Each prompt token
streams all weights again, and each one computes full vocabulary logits that
are then discarded. With B tokens batched, a weight byte is used B times. The
ridge point is about 16 flop/byte for FP32 AVX-512 (B≈32), and about 50 for BF16
AMX at oneDNN's measured 27 TFLOP/s (B≈50).

| Model | FLOP/token | Today | llama.cpp FP32 | AVX-512 FP32 @~5.5 T | AMX BF16 @27 T | AMX INT8 @69 T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1B | 2.5 G | 5.6 | 249 | ~2,200 | ~10,900 | ~27,700 |
| 8B | 15.0 G | 2.7 | 89 | ~370 | ~1,800 | ~4,600 |

(Per socket, modeled upper bounds that ignore attention and overheads. The AMX
columns use measured library throughput, not tile peak.) **AMX's potential is
in prefill**, and in any future batched multi-request decode. Prefill is where
the gap is two to three orders of magnitude. A trivial first step is to skip
the classifier for non-final prompt positions. The classifier is 21% of 1B's
per-token weights and 7% of 8B's, so this alone speeds up even sequential
prefill.

## 5. Bottlenecks, ranked by expected payoff

### 5.1 GEMV kernel (prerequisite for everything else)

Replace the scalar row loop with an AVX-512 kernel:

- 4–8 independent zmm accumulators per row
- 2–4 rows per pass, so each `x` load is reused
- contiguous row blocks per thread (keeps hardware prefetch effective)
- a single horizontal reduction at the end

The single-core streaming measurement (16.5 GB/s) shows that the per-core target
of ~10 GB/s is reachable. For INT8, use VNNI (`vpdpbusd` needs u8×s8, so apply
the sign trick or a +128 offset with a correction). Alternatively, dequantize in
registers and FMA against FP32 activations: 8B Q8_0 at 550 GB/s needs ~9 G
weights/s per core, about 3 µops per 16 weights. This W8A32 path also removes
the activation-quantization error behind the failed 8B Q8_0 budget.
Expected: 8B FP32 goes from 2.75 to ~14 tok/s on one socket of local HBM
(modeled, 56 threads). This reproduces the oneDNN-class GEMV efficiency.

Note that `NATIVE=1` alone does not fix FP32. A vectorized FP32 reduction
changes summation order, so the FP32 reference tolerances (1e-3) need
re-validation, not bitwise equality.

### 5.2 Thread count and binding

Use all 56 cores of a socket per memory domain. With 8 or 32 threads, the HBM
roofline is unreachable regardless of kernel (§1). Keep `OMP_PROC_BIND=close
OMP_PLACES=cores` and bind the CPU set to the socket local to the weights.

### 5.3 Explicit memory placement in flat mode

Weights are currently `mmap`ed from the checkpoint file, so they live in the
**page cache**. Pages already cached stay on whatever node first read them: the
node-local staging `cp`, the validation runs, a previous job. A later `numactl
--membind=2` does not move them. File-backed mappings also use 4 KiB pages
(30 GB ≈ 7.5 M pages) and get no THP.

Proposed design, still zero-dependency:

- Allocate weights with `mmap(MAP_ANONYMOUS)`, call `mbind` via `syscall(SYS_mbind)`
  to the target HBM node, and add `madvise(MADV_HUGEPAGE)`.
- Then `read()` the checkpoint into that memory, or let the owning socket's
  threads first-touch it. `runtime.c`'s `MemoryKind` categories are already the
  intended boundary.
- Per-category policy: linear weights and classifier → local HBM; KV → local HBM
  while it fits; token embedding table (one row per token, and the 1–2 GB FP32
  copy runq expands) → DDR; small activations → local HBM.

A zero-code-change way to test placement now: stage the checkpoint into tmpfs
under the target policy, for example `numactl --membind=2 cp model.bin
/dev/shm/...`. tmpfs pages follow the writer's policy, so the existing `mmap`
then reads from HBM. Remove the file at job end. Check `/dev/shm` capacity first.

### 5.4 Dual socket: tensor parallelism with zero remote weight traffic

Splitting rows across sockets with a plain `omp parallel for` over 112 threads
while weights sit on one node would push half the threads onto the 50–110 GB/s
remote path. That is slower than one socket alone. Instead:

- Split **every matrix by output rows**: Wq/Wk/Wv/W1/W3/Wo/W2 and the classifier.
  Socket s stores its half in its own HBM node and only its own 56 threads read
  it. Input vectors (≤ 57 KB) can be read remotely at negligible cost.
- Use a persistent thread pool with per-socket groups and one node-wide barrier
  per fused stage: QKV, attention, Wo, W1+W3, W2. That is about 5 barriers per
  layer, compared with today's 8 fork/joins.
- Expected (modeled): ~1.9× over one socket, e.g. 8B BF16 ~55–65 tok/s and
  8B Q8_0 ~100+ tok/s at practical efficiency.

### 5.5 Serial work and synchronization (grows in importance after 5.1–5.4)

All of the following run on the master thread today:

- `apply_rope` calls `powf`+`cosf`+`sinf` for every pair, head and layer
  (65,536 per token for 8B, 16,384 for 1B), and recomputes identical frequencies
  each time.
- SwiGLU calls `expf` 458,752 times per 8B token.
- RMSNorm, the residual adds and runq's `quantize` (~0.85 M elements per 8B token)
  are scalar.
- There are 257 (8B) or 129 (1B) OpenMP parallel regions per token.

Estimated serial cost is ~5–8 ms per 8B token. That is about 2% today, ~10% at
the one-socket FP32 roofline (55 ms/token), and ~45% at the two-socket Q8_0
roofline (7 ms/token).

Fixes:

- Precompute RoPE cos/sin per position, or the frequencies once.
- Vectorize and parallelize the elementwise operations, and fuse them into GEMV
  epilogues (SwiGLU after W1/W3, residual after Wo/W2, quantize after norm).
- Use spin barriers in a persistent pool.

These changes are cheap and they are what keeps small models (1B/3B) near the
roofline.

### 5.6 Attention and KV cache

Attention uses the same scalar dot-product chains. It parallelizes only over
query heads (24–32), so it cannot occupy 56–112 cores. Each of the 3–4 query
heads in a GQA group re-reads the same K/V.

KV bytes read per step at position P are P × 64 KiB (1B) or P × 256 KiB
(8B, FP32). For 8B at 32K that is 8.6 GB per step: 29% of FP32 weight traffic,
or more than all of Q8_0's. Fixes, roughly in order:

- Process a whole GQA group per pass, reading K/V once.
- Split positions across threads (flash-decoding with a final softmax merge).
- Store K/V as BF16, halving traffic.
- Lay out `[layer][kv_head][pos][head_dim]` for contiguous streams.
- Keep KV in local HBM and spill the oldest positions to DDR only when the
  HBM budget runs out. At 128K, the 8B FP32 KV is 32 GiB.

This is secondary at the current 2K context. It becomes the main decode
bottleneck in long-context experiments.

### 5.7 Fewer bytes per weight

Decode tokens/s scale ~linearly with bytes/weight: BF16 gives 2× and Q8_0 3.8×
over FP32 (§3). BF16 needs a converter flag plus a kernel (`vdpbf16ps`, or a
16-bit shift to FP32 + FMA). Its accuracy risk is low compared with the current
W8A8. Q8_0 exists but needs 5.1's VNNI/W8A32 kernel and requalification on 8B.

## 6. Recommended order and validation

| Step | Work | Main beneficiary | Expected effect (modeled) |
| --- | --- | --- | --- |
| E1 | Run current binary on flat node: `--cpunodebind=0`, membind DDR 0 vs HBM 2 (tmpfs staging), threads 14/28/56 | Validates analysis | Same tok/s on DDR and HBM (compute-bound) |
| 1 | AVX-512 FP32 GEMV + 56 threads + skip prompt logits | Decode, prefill | 8B ~2.75 → ~14 tok/s on local HBM; first visible HBM/DDR gap (~2×) |
| 2 | Anonymous, `mbind`-placed, huge-page weight loader; per-category policy | Flat mode | Makes HBM placement deliberate and reproducible |
| 3 | Persistent pool, fused elementwise ops, RoPE tables | Small models, all | Removes the ~5–8 ms/token serial floor |
| 4 | BF16 weights; W8A32 or VNNI Q8_0 kernel | Decode | 2× / ~3.8× fewer bytes per token |
| 5 | Two-socket row-split tensor parallelism | Decode | ~1.9×, up to the 1.1 TB/s node roofline |
| 6 | Batched prefill with AVX-512, then AMX BF16/INT8 tiles | Prefill / TTFT | 100–1000× prefill headroom |
| 7 | GQA-aware split-K attention, BF16 KV, KV tiering HBM→DDR | Long context | Keeps decode bandwidth-bound past 8K |

Measured outcome of E1, step 1's kernel and parts of steps 3, 4 and 7: see
[kernels.md](kernels.md#results-on-cresco8-flat-mode). At 56 threads the
scalar engine runs at the same speed on DDR and HBM (8B FP32: 6.9 / 7.5
tok/s). The AVX-512 kernels reach 14.9 tok/s (447 GB/s) on HBM and 8.3 on DDR.
8B Q8_0 reaches 47.9 tok/s (382 GB/s) after prefetch tuning.

For each step, report achieved GB/s, computed as bytes per token × tokens/s,
next to the §3 roofline. Also record uncore `hbm_cas_rd`/`ddr_cas_rd`/UPI counters
using the hw-bench perf event set, so that "HBM was used" is a measurement,
not an inference. Keep the 1B runs at a full socket. At 8 threads every memory
tier looks the same.
