# Prefill gap to vLLM: fixes and results

**Status: run (2026-11-21).** This is phase 2 of the prefill-gap work. Phase 1 was the diagnosis, which made no engine change. Phase 2 implements its ranked fixes on branch `perf/prefill-gap`.
- Commits run from `6bc7722` to `75daab3`, on top of `develop` `684489b`.
- Hardware: cresco8-hbm13 for the A/Bs, cresco8-hbm14 for gates, tests, exports and the attribution.
- Provenance of every job, the gates and the deviations: [the validation record](prefill-gap-validation.md).

The workload is Llama 3.1 8B Instruct, W8A8 (`runq`, int8 weights and per-row int8 activations), weights on HBM, 56 threads per socket.
- P is a 2048-token prompt with one generated token.
- L is an 8192-token prompt with one generated token.
- D is an 86-token prompt with 128 generated tokens.

Each number is tagged as one of:
- **measured**: this campaign, with job id and commit;
- **derived**: arithmetic on measured numbers;
- **modelled**: the estimates of the diagnosis.

The headline table (section 1) comes from one job with 5 interleaved repeats. The per-fix A/Bs of section 2 used 3 repeats and are supporting data.

## Result in one paragraph

Fixes 1-8 keep the Q8_0 gs=64 checkpoint and the W8A8 statistical accuracy (measured, job 5892439, medians of 5):
- P TTFT falls to 0.572x of develop on one socket (2.294 s to 1.312 s) and 0.576x on two (1.385 s to 0.798 s).
- L TTFT falls to 0.450x and 0.485x (14.82 s to 6.67 s; 8.16 s to 3.96 s).
- Decode is unchanged (51.2 / 97.7 tok/s to 51.1 / 96.6).

Fixes 1, 2, 3, 5, 6 and 8 are bitwise. Fixes 4 and 7 change rounding:
- They pass the W8A8 gate against the per-token reference (job 5892385).
- They pass the Hugging Face FP32 gate on the batched prefill path, where they act, at 2048 and 8192 tokens on one and two sockets: rms 0.116-0.138 against 0.110-0.133 for develop, argmax 1.0 (jobs 5892459, 5892460). The per-token (decode) path, which they do not touch, also passes (0.111-0.129, job 5892418).
- They are the new W8A8 defaults.

vLLM is still 2.1x faster at P (0.618 s and 0.369 s) and 2.4x / 2.9x at L. The gap was 3.7x at P. Its GEMMs explain most of what is left:
- In situ, vLLM's GEMMs run 47-72 TOP/s; ours run 21-27 TOP/s on one socket.
- The Q8_0 group format forces one FP32 rescale per 64-wide K step.

Fix 9, the per-channel format, removes that rescale:
- The GEMMs reach 37-60 TOP/s.
- P TTFT is 0.908 s (1S) and 0.586 s (2S): 1.47x and 1.59x of vLLM.
- Decode is 12% faster. The file is 6% smaller; the rest is attributed to less dequantization work per row.
- It fails the Hugging Face gate: rms 0.34-0.39 for RTN per-channel on both paths. Importing the RedHatAI GPTQ checkpoint that vLLM serves does not fix it (batched 0.33-0.40).
- The per-token int8 inputs set the error. The vLLM-equivalent model, with that checkpoint's weights, per-token inputs and BF16 LM head, measures rms 0.395 on the Hugging Face side, so vLLM's W8A8 serving would fail the same gate (modelled, job 5892436). At that accuracy the per-channel file is 1.47x off vLLM at P on one socket; at the gate, the gs64 default keeps about 3x lower rms and is 2.12x off.
- It stays opt-in through the model file.

## 1. Before and after (measured, job 5892439)

One job on cresco8-hbm13 ran the three configurations interleaved within each of 5 repeats. Cells give the median, with min-max in brackets.
- **develop**: tree 684489b with `int8-v3.bin`.
- **final gs64**: tree 23cac73 with `int8-v3.bin` and the new defaults. Its engine sources equal the merged head.
- **final per-channel**: the same tree with `int8-pc-v4.bin`.

The vLLM column is from job 5892201 ([xengine-results.md](xengine-results.md), section 1); its TTFT is client-side over HTTP.

| Workload | S | TTFT s develop | TTFT s final gs64 | TTFT s final per-channel | vLLM TTFT s |
| --- | ---: | --- | --- | --- | ---: |
| P (2048) | 1 | 2.294 [2.281-2.529] | 1.312 [1.278-1.454] | 0.908 [0.904-0.911] | 0.618 |
| P (2048) | 2 | 1.385 [1.347-1.420] | 0.798 [0.775-1.068] | 0.586 [0.583-0.601] | 0.369 |
| L (8192) | 1 | 14.821 [14.784-14.874] | 6.668 [6.636-7.289] | 5.065 [5.058-5.069] | 2.73 |
| L (8192) | 2 | 8.157 [8.131-8.182] | 3.960 [3.875-4.140] | 3.145 [3.140-3.157] | 1.38 |
| D (86 + 128) | 1 | 0.143 [0.141-0.144] | 0.110 [0.110-0.111] | 0.096 [0.094-0.097] | - |
| D (86 + 128) | 2 | 0.103 [0.103-0.108] | 0.088 [0.086-0.113] | 0.081 [0.079-0.083] | - |

| Workload | S | Prefill tok/s develop | Prefill tok/s final gs64 | Prefill tok/s final per-channel | Decode tok/s develop | Decode tok/s final gs64 | Decode tok/s final per-channel |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| P | 1 | 917 [830-923] | 1635 [1471-1681] | 2417 [2407-2428] | - | - | - |
| P | 2 | 1542 [1501-1588] | 2767 [2024-2850] | 3865 [3759-3889] | - | - | - |
| L | 1 | 559 [557-561] | 1261 [1154-1268] | 1673 [1672-1676] | - | - | - |
| L | 2 | 1025 [1022-1029] | 2160 [2062-2207] | 2751 [2740-2761] | - | - | - |
| D | 1 | 949 | 1466 | 1975 | 51.18 [50.98-51.24] | 51.12 [50.93-51.17] | 57.42 [57.37-57.54] |
| D | 2 | 1563 | 2170 | 2668 | 97.71 [97.27-97.90] | 96.58 [96.45-97.09] | 108.38 [108.27-108.69] |

Derived ratios against develop:

| | P 1S | P 2S | L 1S | L 2S |
| --- | ---: | ---: | ---: | ---: |
| final gs64 | 0.572 | 0.576 | 0.450 | 0.485 |
| final per-channel | 0.396 | 0.423 | 0.342 | 0.386 |

Derived ratios against vLLM (remaining gap):

| | P 1S | P 2S | L 1S | L 2S |
| --- | ---: | ---: | ---: | ---: |
| develop | 3.71x | 3.75x | 5.43x | 5.91x |
| final gs64 | 2.12x | 2.16x | 2.44x | 2.87x |
| final per-channel | 1.47x | 1.59x | 1.86x | 2.28x |

Accuracy against the Hugging Face FP32 logits (rms ≤ 0.2, argmax ≥ 0.9, max ≤ 16; all positions; cells give rms / max, argmax 1.0 unless noted). A probe run takes one path:
- **batched** prefill (`compare_engines.py --batch-b M`), where the GEMM, the prefill attention and the SwiGLU epilogue act, so fixes 1-8 act here;
- **per token**, the decode path (GEMV, decode attention), which fixes 4 and 7 do not touch.

| Configuration | Path | 1S-2048 | 1S-8192 | 2S-2048 | 2S-8192 | Gate | Job |
| --- | --- | --- | --- | --- | --- | --- | --- |
| develop (684489b, M=256) | batched | 0.131 / 9.44 | 0.117 / 10.06 | 0.133 / 8.30 | 0.110 / 8.30 | pass | 5892460 |
| final gs64 (08d934c, M=1024, AMX attention, vector exp) | batched | 0.137 / 9.01 | 0.119 / 9.01 | 0.138 / 7.28 | 0.116 / 8.91 | pass | 5892459 |
| develop numerics (9ca3e7e defaults) | per token | 0.131 / 9.44 | 0.117 / 10.06 | 0.133 / 8.30 | 0.110 / 8.30 | pass | 5892395 |
| final gs64 (23cac73) | per token | 0.128 / 8.90 | 0.111 / 8.90 | 0.129 / 9.23 | 0.113 / 9.23 | pass | 5892418 |
| final per-channel RTN | batched | 0.394 / 14.24 | 0.361 / 14.24 | 0.356 / 14.34 | 0.342 / 14.34 | fail | 5892459 |
| final per-channel RTN | per token | 0.385 / 14.40 | 0.361 / 14.40 | 0.352 / 14.16 | 0.336 / 14.16 | fail | 5892418 |

The accuracy cost of AMX attention and the vector exponential is +0.002 to +0.006 rms per cell on the batched path (0.131 → 0.137, 0.117 → 0.119, 0.133 → 0.138, 0.110 → 0.116), with the maximum lower in three cells of four. On the per-token path develop's numbers are the same as batched: its batched prefill is built to match the per-token path.

**Accuracy against vLLM.** vLLM's prefill lead is measured at lower logit fidelity. Its scheme (the RedHatAI GPTQ weights, per-token int8 inputs, BF16 norms, embedding and LM head), modelled on the Hugging Face side, is at rms 0.395 on prompt-2048 and fails the gate (modelled, job 5892436; section 7). The cause is the per-token input quantization: the same weights with FP32 inputs give 0.146. The remaining gap therefore has two readings:
- **At vLLM's accuracy**, the per-channel file (batched rms 0.394, the same scheme family) is 1.47x off vLLM at P on one socket and 1.59x on two.
- **At the gate**, the gs64 default keeps about 3x lower rms (0.137 measured against 0.395 modelled for vLLM) and is 2.12x off at P on one socket, 2.16x on two.

Notes on these numbers:
- **The gs64 TTFT is bimodal.** Most repeats run P on one socket in about 1.28-1.31 s; a few take 1.40-1.45 s. The earlier 3-repeat job 5892420 had two slow repeats and a 1.474 s median; job 5892421 had none (1.299 s). The per-channel file showed no such repeat. The cause is not isolated.
- **Per-channel decode is 12% faster** on both socket counts, which size alone does not explain. The file is only 6% smaller: 8.04 GB against 8.53 GB, with one scale per row instead of one per 64 values. The rest is attributed to less work per row in the decode GEMV: one int32 sum and one rescale per row instead of a rescale per 64-value group. This is derived, not measured: no HBM bytes per token were counted for D.
- **Two-socket decode of the final gs64** is 1.2% below develop (96.6 against 97.7). Its decode path changed only by bitwise scheduling fixes, so this is not attributed to a fix (validation record, deviation 5).

## 2. Each fix: measured gain against the model

Each row comes from its own A/B on the tree that introduced the fix (1S P TTFT, medians, measured). The modelled gain is the diagnosis's estimate against a 2.19 s prefill.

| # | Fix | Commit(s) | Bitwise | Modelled (s) | Measured 1S P TTFT (s) | Δ measured | Job | Decision |
| ---: | --- | --- | --- | ---: | --- | ---: | --- | --- |
| 2, 3, 6 | balanced Q8 GEMM partition; parallel K/V store with K/V caches written once at load; embedding rows and RoPE tables on the team | f6713ca, 6bc7722, 116e95c | yes | -0.31 | 2.294 → 1.971 | -0.32 | 5892371 | default |
| 1 | pipelined AMX-INT8 Q8_0 kernel on tile-ordered activations, transposed weight scales | bbb1f26, de25e40 | yes | -0.55 to -0.6 | 1.966 → 1.778 | -0.19 | 5892377 | default (`LLAMA_Q8_GEMM=chunk` is the way back) |
| 5 | RMSNorm fused with the tile-order quantizer | 86b151b | yes | -0.08 to -0.1 | 1.846 → 1.772 | -0.07 | 5892393 | default |
| 4 | AVX-512 exponential in the SwiGLU epilogue | fba28e6, 870a803 | no (gate) | -0.08 to -0.12 | 1.957 → 1.891 | -0.07 | 5892386 | default (`LLAMA_SWIGLU_EXP=libm` is the way back) |
| 7 | AMX-BF16 flash attention, persistent BF16 K/V copy, fused O update | 15acaa1, 5debcf4, 870a803 | no (gate) | -0.33 to -0.37 | 1.697 → 1.366 | -0.33 | 5892399 (first version 5892387: -0.17) | default for W8A8 (`LLAMA_ATTN=tiled` is the way back) |
| 8 | K-blocked Q8 GEMM (2048 per block, worker-held FP32 accumulators); M = 1024 on AMX-INT8 | bb817cf, 870a803 | yes (every M) | -0.05 | 1.688 → 1.659 (K block); 1.822 → 1.675 (M 256 → 1024) | -0.03; -0.15 | 5892400; 5892402 | default (`LLAMA_PREFILL_BATCH` overrides M) |
| 9 | per-channel INT8 (format 4): one weight scale per output channel, per-token activations | 9ca3e7e, c426ca2, 75daab3 | no (new format) | -0.3 after 1-5 | 1.299 → 0.905 (same tree, `int8-v3.bin` → `int8-pc-v4.bin`) | -0.39 | 5892421 | opt-in through the model file: fails the HF gate |
| 1-8 | all | 6bc7722-23cac73 (engine sources) | - | 2.19 → ~0.77 prefill | 2.294 → 1.312 (prefill 2.233 → 1.253, derived from tok/s) | -0.98 | 5892439 (5 repeats; 5892420: 2.292 → 1.474) | - |

Where the model was wrong:

**Fix 1** gave a third of its model. The prototype's 54 TOP/s assumed L2-resident operands with one chunk per thread. In the engine, a 2048-row prompt at M=256 streamed every weight matrix from HBM 8 times per prefill: 65.7 GB of HBM reads on one socket (section 5). Fix 8 then took back part of that: K blocks kept the accumulators in L1/L2, and M=1024 cut the passes to 2. Even after it, in-situ W13 is 27-33 TOP/s, not 46 (section 3).

**Fix 8** was modelled at -0.05 s from the M sweep of the old kernel. With the new kernel the weight re-streaming dominated, and M=1024 alone gave -0.15 s. Every M from 128 to 2048 gives bitwise the same logits (job 5892402).

**Fix 7's** first version reached half its model:
- Every batch re-packed the whole K/V prefix to BF16.
- The O update was a separate pass.
- Tasks were split by head only.

Keeping a persistent BF16 copy packed once per position, fusing the O update and splitting tasks by head × query block reached the model (P 1S attention 0.449 s → 0.187 s). The result is bitwise to the first version.

## 3. In-situ GEMM rates (derived from the profiled runs)

TOP/s = 2 × tokens × 32 layers × N × K / phase seconds. Each phase includes its fused epilogue (residual add, SwiGLU) and, on two sockets, the Wo/W2 reduction. Every cell comes from one profiled P run, so a cell can land on a slow repeat. The gs64 1S profile of 5892439 did: its forward took 1.41 s against the 1.31 s median.

| Configuration | Job | S | Wqkv | Wo | W13 | W2 | Attention s | Forward s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| develop 684489b (chunk kernel, M=256, tiled attention) | 5892439 | 1 | 14.6 | 18.4 | 20.3 | 17.7 | 0.465 | 2.342 |
| develop 684489b | 5892439 | 2 | 19.7 | 18.9 | 44.1 | 31.3 | 0.240 | 1.321 |
| final gs64 (pipelined + K blocks, M=1024, AMX attention) | 5892439 | 1 | 21.4 | 21.7 | 26.6 | 22.5 | 0.150 | 1.409 |
| final gs64 | 5892439 | 2 | 43.9 | 22.6 | 67.1 | 44.8 | 0.083 | 0.733 |
| final per-channel RTN (`int8-pc-v4.bin`) | 5892439 | 1 | 37.0 | 39.6 | 59.8 | 45.2 | 0.151 | 0.847 |
| final per-channel RTN | 5892439 | 2 | 76.5 | 23.2 | 117.8 | 74.1 | 0.075 | 0.533 |
| vLLM SGL kernel (M=2048, includes its activation quantization) | 5892209 (diagnosis) | 1 | 47.1 | 50.5 | 72.3 | 54.0 | 0.058 | 0.597 |

Supporting profiles (one run each, 3-repeat jobs):

| Configuration | Job | S | Wqkv | Wo | W13 | W2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pipelined, M=256 (wip tree, before K blocks) | 5892400 base | 1 | 22.9 | 27.7 | 25.3 | 26.9 |
| pipelined + K blocks 2048, M=256 | 5892400 kb2048 | 1 | 23.8 | 26.4 | 31.3 | 29.5 |
| final gs64 | 5892420 test | 1 / 2 | 24.1 / 45.9 | 25.4 / 25.1 | 32.7 / 67.5 | 26.1 / 45.0 |
| final gs64 | 5892421 gs64 | 1 / 2 | 23.9 / 46.0 | 25.1 / 31.8 | 32.7 / 67.5 | 26.1 / 37.3 |
| final per-channel RTN | 5892421 rtn | 1 / 2 | 37.4 / 71.8 | 40.0 / 25.7 | 59.8 / 114.1 | 45.2 / 71.3 |
| final per-channel RedHatAI import (`int8-rh-v4.bin`) | 5892421 rh | 1 / 2 | 37.0 / 69.3 | 39.3 / 31.0 | 59.1 / 114.0 | 45.8 / 55.9 |

Wo on two sockets stays at about 23 TOP/s in all three configurations of job 5892439 (develop 18.9, gs64 22.6, per-channel 23.2; 25-32 in the supporting runs), while every other matrix about doubles from one socket to two. The phase includes the exchange and sum of the row-parallel partial outputs, so the reduction, not the kernel, sets that rate. W2 has the same reduction but four times the work per exchanged byte, so the exchange costs it less. Section 10 splits that phase: the kernel scales (48 TOP/s), and the add and the wait for the other socket make up the rest; the interleaved reduction raises Wo to 38.5 TOP/s in situ.

## 4. Attention seconds per phase (measured, job 5892399, instrumented tree `repo-f8a7wip`)

The instrumented build counts TSC cycles per phase in each worker (`LLAMA_ATTN_PROF=1`). The instrumentation is not committed; the instrumented copy of `attention.c` is in the job's scratch tree. The columns are:
- **wall**: the attention wall time of the profiled run;
- **phase columns**: worker-seconds / 56 (derived, mean per worker);
- **sync/idle**: wall minus the mean task time.

`amx0` is commit 15acaa1's kernel and `amx3` the committed one (5debcf4). They give bitwise the same logits.

| Kernel | Prompt, S | Wall s | Pack K/V | Q convert | QK tiles | Mask | Softmax | PV tiles | PV add (O update) | Out | Sync/idle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiled (FP32 AVX-512, the develop default) | P, 1 | 0.449 | - | - | - | - | - | - | - | - | - |
| amx0 (first version) | P, 1 | 0.331 | 0.065 | 0.021 | 0.022 | 0.002 | 0.045 | 0.056 | 0.084 | 0.003 | 0.095 |
| amx3 (committed) | P, 1 | 0.187 | 0.005 | 0.023 | 0.022 | 0.003 | 0.048 | 0.034 | 0.027 | 0.003 | 0.024 |
| tiled | L, 1 | 7.435 | - | - | - | - | - | - | - | - | - |
| amx0 | L, 1 | 4.562 | 1.016 | 0.051 | 0.312 | 0.038 | 0.678 | 0.871 | 1.256 | 0.006 | 1.345 |
| amx3 | L, 1 | 2.334 | 0.018 | 0.058 | 0.335 | 0.043 | 0.718 | 0.505 | 0.423 | 0.006 | 0.238 |

The tiled kernel has no phase counters. Its split is in the diagnosis: the FP32 tiles at 2.5 TFLOP/s.

At P on one socket the committed kernel spends:
- a quarter of its time in the online softmax (vector exp);
- a third in the two tile products;
- 0.023 s converting Q to BF16, which is scalar.

vLLM's attention is 0.058 s per prompt (diagnosis). The follow-ups are bigger key blocks (64 or 128 positions, fewer softmax rescales) and a vector Q convert.

## 5. HBM traffic per prefill (measured, job 5892408)

The counters are uncore `hbm_cas_rd` (32 B per count) and `ddr_cas_rd` (64 B), counted over P and over S, where S is a 86-token prompt with the same model. "P − S" is the prefill of the 1962 extra tokens. Wip tree `repo-f8a7wip` with the K-blocked kernel, median of 3.

| S | M | HBM GB, P | HBM GB, S | HBM GB, P − S | DDR GB, P − S | TTFT s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 85.55 | 19.90 | 65.66 | 0.40 | 1.889 |
| 1 | 1024 (default now) | 48.51 | 19.91 | 28.59 | 0.35 | 1.706 |
| 2 | 256 | 88.58 | 19.65 | 68.92 | -1.53 | 1.046 |
| 2 | 1024 | 53.28 | 19.69 | 33.59 | 1.84 | 0.999 |

The int8 linears and their gs=64 scales are 7.42 GB (derived: 6.98 GB of int8 + 1/16). Going from 8 to 2 passes per 2048 prompt saves 6 × 7.42 = 44.5 GB (modelled); the measured saving is 37.1 GB on one socket and 35.3 GB on two. That settles the re-streaming question: at M=256 the prefill reads almost every weight from HBM once per pass.

## 6. BF16 K/V copy of `LLAMA_ATTN=amx` (derived from the code)

Bytes per token are 2 (K and V) × local KV heads × head_dim × 2 B × layers. For 8B:

| | One socket | Two sockets |
| --- | --- | --- |
| BF16 copy | 2 × 8 × 128 × 2 × 32 = 131,072 B = **128 KiB per token** | 2 × 4 × 128 × 2 × 32 = **64 KiB per token per socket** |
| FP32 cache | 256 KiB per token | 128 KiB per token per socket |

The copy is half the FP32 cache, so the K/V memory grows by 50%. It is sized for the context rounded up to 32 positions, at 1,082,130,432 B (1.01 GiB) at context 8256 on one socket.

It sits in the socket's work region:
- It is allocated only for W8A8 with `LLAMA_ATTN=amx` and active AMX.
- It is keyed by each layer's cache pointer and packed incrementally.
- Any non-AMX writer invalidates it: the per-token decode path, or the tiled fallback.

The FP32 cache cannot be dropped when the knob is on. Decode (split attention, `k_attention_kv`) reads the FP32 cache, and so do the tiled fallback and the per-token path. Dropping it needs:
- a BF16 decode attention kernel;
- BF16 K/V writes in the K/V store;
- a new decode gate, since decode logits would change.

vLLM keeps a BF16 cache only. This was not implemented.

## 7. Fix 9: per-channel INT8 (format 4)

Version 4 checkpoints store one FP32 weight scale per output channel (header group size 0). The engine then quantizes activations per token: one scale per row over the whole K. The AMX-INT8 kernel keeps int32 in the tiles over all of K and rescales once per output block, which is the SGL structure. The file decides the format, so `int8-v3.bin` runs exactly as before and nothing changes silently.

| Artifact | How it was made | Bytes | sha256 | HF gate (rms ≤ 0.2, argmax ≥ 0.9, max ≤ 16) |
| --- | --- | ---: | --- | --- |
| `int8-v3.bin` (Q8_0, gs=64) | `export.py --hf … --version 3 --int8` | 8,532,934,912 | (phase 2 artifact) | pass: batched rms 0.116-0.138 (job 5892459), per token 0.111-0.129 (job 5892418) |
| `int8-pc-v4.bin` (RTN per-channel) | `export.py --hf … --version 3 --int8 --per-channel` (commit 9ca3e7e) | 8,037,591,296 | `64cd68ff…b39abffbc` | **fail**: batched rms 0.342-0.394 (job 5892459), per token 0.336-0.385 (job 5892418), argmax ≥ 0.998, max 14.4 |
| `int8-rh-v4.bin` (RedHatAI GPTQ W8A8 imported, -128 passed through) | `export.py --hf-w8a8 … --version 3 --int8 --per-channel` (commit c426ca2) | 8,037,591,296 | `c213b9d1…94205cf67a7` | **fail**: batched rms 0.339-0.394 (job 5892459); per token 0.76-0.84 (job 5892418), the -128 values on the VNNI decode GEMV; superseded |
| `int8-rh-v4b.bin` (the same, -128 → -127) | the same command (commit 75daab3) | 8,037,591,296 | `e1c50e78…cb721061c36` | **fail**: batched rms 0.333-0.404 (job 5892459), per token 0.345-0.390 (job 5892438), argmax ≥ 0.998, max 14.5 |

Export commands (the snapshot paths are local):

```bash
# RTN per-channel from the Hugging Face BF16 snapshot
python tools/export.py int8-pc-v4.bin --hf <Llama-3.1-8B-Instruct snapshot> --version 3 --int8 --per-channel
# import of a compressed-tensors W8A8 checkpoint (llm-compressor int-quantized: int8 per-channel weights,
# dynamic per-token int8 activations), e.g. RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 revision 024e24cb
python tools/export.py int8-rh-v4.bin --hf-w8a8 <w8a8 snapshot> --version 3 --int8 --per-channel
```

How the import maps the checkpoint:
- **Kept exactly.** The int8 values and the per-row scales of every quantized linear. The BF16 scales are widened to FP32. Q and K rows are un-permuted together with their scales, so the export error is 0.
- **Quantized per channel by the export.** The tensors the checkpoint leaves in BF16: the embedding, the norms and the `lm_head`. Version 4 has one group size for every tensor and no BF16 classifier, so the import cannot keep vLLM's BF16 LM head. The LM head and embedding are our per-channel RTN.
- **Rejected.** Every other scheme: group or float quantization, static or asymmetric activations, KV-cache or output quantization, other ignored modules.

On two sockets, Wo and W2 are split by input columns, and each socket quantizes its own input slice per token: the activation scale is the maximum over that socket's half of the row, not the whole row. This is a deliberate choice:
- It is what a vLLM tensor-parallel rank does: a row-parallel linear quantizes its local input slice per token.
- A global scale needs one cross-socket exchange of the per-token maxima before every Wo and W2, 64 per forward pass on the critical path. Two sockets would still not be bitwise one socket, because the partial sums are added across sockets.
- It does not cost accuracy. The Hugging Face rms of the RTN per-channel file on the batched path is 0.356 / 0.342 on two sockets against 0.394 / 0.361 on one, at 2048 / 8192 tokens (measured, job 5892459).

### The RedHatAI checkpoint (measured, jobs 5892427 and 5892445)

What the checkpoint holds, compared with the Hugging Face original:
- **Config and recipe.** compressed-tensors `int-quantized`: int8 symmetric per-channel static weights, int8 symmetric dynamic per-token inputs, `ignore: [lm_head]`. The recipe is a GPTQModifier alone (dampening 0.01), with no SmoothQuant.
- **The tensors left in BF16 equal the original.** The embedding, the LM head and all 65 RMSNorm weights are identical. No smoothing factor is folded into the norms, and no zero-point or input-scale tensor exists.
- **Weights.** Every int8 matrix is [out, in] with a BF16 [out, 1] scale, about 0.997x of the RTN scale (median). The weights use the full range -128..127.
- **Weight error is not the measure.** GPTQ minimizes the layer output error, not the weight error. Its weights are farther from the original than RTN's: 0.0206 against 0.0163 relative error for `q_proj` of layer 0, 0.0127 against 0.0118 for `down_proj`. The fake-quant rows below decide.

**The -128 values.** The engine's VNNI kernels (decode GEMV in `src/gemv.c`, the AVX-512 Q8 GEMM in `src/gemm.c`) negate the weight where the activation is negative (`vpdpbusd` multiplies unsigned |x| by signed bytes). Negating -128 wraps to -128, so those kernels assume no weight is -128. The RTN export never writes one. The AMX `tdpbssd` path of the batched prefill is exact for -128: the file with the -128 values passes through gives batched rms 0.339-0.394, against 0.333-0.404 for the mapped file (job 5892459). The per-token path is not: its VNNI GEMV gave 0.76-0.84 on the same file (job 5892418). A model file runs on every path, decode included, so commit 75daab3 maps the checkpoint's -128 values to -127, and the import is no longer exact for them.

| Tensor type | Values of -128 | Share of the values | Rows holding one | Share of the rows |
| --- | ---: | ---: | ---: | ---: |
| q_proj | 21,521 | 4.0e-5 | 21,073 | 16.1% |
| k_proj | 5,610 | 4.2e-5 | 5,515 | 16.8% |
| v_proj | 5,369 | 4.0e-5 | 5,258 | 16.0% |
| o_proj | 28,549 | 5.3e-5 | 27,710 | 21.1% |
| gate_proj | 79,577 | 4.2e-5 | 77,502 | 16.9% |
| up_proj | 134,140 | 7.1e-5 | 128,478 | 28.0% |
| down_proj | 25,465 | 1.4e-5 | 24,231 | 18.5% |
| all 224 matrices | 300,231 | 4.3e-5 | 289,767 | 21.1% |

Each mapped value moves by one quantization step, at most 1/128 of its row's largest weight.

### Attribution on the Hugging Face side (modelled, job 5892436)

`hf_fakequant.py` runs the Hugging Face FP32 model on prompt-2048 with fake-quantized weights and linear inputs, and compares it with the FP32 reference under the gate (rms ≤ 0.2, argmax ≥ 0.9, max ≤ 16). These rows model the schemes, not the engine. The RedHatAI rows keep the Hugging Face BF16 embedding and LM head, as vLLM does.

| Variant | Weights | Linear inputs | rms | max | argmax | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `all:g64` | RTN, groups of 64 | groups of 64 | 0.123 | 7.46 | 1.000 | pass |
| `all:pc` | RTN per channel | per token | 0.397 | 14.25 | 0.9985 | fail |
| `all:rh` (with the HF norms, `norms:rh` is identical) | RedHatAI GPTQ | per token | **0.395** | 14.31 | 0.9976 | **fail** |
| `all:rhc` | RedHatAI GPTQ, -128 → -127 | per token | 0.410 | 14.32 | 0.9976 | fail |
| `all:rhw` | RedHatAI GPTQ | FP32 | 0.146 | 8.99 | 1.000 | pass |
| `norms:rh,all:rh,lm:pc,emb:pc` | RedHatAI GPTQ, our per-channel LM head and embedding | per token | 0.400 | 14.27 | 0.9995 | fail |

**The `all:rh` row is the vLLM-equivalent accuracy.** It holds vLLM's weights, its per-token int8 inputs and its BF16 norms, embedding and LM head. At rms 0.395 it fails our rms ≤ 0.2 limit as clearly as our RTN per-channel file does (0.397). vLLM's W8A8 serving of this checkpoint would not pass the gate this campaign holds the engine to; the cross-engine comparison compares speeds at that lower accuracy.

What sets the error:
- **The per-token inputs.** The same GPTQ weights with FP32 inputs pass at 0.146 (`all:rhw`).
- **Not GPTQ against RTN.** RTN and GPTQ per-channel fail by the same amount (0.397 and 0.395).
- **Not the LM head or the embedding.** Our per-channel LM head alone costs rms 0.055 (`lm:pc`), the embedding 0.066 (`emb:pc`); swapping both into the RedHatAI weights moves it from 0.395 to 0.400.
- **The clamp.** Mapping -128 to -127 costs 0.015 (0.395 → 0.410).

One matrix type at a time back to groups of 64, the rest per channel (`all:pc,<type>:g64`):

| Type at gs=64 | none | Wqkv | Wo | W13 | W2 | LM head | embedding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rms | 0.397 | 0.344 | 0.394 | 0.369 | 0.301 | 0.394 | 0.405 |

W2 is the largest single contributor; with GPTQ weights, `all:rh,w2:g64` gives 0.307. No single type brings the per-channel file under 0.2. A mixed file (W2 at gs=64, the rest per channel) is not possible in format 4, which has one group size for every tensor. Its modelled rms of 0.30-0.31 would fail anyway, so it was not built.

### The imported file in the engine (measured, jobs 5892418, 5892438 and 5892459)

| File | Importer | Path | 1S-2048 | 1S-8192 | 2S-2048 | 2S-8192 | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `int8-pc-v4.bin` (RTN) | - | batched | 0.394 / 14.24 / 0.9985 | 0.361 / 14.24 / 0.9996 | 0.356 / 14.34 / 0.9990 | 0.342 / 14.34 / 0.9998 | fail |
| `int8-pc-v4.bin` (RTN) | - | per token | 0.385 / 14.40 / 0.9980 | 0.361 / 14.40 / 0.9995 | 0.352 / 14.16 / 0.9985 | 0.336 / 14.16 / 0.9996 | fail |
| `int8-rh-v4.bin` | c426ca2 (-128 passed through) | batched | 0.394 / 14.39 / 0.9971 | 0.369 / 14.39 / 0.9993 | 0.365 / 14.25 / 0.9995 | 0.339 / 14.25 / 0.9999 | fail |
| `int8-rh-v4.bin` | c426ca2 (-128 passed through) | per token | 0.798 / 13.88 / 0.9980 | 0.837 / 13.88 / 0.9995 | 0.760 / 11.39 / 0.9980 | 0.795 / 12.13 / 0.9995 | fail |
| `int8-rh-v4b.bin` | 75daab3 (-128 → -127) | batched | 0.404 / 14.45 / 0.9985 | 0.363 / 14.45 / 0.9996 | 0.355 / 14.27 / 0.9985 | 0.333 / 14.27 / 0.9996 | fail |
| `int8-rh-v4b.bin` | 75daab3 (-128 → -127) | per token | 0.390 / 14.50 / 0.9995 | 0.360 / 14.50 / 0.9999 | 0.368 / 14.30 / 0.9995 | 0.345 / 14.30 / 0.9999 | fail |

Batched rows: job 5892459 (tree 08d934c, M=1024). Per-token rows: jobs 5892418 and 5892438. Cells give rms / max / argmax against the Hugging Face FP32 logits.

Notes on the import:
- **The first import was wrong for decode.** Its check for values outside -127..127 took `abs()` of an int8 tensor, which wraps -128 to -128, so the -128 values reached the engine unchanged. On the per-token path the VNNI GEMV negates them wrongly: rms 0.76-0.84, twice the Hugging Face model of the same scheme (0.395), which exposed it. On the batched path the AMX tiles are exact and the same file gives 0.339-0.394. Mapping -128 to -127 (75daab3) brings the per-token path to 0.345-0.390.
- **The engine matches the model.** The exact import on the batched path (0.394 at 1S-2048) lands where the Hugging Face model of the same weights and inputs predicts (0.395), and where our RTN file lands. The importer is correct; the scheme fails the gate.
- **Decision.** Format 4 stays opt-in: a version 3 file runs as before, and a version 4 file must be asked for by name. It does not become a default, because no per-channel file passes the Hugging Face gate. Its speed (P TTFT 0.908 s against 1.312 s for gs=64 on one socket, job 5892439) is the measured price of the gs=64 accuracy.

## 8. Flip decisions

| Knob | Before | Now | Way back | Gate | A/B (1S P TTFT) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `LLAMA_Q8_GEMM` | `chunk` | `pipelined` (+ K blocks of 2048 when K > 2048) | `chunk` | bitwise (5892377, 5892400) | 1.966 → 1.778 s; K blocks 1.688 → 1.659 s | default |
| `LLAMA_SWIGLU_EXP` | `libm` | `vector` | `libm` | W8A8 gate 5892385, rms 0.112 / 0.114 (1S / 2S); with the other defaults, HF gate on the batched path 5892459 passes all 4 cells: rms 0.137 / 0.119 / 0.138 / 0.116 at 1S-2048 / 1S-8192 / 2S-2048 / 2S-8192 (develop 0.131 / 0.117 / 0.133 / 0.110, job 5892460) | 1.957 → 1.891 s | default |
| `LLAMA_ATTN` (W8A8) | `tiled` | `amx` | `tiled` | W8A8 gate 5892385, rms 0.136 / 0.147, combined 0.130 / 0.137; HF gate on the batched path 5892459 passes all 4 cells: rms 0.137 / 0.119 / 0.138 / 0.116 at 1S-2048 / 1S-8192 / 2S-2048 / 2S-8192 (develop 0.131 / 0.117 / 0.133 / 0.110, job 5892460) | 1.697 → 1.366 s | default for `runq`; FP32 and BF16 keep `tiled` |
| `LLAMA_PREFILL_BATCH` (W8A8, AMX-INT8) | 256 | 1024 (capped at the context) | `LLAMA_PREFILL_BATCH=256` | bitwise for M 128-2048 (5892402) | 1.822 → 1.675 s | default |
| Fixes 2, 3, 5, 6 (no knob) | - | always | - | bitwise (5892371, 5892393) | -0.32, -0.07 s | default |
| Per-channel format (fix 9) | - | file format 4 | use a version 3 file | HF gate fails on both paths: RTN batched 0.342-0.394 (5892459), per token 0.336-0.385 (5892418); RedHatAI import batched 0.333-0.404 (5892459) | 1.312 → 0.908 s (5892439) | opt-in through the model file, not a default |
| `LLAMA_TP_BATCH_REDUCE` (2S W8A8 prefill) | barrier | `interleaved` (Wo and W2) | `barrier`; `wo` interleaves Wo only | bitwise to develop, 1S and 2S, 2048-all and 8192-last (5898022, 5898042) | 2S P paired -30 ms median (30 pairs), L -161 to -175 ms (5898029, 5898041) | default (section 10) |
| K/V prefetch in the pipelined kernel (`LLAMA_Q8_PF`, wip) | - | dropped | - | bitwise | slower: 1.688 → 1.801 / 1.882 s (5892400) | not committed |

## 9. Follow-ups

1. **Attention**: key blocks of 64 or 128 positions, to halve the online-softmax rescales and the PV adds, and a vector FP32→BF16 Q convert (0.023 s of 0.187 s).
2. **BF16 K/V cache** with a BF16 decode kernel, to drop the FP32 cache (section 6): -128 KiB per token on one socket, plus a decode gate.
3. **Wo on two sockets** (section 3): done on `perf/wo-2s-split`, section 10. Wo reached 38.5 TOP/s in situ. The remaining lever there is the sequence-parallel residual, modelled at -24 ms at P 2S.
4. **GEMM, gs=64**: the kernel is at 21-33 TOP/s in situ against 54 on the L2-resident prototype. The next steps are:
   - B-panel reuse across the M blocks of a pass (vLLM's L2 grouping);
   - one dispatch per layer for QKV.
5. **Fix 9 accuracy.** The per-token input quantization is what fails the gate (section 7). Options, none tried:
   - per-group activation scales over the full-K int32 path, e.g. groups of 512, which keep most of the speed;
   - SmoothQuant-style smoothing folded into the norms, then a re-quantization;
   - a mixed format with W2 at gs=64 (modelled rms 0.30, still failing on its own).

## 10. Two-socket Wo/W2: the reduction interleaved with the projection (branch `perf/wo-2s-split`)

Commits d8948d2 (profiling), 66bae9a, 9f54e92 and 2da8e3c, on `develop` d1b45bf. The validation rows are in [the validation record](prefill-gap-validation.md), jobs 5892710-5898043.

### Diagnosis (measured, job 5892710, d8948d2)

The Wo/W2 phase of P2048 at 2S, split into its parts. Values are ms per prefill (64 calls of M=1024), medians of the 6 socket-runs of 3 profiled runs:

| Matrix | Kernel | Add | Closing barrier | Dispatch | Wait for the other socket |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wo | 46 | 21 | 1.2 | 0.4 | 11 (0.4-66) |
| W2 | 140 | 21 | 16 | 0.4 | 10 (0-51) |

- **The kernel scales.** Wo's kernel runs 48 TOP/s across both sockets, 2.1x the one-socket Wo (98 ms). The in-situ 23 TOP/s of section 3 is the kernel, plus an add about half the kernel's time, plus waiting for the other socket.
- **Where the wait comes from.** It is skew from the phases before the synchronisation point. Socket 1 waits for socket 0 at Wo, and socket 0 waits for socket 1 at W2. This tracks per-socket differences in QKV, quantize, rope and W13.
- **UPI traffic.** Counted P minus S over the whole run: 2.85 / 4.07 GB per direction. The flit size, 6.93 B, is calibrated in job 5892713. The exchanged payload is 2.06 GB per direction (derived), and the add moves it at about 51 GB/s.

### Options (modelled)

The P 2S TTFT was 0.801 s. One socket is unaffected.

| Option | Cross-socket bytes per call | Syncs per Wo | Saving |
| --- | --- | --- | --- |
| Column split (Wo by output rows) | 10.2 vs 16 MiB | 2 vs 1 | -7 ms; needs a second Wo copy. Dropped |
| Reduce-scatter with the residual, then FP32 all-gather | 16 MiB | 2 | 0 |
| Sequence-parallel residual (int8 all-gather after the norm) | -56% per layer | 6 per layer | -24 ms; large refactor. Follow-up |
| Overlap with W13's start | - | - | blocked: the RMSNorm before W13 needs whole reduced rows |
| Interleave the add into the Wo/W2 GEMM per block | 16 MiB | 0 extra | -30 to -59 ms (hides add + closing barrier). Chosen |

### What changed

`LLAMA_TP_BATCH_REDUCE` has three values: `interleaved` (the default), `wo` (Wo only) and `barrier` (the way back). It acts on the two-socket batched W8A8 path only. Decode and one socket are untouched.

How the interleaved mode works:
- The Q8 GEMM gets hooks (`GemmQ8Hooks`, one-pass and K-blocked). Every worker flags each 32-row block of the partial it stores.
- Between its compute units, the worker adds up to 2 of its own blocks whose peer block is flagged too. A block becomes a candidate 4 blocks after the worker stored it, and the next one is prefetched.
- After its range, the worker waits for and adds the rest.
- The add is the barrier form's `residual += p0 + p1`, so the logits are bitwise.

### Results

**Bitwise (measured, jobs 5898022 and 5898042, 9f54e92 / 2da8e3c).** The logits sha256 of develop, `interleaved`, `wo` and `barrier` are identical: 1S and 2S, all 2048 positions of prompt-2048 and the last of prompt-8192. The Hugging Face gate is therefore not needed.

**Where the time went (measured, job 5898024, 9f54e92).** P2048 2S, ms per prefill, medians of the 20 socket-runs of 10 profiled runs. In the interleaved mode the kernel is the phase minus the leader's checks, waits, adds and close (derived).

| Matrix | Mode | Phase | Kernel | Flag checks | Wait for peer | Add | Close |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wo | barrier | 82.9 | 45.1 | - | 8.6 | 22.4 | 3.9 |
| Wo | interleaved | 68.6 | about 40 | 1.2 | 5.0 | 15.9 | 6.5 |
| W2 | barrier | 195.6 | 145.4 | - | 13.7 | 26.8 | 7.5 |
| W2 | interleaved | 187.4 | about 139 | 0.2 | 8.9 | 20.0 | 18.9 |

In-situ rates on socket 0's counters (job 5898023):
- Wo: 22.2 (develop) → 38.5 TOP/s.
- W2: 49.7 → 46.7 TOP/s.
- Other matrices: unchanged.

**The add is not hidden.** Each block's add costs about 3.5 µs on the worker that computed it, against about 10 µs of compute per block. The lag and prefetch sweep (job 5897997) left it at 16-20 ms per matrix. The interleaving removes the UPI-saturating add phase after the GEMM, the second dispatch and the barriers, but not the add work. On W2 the team's closing wait grows (7.5 → 18.9 ms) and takes back most of its add saving.

**TTFT (measured, 15 interleaved repeats, P and L at 2S).** Two-socket runs are bimodal, so the numbers are paired diffs within a repeat against develop, plus medians within each mode:

| Job | Config | P paired median / mean | P faster | P fast mode: n, median | P slow mode: n, median | L median (paired median) |
| --- | --- | --- | --- | --- | --- | --- |
| 5898029 | develop | - | - | 10, 0.782 | 5, 0.840 | 3.929 |
| 5898029 | barrier | +0.8 / -2.3 ms | 6/15 | 10, 0.779 | 5, 0.840 | 3.924 (+3.8 ms) |
| 5898029 | interleaved | -32.0 / -42.6 ms | 12/15 | 12, 0.746 | 3, 0.802 | 3.758 (-175 ms) |
| 5898041 | develop | - | - | 11, 0.781 | 4, 0.843 | 4.038 |
| 5898041 | wo | -19.3 / -2.6 ms | 8/15 | 7, 0.758 | 8, 0.826 | 3.793 (-202 ms) |
| 5898041 | interleaved | +4.6 / -12.4 ms | 7/15 | 7, 0.747 | 8, 0.821 | 3.785 (-161 ms) |

The mode split is per configuration: develop at 0.81 s, the interleaved modes at 0.785 s.

- **Both jobs together, interleaved against develop:** paired median -30.0 ms, mean -27.5 ms, faster in 19 of 30.
  - The modelled saving was -30 to -59 ms. The low end of the measurement reaches the model's low end and is above the 15 ms flag threshold.
  - Each mode's median is lower than develop's: fast -35 ms in both jobs, slow -38 and -22 ms.
  - The fraction of runs in the slow mode moved for it in 5898029 (3 vs 5 of 15) and against it in 5898041 (8 vs 4). Pooled it is 11/30 vs 9/30, within the run-to-run spread.
- **L:** -161 to -175 ms (4.3%); 12-13 of 15 pairs faster.
- **`wo` against `interleaved`:** at P, `wo` is the worse of the two in both modes (fast 0.758 vs 0.747); at L they tie (3.793 vs 3.785). `interleaved` stays the default.
- **5 repeats were not enough** (job 5898023): P 2S was 0.778 / 0.792 / 0.781 for develop / barrier / interleaved, and L read +2.7%. Both are inside the bimodal spread.

**Decode** (job 5898023, 5 repeats) is flat. It takes no new code. 2S: develop 98.5, barrier 97.2, interleaved 97.9 tok/s. 1S: 51.28, 51.21, 51.13.

**UPI bytes rise** (measured over the prefill window, perf stat -I 20, 5 repeats each, 2da8e3c, job 5898043). GB per direction, socket 0 to 1 / socket 1 to 0:

| Mode | Socket 0 → 1 | Socket 1 → 0 |
| --- | ---: | ---: |
| barrier | 2.77 | 2.97 |
| wo | 3.46 | 3.64 |
| interleaved | 3.55 | 4.95 |

The payload is 2.15 GB per direction. The traffic is now spread over the GEMMs, so the links are not the bound. The extra bytes are not attributed. The hypothesis is that reading a block soon after the peer stored it pulls lines out of the peer's caches, and that the hardware prefetchers fetch neighbouring lines the peer will still rewrite.

### Skew and the bimodality (not fixed here)

The 0.4-66 ms waits for the other socket come from per-socket speed differences between synchronisation points, mostly W13 and quantize/rope on one socket.

The bimodality, an ~60-80 ms switch per process run, was captured in job 5898042: 16 profiled `barrier` runs with per-socket cycles/ref-cycles over the prefill window, numastat and AnonHugePages samples. The instrumentation slowed every run (0.92-1.05 s), and only one run fell in the slow mode, so this is a lead, not a finding. In that run:
- W13 is +41 ms on both sockets.
- Quantize is +41 ms on socket 1; rope +10-12 ms on both; QKV +8 ms on socket 0.
- The core clock is not lower (cycles/ref-cycles +0.02).
- Every sampled run held 16.7 GB of huge pages and 7.9 GB on each HBM node.

Hypothesis: a per-run memory-side state (uncore/mesh clock or HBM) slows both sockets' bandwidth-heavy phases together. It is not core frequency, THP or a one-socket placement. That is the next task: uncore clock counters per run, and the mode against run order.

### Follow-ups

- **Sequence-parallel residual:** each socket owns half the residual rows; reduce-scatter of W2, int8 all-gather after each norm. Modelled -24 ms at P 2S, 6 synchronisation points per layer.
- **Hiding the add for real:** a few helper cores per socket would add blocks while the others compute. Modelled -15 ms on Wo and -28 ms on W2 at the cost of 5-12% of the GEMM cores.
- **The 2S bimodality** above.
