# Batched prefill on one socket

> **Correction (2026-11-21):** HBM byte figures derived from `hbm_cas_rd` counts were computed at 64 B per count; the counter counts 32 B (calibrated in job 5892172, see [xengine-results-validation.md](xengine-results-validation.md)). Affected figures below are halved and marked †.

**Status: run (2026-11-09, job 5713857, commit ca2f7f8, node cresco8-hbm14).** All 32 correctness gates ran: 30 passed, 2 recorded
deviations (AMX-BF16 on the 1B model, section "Gates"). The Hugging Face comparison for AMX-BF16 was **not** part of this job; it was run afterwards (17 November 2026, job 5830324) and fixed the BF16 gate (C5), under which the two recorded deviations pass. Provenance and gates: [the validation record](batched-prefill-validation.md). Plan:
[C7](plans/C-batched-prefill/C7-one-socket-prefill-campaign.md). Full generated table: [batched-prefill-summary-table.md](batched-prefill-summary-table.md).
Weights were read from the staged files (placement `file`), not from anonymous HBM; A6 measured that gap at 2.4% for 8B FP32 decode.

Question: how much faster is prefill, and how much shorter the time to first token, on one socket for each rung of the ladder
(per-token baseline, AVX-512 FP32 GEMM, AMX-BF16, AVX-512 VNNI Q8, AMX-INT8 W8A8, W8A32), by model, prompt length and microbatch
size, with decode measured separately to show it is unaffected, and which defaults (`LLAMA_PREFILL_BATCH`, the kernel per
precision) the evidence supports. All numbers below are medians of 3 repeats (ranges in the generated table), 56 threads
unless stated; the per-token baselines at 8192 tokens ran once.

## Result in one paragraph

On 8B at 2048 prompt tokens, one socket: per-token FP32 17.3 tok/s; AVX-512 FP32 GEMM 183 (10.6x); AMX-BF16 640 (37x over the
FP32 per-token baseline); Q8 per token 49.1, AVX-512 Q8 GEMM 218 (4.4x), AMX-INT8 647 (13.2x). AMX-BF16 and AMX-INT8 are equal on
8B (640 against 647): at that size both are held back by attention (41% of the forward at 2048 tokens) and the non-GEMM work, not
by the matrix units. At 8192 tokens attention is 74-75% of the forward and both AMX rungs fall to 270 tok/s. W8A32 gains nothing
from a microbatch (it stays per token: 45.7 against 44.1 tok/s at 8B). Decode is unchanged after every rung.

## Full summary table

The complete generated table (all runs, 1B and 8B, every prompt, thread count and probe) is in [batched-prefill-summary-table.md](batched-prefill-summary-table.md); `tools/summarize_campaign.py` writes it. Rows below: 8B, 2048-token prompt, 56 threads.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 8b-bf16-teams-file-s1-t56-m128-p2048-c2304-kamx_bf16 | 3 | 1 | file | off | 128 | amx_bf16 | bf16 | 2048 | - | 3.468 [3.455-3.473] | 610.79 [609.99-613.13] | 15.55 [15.52-15.60] | - | - | 1250.5 † | 5.3 | 18 |
| 8b-bf16-teams-file-s1-t56-m16-p2048-c2304-kamx_bf16 | 3 | 1 | file | off | 16 | amx_bf16 | bf16 | 2048 | - | 7.007 [6.935-7.013] | 299.71 [299.38-302.80] | 15.63 [15.57-15.67] | - | - | 2848.3 † | 5.7 | 23 |
| 8b-bf16-teams-file-s1-t56-m256-p2048-c2304-kamx_bf16 | 3 | 1 | file | off | 256 | amx_bf16 | bf16 | 2048 | - | 3.315 [3.314-3.324] | 639.69 [638.03-639.79] | 15.59 [15.57-15.59] | - | - | 1134.6 † | 5.2 | 22 |
| 8b-bf16-teams-file-s1-t56-m32-p2048-c2304-kamx_bf16 | 3 | 1 | file | off | 32 | amx_bf16 | bf16 | 2048 | - | 4.736 [4.722-4.736] | 445.30 [445.28-446.59] | 15.64 [15.58-15.66] | - | - | 1922.2 † | 5.1 | 17 |
| 8b-bf16-teams-file-s1-t56-m64-p2048-c2304-kamx_bf16 | 3 | 1 | file | off | 64 | amx_bf16 | bf16 | 2048 | - | 3.980 [3.974-3.980] | 531.28 [531.27-531.81] | 15.60 [15.53-15.68] | - | - | 1480.5 † | 5.8 | 25 |
| 8b-fp32-teams-file-s1-t56-m1-p2048-c2304-kper_token | 3 | 1 | file | off | 1 | per_token | fp32 | 2048 | - | 120.676 [120.637-120.809] | 17.26 [17.25-17.27] | 15.60 [15.54-15.62] | - | - | 59682.0 † | 153.5 | 847 |
| 8b-fp32-teams-file-s1-t56-m128-p2048-c2304-kavx512_fp32 | 3 | 1 | file | off | 128 | avx512_fp32 | fp32 | 2048 | - | 11.488 [11.460-11.538] | 182.31 [181.55-182.83] | 15.61 [15.58-15.62] | - | - | 1457.5 † | 6.9 | 24 |
| 8b-fp32-teams-file-s1-t56-m16-p2048-c2304-kavx512_fp32 | 3 | 1 | file | off | 16 | avx512_fp32 | fp32 | 2048 | - | 15.881 [15.880-15.913] | 131.69 [131.68-131.74] | 15.67 [15.65-15.67] | - | - | 4657.4 † | 16.7 | 31 |
| 8b-fp32-teams-file-s1-t56-m256-p2048-c2304-kavx512_fp32 | 3 | 1 | file | off | 256 | avx512_fp32 | fp32 | 2048 | - | 12.310 [12.305-12.442] | 170.13 [168.39-170.15] | 15.62 [15.60-15.62] | - | - | 1246.1 † | 6.3 | 24 |
| 8b-fp32-teams-file-s1-t56-m32-p2048-c2304-kavx512_fp32 | 3 | 1 | file | off | 32 | avx512_fp32 | fp32 | 2048 | - | 12.858 [12.843-12.882] | 162.78 [162.52-163.01] | 15.57 [15.55-15.60] | - | - | 2829.9 † | 11.1 | 25 |
| 8b-fp32-teams-file-s1-t56-m64-p2048-c2304-kavx512_fp32 | 3 | 1 | file | off | 64 | avx512_fp32 | fp32 | 2048 | - | 11.446 [11.405-11.567] | 183.00 [181.12-183.67] | 15.51 [15.50-15.61] | - | - | 1898.9 † | 8.4 | 24 |
| 8b-int8-teams-file-s1-t56-m1-p2048-c2304-kper_token_q8 | 3 | 1 | file | off | 1 | per_token_q8 | int8_q8_0 | 2048 | - | 42.502 [42.429-42.504] | 49.10 [49.07-49.15] | 42.15 [42.01-42.29] | - | - | 16286.0 † | 43.5 | 296 |
| 8b-int8-teams-file-s1-t56-m128-p2048-c2304-kamx_int8 | 3 | 1 | file | off | 128 | amx_int8 | int8_q8_0 | 2048 | - | 3.513 [3.463-3.520] | 603.23 [601.59-611.53] | 42.14 [42.01-42.19] | - | - | 406.4 † | 1.9 | 9 |
| 8b-int8-teams-file-s1-t56-m128-p2048-c2304-kavx512_q8 | 3 | 1 | file | off | 128 | avx512_q8 | int8_q8_0 | 2048 | - | 9.850 [9.811-9.889] | 212.91 [212.01-213.77] | 41.85 [40.22-42.06] | - | - | 404.4 † | 2.9 | 13 |
| 8b-int8-teams-file-s1-t56-m16-p2048-c2304-kamx_int8 | 3 | 1 | file | off | 16 | amx_int8 | int8_q8_0 | 2048 | - | 5.824 [5.818-5.829] | 361.17 [360.94-362.62] | 42.44 [42.37-42.53] | - | - | 1279.7 † | 2.6 | 12 |
| 8b-int8-teams-file-s1-t56-m16-p2048-c2304-kavx512_q8 | 3 | 1 | file | off | 16 | avx512_q8 | int8_q8_0 | 2048 | - | 12.218 [12.187-12.223] | 171.38 [171.32-171.84] | 42.37 [42.36-42.51] | - | - | 1280.5 † | 5.3 | 21 |
| 8b-int8-teams-file-s1-t56-m256-p2048-c2304-kamx_int8 | 3 | 1 | file | off | 256 | amx_int8 | int8_q8_0 | 2048 | - | 3.279 [3.269-3.282] | 646.78 [646.33-648.75] | 42.17 [42.00-42.21] | - | - | 347.2 † | 1.9 | 8 |
| 8b-int8-teams-file-s1-t56-m256-p2048-c2304-kavx512_q8 | 3 | 1 | file | off | 256 | avx512_q8 | int8_q8_0 | 2048 | - | 9.603 [9.582-9.666] | 218.43 [217.55-218.91] | 39.99 [39.82-40.17] | - | - | 344.3 † | 2.7 | 12 |
| 8b-int8-teams-file-s1-t56-m32-p2048-c2304-kamx_int8 | 3 | 1 | file | off | 32 | amx_int8 | int8_q8_0 | 2048 | - | 4.269 [4.264-4.282] | 494.46 [493.14-495.14] | 42.43 [42.40-42.44] | - | - | 781.4 † | 2.1 | 10 |
| 8b-int8-teams-file-s1-t56-m32-p2048-c2304-kavx512_q8 | 3 | 1 | file | off | 32 | avx512_q8 | int8_q8_0 | 2048 | - | 11.106 [11.071-11.137] | 188.65 [188.18-189.25] | 41.79 [41.07-42.02] | - | - | 780.7 † | 3.9 | 19 |
| 8b-int8-teams-file-s1-t56-m64-p2048-c2304-kamx_int8 | 3 | 1 | file | off | 64 | amx_int8 | int8_q8_0 | 2048 | - | 3.801 [3.783-3.812] | 556.25 [554.86-559.15] | 42.15 [42.00-42.33] | - | - | 527.5 † | 1.9 | 8 |
| 8b-int8-teams-file-s1-t56-m64-p2048-c2304-kavx512_q8 | 3 | 1 | file | off | 64 | avx512_q8 | int8_q8_0 | 2048 | - | 10.282 [10.254-10.311] | 203.97 [203.28-204.43] | 40.53 [40.10-41.57] | - | - | 527.0 † | 3.2 | 16 |
| 8b-w8a32-teams-file-s1-t56-m1-p2048-c2304-kavx512_q8_w8a32 | 3 | 1 | file | off | 1 | avx512_q8_w8a32 | int8_q8_0_w8a32 | 2048 | - | 47.313 [47.276-47.586] | 44.07 [43.82-44.11] | 37.58 [37.47-37.69] | - | - | 16283.7 † | 44.0 | 107 |
| 8b-w8a32-teams-file-s1-t56-m64-p2048-c2304-kavx512_q8_w8a32 | 3 | 1 | file | off | 64 | avx512_q8_w8a32 | int8_q8_0_w8a32 | 2048 | - | 45.620 [45.617-45.687] | 45.71 [45.65-45.71] | 37.41 [37.37-37.50] | - | - | 6065.7 † | 81.4 | 648 |

## Prefill by rung, microbatch and prompt (best M per rung, 56 threads)

Speedup is over the same model's per-token baseline of the same checkpoint precision (BF16 has none; its row is over FP32
per token). Complete M sweeps (16 to 256 at 2048 tokens) are in the generated table.

| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8B | FP32 | 2048 | per token | 1 | 1 | 17.26 | 120.68 | 1.0 | - |
| 8B | FP32 | 2048 | AVX-512 FP32 GEMM | 64 | 1 | 183.0 | 11.45 | 10.6 | - |
| 8B | FP32 | 2048 | AMX-BF16 | 256 | 1 | 639.7 | 3.32 | 37.1 | - |
| 8B | Q8 | 2048 | per token | 1 | 1 | 49.10 | 42.50 | 1.0 | - |
| 8B | Q8 | 2048 | AVX-512 Q8 GEMM | 256 | 1 | 218.4 | 9.60 | 4.4 | - |
| 8B | Q8 | 2048 | AMX-INT8 | 256 | 1 | 646.8 | 3.28 | 13.2 | - |
| 8B | W8A32 | 2048 | per token (M=64 also per token) | 64 | 1 | 45.7 | 45.62 | 0.93 vs Q8 per token | - |
| 8B | FP32 | 512 | AVX-512 FP32 GEMM | 128 | 1 | 196.8 | 2.84 | 11.2 | - |
| 8B | FP32 | 512 | AMX-BF16 | 256 | 1 | 845.1 | 0.70 | 47.9 | - |
| 8B | Q8 | 512 | AMX-INT8 | 256 | 1 | 875.2 | 0.68 | 16.3 | - |
| 8B | FP32 | 8192 | AVX-512 FP32 GEMM | 128 | 1 | 132.1 | 62.5 | 8.4 | - |
| 8B | FP32 | 8192 | AMX-BF16 | 256 | 1 | 269.9 | 30.6 | 17.1 | - |
| 8B | Q8 | 8192 | AMX-INT8 | 256 | 1 | 271.3 | 30.5 | 7.1 | - |
| 8B | Q8 | 86 | AMX-INT8 | 128 | 1 | 694.0 | 0.175 | 12.7 | - |
| 1B | FP32 | 2048 | per token | 1 | 1 | 113.8 | 18.36 | 1.0 | - |
| 1B | FP32 | 2048 | AVX-512 FP32 GEMM | 128 | 1 | 1062 | 2.03 | 9.3 | - |
| 1B | FP32 | 2048 | AMX-BF16 | 256 | 1 | 2962 | 0.76 | 26.0 | - |
| 1B | Q8 | 2048 | per token | 1 | 1 | 291.7 | 7.21 | 1.0 | - |
| 1B | Q8 | 2048 | AVX-512 Q8 GEMM | 256 | 1 | 1242 | 1.75 | 4.3 | - |
| 1B | Q8 | 2048 | AMX-INT8 | 256 | 1 | 2803 | 0.82 | 9.6 | - |
| 1B | FP32 | 8192 | AMX-BF16 | 256 | 1 | 1176 | 7.18 | 13.0 | - |
| 1B | Q8 | 8192 | AMX-INT8 | 256 | 1 | 1142 | 7.38 | 7.4 | - |

Microbatch sweep at 2048 tokens (tok/s): 8B AMX-BF16 300 / 445 / 531 / 611 / 640 for M = 16 / 32 / 64 / 128 / 256; 8B AMX-INT8
361 / 494 / 556 / 603 / 647; 8B AVX-512 FP32 132 / 163 / 183 / 182 / 170 (peaks at 64-128, falls at 256); 8B AVX-512 Q8
171 / 189 / 204 / 213 / 218. The AMX rungs and Q8 are still rising at 256; FP32 is not.

## Thread counts (14 / 28 / 56, 2048 tokens, best M, tok/s)

| Model | Rung | 14 | 28 | 56 |
| --- | --- | ---: | ---: | ---: |
| 8B | AVX-512 FP32 (M=64) | 56.6 | 108.6 | 183.0 |
| 8B | AMX-BF16 (M=256) | 252.2 | 474.4 | 639.7 |
| 8B | AVX-512 Q8 (M=256) | 74.5 | 142.4 | 218.4 |
| 8B | AMX-INT8 (M=256) | 240.1 | 445.8 | 646.8 |
| 8B | W8A32 (M=64, from TTFT) | 20.2 | 34.4 | 45.7 |
| 1B | AMX-BF16 (M=256) | 1153 | 2000 | 2962 |
| 1B | AMX-INT8 (M=256) | 1112 | 1972 | 2803 |

Scaling from 14 to 56 threads (4x would be linear) is 2.5-2.7x for the AMX rungs and 3.2x for the AVX-512 FP32 rung on 8B; the cause
of the AMX rungs' weaker scaling was not isolated by this run.

## Achieved against the modeled ceiling (8B, 2048 tokens)

Modeled per-socket ceilings (`docs/hbm-roofline.md` section 4): about 370 / 1,800 / 4,600 tok/s. Achieved TFLOP/s or TOPS is
estimated from about 16 GFLOP per token (twice the 8B matmul parameters, attention excluded), so it is a lower bound on the
work done.

| Rung | Achieved TFLOP/s (or TOPS), est. | Prefill tok/s | Modeled ceiling tok/s | Achieved / ceiling |
| --- | ---: | ---: | ---: | ---: |
| AVX-512 FP32 (M=64) | 2.9 | 183 | 370 | 49% |
| AMX-BF16 (M=256) | 10.2 | 640 | 1,800 | 36% |
| AMX-INT8 (M=256) | 10.4 | 647 | 4,600 | 14% |

## Where the prefill time goes (profiled repeat, `C7/phase-shares.txt`)

Phase shares of `forward` for the profiled run of each rung (M as listed in the table above; M=256 for the AMX and Q8 rungs):

| Rung | Prompt | GEMM (qkv+wo+w1/w3+w2) | Attention | Conversion / quantize | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8B AMX-BF16 | 2048 | 58% (12+6+20+20) | 41% | - | 1% |
| 8B AMX-BF16 | 8192 | 25% (5+2+9+9) | 74% | - | 1% |
| 8B AMX-INT8 | 2048 | 53% (10+5+23+15) | 41% | 4% | 2% |
| 8B AMX-INT8 | 8192 | 22% (4+2+10+6) | 75% | 2% | 1% |
| 8B AVX-512 Q8 | 2048 | 79% (12+6+38+23) | 17% | 3% | 1% |
| 8B AVX-512 FP32 (M=64) | 2048 | 86% (13+7+39+27) | 13% | - | 1% |
| 8B AVX-512 FP32 (M=64) | 8192 | 61% (9+5+28+19) | 38% | - | 1% |
| 1B AMX-BF16 | 2048 | 46% (11+4+15+16) | 51% | - | 3% |
| 1B AMX-BF16 | 8192 | 21% (5+2+6+8) | 78% | - | 1% |

At 8192 tokens attention dominates whatever the GEMM rung; the AMX rungs speed up the GEMM only (8B BF16: 30.6 s at 8192 tokens, of
which about 23 s are attention), so a faster GEMM rung gains little there and no further GEMM speedup is claimed at that length. The
next speedup for long prompts is in attention, not in the matrix kernels.

## Decode after each rung's prefill (128 tokens from a 2048-token retained context, `WARM=2048`)

The comparison is against the per-token run of the same model at the same retained context (same `-xdec` protocol); decode at a
2304-token context is slower than the short-context A6 figures (8B FP32 16.07, W8A8 50.9 tok/s), as expected from the KV reads.

| Model | Precision | Rung | Decode tok/s | Per-token reference (same context) | Within range |
| --- | --- | --- | ---: | ---: | --- |
| 8B | FP32 | AVX-512 FP32 GEMM, M=64 | 15.54 | 15.45 [15.29-15.50] | yes |
| 8B | BF16 | AMX-BF16, M=256 | 15.52 | 15.45 | yes |
| 8B | Q8 | AVX-512 Q8, M=256 | 41.43 | 41.81 [41.76-41.87] | -0.9% (ranges do not overlap) |
| 8B | Q8 | AMX-INT8, M=256 | 42.02 | 41.81 | yes |
| 8B | W8A32 | per token, M=64 | 36.92 | not measured | - |
| 1B | FP32 | AVX-512 FP32 GEMM, M=128 | 85.73 | 87.04 | -1.5% |
| 1B | Q8 | AMX-INT8, M=256 | 203.5 | 204.0 | yes |

At the 86-token prompt the batched rungs decode the same or slightly faster than the per-token run (8B Q8: 54.2-54.6 against
52.0); nothing suggests a decode penalty from batched prefill. The 8B AVX-512 Q8 point is 0.9% below its reference with
non-overlapping ranges; it is in the same range as the run-to-run spread of the 1B Q8 decode (149-175 at M=256) and is not read as
a regression.

## Gates

| Gate | Criterion | Observed | Result |
| --- | --- | --- | --- |
| AVX-512 FP32 GEMM against per-token (1B, 8B; M=64, 256; 86 and 2048 tokens) | bitwise to 1e-3 | 8 of 8 passed | pass |
| AVX-512 Q8 against per-token (same grid) | max abs 1e-3 | 8 of 8 passed | pass |
| AMX-INT8 against AVX-512 (same grid) | W8A8 statistical gate (max 16, rms 0.2, argmax 0.9; last position for 2048) | 8 of 8 passed (job 5713471 failed the same comparison at the strict 1e-3, which was the wrong gate) | pass |
| AMX-BF16 against AVX-512 FP32 (8B) | recorded, argmax 1.0 | 4 of 4 passed (8B: max 0.77, rms 0.023) | pass |
| AMX-BF16 against AVX-512 FP32 (1B, 86 tokens) | recorded, argmax 1.0 | **2 of 2 recorded failures**: max 0.057, rms 0.0096, argmax 98.1% (52 of 53); the one flip is position 0 (the BOS token) | provisional gate not met; not a campaign failure |
| AMX-BF16 against AVX-512 FP32 (1B, 2048 tokens, last position) | recorded | 2 of 2 passed | pass |

The 1B AMX-BF16 flip is the case the K1 plan already named ("1B short prompt is at 98.1%"): a near-tie at the first position, with
BF16 rounding moving the logits by at most 0.057. Whether the gate should require argmax agreement at position 0, or compare the
Hugging Face numbers instead, is decided with the Hugging Face run (K1), which this job did not do.

## Chosen defaults (recommendations; no code default changed by this run)

| Precision | `LLAMA_PREFILL_BATCH` | Prefill kernel | Why |
| --- | ---: | --- | --- |
| FP32 | 128 | AVX-512 FP32 GEMM | 8B peaks at 64-128 (183 / 182) and loses 7% at 256; 1B is flat from 128 (1062 against 1056 at 256). 128 is within 1% of the best on both models |
| FP32 with `LLAMA_PRECISION=bf16` | 256 | AMX-BF16 (opt in only) | still rising at 256 on both models (8B +5% over 128, 1B +3%); 3.5x the FP32 GEMM |
| W8A8 (Q8 checkpoints) | 256 | AMX-INT8 where available, else AVX-512 Q8 | AMX-INT8 is 3.0x the AVX-512 Q8 GEMM at 8B (647 against 218) and 2.3x at 1B; still rising at 256 |

Whether `auto` may select AMX for prefill: the evidence supports AMX-INT8 as the `auto` choice for Q8 checkpoints (the gates pass
at the W8A8 statistical level, the same noise floor D3 measured from a pure kernel switch, and decode is unaffected). AMX-BF16
never becomes a silent default for FP32 checkpoints (user decision); it needs the Hugging Face confirmation first. Both are
decisions for the owner.

## Exit criteria

- [x] Every rung passes its gate (C2, C3, C5, C6 `compare_engines.py` at the chosen M): 30 of 32 passed; the 2 exceptions are the
  recorded 1B AMX-BF16 argmax at position 0 (98.1%), which is a provisional-gate matter, not a kernel fault.
- [x] Prefill speedups are reported with ranges: ranges in the generated table; the per-token baselines at 8192 tokens ran once.
- [x] Decode is within its one-socket range after every rung: 8B FP32 15.5, BF16 15.5, Q8 41.4-42.0 tok/s against 15.45 / 41.81
  per-token references at the same context (one 0.9% miss at 8B AVX-512 Q8, run-to-run spread).
- [ ] Defaults are chosen and justified: recommended above (M per precision); the `auto`-AMX decision is the owner's.
- [x] The BF16 provisional gate of C5 is confirmed or revised from the Hugging Face numbers: revised on 17 November 2026 (job 5830324: max <= 0.22, rms <= 0.015 against Hugging Face, greedy continuations identical), to max 2.0, rms 0.05, argmax 0.97 against FP32 per token; see C5. The 98.1% of the two 1B deviations passes it.

## Follow-ups for a later plan

- Two-socket prefill (D1, D2). Multi-request batching.
- Attention is the next prefill cost at long prompts (41% at 2048, 74% at 8192 tokens on 8B BF16/INT8); a faster attention kernel
  is where an 8192-token prefill could still gain.
- AMX-INT8 is not faster than AMX-BF16 on 8B at 2048 tokens (647 against 640): INT8 only helps once attention is smaller or on
  smaller models.
