# NUMA placement, two-socket tensor parallelism and batched prefill: final results

> **Correction (2026-11-21):** HBM byte figures derived from `hbm_cas_rd` counts were computed at 64 B per count; the counter counts 32 B (calibrated in job 5892172, see [xengine-results-validation.md](xengine-results-validation.md)). Affected figures below are halved and marked †.

**Status: run (2026-11-17, jobs 5845031 and 5845032, commit 8f3114f, node cresco8-hbm14).** The headline points of every feature ran
from one commit and one node state, 3 interleaved repeats, 288 reports (1B and 8B), all `completed`, placement verified in every
report. Provenance and deviations: [the validation record](numa-tp-prefill-results-validation.md). Plan:
[D2](plans/D-integration/D2-final-campaign-and-decision.md). Branch: `feature/integ-d2-final-campaign` (scripts and results documents
only). The generated tables of both campaign directories are in [numa-tp-prefill-results-summary-table.md](numa-tp-prefill-results-summary-table.md).

Closes the plan (milestone M5): one controlled campaign over the whole feature set and a decision for each feature: a default, an
option, or a documented capacity result. Prefill and decode are always measured and reported separately. Inputs: A6, B7, C7 and D1
(their results documents), the study (`numa-tensor-parallel-study.md` sections 8.3, 10.3 and 10.4) and the roofline (`hbm-roofline.md`
sections 3 and 4). Medians of 3 repeats (ranges in the tables), 56 threads per socket, weights on HBM nodes 2 (and 3), THP off,
`D2_M1="avx512_fp32=128 amx_bf16=256 avx512_q8=256 amx_int8=256"`, `D2_M2="avx512_fp32=64 amx_bf16=256 avx512_q8=256 amx_int8=256"`.

## Result in one paragraph

Every feature beats the baseline it replaces, and two sockets win almost everywhere. On 8B at a 2048-token prompt the time to first
token falls from 121.3 s (FP32, per token, one socket) to 10.5 s (AVX-512 GEMM), 2.94 s (AMX-BF16) and 1.61 s (AMX-BF16 on two
sockets, 75x); Q8 falls from 41.2 s to 2.82 s (AMX-INT8) and 1.62 s on two sockets (25x). At 8192 tokens FP32 falls from 534.9 s to
11.8 s (45x), Q8 from 220 s to 11.8 s (19x). Two sockets give 1.7-2.15x the prefill of one socket on 8B at 2048 and 8192 tokens (2.05-2.15x
at 8192, above the 2x of the hardware: not isolated) and 1.4-1.8x at 86 tokens. Decode is 1.54-1.91x on all 12 model, precision and context cells, beyond
run-to-run variability (8B FP32 30.6 / 25.3 tok/s against 16.5 / 13.3 at contexts of 512 / 8192, 8B W8A8 91.0 / 50.6 against 49.4 / 28.8). The only
cells where two sockets do not win beyond variability are two 1B prefills of the 86-token prompt (AMX-BF16 4375 against 3482 tok/s, AVX-512
FP32 1798 against 1225; the ranges overlap). AMX-INT8 and AMX-BF16 are equal on 8B (754 and 722 tok/s on one socket, 1332 and 1344 on two, at 2048 tokens), and on 1B INT8 is
slightly below BF16. Achieved prefill is 10-55% of the modeled per-socket ceilings; decode is 56-90% of the HBM roofline, the gap lying in
the kernels and attention, not in synchronization.

## Full summary table

`tools/summarize_campaign.py` writes this table to `summary.md` of each campaign directory; the complete tables of both jobs are in
[numa-tp-prefill-results-summary-table.md](numa-tp-prefill-results-summary-table.md). Rows below: 1B and 8B, 2048-token prompt, one and two sockets.

| Configuration | Runs | Sockets | Placement | THP | M | Kernel | Precision | Prompt | Warm ctx | TTFT s | Prefill tok/s | Decode tok/s | Decode phases | Reduce+wait | HBM read GB | DDR read GB | UPI rx data flits (M) |
| --- | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 1b-bf16-teams-hbm-s1-t56-m256-p2048-kamx_bf16 | 3 | 1 | hbm (unverified) | off | 256 | amx_bf16 | bf16 | 2048 | - | 0.662 [0.658-0.676] | 3456.23 [3380.58-3480.83] | 87.83 [87.06-87.89] | - | - | 177.6 † | 0.5 | 3 |
| 1b-bf16-teams-hbm-s2-t56-m256-p2048-kamx_bf16 | 3 | 2 | file | off | 256 | amx_bf16 | bf16 | 2048 | - | 0.409 [0.395-0.430] | 5918.62 [5899.66-6157.50] | 157.21 [155.26-161.76] | - | - | 180.0 † | 3.9 | 287 |
| 1b-fp32-teams-hbm-s1-t56-m1-p2048-kper_token | 3 | 1 | hbm (verified) | off | 1 | per_token | fp32 | 2048 | - | 18.268 [18.252-18.318] | 114.35 [114.03-114.50] | 88.93 [88.55-89.91] | - | - | 8251.3 † | 5.9 | 73 |
| 1b-fp32-teams-hbm-s1-t56-m128-p2048-kavx512_fp32 | 3 | 1 | hbm (unverified) | off | 128 | avx512_fp32 | fp32 | 2048 | - | 1.786 [1.782-1.791] | 1205.95 [1203.20-1208.52] | 87.57 [87.23-88.18] | - | - | 226.8 † | 0.6 | 5 |
| 1b-fp32-teams-hbm-s2-t56-m1-p2048-kper_token | 3 | 2 | file | off | 1 | per_token | fp32 | 2048 | - | 12.692 [12.684-12.791] | 164.78 [163.50-164.86] | 161.90 [160.99-163.60] | - | - | 10248.9 † | 13.7 | 1584 |
| 1b-fp32-teams-hbm-s2-t56-m64-p2048-kavx512_fp32 | 3 | 2 | file | off | 64 | avx512_fp32 | fp32 | 2048 | - | 1.099 [1.099-1.100] | 1997.79 [1994.04-1998.71] | 171.68 [164.67-172.94] | - | - | 282.4 † | 4.0 | 288 |
| 1b-int8-teams-hbm-s1-t56-m1-p2048-kper_token_q8 | 3 | 1 | hbm (verified) | off | 1 | per_token_q8 | int8_q8_0 | 2048 | - | 6.938 [6.911-7.006] | 302.67 [299.75-303.87] | 226.01 [225.19-227.24] | - | - | 2178.6 † | 2.7 | 37 |
| 1b-int8-teams-hbm-s1-t56-m256-p2048-kamx_int8 | 3 | 1 | hbm (unverified) | off | 256 | amx_int8 | int8_q8_0 | 2048 | - | 0.715 [0.712-0.718] | 3181.22 [3162.53-3254.49] | 228.80 [217.89-241.90] | - | - | 54.3 † | 0.3 | 2 |
| 1b-int8-teams-hbm-s1-t56-m256-p2048-kavx512_q8 | 3 | 1 | hbm (unverified) | off | 256 | avx512_q8 | int8_q8_0 | 2048 | - | 1.660 [1.642-1.663] | 1301.62 [1299.01-1326.38] | 201.74 [198.10-230.41] | - | - | 54.4 † | 0.4 | 5 |
| 1b-int8-teams-hbm-s2-t56-m1-p2048-kper_token_q8 | 3 | 2 | file | off | 1 | per_token_q8 | int8_q8_0 | 2048 | - | 4.977 [4.970-5.006] | 423.32 [422.56-423.83] | 393.50 [389.15-397.00] | - | - | 2572.9 † | 7.0 | 1428 |
| 1b-int8-teams-hbm-s2-t56-m256-p2048-kamx_int8 | 3 | 2 | file | off | 256 | amx_int8 | int8_q8_0 | 2048 | - | 0.446 [0.441-0.457] | 5351.63 [5206.36-5407.64] | 401.65 [401.54-442.57] | - | - | 53.4 † | 1.8 | 312 |
| 1b-int8-teams-hbm-s2-t56-m256-p2048-kavx512_q8 | 3 | 2 | file | off | 256 | avx512_q8 | int8_q8_0 | 2048 | - | 0.924 [0.910-0.943] | 2403.21 [2347.52-2436.50] | 296.36 [292.94-417.79] | - | - | 52.3 † | 1.1 | 270 |
| 8b-bf16-teams-hbm-s1-t56-m256-p2048-kamx_bf16 | 3 | 1 | hbm (unverified) | off | 256 | amx_bf16 | bf16 | 2048 | - | 2.944 [2.943-2.946] | 722.12 [722.11-722.79] | 15.71 [15.62-15.79] | - | - | 1146.2 † | 19.4 | 18 |
| 8b-bf16-teams-hbm-s2-t56-m256-p2048-kamx_bf16 | 3 | 2 | file | off | 256 | amx_bf16 | bf16 | 2048 | - | 1.611 [1.591-1.612] | 1343.64 [1342.52-1360.27] | 30.74 [30.67-30.91] | - | - | 1132.1 † | 47.5 | 2770 |
| 8b-fp32-teams-hbm-s1-t56-m1-p2048-kper_token | 3 | 1 | hbm (verified) | off | 1 | per_token | fp32 | 2048 | - | 121.309 [121.143-121.358] | 17.17 [17.17-17.20] | 15.81 [15.76-15.81] | - | - | 59656.1 † | 55.2 | 273 |
| 8b-fp32-teams-hbm-s1-t56-m128-p2048-kavx512_fp32 | 3 | 1 | hbm (unverified) | off | 128 | avx512_fp32 | fp32 | 2048 | - | 10.504 [10.500-10.506] | 199.37 [199.32-199.46] | 15.69 [15.61-15.73] | - | - | 1463.1 † | 17.4 | 23 |
| 8b-fp32-teams-hbm-s2-t56-m1-p2048-kper_token | 3 | 2 | file | off | 1 | per_token | fp32 | 2048 | - | 67.073 [67.053-67.133] | 31.07 [31.04-31.08] | 31.05 [30.98-31.11] | - | - | 63871.0 † | 83.4 | 5909 |
| 8b-fp32-teams-hbm-s2-t56-m64-p2048-kavx512_fp32 | 3 | 2 | file | off | 64 | avx512_fp32 | fp32 | 2048 | - | 5.392 [5.386-5.397] | 390.52 [390.09-390.92] | 30.33 [30.10-30.56] | - | - | 1892.5 † | 62.0 | 3651 |
| 8b-int8-teams-hbm-s1-t56-m1-p2048-kper_token_q8 | 3 | 1 | hbm (verified) | off | 1 | per_token_q8 | int8_q8_0 | 2048 | - | 41.166 [41.042-41.493] | 50.66 [50.25-50.80] | 44.00 [40.99-44.04] | - | - | 16270.6 † | 18.7 | 110 |
| 8b-int8-teams-hbm-s1-t56-m256-p2048-kamx_int8 | 3 | 1 | hbm (unverified) | off | 256 | amx_int8 | int8_q8_0 | 2048 | - | 2.823 [2.799-2.829] | 753.90 [751.95-759.85] | 41.79 [41.07-42.41] | - | - | 357.1 † | 2.0 | 11 |
| 8b-int8-teams-hbm-s1-t56-m256-p2048-kavx512_q8 | 3 | 1 | hbm (unverified) | off | 256 | avx512_q8 | int8_q8_0 | 2048 | - | 9.395 [9.341-9.410] | 223.04 [222.68-224.33] | 44.07 [42.63-44.31] | - | - | 344.2 † | 2.2 | 13 |
| 8b-int8-teams-hbm-s2-t56-m1-p2048-kper_token_q8 | 3 | 2 | file | off | 1 | per_token_q8 | int8_q8_0 | 2048 | - | 23.625 [23.607-23.805] | 88.35 [87.67-88.41] | 85.80 [84.88-86.15] | - | - | 17271.0 † | 22.4 | 4814 |
| 8b-int8-teams-hbm-s2-t56-m256-p2048-kamx_int8 | 3 | 2 | file | off | 256 | amx_int8 | int8_q8_0 | 2048 | - | 1.623 [1.614-1.635] | 1332.16 [1318.86-1337.24] | 83.31 [80.95-83.67] | - | - | 355.6 † | 10.9 | 1269 |
| 8b-int8-teams-hbm-s2-t56-m256-p2048-kavx512_q8 | 3 | 2 | file | off | 256 | avx512_q8 | int8_q8_0 | 2048 | - | 4.848 [4.831-4.866] | 434.48 [432.80-436.03] | 81.78 [76.01-81.89] | - | - | 343.4 † | 19.7 | 1841 |

## 1. Prefill

Per precision x sockets x prompt cell; speedup over the one-socket per-token baseline of the same precision (BF16 has none: its rows
are compared with FP32). `D2_M1` and `D2_M2` give M per rung (C7 for one socket, D1 for two): FP32 AVX-512 128 and 64, AMX-BF16 256 and 256,
AVX-512 Q8 256 and 256, AMX-INT8 256 and 256. 8B, every prompt:

| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8b | bf16 | 86 | amx_bf16 | 256 | 1 | 769.63 [769.53-783.04] | 0.164 [0.162-0.165] | - | - |
| 8b | bf16 | 86 | amx_bf16 | 256 | 2 | 1079.32 [1061.79-1252.11] | 0.134 [0.123-0.136] | - | yes |
| 8b | bf16 | 2048 | amx_bf16 | 256 | 1 | 722.12 [722.11-722.79] | 2.944 [2.943-2.946] | - | - |
| 8b | bf16 | 2048 | amx_bf16 | 256 | 2 | 1343.64 [1342.52-1360.27] | 1.611 [1.591-1.612] | - | yes |
| 8b | bf16 | 8192 | amx_bf16 | 256 | 1 | 330.04 [329.90-330.14] | 25.097 [25.095-25.103] | - | - |
| 8b | bf16 | 8192 | amx_bf16 | 256 | 2 | 709.34 [709.12-709.63] | 11.771 [11.764-11.772] | - | yes |
| 8b | fp32 | 86 | avx512_fp32 | 128 | 1 | 226.12 [224.87-226.16] | 0.433 [0.431-0.434] | 13.49 | - |
| 8b | fp32 | 86 | avx512_fp32 | 64 | 2 | 365.16 [360.68-365.38] | 0.286 [0.286-0.289] | 21.78 | - |
| 8b | fp32 | 86 | per_token | 1 | 1 | 16.76 [16.68-16.98] | 5.183 [5.122-5.209] | 1.00 | - |
| 8b | fp32 | 86 | per_token | 1 | 2 | 30.81 [30.75-30.82] | 2.842 [2.841-2.849] | 1.84 | yes |
| 8b | fp32 | 2048 | avx512_fp32 | 128 | 1 | 199.37 [199.32-199.46] | 10.504 [10.500-10.506] | 11.61 | - |
| 8b | fp32 | 2048 | avx512_fp32 | 64 | 2 | 390.52 [390.09-390.92] | 5.392 [5.386-5.397] | 22.74 | - |
| 8b | fp32 | 2048 | per_token | 1 | 1 | 17.17 [17.17-17.20] | 121.309 [121.143-121.358] | 1.00 | - |
| 8b | fp32 | 2048 | per_token | 1 | 2 | 31.07 [31.04-31.08] | 67.073 [67.053-67.133] | 1.81 | yes |
| 8b | fp32 | 8192 | avx512_fp32 | 128 | 1 | 148.41 [148.34-148.42] | 55.623 [55.600-55.635] | 9.65 | - |
| 8b | fp32 | 8192 | avx512_fp32 | 64 | 2 | 303.91 [303.91-303.94] | 27.245 [27.239-27.259] | 19.75 | - |
| 8b | fp32 | 8192 | per_token | 1 | 1 | 15.39 [15.36-15.40] | 534.907 [534.402-535.775] | 1.00 | - |
| 8b | fp32 | 8192 | per_token | 1 | 2 | 27.94 [27.93-27.99] | 294.650 [294.109-294.741] | 1.82 | yes |
| 8b | int8 | 86 | amx_int8 | 256 | 1 | 860.53 [803.67-877.13] | 0.152 [0.150-0.160] | 15.78 | - |
| 8b | int8 | 86 | amx_int8 | 256 | 2 | 1262.05 [953.65-1359.06] | 0.116 [0.111-0.143] | 23.14 | yes |
| 8b | int8 | 86 | avx512_q8 | 256 | 1 | 261.24 [239.47-261.78] | 0.383 [0.381-0.411] | 4.79 | - |
| 8b | int8 | 86 | avx512_q8 | 256 | 2 | 435.99 [419.40-439.13] | 0.245 [0.244-0.258] | 7.99 | yes |
| 8b | int8 | 86 | per_token_q8 | 1 | 1 | 54.55 [54.10-56.70] | 1.629 [1.574-1.642] | 1.00 | - |
| 8b | int8 | 86 | per_token_q8 | 1 | 2 | 95.36 [94.27-96.72] | 0.949 [0.940-0.960] | 1.75 | yes |
| 8b | int8 | 2048 | amx_int8 | 256 | 1 | 753.90 [751.95-759.85] | 2.823 [2.799-2.829] | 14.88 | - |
| 8b | int8 | 2048 | amx_int8 | 256 | 2 | 1332.16 [1318.86-1337.24] | 1.623 [1.614-1.635] | 26.30 | yes |
| 8b | int8 | 2048 | avx512_q8 | 256 | 1 | 223.04 [222.68-224.33] | 9.395 [9.341-9.410] | 4.40 | - |
| 8b | int8 | 2048 | avx512_q8 | 256 | 2 | 434.48 [432.80-436.03] | 4.848 [4.831-4.866] | 8.58 | yes |
| 8b | int8 | 2048 | per_token_q8 | 1 | 1 | 50.66 [50.25-50.80] | 41.166 [41.042-41.493] | 1.00 | - |
| 8b | int8 | 2048 | per_token_q8 | 1 | 2 | 88.35 [87.67-88.41] | 23.625 [23.607-23.805] | 1.74 | yes |
| 8b | int8 | 8192 | amx_int8 | 256 | 1 | 338.96 [337.62-339.68] | 24.443 [24.386-24.536] | 9.06 | - |
| 8b | int8 | 8192 | amx_int8 | 256 | 2 | 710.18 [706.29-714.20] | 11.756 [11.695-11.821] | 18.97 | yes |
| 8b | int8 | 8192 | avx512_q8 | 256 | 1 | 159.19 [158.92-159.23] | 51.868 [51.833-51.933] | 4.25 | - |
| 8b | int8 | 8192 | avx512_q8 | 256 | 2 | 337.83 [336.30-338.15] | 24.516 [24.494-24.625] | 9.03 | yes |
| 8b | int8 | 8192 | per_token_q8 | 1 | 1 | 37.43 [37.37-37.44] | 219.950 [219.875-220.300] | 1.00 | - |
| 8b | int8 | 8192 | per_token_q8 | 1 | 2 | 64.18 [64.15-64.21] | 128.336 [128.287-128.395] | 1.71 | yes |

1B, 2048-token prompt (the 86 and 8192-token rows are in the generated table):

| Model | Precision | Prompt | Rung | M | Sockets | Prefill tok/s | TTFT s | Speedup vs 1-socket per-token | 2s wins |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1b | bf16 | 2048 | amx_bf16 | 256 | 1 | 3456.23 [3380.58-3480.83] | 0.662 [0.658-0.676] | - | - |
| 1b | bf16 | 2048 | amx_bf16 | 256 | 2 | 5918.62 [5899.66-6157.50] | 0.409 [0.395-0.430] | - | yes |
| 1b | fp32 | 2048 | avx512_fp32 | 128 | 1 | 1205.95 [1203.20-1208.52] | 1.786 [1.782-1.791] | 10.55 | - |
| 1b | fp32 | 2048 | avx512_fp32 | 64 | 2 | 1997.79 [1994.04-1998.71] | 1.099 [1.099-1.100] | 17.47 | - |
| 1b | fp32 | 2048 | per_token | 1 | 1 | 114.35 [114.03-114.50] | 18.268 [18.252-18.318] | 1.00 | - |
| 1b | fp32 | 2048 | per_token | 1 | 2 | 164.78 [163.50-164.86] | 12.692 [12.684-12.791] | 1.44 | yes |
| 1b | int8 | 2048 | amx_int8 | 256 | 1 | 3181.22 [3162.53-3254.49] | 0.715 [0.712-0.718] | 10.51 | - |
| 1b | int8 | 2048 | amx_int8 | 256 | 2 | 5351.63 [5206.36-5407.64] | 0.446 [0.441-0.457] | 17.68 | yes |
| 1b | int8 | 2048 | avx512_q8 | 256 | 1 | 1301.62 [1299.01-1326.38] | 1.660 [1.642-1.663] | 4.30 | - |
| 1b | int8 | 2048 | avx512_q8 | 256 | 2 | 2403.21 [2347.52-2436.50] | 0.924 [0.910-0.943] | 7.94 | yes |
| 1b | int8 | 2048 | per_token_q8 | 1 | 1 | 302.67 [299.75-303.87] | 6.938 [6.911-7.006] | 1.00 | - |
| 1b | int8 | 2048 | per_token_q8 | 1 | 2 | 423.32 [422.56-423.83] | 4.977 [4.970-5.006] | 1.40 | yes |

Two sockets over one socket at the same rung (prefill tok/s ratio, 2 sockets / 1 socket, same precision):

| Model | Rung | 86 tokens | 2048 tokens | 8192 tokens |
| --- | --- | ---: | ---: | ---: |
| 8B | FP32 per token | 1.84 | 1.81 | 1.82 |
| 8B | AVX-512 FP32 GEMM (M 128 / 64) | 1.61 | 1.96 | 2.05 |
| 8B | AMX-BF16 | 1.40 | 1.86 | 2.15 |
| 8B | Q8 per token | 1.75 | 1.74 | 1.71 |
| 8B | AVX-512 Q8 | 1.67 | 1.95 | 2.12 |
| 8B | AMX-INT8 | 1.47 | 1.77 | 2.10 |
| 1B | AVX-512 FP32 GEMM (M 128 / 64) | 1.47 | 1.66 | 1.72 |
| 1B | AMX-BF16 | 1.26 | 1.71 | 1.92 |
| 1B | AVX-512 Q8 | 1.60 | 1.85 | 1.91 |
| 1B | AMX-INT8 | 1.24 | 1.68 | 1.89 |

Reading. The gain of two sockets grows with the prompt (short prompts are limited by the serial part of a microbatch and by the
exchange of an M x 4096 vector twice per layer; long prompts by attention, which each socket does for its own heads and KV). At 8192
tokens the 8B rungs exceed 2.0x (2.05-2.15x): the cause was not isolated here (candidates: half the KV and score traffic per socket, and
the per-socket caches), so it is reported as measured. Against C7 (staged files, commit ca2f7f8) the one-socket 8B rungs are faster:
AMX-BF16 722 against 640 tok/s and AMX-INT8 754 against 647 at 2048 tokens, AVX-512 FP32 199 against 183 (+9% to +17%, mostly the row-parallel
RMSNorm/RoPE of D1, measured at +12-16% on one socket in job 5713378); the per-token rungs, whose code did not change, agree with C7 within 0.5% (FP32
17.17 against 17.26) and +3% (Q8 50.66 against 49.10).

## 2. Decode

From retained contexts of 512 and 8192 tokens (128 tokens timed, `--warm-context`), 56 threads per socket, FP32, W8A8 (`int8`) and W8A32.
Compared with A6 (8B FP32 15.69 owned HBM and 16.07 staged at a short context; W8A8 50.76 and 50.91) and `docs/kernels.md` (14.90 and 44.49,
the first campaign): the one-socket 8B figures at a context of 512 are 16.45 and 49.41, the difference to A6 being the retained context
and later kernel work.

| Model | Precision | Context | 1x56 | 2x56 | Gain 2x56 / 1x56 | Beyond variability |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1B | fp32 | 512 | 90.69 [90.69-91.38] | 172.89 [170.25-173.44] | 1.91 | yes |
| 1B | fp32 | 8192 | 71.16 [70.44-71.18] | 129.61 [128.61-130.24] | 1.82 | yes |
| 1B | int8 | 512 | 262.18 [254.76-266.67] | 472.87 [472.59-473.90] | 1.80 | yes |
| 1B | int8 | 8192 | 122.44 [122.36-123.18] | 195.96 [195.50-196.24] | 1.60 | yes |
| 1B | w8a32 | 512 | 237.58 [236.33-239.99] | 368.39 [359.07-372.93] | 1.55 | yes |
| 1B | w8a32 | 8192 | 116.62 [116.30-117.24] | 195.86 [195.17-195.96] | 1.68 | yes |
| 8B | fp32 | 512 | 16.45 [16.44-16.47] | 30.57 [30.50-30.67] | 1.86 | yes |
| 8B | fp32 | 8192 | 13.31 [13.30-13.46] | 25.29 [25.21-25.43] | 1.90 | yes |
| 8B | int8 | 512 | 49.41 [49.01-49.47] | 90.95 [90.72-91.50] | 1.84 | yes |
| 8B | int8 | 8192 | 28.83 [28.82-29.01] | 50.58 [50.40-51.22] | 1.75 | yes |
| 8B | w8a32 | 512 | 43.62 [42.91-44.31] | 81.27 [81.15-81.44] | 1.86 | yes |
| 8B | w8a32 | 8192 | 30.64 [30.40-30.70] | 47.26 [45.99-47.52] | 1.54 | yes |

Decode after each rung's prefill is unchanged by the prefill choice (the 32-token decode of the prefill cells above, 8B FP32 on one socket:
15.68-15.72 tok/s after per-token, AVX-512 and AMX-BF16 prefill of 86 tokens; Q8 50.5-51.0 after per-token, AVX-512 Q8 and AMX-INT8). One
caution: the three two-socket 1B W8A8 runs from a retained context of 8192 tokens ended on a stop token after 29 generated tokens (not
128), so their 195.96 tok/s covers 29 tokens (the validation record lists it).

## 3. Crossover tables (study 10.4)

Decode, every model, precision and context of the matrix (gain of two sockets x 56 over one socket x 56):

| Model | Precision | Context | Gain | Two sockets win |
| --- | --- | ---: | ---: | --- |
| 1B | fp32 | 512 | 1.91 | yes |
| 1B | fp32 | 8192 | 1.82 | yes |
| 1B | int8 | 512 | 1.80 | yes |
| 1B | int8 | 8192 | 1.60 | yes |
| 1B | w8a32 | 512 | 1.55 | yes |
| 1B | w8a32 | 8192 | 1.68 | yes |
| 8B | fp32 | 512 | 1.86 | yes |
| 8B | fp32 | 8192 | 1.90 | yes |
| 8B | int8 | 512 | 1.84 | yes |
| 8B | int8 | 8192 | 1.75 | yes |
| 8B | w8a32 | 512 | 1.86 | yes |
| 8B | w8a32 | 8192 | 1.54 | yes |

Prefill: two sockets win beyond variability in 34 of the 36 (model, rung, prompt) comparisons that D2 runs (per-token, AVX-512 and AMX
rungs of both precisions, prompts of 86, 2048 and 8192 tokens, both models; the generated tables mark the same-M ones, the AVX-512 FP32 pair
has different M on the two socket counts and was compared from the ranges). The exceptions are 1B at 86 tokens: AMX-BF16 (4375 [2465-4503]
against 3482 [2179-3500] tok/s) and AVX-512 FP32 (1798 [1127-1824] against 1225 [888-1243]), where a prefill of 68-120 ms is dominated by start-up noise
(each of the four has one slow repeat).
The gain is the smallest where the microbatch work is shortest (86 tokens: 1.24-1.84x) and the largest at 8192 tokens (1.71-2.15x).

Where synchronization dominates: nowhere. B7's 30 profiled decode runs never reach a revisit trigger (reductions plus waits at most
0.022 ms per token), and in D2's prefill the exchange is 1-7% of the layer time (D1 results). Where attention dominates: long contexts
and long prompts: 8B AMX rungs spend 41% of a one-socket forward in attention at 2048 tokens and 74% at 8192 (C7), which is why the 8192-token
prefill (330-339 tok/s on one socket, 709-710 on two) is half the rate of the 2048-token one and why the gain of the matrix kernels there is
small; decode gains fall at long contexts for 1B (1.60 for W8A8 at 8192, B7: 1.26 at 32768) and for 8B W8A32 (1.54) for the same reason.
Capacity-only gains: none needed here, since every checkpoint and KV of the matrix fits in one socket's 64 GiB of HBM (8B FP32 weights 30 GB
on one socket, 15 GB per socket on two).

## 4. Placement evidence

All 288 reports carry their region table with page-level verification (`move_pages`): 144 `embedding` (DDR node 0), 432 `weights`, 864 `kv`
(432 `kv.k` + 432 `kv.v`) and 108 `packed` regions, and 576 of the 720 `work` regions are `full`, with **0 mismatched pages** in every one. Weights,
embedding, packed and work regions have no missing page; the KV regions report pages never touched (`pages_missing`, up to 49,609 of 270,336 pages,
the context beyond the prompt and the generated tokens, which is allocated but not written), and none placed elsewhere. Weights, KV and packed shards sit on the requested HBM node (2 for socket 0, 3 for
socket 1); the embedding table, which only a row is read from per token, is kept on DDR node 0. The other 144 `work` regions (the 1.0-70 MB
scratch of one-socket batched runs, node 2) are `unverified`: the one-socket batched path does not verify its work region, and the generated
tables mark those rows `hbm (unverified)`; the two-socket runs verify theirs. Two-socket reports state `placement: file` in their runtime
field while their regions are verified on nodes 2 and 3: the shard loader of `LLAMA_SOCKETS=2` stages from the file and places each socket's shard itself
(the field names the loader, not the residency). THP was off for all runs (the node's setting is `[always]`, which backs the weights and packed shards with huge pages anyway:
`thp_bytes` is 99.6-100% of the region size in every report; A6 found the explicit setting neutral). Memory traffic, 8B FP32 decode from a retained context of 512
tokens (HBM read GB over the run, halved †): 4119 on one socket, 4110 on two, so the weights are read once in total and split between the sockets; DDR read
33 and 51 GB (the embedding rows and the copy); UPI rx data flits 33 M on one socket and 2583 M on two, about 0.6% of the 456,259 M
that 112 threads reading remote weights produced in `docs/kernels.md` (another context length), i.e. no weight traffic crosses the link (B7 measured 0.8-0.9% against its control).

## 5. Numerical evidence

D2 reran the headline points and compared performance; it does not repeat the comparisons of the earlier phases, which passed on earlier
commits (listed here, with the source differences since: `src/numa.c` only, the A6 far-HBM diagnostic override, plus the campaign scripts):

| Comparison | Gate | Observed | Result |
| --- | --- | --- | --- |
| FP32 two sockets against one, batched | max abs dlogit <= 1e-3 | 7e-5 (1B and 8B, 86 and 2048 tokens); Hugging Face with two sockets 1B 2.5e-5, 8B 7.7e-5, argmax 100% (D1 jobs 5704689, 5837867) | pass |
| W8A8 and AMX-INT8 two sockets against one | statistical gate (rms 0.2, argmax 0.9, max 16) | 20 of 20 D1 comparisons; rms 0.10-0.14 on 8B, the noise floor shown by D3 for a pure kernel switch | pass |
| W8A32 two sockets against one | strict 1e-3 | 6.7e-5 (B7 gates, job 5704682); B6 Hugging Face re-run passes | pass |
| BF16 against FP32 / Hugging Face | max 2.0, rms 0.05, argmax 0.97 (fixed by K1) | against FP32: max 0.057-1.48, rms 0.0096-0.024, argmax 98.1-100%; against Hugging Face 8B max 0.22, rms 0.015, 100% argmax (job 5830324) | the gate is confirmed and revised from the data; pass |
| `anon-hbm` against staged | bitwise | bitwise on the 4 comparisons (A6, jobs 5713402, 5713411) | pass |
| packed4 against rows | bitwise | max abs 0 on the real 8B AVX-512 kernels (B7 gates, job 5704682) | pass |

The BF16 provisional gate of C5 is **confirmed in its revised form** (max 2.0, rms 0.05, argmax 0.97, `BF16_GATE`), after the Hugging Face run. The
campaign itself ran no logit comparison (`SKIP_TEST=1`, `HF_GATE=0`).

## 6. Modeled against measured

The roofline (`hbm-roofline.md` sections 3 and 4) gives per-socket upper bounds that ignore attention, conversion and overheads, doubled for two sockets.

Decode, context 512 (tok/s, achieved / roofline at 550 GB/s per socket, 1100 GB/s for two; the KV is excluded from the roofline):

| Model | Precision | 1x56 measured | 1x56 roofline | Share | 2x56 measured | 2x56 roofline | Share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | FP32 | 16.45 | 18.3 | 90% | 30.57 | 36.6 | 84% |
| 8B | W8A8 | 49.41 | 69 | 72% | 90.95 | 138 | 66% |
| 1B | FP32 | 90.69 | 111 | 82% | 172.89 | 222 | 78% |
| 1B | W8A8 | 262.18 | 419 | 63% | 472.87 | 838 | 56% |

The study 8.3 model of two sockets (27.19 ms per token for 8B FP32, 36.8 tok/s) is reached to 83%: the measured 32.7 ms is in the phases (the FFN is
22 of 32.5 ms in B7's profile), not in waiting: synchronization is under 0.1% of the token. FP32 sits near the roofline because its kernel streams at 81-90% of
the HBM read rate; the Q8 kernels reach 64-75% of it (`docs/kernels.md` Q8_0 tuning) and 1B is limited by small matrices, 16 layers of 2048-wide GEMVs. At a context of 8192
the share falls (8B FP32 13.31 against 18.3, 25.29 against 36.6) because the roofline excludes the KV reads (B7: attention 9.8 of 39.3 ms at 8192).

Prefill, 2048 tokens (tok/s, per-socket ceilings of the roofline section 4: 8B 370 / 1800 / 4600 and 1B 2200 / 10900 / 27700 for AVX-512 FP32 / AMX-BF16 / AMX-INT8):

| Model | Rung | 1x56 | Share of ceiling | 2x56 | Share of 2 x ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| 8B | AVX-512 FP32 | 199 | 54% | 391 | 53% |
| 8B | AMX-BF16 | 722 | 40% | 1344 | 37% |
| 8B | AMX-INT8 | 754 | 16% | 1332 | 14% |
| 1B | AVX-512 FP32 | 1206 | 55% | 1998 | 45% |
| 1B | AMX-BF16 | 3456 | 32% | 5919 | 27% |
| 1B | AMX-INT8 | 3181 | 11% | 5352 | 10% |

The gap to the ceilings is attention and the non-GEMM work, not the matrix units: on 8B at 2048 tokens attention is 41% of the AMX forward and
the GEMMs 53-58% (C7 profile); a GEMM that is twice as fast cannot make the forward more than 1.4x faster. That is also why AMX-INT8 (ceiling 2.5x
BF16) is no faster than BF16 on 8B: the rungs differ only in the GEMM share. D1 measured the serial per-row work (RMSNorm 18% of the
AMX-BF16 token) and removed it, which accounts for most of the +9% to +17% against C7 here. At 8192 tokens the share falls to 7-18% (8B AMX-BF16 330 tok/s
on one socket) as attention reaches 74%.

## 7. Decisions

Code defaults are not changed by this run; the table says which default the evidence supports (the owner flips them).

Since 20 November 2026 the code follows this table: the recommended defaults are implemented ([architecture.md](architecture.md)
§2.1), and packed4 was removed rather than kept as an option ([audit-2026-11.md](audit-2026-11.md) F1, F4).

| Feature | Becomes default when | Evidence | Decision |
| --- | --- | --- | --- |
| Engine-owned placement (`LLAMA_PLACEMENT=hbm`) | not slower than staged HBM, and verified | Verified in 288 of 288 reports, 0 mismatched pages. Against staged: A6 8B FP32 decode -2.4% (15.69 against 16.07; marginal miss), Q8 equal; D2 per-token prefill against C7's staged files: FP32 -0.5%, Q8 +3% (different commits) | **Option** (`file` stays the default for one socket). Required for the two-socket path, which always places and verifies its own shards. Use it where residency must be proven and for capacity |
| Two sockets (`LLAMA_SOCKETS=2`) | lower latency than the best one-socket configuration beyond variability, per model / precision / context | Decode 12 of 12 cells (1.54-1.91x), prefill 34 of 36 (1.24-2.15x); no synchronization cost (B7) | **Default on a dual-socket HBM node** (the rule is met everywhere D2 looked, except two 1B cells at 86 tokens, whose 70-120 ms prefills are noise-limited). Keep 1 where the node has one socket or a different topology |
| Batched prefill (`LLAMA_PREFILL_BATCH=M`) | faster TTFT than per-token beyond variability, gates passing | 8B 2048 tokens: 121 s to 10.5 s (FP32), 41 s to 9.4 s (Q8), with every gate passing and decode unchanged | **Default**: M = 128 (FP32, one socket), 64 (FP32, two sockets), 256 for Q8 and for the AMX rungs |
| AMX-INT8 for Q8 prefill | faster than AVX-512 Q8 batched, gate passing | 3.4x on one socket (754 against 223 tok/s, 8B 2048 tokens) and 3.1x on two (1332 against 434); faster in every one of the 12 cells (2 models x 3 prompts x 2 socket counts), 1.6-3.4x; W8A8 statistical gate passes, Hugging Face recorded (K1) | **Default for Q8 checkpoints when AMX is present** (`auto`), AVX-512 VNNI otherwise |
| AMX-BF16 | never a silent default for FP32 checkpoints (user decision); recommended when faster within its gate | 3.6x the AVX-512 FP32 GEMM on one socket (722 against 199 tok/s) and 3.4x on two (1344 against 391); gate max 2.0 / rms 0.05 / argmax 0.97 met, Hugging Face 8B max 0.22 | **Opt-in** (`LLAMA_PRECISION=bf16`), recommended in the documentation for prefill-heavy use |
| `LLAMA_FP32_LAYOUT=packed4` | bitwise equal and faster beyond the range (B7) | bitwise equal; +3.2% at context 512 only, a tie at 8192 and 32768, 14 GB more HBM per socket (B7) | **Not adopted**; stays an option |
| Prefill attention (`LLAMA_ATTN`) | faster TTFT than `split` beyond variability on the rung, decode unchanged, gates passing, and the determinism comparison (job 5864239) passing | E4 (job 5864134, commit 5a354ea): `tiled` faster in 32 of 32 cells; 8B 8192 tokens one socket 1.55x on AMX-BF16 (16.2 against 25.1 s), 1.50x on AMX-INT8, 1.16-1.18x on AVX-512 FP32; 2048 tokens 1.04-1.32x; decode ratio 0.997; placement verified in 256 of 256 reports; 30 of 30 logit gates pass; determinism rerun (job 5864239) 12 of 12 identical ([attention-results.md](attention-results.md) section 2) | **Recommended `tiled` on every rung tested** (AMX-BF16, AMX-INT8, AVX-512 FP32; largest gain on the AMX rungs at long prompts). The determinism rerun passed and the owner switched the code default to `tiled` (2026-11-19); `LLAMA_ATTN=split` reproduces earlier runs. The two-socket gain at 8192 tokens falls from 2.0-2.13x to at most 1.95x on 8B |
| Decode attention (`LLAMA_ATTN_DECODE`) | faster decode than `split` beyond variability, 512 not slower, gates passing | E5 (job 5870935, commit 9d3f72b): `group` faster in 16 of 16 cells at 8192 and 32768 tokens (1.007-1.371x); at 512 all 8 cells overlap `split`, 2 below the B7 range (8B one socket, loss in the weight phases); 25 of 25 correctness gates pass ([decode-attention-results.md](decode-attention-results.md)) | **Stays `split`** pending the 512 rerun and the owner's decision; `group` is a candidate for long contexts |
| K/V cache layout (`LLAMA_KV_LAYOUT=rows\|head`, E5.4) | bitwise equal to `rows` on every path; decode and prefill not slower; the E5 value gates (1B 32768 one socket 1.3x with `group`, two-socket W8A8 1.5x) | Knob on `develop`, default `head` since the owner's decision of 2026-11-20 (`rows` before); rules and results in [decode-attention-results.md](decode-attention-results.md) section 9; E5.4 (jobs 5876167, 5876582, 5876584, 5876592): C1, C2 (bitwise, 48 of 48 logit comparisons, 16 of 16 generations), V1 (1.34x FP32, 1.65x W8A8), V2 (1.87x) and V4 pass; V3 fails in 1 of 16 cells (8B FP32 one socket `group` at 512, 0.01 tok/s) | **Adopted as the engine default** (owner decision 2026-11-20, V3 cell waived: it is the `group` effect at 512, and with the default `split` head is faster in that cell, 16.60 against 16.45 tok/s); campaigns pin `rows` so their numbers stay comparable with earlier ones; `LLAMA_KV_LAYOUT=rows` restores the old layout |

## Exit criteria

- [x] Every row of the decision table has a decision backed by D2 evidence: six of six rows above, each citing the D2 cells (placement also cites A6 for the staged comparison, which D2 does not rerun).
- [x] The README milestones show M1 to M5 done: M1-M5 complete in the plans README (17 November 2026).
- [x] `docs/kernels.md` "Not in this change" updated: the four items it listed now point at their results documents.

## Follow-ups for a later plan

- Attention is the next cost at long prompts and contexts (41% of the forward at 2048 tokens, 74% at 8192 on 8B AMX; 65-81% of a 1B decode token at 32768): an AMX or fused attention kernel and KV-aware scheduling.
- The two-socket prefill gain above 2x at 8192 tokens (2.05-2.15x on 8B) deserves a counter-based explanation.
- BF16 checkpoints or KV, KV tiering across HBM and DDR, multi-request batching (batched decode), exchange/compute overlap and asymmetric FFN shards (B7 follow-ups) are candidates for a later plan.
- The two-socket W8A8 continuation can diverge from the one-socket one (1B, 8192-token context: stop token after 29 tokens in three of three repeats): within the statistical gate, but a determinism comparison of the generated text across kernels is not part of the gates.
