# Reference numbers for the hybrid-memory campaign

Compiled 24 November 2026 from `develop` at b9b9bbb, plus the earlier 70B comparison work (older tree). This page lists every number the sources state. It is a plain reference: no conclusions and no recommendations.

## Sources

| Short name | Source |
| --- | --- |
| capacity study | `docs/plans/hbm-capacity-study.md` on develop (plan, 21 Nov; its numbers are predictions unless labelled measured) |
| status | `docs/status-2026-11-22.md` on develop |
| roofline | `docs/hbm-roofline.md` on develop (early Phase 2 engine; several figures predate the current kernels) |
| xengine | `docs/xengine-results.md` on develop (cross-engine campaign, tree a3402b6, jobs 5892196, 5892201, 5892202) |
| prefill-gap | `docs/prefill-gap.md` on develop (tree 684489b before, 23cac73 after) |
| gs64 | `docs/gs64-gemm.md` on develop (job 5910223, 5913457 context) |
| gs32 | `docs/gs32-amx.md` on develop; its section 6 is the 24 Nov gs32 decode GEMV work (no separate document exists on develop) |
| numa-tp | `docs/numa-tp-prefill-results.md` on develop (tree 8f3114f, jobs 5845031 and 5845032, 17 Nov) |
| earlier 70B comparison work (older tree) | Earlier 70B comparison work on an older tree, with a pilot study, a matched replication and a three-engine comparison. Cited by section heading only. |

## Conventions

- Labels: **measured** (from a cluster run), **modelled** (a prediction from a model), **derived** (arithmetic on other numbers or on architecture), **estimated** (planning figure or rough value, including planned tolerances and assumptions).
- Job ids are given as the source gives them. "none given" means the source names no job for that number.
- P is a 2048-token prompt with one generated token. L is an 8192-token prompt with one generated token (for decode at L, the depth is 8192). D is an 86-token prompt with 128 generated tokens. Decode tok/s for the cross-engine table is the D workload unless stated.
- Ranges in brackets are min to max of the repeats, as the source gives them.
- 1S and 2S mean one or two sockets, 56 cores per socket. HBM means Flat mode with all tensors on HBM. DDR means Flat mode with all tensors on DDR. Cache means Cache mode on a different physical node (no privileged counters, no anchor run).
- Status column: "current" means not superseded by a later measurement on develop. "stale" means the source says the number predates a change that alters it, and gives the reason. "older tree" means the number was measured on a tree other than develop.

---

## 1. Llama 3.1 70B

### 1.1 Sizes, KV, capacity and placement arithmetic

| ID | Quantity | Value | Configuration | Label | Job | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 1.01 | Layers x KV heads x head dim | 80 x 8 x 128 | 70B Instruct | derived | none given | capacity study section 2 table |
| 1.02 | Q8_0 checkpoint file | 75.0 GB (69.8 GiB); 74,967,187,712 B including the untied input embedding | 70B, Q8_0 group 64 (v3) | measured (bytes, older tree); derived (GB) | none given | earlier 70B comparison work (older tree), "Phase accounting and performance model" |
| 1.03 | HBM-resident weights, both sockets | 68.8 GiB (34.4 GiB per socket) | 70B Q8_0, embedding in DDR | derived | none given | capacity study section 2 table |
| 1.04 | Weight bytes per decode token, logical | 73.85 GB (73.856 GB including classifier, FP32 norms and one embedding row) | 70B Q8_0 group 64 | derived | none given | capacity study section 2 table; earlier 70B comparison work (older tree), "Phase accounting" |
| 1.05 | Weight bytes per decode token per socket | 36.9 GB | 70B, 2 sockets | derived | none given | capacity study section 5 |
| 1.06 | FP32 file size | 282 GB | 70B FP32 | derived | none given | capacity study section 2 table |
| 1.07 | KV per token, FP32 | 640 KiB across the node; 320 KiB per socket | 70B, 2 sockets | derived | none given | capacity study section 2; earlier 70B comparison work (older tree), "Phase accounting" |
| 1.08 | KV per token, BF16 and INT8 (node) | 320 KiB and 160 KiB | 70B | derived | none given | capacity study section 2 table |
| 1.09 | Context limit | checkpoint maximum 131072; engine default min(model, 4096); beyond the maximum the engine refuses | 70B 3.1 / 3.3 checkpoint | derived | none given | capacity study section 3 |
| 1.10 | Context at the per-socket budget edge, FP32 KV | 69k tokens | 70B Q8_0, 2 sockets, 56 GiB HBM budget per socket (112 GiB node), weights and work included | derived | none given | capacity study section 2 table |
| 1.11 | Same, 64 GiB per socket | 95k tokens | 70B Q8_0, 2 sockets, nominal 128 GiB node | derived | none given | capacity study section 2 table |
| 1.12 | Same, 119 GiB node (128 GB decimal) | 80k tokens | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2 table |
| 1.13 | Same, BF16 KV, 56 GiB per socket | 139k tokens (beyond 131072) | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2 table |
| 1.14 | Same, INT8 KV, 56 GiB per socket | 277k tokens | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2 table |
| 1.15 | Node total of weights, work and KV at 65,536 positions (FP32 KV) | 109.8 GiB (98% of 112) | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2, edge table |
| 1.16 | Same at 98,304 positions | 129.8 GiB (116%) | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2, edge table |
| 1.17 | Same at 131,072 positions | 149.8 GiB (134%) | 70B Q8_0, 2 sockets | derived | none given | capacity study section 2, edge table |
| 1.18 | KV per socket at 8192 / 32768 / 65536 / 98304 / 131072 positions | 2.5 / 10 / 20 / 30 / 40 GiB | 70B FP32 KV, 2 sockets | derived | none given | capacity study section 5 table |
| 1.19 | Work buffers per socket | about 0.5 GiB at a 256-row batch | 70B, 2 sockets | estimated (first job measures it) | none given | capacity study section 2 |
| 1.20 | AMX-INT8 packed copy per socket | 31.9 GiB; with 34.4 GiB of weights, 66 GiB, over the budget at any context | 70B Q8_0, TP_PACK_INT8 | derived | none given | capacity study section 2 |
| 1.21 | Read per token at 131072 positions, per socket | 40 GiB of KV against 34 GiB of weights (decode is KV-dominated from about 64k) | 70B FP32 KV, 2 sockets | derived | none given | capacity study section 2, consequences |
| 1.22 | Weight-first hybrid planner charge at a 98,304 reservation | 80 weight / 57 KV layers in HBM; 55.84 GiB charged per HBM node | 70B VNNI, 2 sockets | measured | none given (pilot); 5904621 | earlier 70B comparison work (older tree), "Original pilot results" and "Fixed active history" |
| 1.23 | Strict HBM at the same 98,304 reservation | rejected: socket 0 needs 65.57 GiB against the 56 GiB budget | 70B, strict HBM, 2 sockets | measured | none given (pilot) | earlier 70B comparison work (older tree), "AMX packs and reservation pilot" |
| 1.24 | Strict HBM at 16,384 reservation (admissible control) | 40.56 / 39.52 GiB charged per socket; 80 / 80 layers | 70B, 2 sockets | measured | 5904621 | earlier 70B comparison work (older tree), "Fixed active history" |
| 1.25 | KV-heavy plan at 98,304 reservation | 60 / 79 layers; 55.62 GiB charged per socket | 70B, 2 sockets | measured | 5904621 | earlier 70B comparison work (older tree), "Fixed active history" |
| 1.26 | Real warm-up of 73,728 positions, weight-first plan | 45.06 GiB active KV of 60 GiB reserved; 55.87 GiB charged per socket; packed weights in DDR | 70B, 98,304 reservation | measured | 5904744 | earlier 70B comparison work (older tree), "Real workload beyond the practical HBM budget" |
| 1.27 | Peak RSS, AMX pilot (2113-token prompt, 8192 reservation) | 136.2 GiB | 70B AMX hybrid | measured | none given (pilot) | earlier 70B comparison work (older tree), "AMX packs and reservation pilot" |
| 1.28 | Peak RSS, VNNI auto (98,304 reservation, 8192 warm) | 77.5 GiB | 70B VNNI hybrid | measured | none given (pilot) | earlier 70B comparison work (older tree) |
| 1.29 | Peak RSS, real 73,728-position run | 181.30 GiB | 70B, weight-first, split and grouped | measured | 5904744; 5904763 | earlier 70B comparison work (older tree) |
| 1.30 | Peak RSS, primed synthetic 130,944 positions | about 150 GiB | 70B VNNI, 131,072 reservation | measured | 5904765 | earlier 70B comparison work (older tree) |
| 1.31 | Packed AMX matrices held in DDR | about 63.75 GiB | 70B AMX hybrid | measured | none given (pilot) | earlier 70B comparison work (older tree), "AMX packs and reservation pilot" |
| 1.32 | Active KV at the end, VNNI auto pilot | 5.08 GiB active of 60.00 GiB reserved | 70B, 98,304 reservation | measured | none given (pilot) | earlier 70B comparison work (older tree) |
| 1.33 | Conversion RAM peak (FP32 in torch) | about 300 GB; 1 TB node or 515 GB standard node | 70B Q8_0 conversion | estimated | none given | capacity study section 6 |
| 1.34 | Conversion time | 1-2 h CPU-only, about 3 node-hours | 70B Q8_0 conversion | estimated | none given | capacity study sections 6 and 7 |
| 1.35 | Mirror snapshot size | 141.1 GB (30 safetensors shards) | 70B Instruct, mirror and official repositories | measured | none given | capacity study section 6 |
| 1.36 | Model store free space / scratch free space | 96 GB total with 55 GB free / 1.7 PB free | cluster storage | measured | none given | capacity study section 6 |
| 1.37 | Node pool | 2 Flat HBM nodes and 12 Cache HBM nodes, about 1 TB DDR each | cluster | estimated (plan input) | none given | capacity study section 6 |
| 1.38 | Node-hours budget of the planned study | about 40 node-hours, about 15 jobs | 70B capacity study | estimated | none given | capacity study sections 6 and 7 |
| 1.39 | Real warm-up to 131072 positions | about 3 h | 70B VNNI | estimated | none given | capacity study section 3 (R5) |
| 1.40 | Real warm-up of 73,728 positions, duration | 1,970.65 s | 70B, weight-first, real | measured | 5904744 | earlier 70B comparison work (older tree) |
| 1.41 | Checkpoint load time at start | 106 s first start; about 33-36 s on later starts | 70B Q8_0 | measured | 5904620 | earlier 70B comparison work (older tree), "Matched 70B replication" |
| 1.42 | Load time, planned | 2-5 min per process start | 70B from scratch | estimated | none given | capacity study section 6 |
| 1.43 | 405B reference (not run) | Q8_0 file 431 GB; FP32 1623 GB; about 1 tok/s DDR-only | 405B | derived / modelled | none given | capacity study sections 1, 2 and 7 |

### 1.2 Decode and TTFT measured on the older tree (two sockets unless stated)

| ID | Quantity | Value | Configuration | Label | Job | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 1.44 | Decode, HBM only | 10.91 tok/s [10.87-10.96] | 70B Q8_0 group 64, VNNI, FP32 KV, reservation 8192, warm 512, 35-token prompt, 128 generated, older tree 1bc9189 | measured | 5892715 | earlier 70B comparison work (older tree), "Original pilot results" |
| 1.45 | Decode, hybrid 40 weight and 40 KV layers in HBM | 8.18 tok/s [8.16-8.19] | same | measured | 5892715 | same |
| 1.46 | Decode, DDR only | 6.25 tok/s [6.24-6.26] | same | measured | 5892715 | same |
| 1.47 | Read rate HBM / DDR, HBM only | 805.4 / 0.4 GB/s; 74.38 / 0.04 GB per step | same | measured | 5892715 | same |
| 1.48 | Read rate HBM / DDR, hybrid | 309.0 / 298.3 GB/s; 37.94 / 36.62 GB per step | same | measured | 5892715 | same |
| 1.49 | Read rate HBM / DDR, DDR only | 0.2 / 464.4 GB/s; 0.03 / 74.52 GB per step | same | measured | 5892715 | same |
| 1.50 | Total read per step, three placements | 74.42, 74.56 and 74.55 GB | same | measured | 5892715 | same |
| 1.51 | Serial timing model, hybrid | 7.99 tok/s (measured 8.18, residual 2.4%) | same | modelled | 5892715 | same |
| 1.52 | Isolated step times (HBM plus DDR) | 46.9 ms and 78.7 ms; perfect-overlap bound 12.70 tok/s | same | derived | 5892715 | same |
| 1.53 | TTFT, 35-token prompt, HBM only (two repeats) | 0.83 s and 15.31 s | same; the source calls these pairs unstable and confounded | measured | 5892715 | same |
| 1.54 | TTFT, 35-token prompt, hybrid (two repeats) | 13.28 s and 0.82 s | same | measured | 5892715 | same |
| 1.55 | TTFT, 35-token prompt, DDR only (two repeats) | 9.86 s and 0.83 s | same | measured | 5892715 | same |
| 1.56 | Replicate, decode HBM only | 10.90 tok/s [10.84-10.93] | 70B Q8_0 group 64, VNNI, pre-port revision (job 5904620), node hbm14, three repeats in rotated order | measured | 5904620 | earlier 70B comparison work (older tree), "Matched 70B replication" |
| 1.57 | Replicate, TTFT HBM only | 0.816 s [0.812-0.837] | same | measured | 5904620 | same |
| 1.58 | Replicate, decode hybrid 40/40 | 8.14 tok/s [8.09-8.14] | same | measured | 5904620 | same |
| 1.59 | Replicate, TTFT hybrid | 0.828 s [0.819-0.842] | same | measured | 5904620 | same |
| 1.60 | Replicate, decode DDR only | 6.23 tok/s [6.22-6.24] | same | measured | 5904620 | same |
| 1.61 | Replicate, TTFT DDR only | 0.829 s [0.828-0.835] | same | measured | 5904620 | same |
| 1.62 | Replicate, read rates HBM / DDR | HBM only 808.5 / 0.4 GB/s; hybrid 306.4 / 297.9 GB/s; DDR only 0.2 / 463.3 GB/s | same | measured | 5904620 | same |
| 1.63 | Replicate, write rates HBM / DDR | HBM only 0.52 / 0.15 GB/s; hybrid 0.44 / 0.14 GB/s; DDR only 0.20 / 0.59 GB/s | same | measured | 5904620 | same |
| 1.64 | Replicate, energy per step (package plus DRAM), gross / idle-adjusted | HBM only 68.1 / 36.8 J; hybrid 98.5 / 56.6 J; DDR only 133.9 / 79.3 J | same | measured (power), derived (per step) | 5904620 | same |
| 1.65 | Replicate, timing-only serial estimate, hybrid | 7.96 tok/s (within 2.2% of the 8.14 median) | same | modelled | 5904620 | same |
| 1.66 | Replicate, weights of 40 layers in DDR, KV in HBM | 8.14 tok/s; 36.42 GB DDR per step (single run) | same | measured | 5904620 | same |
| 1.67 | Replicate, KV of 40 layers in DDR, weights in HBM | 10.88 tok/s; 0.24 GB DDR per step (single run) | same | measured | 5904620 | same |
| 1.68 | Replicate, active unhalted cores during decode | 111.9-112.0 across both sockets | same | measured | 5904620 | same |
| 1.69 | Replicate, core clocks sockets 0 / 1 | hybrid 1.79 / 1.87 GHz; HBM only 1.67 / 1.75 GHz; DDR only 1.69 / 1.78 GHz | same | measured | 5904620 | same |
| 1.70 | Replicate, receive data flits per step, both sockets | 9.0-9.2 million | same | measured | 5904620 | same |
| 1.71 | Profiled decode time share, hybrid, short context | W1/W3 49%; W2 27% (including its fused reduction); attention 2.5% | same | measured | 5904727 | same, "Matched 70B replication" |
| 1.72 | Independent 8B check, predicted vs observed at 8, 16, 24 HBM layers | predicted 58.52 / 66.84 / 77.91 tok/s; observed 60.89 / 70.67 / 81.18 tok/s (residuals 4.0-5.7%) | 8B Instruct, same kernel family | modelled vs measured | 5892711 | earlier 70B comparison work (older tree), "Original pilot results" |
| 1.73 | AMX hybrid, 8192 reservation, 2113-token prompt | TTFT 23.26 s; decode 10.41 tok/s; prefill 91.1 tok/s (single run); KV 5.00 GiB reserved, 1.37 GiB active | 70B AMX, weights and KV in HBM, packs in DDR | measured | none given (pilot) | earlier 70B comparison work (older tree), "AMX packs and reservation pilot" |
| 1.74 | AMX DDR, same prompt | TTFT 21.64 s; decode 6.07 tok/s; prefill 98.0 tok/s (single run) | 70B AMX, all DDR | measured | none given (pilot) | same |
| 1.75 | VNNI auto, 98,304 reservation, 8192 warm, 35-token prompt | TTFT 1.06 s; decode 9.37 tok/s; 80 / 57 layers | 70B VNNI, 2 sockets | measured | none given (pilot) | same |
| 1.76 | AMX, weights and KV in HBM, packs in DDR (three repeats, 2113-token prompt) | prefill 86.50 tok/s [86.05-108.46]; TTFT 24.60 s [19.55-24.62]; decode 10.42 tok/s [10.40-10.46] | 70B AMX, second Flat node | measured | 5904622 | earlier 70B comparison work (older tree), "AMX packs and reservation pilot" |
| 1.77 | AMX, all data and packs in DDR (three repeats) | prefill 84.43 tok/s [80.86-99.56]; TTFT 25.09 s [21.30-26.20]; decode 6.06 tok/s [6.04-6.07] | 70B AMX, second Flat node | measured | 5904622 | same |
| 1.78 | Decode ratio, AMX HBM-resident vs DDR | 1.72 (1.71 in the pilot) | 70B AMX | derived | 5904622 | same |
| 1.79 | Weight-first vs KV-heavy decode at 8192 active positions, 98,304 reservation | 9.36 vs 8.36 tok/s; weight-first 12.0% faster | 70B VNNI, 2 sockets | measured (ratio derived) | 5904621 | earlier 70B comparison work (older tree), "Fixed active history" |
| 1.80 | Strict HBM control at 8192 active positions | 9.49 tok/s; TTFT 1.073 s; reads 79.41 / 0.04 GB per step | 70B, 16,384 reservation | measured | 5904621 | same |
| 1.81 | Weight-first at 8192 active positions | 9.36 tok/s; TTFT 1.046 s; reads 78.05 / 1.60 GB per step | 70B, 98,304 reservation, 80 / 57 | measured | 5904621 | same |
| 1.82 | KV-heavy at 8192 active positions | 8.36 tok/s; TTFT 1.055 s; reads 61.15 / 18.42 GB per step | 70B, 98,304 reservation, 60 / 79 | measured | 5904621 | same |
| 1.83 | Grouped attention gain at 8192 active positions | 9.75 vs 9.36 tok/s (+4.2%) | 70B, weight-first, real | measured | 5904761 | earlier 70B comparison work (older tree), "Controlled concurrency" and "Fixed active history" |

### 1.3 Deep-context decode (70B, weight-first and KV-heavy, split and grouped attention)

Measured rows with primed synthetic prefixes are marked "primed". Real warm-ups are marked "real". The "split" and "grouped" columns name the decode attention kernel.

| ID | Warm positions / reservation | Policy (HBM weight / KV layers) | Kernel | Decode tok/s | Other values | Label | Job | Source |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- |
| 1.84 | 73,728 real / 98,304 | weight-first, 80 / 57 | split | 4.80 (63 steps); 4.805 in the validation | HBM / DDR 109.57 / 13.85 GB per step; attention 116.3 of 208.1 ms per step (55.9%); TTFT 2.14 s; HBM 55.87 GiB per socket | measured (real) | 5904744; 5904763 | earlier 70B comparison work (older tree), "Real workload beyond the practical HBM budget" and "Real grouped-attention validation" |
| 1.85 | 73,728 real / 98,304 | weight-first, 80 / 57 | grouped | 5.720 | 174.81 ms per step; attention 84.84 ms; HBM / DDR 109.00 / 13.91 GB per step; gross energy 132.87 J per step (split 157.69 J); TTFT 2.12 s | measured (real) | 5904763 | same |
| 1.86 | 64,512 primed / 65,536 | strict HBM, 80 / 80 | split | 5.31 | active KV 39.43 GiB | measured (primed) | 5904760 | earlier 70B comparison work (older tree), "Validated capacity frontier" |
| 1.87 | 73,728 primed / 98,304 | weight-first, 80 / 57 | split | 4.89 (source prints range 4.892-4.893, which does not contain the median) | active KV 45.06 GiB | measured (primed) | 5904760 | same |
| 1.88 | 73,728 primed / 98,304 | KV-heavy, 60 / 79 | split | 4.68 [4.679-4.681] | active KV 45.06 GiB | measured (primed) | 5904760 | same |
| 1.89 | 73,728 primed / 98,304 | DDR only, 0 / 0 | split | 3.53 | | measured (primed) | 5904760 | same |
| 1.90 | 73,728 primed / 98,304 | exact-backend AMX validation, weight-first | split | 4.865 | 1.3% above the real 4.805 on the other node; controller reads 109.56 / 13.86 GB per step | measured (primed) | 5904760 | same |
| 1.91 | 73,728 primed / 98,304 | weight-first, 80 / 57 | grouped | 5.789 and 5.797 (two runs) | +18.4% over split | measured (primed) | 5904761 | same, "Validated capacity frontier" |
| 1.92 | 73,728 primed / 98,304 | KV-heavy, 60 / 79 | grouped | 5.531 and 5.564 (two runs) | | measured (primed) | 5904761 | same |
| 1.93 | 98,176 primed / 98,304 | weight-first, 80 / 57 | split | 4.16 [4.160-4.162] | active KV 59.98 GiB | measured (primed) | 5904760 | same |
| 1.94 | 98,176 primed / 98,304 | weight-first, 80 / 57 | grouped | 4.99 | | measured (primed) | 5904765 | same, "Near the full 128K reservation" |
| 1.95 | 98,176 primed / 98,304 | KV-heavy, 60 / 79 | split | 4.00 [4.002-4.003] | | measured (primed) | 5904760 | same |
| 1.96 | 130,944 primed / 131,072 | weight-first, 80 / 43 | split | 3.42 | 55.96 GiB per socket; reads 122.55 / 39.58 GB per step; active KV 79.98 GiB | measured (primed) | 5904765 | same |
| 1.97 | 130,944 primed / 131,072 | weight-first, 80 / 43 | grouped | 4.13 | 55.96 GiB per socket; reads 122.00 / 39.56 GB per step | measured (primed) | 5904765 | same |
| 1.98 | 130,944 primed / 131,072 | KV-heavy, 60 / 59 | grouped | 4.02 | 55.50 GiB per socket; reads 120.95 / 40.57 GB per step | measured (primed) | 5904765 | same |
| 1.99 | 130,944 primed / 131,072 | DDR only, 0 / 0 | grouped | 2.95 | reads 0.07 / 161.53 GB per step | measured (primed) | 5904765 | same |
| 1.100 | 8,192 real / 8,192 | weight-first, VNNI | split | 9.36 (9.37 in the pilot) | | measured (real) | 5904621; pilot (none given) | earlier 70B comparison work (older tree) |
| 1.101 | 8,192 primed / 8,192 | weight-first, VNNI | split, with 1,024-token real primer | 9.22 (within 1.6% of 9.36 real) | HBM / DDR 78.06 / 1.61 GB per step (real 78.05 / 1.60) | measured (primed) | 5904757 | earlier 70B comparison work (older tree), "Startup pause and proxy rejection" |
| 1.102 | 8,192 primed / 8,192 | weight-first, VNNI | split, no primer | 3.97 (5904757); 3.93 (5904751) | the unprimed run had a 12.6 s near-idle interval | measured (primed) | 5904757; 5904751 | same |
| 1.103 | 8,192 primed (validation) | VNNI, weight-first | split | 9.398 (real 9.362 on the same node) | | measured (primed) | 5904760 | earlier 70B comparison work (older tree), "Validated capacity frontier" |
| 1.104 | 8,192 real / 8,192 | weight-first, VNNI | grouped | 9.75 | | measured (real) | 5904761 | same |

### 1.4 Model predictions for 70B (modelled, derived or estimated)

| ID | Quantity | Value | Configuration | Label | Source |
| --- | --- | --- | --- | --- | --- |
| 1.106 | Decode, all HBM, 8192 positions | 9.4 tok/s | 70B Q8_0, 2 sockets, FP32 KV | modelled | capacity study section 5 table |
| 1.107 | Decode, HBM-first (KV overflow to DDR), 8192 positions | 9.4 tok/s | same | modelled | same |
| 1.108 | Decode, weights in HBM and all KV in DDR, 8192 positions | 8.2 tok/s | same | modelled | same |
| 1.109 | Decode, all DDR, 8192 positions | 5.5 tok/s | same | modelled | same |
| 1.110 | Decode, all HBM, 32768 positions | 6.8 tok/s | same | modelled | same |
| 1.111 | Decode, HBM-first, 32768 positions | 6.8 tok/s | same | modelled | same |
| 1.112 | Decode, weights HBM and KV DDR, 32768 positions | 4.9 tok/s | same | modelled | same |
| 1.113 | Decode, all DDR, 32768 positions | 3.7 tok/s | same | modelled | same |
| 1.114 | Decode, all HBM, 65536 positions | 5.0 tok/s | same | modelled | same |
| 1.115 | Decode, HBM-first, 65536 positions | 5.0 tok/s | same | modelled | same |
| 1.116 | Decode, weights HBM and KV DDR, 65536 positions | 3.1 tok/s | same | modelled | same |
| 1.117 | Decode, all DDR, 65536 positions | 2.6 tok/s | same | modelled | same |
| 1.118 | Decode, all HBM, 98304 positions | does not fit | same | modelled | same |
| 1.119 | Decode, HBM-first, 98304 positions | 3.3 tok/s (70% of KV in HBM) | same | modelled | same |
| 1.120 | Decode, weights HBM and KV DDR, 98304 positions | 2.3 tok/s | same | modelled | same |
| 1.121 | Decode, all DDR, 98304 positions | 2.0 tok/s | same | modelled | same |
| 1.122 | Decode, all HBM, 131072 positions | does not fit | same | modelled | same |
| 1.123 | Decode, HBM-first, 131072 positions | 2.4 tok/s (53% of KV in HBM) | same | modelled | same |
| 1.124 | Decode, weights HBM and KV DDR, 131072 positions | 1.8 tok/s | same | modelled | same |
| 1.125 | Decode, all DDR, 131072 positions | 1.7 tok/s | same | modelled | same |
| 1.126 | Decode, one socket, DDR only, 4096 / 32768 / 131072 positions | 2.9 / 1.9 / 0.8 tok/s | 70B Q8_0, 1 socket | modelled | capacity study section 5 |
| 1.127 | Decode, one socket, Cache mode, 4096 / 32768 / 131072 positions | 1.3 / 1.0 / 0.6 tok/s (from the measured 98 GB/s Cache-mode rate of the roofline) | 70B Q8_0, 1 socket, 74 GB working set | modelled | capacity study section 5 |
| 1.128 | Decode, two sockets, Cache mode | about 0.8 x the HBM curve up to about 90k positions; about 1.2 tok/s at 131072 | 70B Q8_0, 2 sockets | modelled | capacity study section 5 |
| 1.129 | Decode, one socket, FP32 weights, DDR only (not run) | about 0.9 tok/s | 70B FP32, 2 sockets | modelled | capacity study section 1 |
| 1.130 | Per-socket decode at 70B, 1 socket Q8_0 | does not fit in HBM (69 GiB of weights alone) | 70B, 1 socket | derived | capacity study section 2 table |
| 1.131 | Planned tolerance, decode proxy check | 3% in decode rate, 5% in total controller reads per step | synthetic-versus-real check | estimated | earlier 70B comparison work (older tree), "Capacity-frontier measurement" |
| 1.132 | Planned stop rule | if H1 fails by more than 30% in S1, fix the model or the placement before 70B | capacity study section 6 | estimated | capacity study section 6 |
| 1.133 | Planned H1 agreement | measured curve within 15% of the modelled table for all placements | capacity study section 5 | estimated | capacity study section 5 |
| 1.134 | Planned node-hours per stage | S3-S8 six 70B jobs, 18 node-hours; S9-S10 three jobs, 9 node-hours; reruns 6 node-hours | 70B capacity study | estimated | capacity study section 6 |

### 1.5 Cross-engine, 70B (earlier 70B comparison work, older tree; both prompts are 547 and 2113 tokens, 128 greedy tokens, decode over 127 steps)

#### 1.5a FP32 on DDR

| ID | Configuration (engine, prompt) | TTFT s | Decode tok/s | Label | Job | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1.135 | llama3.x.c, 547 tokens | 13.625 [13.624-13.638] | 1.766 [1.765-1.766] | measured | 5910133 | earlier 70B comparison work (older tree), "FP32 on DDR" |
| 1.136 | vLLM 0.34.0+cpu, 547 tokens | 10.018 [10.014-10.020] | 1.680 [1.680-1.681] | measured | 5910177 | same |
| 1.137 | llama.cpp, 547 tokens | 22.704 [22.702-24.684] | 1.027 [1.014-1.036] | measured | 5912354 | same |
| 1.138 | llama3.x.c, 2113 tokens | 52.996 [52.995-52.998] | 1.757 [1.757-1.757] | measured | 5910133 | same |
| 1.139 | vLLM, 2113 tokens | 37.782 [37.774-37.784] | 1.671 [1.670-1.671] | measured | 5910177 | same |
| 1.140 | llama.cpp, 2113 tokens | 88.630 [88.559-88.707] | 1.032 [1.029-1.046] | measured | 5912354 | same |
| 1.141 | Resident memory (all DDR), llama3.x.c / vLLM (two workers) / llama.cpp | 271,904 MiB / 291,576 MiB / 274,465 MiB | | measured | 5910133; 5910177; 5912354 | same |
| 1.142 | Ratio, llama3.x.c decode vs vLLM and vs llama.cpp | 1.051-1.052x (5.1-5.2% faster) and 1.70-1.72x | | derived | 5910133; 5910177; 5912354 | same |

#### 1.5b Q8_0 with identical quantized weights (stale)

Status: stale for both engines. The llama3.x.c 70B Q8_0 file was run on the portable group-32 kernels (the gs32 merges of 24 Nov, see section 2, rows 2.37 to 2.39, give AMX prefill and VNNI decode). The gs32 document asks that these prefill runs be rerun on the current tree.

| ID | Memory, engine, prompt | TTFT s | Decode tok/s | Label | Job | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| 1.143 | DDR, llama3.x.c, 547 | 84.236 [84.222-84.753] | 5.068 [5.064-5.069] | measured; stale (portable group-32 kernels) | 5910211 | earlier 70B comparison work (older tree), "Q8_0 with identical quantized weights" |
| 1.144 | DDR, llama.cpp, 547 | 5.928 [5.820-6.069] | 3.100 [3.066-3.131] | measured | 5910211 | same |
| 1.145 | DDR, llama3.x.c, 2113 | 326.208 [326.052-327.975] | 4.489 [4.471-4.494] | measured; stale (portable group-32 kernels) | 5910211 | same |
| 1.146 | DDR, llama.cpp, 2113 | 23.716 [23.319-23.793] | 3.037 [3.033-3.068] | measured | 5910211 | same |
| 1.147 | HBM, llama3.x.c, 547 | 84.484 [84.484-84.499] | 5.343 [5.341-5.346] | measured; stale (portable group-32 kernels) | 5910211 | same |
| 1.148 | HBM, llama.cpp, 547 | 17.068 [16.129-17.354] | 2.704 [2.683-2.733] | measured | 5913407 (repeat of the cell; job 5910211 reached its time limit in it) | same |
| 1.149 | HBM, llama3.x.c, 2113 | 326.997 [326.994-327.051] | 5.290 [5.289-5.291] | measured; stale (portable group-32 kernels) | 5910211 | same |
| 1.150 | HBM, llama.cpp, 2113 | 67.208 [65.885-68.249] | 2.639 [2.626-2.672] | measured | 5913407 | same |
| 1.151 | Derived, llama3.x.c HBM over DDR decode gain | +5.4% at 547 tokens; +17.8% at 2113 tokens | | derived | 5910211 | same |
| 1.152 | Derived, llama.cpp HBM over DDR decode | -12.8% to -13.1%; TTFT 2.83-2.88x | | derived | 5913407 | same |
| 1.153 | Resident memory, Q8_0 DDR, llama3.x.c / llama.cpp | 78,475 MiB / 85,109 MiB | | measured | 5913407 | same |
| 1.154 | Resident memory, Q8_0 HBM, llama3.x.c / llama.cpp | 78,462 MiB / 85,106 MiB (under 13 MiB on DDR in either) | | measured | 5913407 | same |

---

## 2. Llama 3.1 8B: HBM, DDR and Cache mode

### 2.1 Cross-engine, Flat HBM, Flat DDR and Cache (xengine; tree a3402b6; Llama 3.1 8B Instruct; W8A8 for llama3.x.c and vLLM, Q8_0 for llama.cpp)

Decode D and decode at depth 8192 (L) in tok/s; TTFT for P (2048 tokens) and L (8192 tokens) in seconds. Each cell: median [min-max] of three repeats. Jobs: HBM 5892201, DDR 5892202, Cache 5892196.

| ID | Engine | Tier | Sockets | Decode D tok/s | Decode L tok/s | TTFT P s | TTFT L s | Label | Job | Source | Status |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.01 | llama3.x.c | HBM | 1 | 51.9 [51.8-52.6] | 37.2 [37.2-37.5] | 2.258 [2.214-2.266] | 14.75 [14.72-14.78] | measured | 5892201 | xengine headline and section 1 | decode current; TTFT stale (before prefill-gap fixes) |
| 2.02 | llama3.x.c | HBM | 2 | 99.1 [97.2-100.2] | 66.4 [65.6-66.4] | 1.366 [1.330-1.412] | 8.10 [8.06-8.13] | measured | 5892201 | same | decode current; TTFT stale |
| 2.03 | llama3.x.c | DDR | 1 | 29.8 [29.7-29.8] | 22.2 [22.1-22.3] | 2.526 [2.513-2.563] | 15.91 [15.81-15.92] | measured | 5892202 | same | decode current; TTFT stale |
| 2.04 | llama3.x.c | DDR | 2 | 55.9 [55.8-56.0] | 40.4 [40.2-40.4] | 1.494 [1.481-1.512] | 8.80 [8.77-8.80] | measured | 5892202 | same | decode current; TTFT stale |
| 2.05 | llama3.x.c | Cache | 1 | 46.3 [45.7-46.6] | 32.5 [32.5-32.5] | 2.523 [2.509-2.980] | 16.20 [16.15-16.63] | measured; LLAMA_PLACEMENT=ddr, no binding | 5892196 | same | TTFT stale; Cache node has no counters or anchor |
| 2.06 | llama3.x.c | Cache | 2 | 79.2 [79.1-79.7] | 55.6 [53.9-56.8] | 1.590 [1.568-1.756] | 8.75 [8.53-8.83] | measured | 5892196 | same | TTFT stale |
| 2.07 | llama.cpp | HBM | 1 | 40.6 [40.5-40.9] | 29.1 [29.0-29.3] | 4.562 [4.560-4.602] | 29.82 [29.81-29.91] | measured | 5892201 | same | current (engine unchanged) |
| 2.08 | llama.cpp | HBM | 2 | 23.9 [23.8-23.9] | 20.1 [20.1-20.6] | 3.814 [3.795-3.843] | 21.45 [21.38-21.78] | measured | 5892201 | same | current |
| 2.09 | llama.cpp | DDR | 1 | 24.9 [24.9-25.1] | 20.0 [19.7-20.8] | 4.599 [4.579-4.615] | 29.93 [29.42-30.87] | measured | 5892202 | same | current |
| 2.10 | llama.cpp | DDR | 2 | 29.7 [29.6-30.0] | 22.6 [22.6-24.1] | 3.189 [3.162-3.194] | 17.92 [17.91-18.20] | measured | 5892202 | same | current |
| 2.11 | llama.cpp | Cache | 1 | 32.8 [32.8-32.9] | 24.7 [24.7-25.3] | 4.632 [4.626-4.656] | 29.79 [29.69-30.24] | measured | 5892196 | same | current |
| 2.12 | llama.cpp | Cache | 2 | 19.3 [16.7-19.7] | 15.9 [14.2-16.6] | 5.707 [5.339-5.736] | 31.36 [29.32-32.66] | measured; untuned layout (112 threads, no interleave) | 5892196 | same | current, but not a Cache-mode result |
| 2.13 | vLLM, patched worker mapping ranks to HBM | HBM | 1 | 40.5 [40.3-40.6] | 35.3 [35.2-35.3] | 0.618 [0.611-0.633] | 2.73 [2.72-2.74] | measured | 5892201 | same | current; patched, not stock |
| 2.14 | vLLM, patched worker mapping ranks to HBM | HBM | 2 | 61.2 [60.5-61.4] | 51.8 [51.6-52.0] | 0.369 [0.367-0.369] | 1.38 [1.38-1.39] | measured | 5892201 | same | current; patched |
| 2.15 | vLLM stock + numactl --membind (control) | HBM | 1 | 24.7 [24.7-24.8] | not run | not run | not run | measured; 99.9% of reads from DDR | 5892201 | same | control only |
| 2.16 | vLLM, stock | DDR | 1 | 24.9 [24.8-25.0] | 22.2 [22.1-22.2] | 0.740 [0.730-0.746] | 3.13 [3.11-3.13] | measured | 5892202 | same | current |
| 2.17 | vLLM, stock | DDR | 2 | 41.2 [41.1-41.5] | 36.7 [36.6-36.7] | 0.456 [0.453-0.460] | 1.68 [1.67-1.68] | measured | 5892202 | same | current |
| 2.18 | vLLM, stock | Cache | 1 | 34.1 [33.9-34.2] | 30.3 [30.0-30.4] | 0.668 [0.664-0.683] | 2.90 [2.90-2.92] | measured | 5892196 | same | current |
| 2.19 | vLLM, stock | Cache | 2 | 50.2 [49.1-50.3] | 43.8 [42.2-43.8] | 0.399 [0.393-0.402] | 1.47 [1.46-1.47] | measured | 5892196 | same | current |

### 2.2 llama3.x.c development history on Flat HBM, Q8_0 group 64 unless stated (prefill-gap, gs64, gs32)

| ID | Configuration | Sockets | Decode D tok/s | TTFT P s | TTFT L s | Label | Job | Source | Status |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 2.20 | develop 684489b, before fixes 1-9 | 1 | 51.18 [50.98-51.24] | 2.294 [2.281-2.529] | 14.821 [14.784-14.874] | measured | 5892439 | prefill-gap section 1 | superseded (prefill fixes) |
| 2.21 | develop 684489b, before fixes | 2 | 97.71 [97.27-97.90] | 1.385 [1.347-1.420] | 8.157 [8.131-8.182] | measured | 5892439 | prefill-gap section 1 | superseded |
| 2.22 | final gs64 defaults, tree 23cac73 | 1 | 51.12 [50.93-51.17] | 1.312 [1.278-1.454] | 6.668 [6.636-7.289] | measured | 5892439 | prefill-gap section 1; status section 2 | superseded by 2.26 |
| 2.23 | final gs64 defaults | 2 | 96.58 [96.45-97.09] | 0.798 [0.775-1.068] | 3.960 [3.875-4.140] | measured | 5892439 | prefill-gap section 1 | superseded by 2.27 |
| 2.24 | per-channel format 4 (opt-in; fails the HF gate) | 1 | 57.42 [57.37-57.54] | 0.908 [0.904-0.911] | 5.065 [5.058-5.069] | measured | 5892439 | prefill-gap section 1 and 7 | opt-in, not a default |
| 2.25 | per-channel format 4 (opt-in; fails the HF gate) | 2 | 108.38 [108.27-108.69] | 0.586 [0.583-0.601] | 3.145 [3.140-3.157] | measured | 5892439 | prefill-gap section 1 and 7 | opt-in, not a default |
| 2.26 | gs64, row block outer (current default), tree 9c292f6 | 1 | not measured in this job | 1.170 [1.141-1.199] | 6.079 [6.060-6.197] | measured (15 interleaved repeats) | 5910223 | gs64 sections 3.2 and 4 | current |
| 2.27 | gs64, row block outer (current default) | 2 | not measured in this job | 0.690 [0.669-0.702] | 3.494 [3.467-3.563] | measured (15 interleaved repeats) | 5910223 | gs64 section 3.2 | current |
| 2.28 | gs64 AMX reference for the gs32 A/B, tree 4d13eb9 | 1 | 51.07 (decode in job 5913494; see 2.30) | 1.186 | 6.124 | measured (median of 5 interleaved repeats) | 5913457 | gs32 section 0 table | current |
| 2.29 | gs64 AMX reference for the gs32 A/B | 2 | 97.23 (see 2.31) | 0.703 | 3.558 | measured | 5913457 | gs32 section 0 table | current |
| 2.30 | gs64 VNNI decode reference | 1 | 51.07 [50.92-51.19] | not given | not given | measured | 5913494 | gs32 section 6 | current |
| 2.31 | gs64 VNNI decode reference | 2 | 97.23 [96.75-98.13] | not given | not given | measured | 5913494 | gs32 section 6 | current |
| 2.32 | Interleaved reduction repeats, 1S (develop; barrier; interleaved) | 1 | 51.28; 51.21; 51.13 (five repeats) | not given | not given | measured | 5898023 | prefill-gap section 10 | current |
| 2.33 | Interleaved reduction repeats, 2S (develop; barrier; interleaved) | 2 | 98.5; 97.2; 97.9 | not given | not given | measured | 5898023 | prefill-gap section 10 | current |
| 2.34 | Two-socket prefill, develop, two modes | 2 | not given | fast mode 0.781 s (11 of 15); slow mode 0.843 s (4 of 15); median 0.801 s | 3.929 s (job 5898029 develop) and 4.038 s (job 5898041 develop) | measured | 5898029; 5898041 | prefill-gap section 10 | the slow mode is a bimodality (bimodality.md) |
| 2.35 | Interleaved reduction, paired P 2S change against develop | 2 | not given | -30.0 ms paired median (30 pairs, faster in 19 of 30) | L: -161 to -175 ms | measured | 5898029; 5898041 | prefill-gap section 10 | current |
| 2.36 | gs32 file, portable kernels (before the gs32 merges) | 1 | 24.57 [24.49-24.83] | 60.067 | not given | measured | 5913457 (TTFT); 5913494 (decode) | gs32 sections 0 and 6 | stale: portable kernels |
| 2.37 | gs32 file, portable kernels | 2 | 47.79 [47.64-47.90] | 30.273 | not given | measured | 5913457; 5913494 | gs32 sections 0 and 6 | stale: portable kernels |
| 2.38 | gs32 file, AMX-INT8 half-shape tiles (24 Nov) | 1 | see 2.39 | 1.763 | 8.570 | measured | 5913457 | gs32 section 0 table | current |
| 2.39 | gs32 file, AMX-INT8 half-shape tiles (24 Nov) | 2 | see 2.40 | 0.988 | 4.686 | measured | 5913457 | gs32 section 0 table | current |
| 2.40 | gs32 file, VNNI pair decode GEMV (24 Nov) | 1 | 48.04 [47.97-48.14] | not given | not given | measured | 5913494 | gs32 section 6 | current |
| 2.41 | gs32 file, VNNI pair decode GEMV (24 Nov) | 2 | 91.16 [91.04-91.94] | not given | not given | measured | 5913494 | gs32 section 6 | current |
| 2.42 | Derived, gs32 over gs64, AMX prefill ratio | 1 and 2 | | 1.49 (P 1S); 1.41 (P 2S); 1.40 (L 1S); 1.32 (L 2S) | | derived | 5913457 | gs32 section 0 table | current |
| 2.43 | Derived, gs32 VNNI decode over gs64 | 1 and 2 | 0.94 of gs64 at both | | | derived | 5913494 | gs32 section 6 | current |

### 2.3 Per-fix TTFT, P 1S (prefill-gap section 2; each fix measured on its own tree)

| ID | Change | TTFT P 1S before s | TTFT P 1S after s | Change s | Job | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2.44 | Fixes 2, 3, 6 (balanced GEMM partition; parallel K/V store; embedding and RoPE on the team) | 2.294 | 1.971 | -0.32 | 5892371 | default |
| 2.45 | Fix 1 (pipelined AMX-INT8 kernel) | 1.966 | 1.778 | -0.19 | 5892377 | default |
| 2.46 | Fix 5 (RMSNorm fused with the quantizer) | 1.846 | 1.772 | -0.07 | 5892393 | default |
| 2.47 | Fix 4 (AVX-512 exponential in SwiGLU; gated) | 1.957 | 1.891 | -0.07 | 5892386 | default |
| 2.48 | Fix 7 (AMX-BF16 flash attention; gated; first version 5892387 gave -0.17) | 1.697 | 1.366 | -0.33 | 5892399 | default for W8A8 |
| 2.49 | Fix 8 (K-blocked GEMM, K block 2048) | 1.688 | 1.659 | -0.03 | 5892400 | default |
| 2.50 | Fix 8 (M = 1024 instead of 256) | 1.822 | 1.675 | -0.15 | 5892402 | default |
| 2.51 | Fix 9 (per-channel format, opt-in) | 1.299 | 0.905 | -0.39 | 5892421 | opt-in |
| 2.52 | All fixes 1-8 (five repeats) | 2.294 | 1.312 | -0.98 | 5892439 | default |
| 2.53 | All fixes 1-8 (three repeats, the other job) | 2.292 | 1.474 | | 5892420 | superseded by 5892439 |

### 2.4 Hugging Face and vLLM-equivalent accuracy (rms against FP32 logits; gate rms <= 0.2, argmax >= 0.9)

| ID | Configuration | Path | rms 1S-2048 | rms 1S-8192 | rms 2S-2048 | rms 2S-8192 | Gate | Label | Job |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2.54 | develop 684489b | batched | 0.131 | 0.117 | 0.133 | 0.110 | pass | measured | 5892460 |
| 2.55 | final gs64 (08d934c) | batched | 0.137 | 0.119 | 0.138 | 0.116 | pass | measured | 5892459 |
| 2.56 | final gs64 (23cac73) | per token | 0.128 | 0.111 | 0.129 | 0.113 | pass | measured | 5892418 |
| 2.57 | per-channel RTN (format 4) | batched | 0.394 | 0.361 | 0.356 | 0.342 | fail | measured | 5892459 |
| 2.58 | vLLM's scheme (RedHatAI GPTQ, per-token int8 inputs), fake-quant on the HF model | per model | 0.395 | | | | fail | modelled | 5892436 |
| 2.59 | Same weights, FP32 inputs | per model | 0.146 | | | | pass | modelled | 5892436 |

### 2.5 numa-tp (17 Nov, tree 8f3114f, HBM placement; stale for prefill because the prefill fixes came later)

TTFT in seconds, 3 repeats, medians. Job ids 5845031 and 5845032 are given together by the source; the per-row job is not given.

| ID | Rung (M) | TTFT P 1S | TTFT P 2S | TTFT L 1S | TTFT L 2S | Label | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2.60 | FP32 per token (M = 1) | 121.309 [121.143-121.358] | 67.073 [67.053-67.133] | 534.907 [534.402-535.775] | 294.650 [294.109-294.741] | measured | stale: pre-prefill-gap |
| 2.61 | AVX-512 FP32 GEMM (M 128 on 1S, 64 on 2S) | 10.504 [10.500-10.506] | 5.392 [5.386-5.397] | 55.623 [55.600-55.635] | 27.245 [27.239-27.259] | measured | stale |
| 2.62 | AMX-BF16 (M 256) | 2.944 [2.943-2.946] | 1.611 [1.591-1.612] | 25.097 [25.095-25.103] | 11.771 [11.764-11.772] | measured | stale |
| 2.63 | Q8 per token (M = 1) | 41.166 [41.042-41.493] | 23.625 [23.607-23.805] | 219.950 [219.875-220.300] | 128.336 [128.287-128.395] | measured | stale |
| 2.64 | AVX-512 Q8 (M 256) | 9.395 [9.341-9.410] | 4.848 [4.831-4.866] | 51.868 [51.833-51.933] | 24.516 [24.494-24.625] | measured | stale |
| 2.65 | AMX-INT8 (M 256) | 2.823 [2.799-2.829] | 1.623 [1.614-1.635] | 24.443 [24.386-24.536] | 11.756 [11.695-11.821] | measured | stale |
| 2.66 | Prefill tok/s, AMX-INT8 | 753.90 | 1332.16 | 338.96 | 710.18 | measured | stale |
| 2.67 | Prefill tok/s, AMX-BF16 | 722.12 | 1343.64 | 330.04 | 709.34 | measured | stale |

Decode for the same tree. Rows 2.68 to 2.73 are retained contexts of 512 and 8192 tokens with 128 timed tokens. Rows 2.74 to 2.77 are after a 2048-token prompt and a 32-token decode.

| ID | Precision, context | Decode tok/s 1S | Decode tok/s 2S | Label | Status |
| --- | --- | --- | --- | --- | --- |
| 2.68 | FP32, 512 | 16.45 [16.44-16.47] | 30.57 [30.50-30.67] | measured | stale |
| 2.69 | FP32, 8192 | 13.31 [13.30-13.46] | 25.29 [25.21-25.43] | measured | stale |
| 2.70 | W8A8 (int8), 512 | 49.41 [49.01-49.47] | 90.95 [90.72-91.50] | measured | stale |
| 2.71 | W8A8 (int8), 8192 | 28.83 [28.82-29.01] | 50.58 [50.40-51.22] | measured | stale |
| 2.72 | W8A32, 512 | 43.62 [42.91-44.31] | 81.27 [81.15-81.44] | measured | stale |
| 2.73 | W8A32, 8192 | 30.64 [30.40-30.70] | 47.26 [45.99-47.52] | measured | stale |
| 2.74 | After 2048-token prompt, AMX-INT8 | 41.79 [41.07-42.41] | 83.31 [80.95-83.67] | measured | stale |
| 2.75 | After 2048-token prompt, per-token Q8 | 44.00 [40.99-44.04] | 85.80 [84.88-86.15] | measured | stale |
| 2.76 | After 2048-token prompt, AMX-BF16 | 15.71 [15.62-15.79] | 30.74 [30.67-30.91] | measured | stale |
| 2.77 | After 2048-token prompt, FP32 per token | 15.81 [15.76-15.81] | 31.05 [30.98-31.11] | measured | stale |

### 2.6 Older 8B references (roofline document, early Phase 2 engine)

| ID | Configuration | Value | Label | Job | Source | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 2.78 | 8B FP32, Cache mode baseline, sequential prefill, 32 threads | 2.75 tok/s | measured | none given | roofline section 2 | stale (scalar kernel) |
| 2.79 | llama.cpp 8B FP32, 32 threads, same harness | 4.84 tok/s | measured | none given | roofline section 2 | stale |
| 2.80 | 8B FP32 scalar engine, 56 threads, DDR and HBM | 6.9 and 7.5 tok/s (same speed on both; order as the source gives it) | measured | none given | roofline section 6 | stale |
| 2.81 | 8B FP32, AVX-512 kernels, 56 threads, HBM | 14.9 tok/s (447 GB/s) | measured | none given | roofline section 6 | stale |
| 2.82 | 8B FP32, AVX-512 kernels, 56 threads, DDR | 8.3 tok/s | measured | none given | roofline section 6 | stale |
| 2.83 | 8B Q8_0, after prefetch tuning, HBM | 47.9 tok/s (382 GB/s) | measured | none given | roofline section 6 | stale (superseded by 2.39 and 2.40) |
| 2.84 | 8B FP32 expected on one socket of HBM after the AVX-512 kernel | about 14 tok/s | modelled | none given | roofline section 5.1 | superseded by 2.81 |

### 2.7 8B bytes, KV and capacity (file sizes measured; the rest derived)

| ID | Quantity | Value | Configuration | Label | Source |
| --- | --- | --- | --- | --- | --- |
| 2.85 | Q8 file, group 64 (int8-v3) | 8,532,934,912 B (8.53 GB) | 8B | measured | prefill-gap section 7 table |
| 2.86 | Q8_0 GGUF, llama.cpp | 8.54 GB | 8B | measured | xengine headline |
| 2.87 | Q8 file, group 32 (repacked GGUF) | 9,034,809,600 B | 8B | measured | gs32 section 5 |
| 2.88 | Per-channel format 4 | 8,037,591,296 B | 8B | measured | prefill-gap section 7 table |
| 2.89 | Weight bytes per decode token, nominal | 7.97 GB (7,974,764,544 B: 32 layers 7.416 GB plus LM head 0.558 GB) | 8B Q8_0 | derived | xengine section 2 |
| 2.90 | Weight bytes per decode token, counter (llama3.x.c) | 7.72 (HBM 1S), 7.86 (DDR 1S), 7.46 (HBM 2S), 7.71 (DDR 2S) GB per token | 8B, decode window | measured | xengine section 2 table (jobs 5892201, 5892202) |
| 2.91 | Counter bandwidth, llama3.x.c | HBM 1S 400 GB/s (73% of roofline); DDR 1S 232 GB/s (89%); HBM 2S 354 + 353 GB/s per socket (707 total, 64%); DDR 2S 213 + 213 GB/s (426 total, 81%) | 8B | measured | xengine section 2 table |
| 2.92 | KV per token, FP32 | 256 KiB across the node; 128 KiB per socket | 8B | derived | capacity study section 2 table |
| 2.93 | KV per token, BF16 and INT8 (node) | 128 KiB and 64 KiB | 8B | derived | capacity study section 2 table |
| 2.94 | BF16 K/V copy under LLAMA_ATTN=amx | 128 KiB per token on 1S; 64 KiB per socket on 2S | 8B | derived | prefill-gap section 6 |
| 2.95 | HBM weights, Q8 | 7.4 GiB node; 3.7 GiB per socket | 8B | derived | capacity study section 2 table |
| 2.96 | HBM weights, FP32 on one socket | 28.0 GiB | 8B FP32 | derived | capacity study section 2 table |
| 2.97 | Context at budget edge, FP32 KV, FP32 weights on 1S, 56 GiB | 113k tokens | 8B | derived | capacity study section 2 table |
| 2.98 | Context at budget edge, Q8 weights on 2S, FP32 KV | 430k (56 GiB per socket); 500k (64 GiB); 460k (119 GiB node) | 8B | derived | capacity study section 2 table |
| 2.99 | Decode energy, llama3.x.c, 1S HBM / 1S DDR | 10.96 / 21.03 J per token (net 4.82 / 10.54) | 8B, package plus DRAM, both sockets counted | measured (power), derived (energy) | xengine section 6 (job 5892201, 5892202) |
| 2.100 | Decode energy, llama3.x.c, 2S HBM / 2S DDR | 8.02 / 15.22 J per token (net 4.73 / 9.60) | 8B | measured (power), derived (energy) | xengine section 6 |
| 2.101 | Cache-node energy | not measured (the Cache node has no RAPL counters) | 8B Cache | not measured | xengine section 6 |

### 2.8 Derived cross-engine ratios (xengine, same tree as 2.01 to 2.19)

| ID | Ratio | Value | Label | Source |
| --- | --- | --- | --- | --- |
| 2.102 | HBM over DDR, decode D, 1S: llama3.x.c / llama.cpp / vLLM patched over stock | 1.74 / 1.63 / 1.63 | derived | xengine section 7 |
| 2.103 | HBM over DDR, decode D, 2S: llama3.x.c / llama.cpp / vLLM patched over stock | 1.77 / 0.81 / 1.48 | derived | xengine section 7 |
| 2.104 | HBM over DDR, prefill P, 1S: llama3.x.c / llama.cpp / vLLM | 1.12 / 1.01 / 1.20 | derived | xengine section 7 |
| 2.105 | Cache over DDR, decode D, 1S / 2S (llama3.x.c) | 1.55 / 1.42 | derived | xengine section 7 |
| 2.106 | Cache over HBM, decode D, 1S / 2S (llama3.x.c) | 0.89 / 0.80 | derived | xengine section 7; capacity study section 5 |
| 2.107 | Share of HBM decode gain kept in Cache mode, llama3.x.c 1S / 2S | 75% / 54% | derived | xengine section 7 |
| 2.108 | Two-socket decode gain, HBM, llama3.x.c / llama.cpp / vLLM patched | 1.91 / 0.59 / 1.51 | derived | xengine section 4 |
| 2.109 | Two-socket decode gain, DDR, llama3.x.c / llama.cpp / vLLM | 1.88 / 1.19 / 1.65 | derived | xengine section 4 |
| 2.110 | Node-to-node anchor, llama3.x.c 1S HBM decode, node A over node B | 1.004 (prefill 86 tokens 0.997) | measured (ratio derived) | xengine sections 7 and 8 (jobs 5892201, 5892202) |

---

## 3. Machine ceilings and model calibration

| ID | Resource | Value | Conditions | Label | Source |
| --- | --- | --- | --- | --- | --- |
| 3.01 | Local HBM read, AVX-512, one socket | 16.5 (1 thread), 223 (14), 387 (28), 552 (56) GB/s | Flat, per socket | measured | roofline section 1 |
| 3.02 | Local HBM copy, non-temporal stores | 20.5 (1), 269 (14), 446 (28), 616 (56) GB/s | Flat, per socket | measured | roofline section 1 |
| 3.03 | Local DDR read | 17.0 (1), 213 (14), 268 (28), 262 (56) GB/s | Flat, per socket | measured | roofline section 1 |
| 3.04 | Remote HBM read, socket 0 cores to socket 1 HBM | 112 (14), 111 (28), 50 (56) GB/s | Flat | measured | roofline section 1 |
| 3.05 | Remote DDR read | 124 (14), 124 (28), 121 (56) GB/s | Flat | measured | roofline section 1 |
| 3.06 | Both sockets, local HBM, concurrently | 548 + 557, about 1.1 TB/s per node (the two-socket roofline uses 1104) | Flat | measured | roofline section 1 and section 3 |
| 3.07 | Cache mode, read set 8 GB / 96 GB | 560 GB/s / 98 GB/s | Cache mode, same harness | measured | roofline section 1 |
| 3.08 | oneDNN GEMV, K = 4096, N = 14336, M = 1, local HBM | BF16 327 GB/s; INT8 358 GB/s | Flat | measured | roofline section 1 |
| 3.09 | oneDNN GEMV, M = 1, local DDR | BF16 229 GB/s; INT8 236 GB/s | Flat | measured | roofline section 1 |
| 3.10 | oneDNN GEMV over a 17 GB pool, Cache mode | 277 GB/s | Cache mode | measured | roofline section 1 |
| 3.11 | Compute peaks at 56 cores | AVX-512 FP32 FMA 8.8 TFLOP/s; AVX-512 BF16 (vdpbf16ps) 4.2 TFLOP/s; AVX-512 VNNI INT8 35.9 TOP/s; AMX BF16 112 TFLOP/s; AMX INT8 225 TOP/s | one socket | measured | roofline section 1 |
| 3.12 | oneDNN prefill shape, M = 256 | BF16 27 TFLOP/s; INT8 69 TOP/s | local HBM | measured | roofline section 1 |
| 3.13 | AMX tile throughput per core (16 x 16 x 64 tile every 16 cycles) | 1024 MACs per AMX-busy cycle | hardware assumption | estimated | xengine section 5 |
| 3.14 | Ideal AMX-busy cycles per prefilled token, 8B, 2048 prompt | 6.82 million (6.979e9 MACs over 1024 per cycle) | 8B | derived | xengine section 5 |
| 3.15 | Reference clock used to convert cycles | 1.9 GHz (assumed); 2.0 GHz nominal (prototype) | | estimated | xengine section 5; gs64 section 1 |
| 3.16 | Decode roofline, 8B Q8_0 at 7.97 GB per token | 1 socket DDR 33; 1 socket HBM 69; practical HBM (440 GB/s) 55; 2 sockets HBM 138 tok/s | | modelled | roofline section 3 |
| 3.17 | Decode roofline, 8B FP32 at 30.02 GB per token | 1 socket DDR 8.7; 1 socket HBM 18.3; practical HBM 14.7; 2 sockets HBM 36.6 tok/s | | modelled | roofline section 3 |
| 3.18 | Decode roofline, 70B at 73.85 GB per token | not stated in the source | | n/a | capacity study section 5 (per socket 36.9 GB) |
| 3.19 | Prefill ceilings per socket, 8B | AVX-512 FP32 370; AMX BF16 1800; AMX INT8 4600 tok/s | upper bounds, ignore attention and overheads | modelled | roofline section 4; numa-tp section 6 |
| 3.20 | Model rate, HBM weight stream | 397 GB/s per socket (from 99.1 tok/s x 7.97 GB / 2) | calibration on 8B W8A8 2S | derived | capacity study section 5 |
| 3.21 | Model rate, HBM attention | 200 GB/s (measured 186-225 at 8192 and 32768 positions) | 8B 2S | measured (rate), derived | capacity study section 5 |
| 3.22 | Model rate, DDR weight stream | 238 GB/s (from 29.8 tok/s x 7.97 GB) | 8B 1S DDR | derived | capacity study section 5 |
| 3.23 | Model rate, DDR attention | 95 GB/s (200 x 262 / 552) | assumed | estimated | capacity study section 5 |
| 3.24 | Model check, 8B W8A8 2S predictions at 512 / 8192 / 32768 positions | 96 / 65 / 32 tok/s modelled; measured 89.6-92.4 / 50.6-58.9 / 29.3-34.1 (split and group kernels) | model is 5-10% optimistic | modelled, measured | capacity study section 5 (measured values as quoted there) |
| 3.25 | Relative decode speed vs fraction f of bytes from DDR, roofline ratio 552/262 | f = 0.1, 0.25, 0.5, 0.75, 1: 0.90, 0.78, 0.64, 0.55, 0.47 | serial reads | modelled | capacity study section 5 |
| 3.26 | Same, engine-calibrated ratio 397/238 | f = 0.1, 0.25, 0.5, 0.75, 1: 0.94, 0.86, 0.75, 0.67, 0.60 | serial reads | modelled | capacity study section 5 |
| 3.27 | Counter-measured 8B HBM and DDR streams, both sockets, raw | HBM 1,227 GB/s; DDR 534 GB/s (4 GiB streams) | 8B job, two sockets | measured | earlier 70B comparison work (older tree), "Original pilot results" (job 5892711) |
| 3.28 | Concurrency control, aggregate, serial then concurrent | serial 308.7 [308.3-309.4] GB/s; concurrent 490.7 [469.1-499.1] GB/s | two groups of 28 cores on one socket, HBM and DDR | measured | earlier 70B comparison work (older tree), "Controlled concurrency" (job 5904726) |
| 3.29 | Concurrency control, per-group rates during own interval | serial HBM 365.2 / DDR 268.3; concurrent 298.7 / 245.3 GB/s; common-time fraction 0.80 | same | measured | same |
| 3.30 | Sub-token controller bins | HBM-dominant median 811 GB/s; DDR-dominant median 483 GB/s; Pearson -0.986 | 70B hybrid, 10 ms bins | measured | earlier 70B comparison work (older tree), "Sub-token controller activity" (job 5904742) |
| 3.31 | Node-to-node anchor | 8B decode 1.004; prefill 0.997 | node A (5892201) over node B (5892202) | measured (ratio) | xengine section 8 |
| 3.32 | HBM CAS unit | 32 B per count (calibrated); DDR 64 B | counters | measured | xengine section 3 |
| 3.33 | UPI ceiling | not measured in the listed sources | | n/a | |
| 3.34 | UPI observed, llama.cpp remote traffic (8B, 2S HBM) | 106 GB/s total (58.7 and 47.5 GB/s received); 4.44 GB per token | interleaved over two nodes | measured | xengine section 4 |
| 3.35 | UPI observed, remote HBM roofline quoted against the above | about 50 GB/s, against 53 GB/s average per socket | | measured | xengine section 4; roofline section 1 |
| 3.36 | UPI per token, llama3.x.c 2S HBM / DDR | 0.015 / 0.016 GB per token | 8B | measured | xengine section 4 |
| 3.37 | UPI per token, vLLM patched 2S HBM | 0.008 GB per token | 8B | measured | xengine section 4 |
| 3.38 | UPI per prefill, develop, P minus S (diagnosis) | 2.85 and 4.07 GB per direction; payload 2.06 GB per direction | 8B, 2S, barrier reduction | measured; payload derived | prefill-gap section 10 |
| 3.39 | UPI per prefill, interleaved reduction, 2S | socket 0 to 1: 3.55 GB; socket 1 to 0: 4.95 GB (barrier 2.77 / 2.97; wo 3.46 / 3.64) | 8B, 2S | measured | prefill-gap section 10 (job 5898043) |
| 3.40 | UPI flit size | 6.93 B, calibrated (see 4.14) | | measured | prefill-gap section 10 (job 5892713) |
| 3.41 | Package power per active socket, prefill and decode, all engines and tiers | 339-366 W | 8B, all engines | measured | xengine section 6 |
| 3.42 | Idle baseline per socket, node A: package / DRAM | 140.1 / 133.5 W (package socket 0 / 1); DRAM 22.1 / 20.5 W | node A (5892201) | measured | xengine section 6 |
| 3.43 | Idle baseline per socket, node B: package / DRAM | 135.9 / 134.1 W; DRAM 21.1 / 19.1 W | node B (5892202) | measured | xengine section 6 |
| 3.44 | DRAM domain rise with DDR traffic | about 46 W per socket at 232 GB/s, about 0.2 W per GB/s, about 25 pJ per bit | 8B DDR decode | measured (power), derived (per bit) | xengine section 6 |
| 3.45 | HBM power visibility | HBM energy is inside the package value or not measured; not in the DRAM domain | | measured (inference) | xengine section 6 |
| 3.46 | HBM capacity per socket, nominal | 64 GB (2 sockets 128 GB decimal, 119 GiB) | Flat node | estimated (nominal) | capacity study section 1 |
| 3.47 | HBM budget per socket used by the engine | 56 GiB (LLAMA_HBM_BUDGET_GIB default); 60 GiB proposed for the HBM-only edge rows (accepted by the engine, not yet run) | 2 sockets | stated (engine setting) | capacity study sections 2 and 7 |
| 3.48 | Cores | 56 per socket (112 per node); 110 used by vLLM on 2 sockets; 64 by llama.cpp; 55 per socket for the 70B three-engine runs | Flat | measured | xengine section 9; earlier 70B comparison work (older tree), "Setup" |

---

## 4. Values reported in two sources with different values

Rows 4.01 to 4.14 list the same quantity with two or more values. Differences come from different trees, jobs or definitions, as the note states. Nothing here picks a winner.

| ID | Quantity | Value A (source, job, label) | Value B (source, job, label) | Note |
| --- | --- | --- | --- | --- |
| 4.01 | 8B, Flat HBM, 1S decode | 51.9 tok/s (xengine, 5892201, measured, tree a3402b6) | 51.18 tok/s (prefill-gap develop 684489b, 5892439, measured); status file repeats 51.2 | Same tier and workload label, different tree and job. Both are measured. |
| 4.02 | 8B, Flat HBM, 2S decode | 99.1 tok/s (xengine, 5892201, measured) | 97.71 tok/s (prefill-gap develop, 5892439, measured) | Different tree and job. |
| 4.03 | 8B, Flat HBM, P TTFT 1S | 2.258 s (xengine, 5892201, measured) | 2.294 s (prefill-gap develop, 5892439, measured); 2.823 s (numa-tp AMX-INT8, 5845031 / 5845032, measured) | Three trees. The numa-tp value is the AMX-INT8 rung. |
| 4.04 | 8B, Flat HBM, P TTFT 2S | 1.366 s (xengine, 5892201, measured) | 1.385 s (prefill-gap develop, 5892439); 1.623 s (numa-tp AMX-INT8, 5845031 / 5845032) | Three trees. |
| 4.05 | 8B, Flat HBM, L TTFT 1S | 14.75 s (xengine, 5892201, measured) | 14.821 s (prefill-gap develop, 5892439); 24.443 s (numa-tp AMX-INT8, 5845031 / 5845032) | The numa-tp value is about 66% higher; no cause is given in the sources. |
| 4.06 | 8B, Flat HBM, L TTFT 2S | 8.10 s (xengine, 5892201, measured) | 8.157 s (prefill-gap develop, 5892439); 11.756 s (numa-tp AMX-INT8) | Same as 4.05. |
| 4.07 | gs64 final, P TTFT 1S | 1.312 s (prefill-gap, 5892439, measured) | 1.170 s (gs64, 5910223, measured); 1.186 s (gs32 A/B reference, 5913457, measured) | Same default described in three documents, different trees. |
| 4.08 | gs64 final, L TTFT 1S | 6.668 s (prefill-gap, 5892439) | 6.079 s (gs64, 5910223); 6.124 s (gs32 reference, 5913457) | As 4.07. |
| 4.09 | gs64 final, P TTFT 2S | 0.798 s (prefill-gap, 5892439) | 0.690 s (gs64, 5910223); 0.703 s (gs32 reference, 5913457) | As 4.07. |
| 4.10 | gs64 final, L TTFT 2S | 3.960 s (prefill-gap, 5892439) | 3.494 s (gs64, 5910223); 3.558 s (gs32 reference, 5913457) | As 4.07. |
| 4.11 | Decode 1S and 2S, gs64 | 51.12 and 96.58 tok/s (prefill-gap, 5892439) | 51.07 and 97.23 tok/s (gs32 section 6, 5913494); 51.13 and 97.9 tok/s (interleaved reduction, 5898023) | Repeats on different jobs. Differences are within the run-to-run ranges given. |
| 4.12 | 70B HBM only decode, and hybrid and DDR only | 10.91 / 8.18 / 6.25 tok/s (older tree pilot, 5892715, measured) | 10.90 / 8.14 / 6.23 tok/s (older tree replicate, 5904620, measured) | Same configuration, two jobs. Also TTFT 0.83 s (pilot, 35-token prompt) and 0.816 s (replicate). |
| 4.13 | 70B HBM only decode at depth | 10.91 tok/s at about 611 live positions (5892715, measured) | 9.4 tok/s at 8192 positions (capacity study, modelled) | Different depth, so not a conflict. Listed because both are the HBM-only decode of 70B. |
| 4.14 | UPI flit size | 6.93 B, calibrated (prefill-gap section 10, job 5892713, measured) | 64/9 B = 7.11 B, a documented assumption (xengine section 4, estimated) | Different basis. xengine's UPI byte counts use the assumed value. |
| 4.15 | Read rate per socket, 8B, 2S HBM weight stream | 397 GB/s (capacity study section 5, derived from 99.1 tok/s) | 354 GB/s per socket (xengine section 2, counter, measured; 707 total) | Derived from decode versus counter-measured bandwidth. |
| 4.16 | Read rate per socket, 8B, 1S DDR weight stream | 238 GB/s (capacity study section 5, derived) | 232 GB/s (xengine section 2, counter, measured) | As 4.15. |
| 4.17 | Group accumulations per 8B token | 218 million (xengine section 5, derived for 32-element groups) | 109 million (gs32 section 5 correction, 6.979e9 / 64; the 8B file uses group 64) | The xengine number assumed group 32. The gs32 document gives the correction. |
| 4.18 | Roofline quoted in the capacity study | 550 GB/s per socket HBM (capacity study section 5) | 552 GB/s (roofline section 1, measured) | Rounding. |
| 4.19 | Exchanged payload per direction, two-socket prefill | 2.06 GB (prefill-gap section 10, diagnosis, derived) | 2.15 GB (prefill-gap section 10, interleaved-reduction results, measured) | Same section, two values. |

## Notes

- The 70B decode and TTFT figures above come from the older tree at its own kernels. They are not re-measured on develop. The develop prefill kernels differ (see section 2.2).
- The earlier 70B comparison work (older tree) has no P or L prompt. Its prompts are 35 or 547 tokens (pilot and warm-up, not a P or L workload), and 2113 tokens (the AMX campaign prompt). Its TTFT definition starts at request submission with token ids and stops at the first greedy token.
- The earlier 70B comparison work (older tree) defines decode as 127 divided by the time from the first to the last generated token. xengine defines decode and TTFT per engine (native timers; vLLM over local HTTP); see its section 1 and note b.
