# HBM capacity study: model plus KV cache at and beyond 128 GB of on-package HBM

Plan written 21 November 2026, before any run, from `develop` at 684489b. It is the reviewer-facing statement of what will
be measured, in what order and with which engine changes. The study runs after the prefill-optimisation task closes; every
number marked **modelled** or **derived** below is a prediction to be compared with a measurement, not a result. Machine
ceilings come from [`docs/hbm-roofline.md`](../hbm-roofline.md), engine reference points from
[`docs/xengine-results.md`](../xengine-results.md) and [`docs/decode-attention-results.md`](../decode-attention-results.md).
Results will go to `docs/hbm-capacity-results.md` with a validation companion, following the repository's pairs.

## 1. Question and scope

> What happens to decode speed, time to first token and energy when the model plus KV cache of a single stream grows
> from comfortably inside the node's 128 GB of HBM (2 sockets x 64 GB) to past it, and which placement of the overflow
> (KV, weights, or the hardware cache) degrades least?

| Axis | Values |
| --- | --- |
| Models | Llama 3.1 8B (cheap first step, emulated capacity), Llama 3.1 70B Instruct (the real edge); 405B only as an infeasible reference |
| Precision | Q8_0 W8A8 (group 64, 1.0625 B/weight); FP32 KV today, BF16/INT8 KV only if built (section 3) |
| Sockets | 1 and 2 (tensor parallel, each socket reads only its own shard) |
| Context | 4096 up to 131072 (the checkpoint limit); decode measured at a retained context |
| Placement | HBM only, HBM then DDR (KV spilled, weights spilled), DDR only, Cache mode |
| Out of scope | Batching and multi-request serving, other model families, accuracy beyond a sanity gate, SNC4 |

## 2. Capacity arithmetic

All figures **derived** from the architecture (`tools/export.py` and `src/runtime.c` layouts, `src/tp.c` budget). GiB = 2^30.
Q8_0 file = 1 B per weight + 4 B scale per 64; the untied input embedding (1.0 GiB at 70B) stays in DDR, so HBM holds the
linear layers plus the classifier. 2S budget: 56 GiB per socket by default (`LLAMA_HBM_BUDGET_GIB`), nominal 64.

| Model | Layers x KV heads x head_dim | Q8_0 file | HBM weights, 2S total / per socket | Bytes per decode token | FP32 file | KV per token FP32 / BF16 / INT8 | KV per token per socket (FP32) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8B | 32 x 8 x 128 | 8.5 GB | 7.4 GiB / 3.7 | 7.97 GB | 32.1 GB | 256 / 128 / 64 KiB | 128 KiB |
| 70B | 80 x 8 x 128 | 75.0 GB (69.8 GiB) | 68.8 GiB / 34.4 | 73.85 GB | 282 GB | 640 / 320 / 160 KiB | 320 KiB |
| 405B (reference) | 126 x 8 x 128 | 431 GB | 401 GiB / 200 | 429 GB | 1623 GB | 1008 / 504 / 252 KiB | 504 KiB |

BF16 weights are not an engine format (BF16 is a prefill-packing mode of an FP32 file), so 70B exists only as Q8_0 (FP32
282 GB cannot sit in HBM; it would be a DDR-only curiosity at about 0.9 tok/s on two sockets, not run). 405B Q8_0 is 3.6x
the node budget before any KV, and its FP32 conversion needs more than the node's RAM: reference only.

**Work buffers** (`work_bytes_for`, `tp_exchange_bytes`): about 0.5 GiB per socket at 70B with a 256-row batch (activations,
`n_heads/2 x context` scores, K/V staging, 32 MiB attention workspace); the first job measures it. **AMX-INT8 packed copy**
(`TP_PACK_INT8`): about 1 B per weight again, 31.9 GiB per socket at 70B, which with the 34.4 GiB of weights is **66 GiB, over
the budget at any context**. 70B therefore runs `LLAMA_KERNELS=avx512` (VNNI prefill, no packed copy; 8B 2S: 434 against
1332 tok/s with AMX, [numa-tp-prefill-results.md](../numa-tp-prefill-results.md)) unless the packed copy is placed in DDR (R4).
The KV budget counts the **full `--context-length` capacity** at load, filled or not.

Context at which weights + KV + work reach each limit (2S, per socket equals half the node figure):

| Model, KV dtype | 56 GiB per socket (default budget, 112 GiB node) | 64 GiB per socket (nominal, 128 GiB node) | 128 GB decimal node (119 GiB) |
| --- | ---: | ---: | ---: |
| 70B, FP32 KV | **69k tokens** | 95k | 80k |
| 70B, BF16 KV | 139k (beyond 131072) | 191k | 160k |
| 70B, INT8 KV | 277k | 382k | 320k |
| 8B Q8, FP32 KV, 2S | 430k (never) | 500k | 460k |
| 8B FP32 weights, FP32 KV, **1S** (28.0 GiB of HBM weights) | **113k** | 145k (fits at 131072) | 127k |
| 70B Q8 on **1S** | never fits (34.4 GiB x 2 = 69 GiB of weights alone) | never | never |

Model/context combinations around the edge (2S, Q8; node totals of weights + work + KV, against the 112 GiB budget):

| Position | Combination | Weights + work | KV | Total | vs 112 GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Far under | 70B, 8192, FP32 KV | 69.8 | 5.0 | 74.8 | 67% |
| Under | 70B, 32768, FP32 KV | 69.8 | 20.0 | 89.8 | 80% |
| **Just under** | 70B, 65536, FP32 KV; 70B, 131072, BF16 KV | 69.8 | 40.0 | 109.8 | 98% |
| **At** | 70B, 69k, FP32 KV | 69.8 | 43.2 | 113 | 100% |
| **Over** | 70B, 98304, FP32 KV | 69.8 | 60.0 | 129.8 | 116% |
| **Far over** | 70B, 131072, FP32 KV | 69.8 | 80.0 | 149.8 | 134% |
| Natural 1S case | 8B FP32 weights, 1S, 131072, FP32 KV | 28.5 | 32.0 | 60.5 | 108% of 56 |
| Far under | 8B Q8, 131072, FP32 KV | 8.4 | 32.0 | 40.4 | 36% |

Consequences. (1) With the checkpoint limit at 131072 tokens, **only 70B (and 8B FP32 on one socket) can cross the edge
by context alone**; 8B Q8 has to be pushed over by emulation (R2, a forced split of a chosen fraction of bytes to DDR).
(2) 70B decode is KV-dominated from about 64k: at 131072 a socket reads 40 GiB of KV per token against 34 GiB of weights.
(3) Contexts above 32768 are untested in the engine (decode campaigns stop there), so 8B at 131072 is the first step.

## 3. Engine behaviour today and the work it needs

From the source at 684489b; "verified" means a test or measurement in the repository exercises it.

| Question | Today | Verified |
| --- | --- | --- |
| Model families | Bias-free Llama with SwiGLU, `head_dim = dim / heads`; RoPE `default` or `llama3` (factor, low/high frequency factors, original 8192); 70B shape (8192, 80, 64/8, 28672, vocab 128256) passes the converter checks | 1B and 8B converted and run; 70B never converted |
| Max context | `--context-length` up to the checkpoint's `max_position_embeddings` (131072 for 3.1/3.3); default min(model, 4096); beyond it the engine refuses | decode to 32768 only |
| Converter | `tools/export.py --hf SNAPSHOT --version 3 --int8` loads the whole model as FP32 in torch (about 282 GB for 70B), then quantises; rejects pre-quantised checkpoints (so a published W8A8 is unusable) | 1B, 8B |
| Over budget, `hbm` placement | **Fatal at load**: `fail("socket N needs ... but LLAMA_HBM_BUDGET_GIB allows ...")` (`check_budget`, `budget_claim`). No fallback, no spill | parse error tested; **the over-budget failure itself has no test** |
| What the budget counts | Per socket: weight shards, packed copies, both KV caches at full capacity, work, exchange. Order on one socket: weights, KV, work | by reading |
| Raising the budget | `LLAMA_HBM_BUDGET_GIB=60` is accepted; regions are `mbind`-ed to the node, so a node that really runs out would OOM, not spill (expected from bind semantics) | no |
| `ddr` placement | Weights, KV, work and packed copies all bound to the socket's DDR node; no budget check | yes (A-track, xengine) |
| `file` placement | One socket: checkpoint mapping, unbound. Two sockets: shards copied, first-touched by each socket's team (lands in DDR) | yes |
| Mixed placement | **None.** One `mode` per run: weights, KV, packed and work share the node (`placement_node`) | by reading |
| KV dtype | **FP32 only** (`sizeof(float)` in `tp.c`, `runtime.c`); `LLAMA_ATTN_KV=bf16-shadow` was a planned prompt-only option, not present | by reading |
| KV layout | Head-major `[layer][kv_head][position][head_dim]` per socket, one region for K and one for V; pages appear on first write | yes |
| Cache mode | Run with `LLAMA_PLACEMENT=ddr`, no binding; HBM invisible to the engine | yes (8B, in-cache only) |

| ID | Work | Why | Need |
| --- | --- | --- | --- |
| R1 | Convert 70B (3.1 Instruct) to Q8_0 v3; check header, shard divisibility (hidden 14336 per socket, divisible by 64), RoPE metadata, and the HF-reference sanity gate (argmax and top-5 on short prompts; FP32 HF reference needs about 300 GB RAM) | Nothing else runs without it | **Required** |
| R2 | **KV tiering knob** `LLAMA_KV_PLACEMENT=hbm\|ddr\|auto` plus a fraction (layer-granular first: layers below L* in HBM, the rest in DDR, per-socket `mbind` on sub-ranges of the KV region); `auto` = HBM until the budget, then DDR, replacing the fatal error; report the split in `--json` and verify pages | Needed for every spill row; with a fraction it also emulates capacity at 8B | **Required** |
| R3 | **Weight tiering** `LLAMA_WEIGHTS_PLACEMENT` with a layer fraction (second weight region on the DDR node; the loader already keeps up to four segments) | The "weights spilled" rows | **Required** (can follow R2 if the KV rows come first) |
| R4 | Packed AMX-INT8 copy in DDR (or built on the fly per prefill batch) so 70B keeps AMX prefill | Prefill speed at 70B; not needed for decode | Optional |
| R5 | Synthetic KV fill (`--warm-context N` without model compute: deterministic data written to every position's pages) and a context ladder in one process | Real warm-up at 70B VNNI, about 40-50 tok/s plus quadratic attention, is roughly 3 h for 131072 (estimated), more than one job | **Required above 32768**, real fill kept at 2-3 points as a check |
| R6 | KV dtype BF16 (then INT8 with per-row scales) in `k_kv_store` and the decode/prefill attention kernels | Moves the edge 2x/4x (section 2) and halves attention bytes; an explicit precision mode, owner decision | Optional |
| R7 | Test for the over-budget failure and for each new placement path (virtual topologies, `tests/test_numa.py`) | Gate for R2-R3 | **Required** with R2-R3 |
| R8 | Per-run capture of CAS, UPI and RAPL (reuse `scripts/xengine.sbatch` events, HBM CAS 32 B per count, DDR 64 B) and `numastat` per tier | Section 4 metrics | **Required** (script only) |

Defaults never change: all knobs are opt-in, FP32 KV and the current fatal budget stay the default until a decision table says otherwise.

## 4. Experiment matrix

Decode: warm to the context (real fill for the gate points, synthetic after R5), 8 untimed then 64 timed tokens, `group`
attention is **not** assumed (default `split`; run both on the first 8B cell and keep the faster for the rest), 3 repeats,
medians with ranges. TTFT: prompts of 2048, 8192, 32768 (real) at each placement that is exercised. Greedy, exact lengths.

| Stage | Model | Sockets | Placement | Contexts | Purpose |
| --- | --- | --- | --- | --- | --- |
| S0 | 8B Q8 | 1, 2 | HBM | 4096, 32768, 65536, 131072 | Engine at 128K (untested), work buffer size, real HBM node size, budget behaviour (check the fatal message) |
| S1 | 8B Q8 | 2 | HBM weights, KV fraction f in DDR (0, .25, .5, .75, 1) | 32768, 131072 | **Emulated capacity**: measures decode against DDR byte fraction; validates the model of section 5 cheaply |
| S1b | 8B Q8 | 2 | weights fraction g in DDR (0, .25, .5, 1), KV in HBM | 8192 | Same for the weights side |
| S2 | 8B FP32 | 1 | HBM only to 100k; HBM-first above | 98304, 114688, 122880, 131072 | The one natural crossing at 8B (about 113k) |
| S3 | 70B Q8 | 2 | HBM only | 4096, 8192, 32768, 65536 | Under and just under; baseline curve |
| S4 | 70B Q8 | 2 | HBM-first then DDR KV (`auto`) | 65536, 73728, 81920, 98304, 131072 | **The edge**: KV fraction in DDR 0, .06, .16, .30, .47 |
| S5 | 70B Q8 | 2 | weights in HBM, all KV in DDR | 8192, 32768, 65536, 131072 | KV-spill bound |
| S6 | 70B Q8 | 2 | KV in HBM, a fraction g of weights in DDR (.1, .25, .5) | 8192, 32768 | Weights-spill sensitivity |
| S7 | 70B Q8 | 2 | DDR only | 4096, 32768, 131072 | Lower bound |
| S8 | 70B Q8 | 1 | DDR only | 4096, 32768, 65536 | One socket cannot hold it in HBM; DDR reference |
| S9 | 70B Q8 | 1, 2 | Cache mode, `ddr` placement, no binding | 4096, 32768, 65536, 98304, 131072 | Hardware cache across 64 GiB per socket: 1S is over the cache at every context, 2S crosses near 90k |
| S10 | 70B Q8 | 2 | HBM, AVX-512 vs AMX | 2048, 8192 prompt | Prefill TTFT (AMX only after R4) |

Metrics per run, from the engine report and `perf stat -a --per-socket -I 200` as in the cross-engine plan:

| Metric | Source | Notes |
| --- | --- | --- |
| Decode tok/s vs context, TTFT | engine report | median of 3, ranges |
| Bytes per token and tier split | `hbm_cas_rd` x 32 B, `ddr_cas_rd` x 64 B over the decode window; `numastat -p` for residency | measured DDR fraction f must match the placement; calibrate on S0 |
| Remote traffic | UPI rx/tx flits | expected near zero (shards are socket-local); a remote KV page would show here |
| Power, energy per token | RAPL package and DRAM, idle baseline subtracted | HBM is inside the package domain or unmeasured, not in DRAM ([xengine-results.md](../xengine-results.md) section 6); Cache nodes have no privileged counters, so timing only |
| Placement proof | `LLAMA_KV_VERIFY=1`, `tools/check_placement.py`, `LLAMA_PLACEMENT_STRICT` | per region, per socket |

## 5. Model to compare against (modelled)

Per socket, bytes per decode token = weights W (36.9 GB at 70B) + KV read C x kv_tok (320 KiB per token at 70B). Time =
W / B_w + KV / B_a, with each tier's effective rates **calibrated on measurements, not on peaks**: HBM weight stream 397 GB/s
(8B W8A8 2S, 99.1 tok/s against the 138 roofline), HBM attention 200 GB/s (8B 2S, 186-225 measured at 8192 and 32768), DDR
weight stream 238 GB/s (8B 1S DDR, 29.8 tok/s), DDR attention 200 x 262/552 = 95 GB/s (assumed). Check on 8B W8A8 2S: model
96 / 65 / 32 tok/s at 512 / 8192 / 32768 against measured 89.6-92.4 / 50.6-58.9 / 29.3-34.1 (`split`-`group`), so the model
is about 5-10% optimistic and the study reports the residual.

Relative decode speed against the fraction f of per-token bytes read from DDR (serial reads, every core streams both tiers):
`1 / ((1 - f) + f x B_hbm / B_ddr)`.

| f | 0 | .1 | .25 | .5 | .75 | 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Roofline ratio 552/262 | 1.00 | 0.90 | 0.78 | 0.64 | 0.55 | 0.47 |
| Engine-calibrated 397/238 | 1.00 | 0.94 | 0.86 | 0.75 | 0.67 | 0.60 |

Modelled 70B, Q8, 2S, FP32 KV, tok/s (HBM-first fills the KV positions that fit the 56 GiB budget, the rest go to DDR):

| Context | KV per socket | All HBM | HBM-first (KV overflow in DDR) | Weights HBM, all KV DDR | All DDR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 2.5 GiB | 9.4 | 9.4 | 8.2 | 5.5 |
| 32768 | 10 GiB | 6.8 | 6.8 | 4.9 | 3.7 |
| 65536 | 20 GiB | 5.0 | 5.0 | 3.1 | 2.6 |
| 98304 | 30 GiB | does not fit | 3.3 (70% of KV in HBM) | 2.3 | 2.0 |
| 131072 | 40 GiB | does not fit | 2.4 (53%) | 1.8 | 1.7 |

One socket, 70B, DDR only: 2.9 / 1.9 / 0.8 tok/s at 4096 / 32768 / 131072. Cache mode: in-cache 8B measured 0.89 (1S) and
0.80 (2S) of Flat HBM; once the per-socket working set exceeds the 64 GiB cache the roofline file measured 98 GB/s, which
gives 1S 70B (74 GB working set) 1.3 / 1.0 / 0.6 tok/s at 4096 / 32768 / 131072, and 2S about 0.8 x the HBM curve up to
roughly 90k, then falling towards 98 GB/s per socket (about 1.2 tok/s at 131072). Hypotheses to test: **H1** the
measured curve follows the table within 15% for all placements; **H2** KV spill costs less than weight spill per byte
because attention already runs at about 36% of HBM peak; **H3** HBM-first beats Cache mode at the edge because Cache mode's
conflict misses start below 64 GiB; **H4** a dedicated core subset streaming the spill concurrently could beat the serial
model (HBM and DDR controllers are independent) and is worth a prototype only if H1 holds.

## 6. Prerequisites and logistics

| Item | State and plan |
| --- | --- |
| 70B checkpoint on the cluster | **None.** The cluster holds only 8B and 1B Instruct snapshots (18 GB). Official `meta-llama/Llama-3.1-70B-Instruct` and `Llama-3.3-70B-Instruct` are **gated (manual approval)** and the cluster has no token |
| Ungated copies | `NousResearch/Meta-Llama-3.1-70B-Instruct` and `unsloth/Llama-3.3-70B-Instruct`: same 30 safetensors shards, same sizes (141.1 GB) as the official repositories; the official file hashes are hidden for gated repositories, so byte identity cannot be compared. Config of the 3.3 copy matches 70B (RoPE llama3 factor 8, 131072, 8 KV heads, untied). Download only `*.safetensors`, configs and tokenizer (141 GB; the Nous repository also holds a second 141 GB `original/` copy) |
| Published W8A8 (`RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8`, 72.7 GB, compressed-tensors) | **Not usable**: the converter refuses external quantisation and per-channel scales differ from Q8_0 groups. The C engine needs original weights |
| Disk | The model store reports 96 GB size with 55 GB free (a limited fileset; it cannot hold 141 GB). The scratch filesystem reports 1.7 PB free with no user quota; its purge policy is unknown. Plan: snapshot 141 GB and Q8 checkpoint 75 GB on scratch, copy the 75 GB checkpoint to the store only if it fits after cleanup |
| Conversion | Needs about 300 GB RAM at peak (FP32 in torch), so a 1 TB Flat or Cache node, or a 515 GB standard node; estimate 1-2 h CPU-only, one job; a streaming converter (per-tensor BF16 to Q8_0) is optional hardening |
| Nodes | 2 Flat HBM nodes and 12 Cache HBM nodes, about 1 TB DDR each; jobs run one at a time on the exclusive node, each under 3 h; everything built and tested inside jobs |
| Anchor | The two Flat nodes agreed to 1.004 on the 8B anchor; rerun one 8B cell on each at the start to cover node-to-node difference. The Cache node has no counters and no anchor, so Cache rows carry a node-to-node caveat as before |
| Load time | 75 GB read from the parallel filesystem plus shard copy: estimated 2-5 min per process start, so each (placement, sockets) setting is one process with a context ladder |

Estimated budget (node-hours, **estimated**): R1 conversion 3; S0-S2 8B, two jobs 6; S3-S8 70B on Flat nodes, six jobs 18;
S9-S10 Cache and prefill, three jobs 9; reruns and the anchor 6; total about **40 node-hours, about 15 jobs**, plus engine
work R2, R3, R5, R7, R8 (a few days) done and tested locally before the first job.

Order: R1 and the download (parallel to the engine work) -> R2, R7, R8 -> S0/S1/S2 (8B, validates model and tooling) ->
R3, R5 -> S3-S8 -> S9/S10. Stop rule: if H1 fails by more than 30% in S1, fix the model or the placement before 70B.

## 7. Open questions for the owner

| # | Question | Default if unanswered |
| --- | --- | --- |
| 1 | Which 70B: 3.1 Instruct (continuity with the 8B campaigns) or 3.3 Instruct; is an ungated mirror acceptable, and may 141 GB be downloaded to the scratch area (licence acceptance is the owner's) | 3.1 Instruct from the mirror, after approval |
| 2 | Is BF16 (then INT8) KV an acceptable explicit precision mode (R6)? It moves the 70B edge from 69k to beyond 131072 tokens | Not built; study FP32 KV only |
| 3 | Approve opt-in placement knobs R2/R3 in the engine (defaults unchanged, fatal budget stays unless `auto` is chosen) | Yes |
| 4 | 70B prefill without AMX (VNNI, no packed copy) acceptable, or build R4 for AMX at 70B | VNNI; R4 only if TTFT matters |
| 5 | Real 70B warm-up to 131072 (about 3 h, needs splitting) or synthetic KV fill after a few real checks (R5) | Synthetic, real at 8k/32k/65k |
| 6 | Budget cap: about 40 node-hours acceptable; may `LLAMA_HBM_BUDGET_GIB` be raised to 60 for the HBM-only edge rows | 40 h; 56 stays the default |
| 7 | Is "128 GB" decimal (119 GiB) or binary (128 GiB) for the headline edge; usable HBM per node is measured in S0 | Report both, headline on the measured usable size |
| 8 | Cache nodes have no privileged counters: accept timing-only Cache rows, or ask the cluster staff for counter access | Timing only |
| 9 | Include a 405B reference run (DDR only, about 1 tok/s) to show infeasibility | No; arithmetic only |
