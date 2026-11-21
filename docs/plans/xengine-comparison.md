# Cross-engine comparison: llama3.x.c vs llama.cpp vs vLLM (CPU) on Xeon Max

Plan written 21 November 2026, before any run, from `develop` at a3402b6. It is the reviewer-facing statement of what will
be measured and how. Numbers quoted here are the machine ceilings from [`docs/hbm-roofline.md`](../hbm-roofline.md) and the
engine reference points from [`docs/numa-tp-prefill-results.md`](../numa-tp-prefill-results.md); nothing below is a result
of this campaign. Results go to `docs/xengine-results.md` with a validation companion, every number labelled **measured**,
**derived** or **modelled** with job id and commit, following the repository's results/validation pairs.

## 1. Question and scope

> How much does HBM improve CPU LLM inference over DDR, and does an engine that places memory correctly, splits the model
> across both sockets and keeps HBM bandwidth saturated with AMX turn that advantage into throughput better than
> general-purpose engines do?

| Axis | Values |
| --- | --- |
| Model | Llama 3.1 8B Instruct, **one Hugging Face revision for every engine** (recorded in the environment file) |
| Precision | INT8 weights and AMX-INT8 prefill in every engine (table in 1.1); no FP32/BF16 weights, no accuracy study |
| Sockets | 1 (socket 0, 56 cores) and 2 (112 cores) |
| Memory | Flat HBM, Flat DDR, Cache mode (section 3) |
| Load | Single request, batch 1, greedy, exact lengths (section 4) |
| Out of scope | Other models, multi-request serving, SNC4, SGLang (possible later addition) |

### 1.1 Engines and pins

Pins are **tentative** until the setup jobs confirm that each builds or installs and runs; the confirmed values go into the
environment record and the results document.

| Engine | Version (tentative) | Weights | Prefill path | Decode path |
| --- | --- | --- | --- | --- |
| llama3.x.c | `develop` HEAD, commit recorded | `runq` Q8_0 (32-element groups, about 8.5 bit/weight), W8A8 | AMX-INT8 | AVX-512 kernels, engine-owned NUMA placement |
| llama.cpp | v1.0.0 | Q8_0 GGUF converted by its own tools from the same snapshot | AMX extra buffer (AMX-INT8) | AVX512-VNNI |
| vLLM CPU backend | v0.34.0 | Published RedHatAI W8A8 checkpoint of the same model: per-channel int8 weights, dynamic per-token int8 activations, bf16 LM head | SGL AMX kernel for M > 4 (AMX) | VNNI (M <= 4) |

Builds and installs happen inside jobs, never on a login node. If the published W8A8 checkpoint turns out not to be
derived from the same base revision, the fallback is to quantise with llm-compressor and record that.

## 2. Nodes and job split

Nodes are labelled only by role; no host names appear in the repository.

| Label | Mode | Contents | Why |
| --- | --- | --- | --- |
| node A | Flat | All engines, all weights, KV and buffers on **HBM** | Engine comparison at the HBM ceiling |
| node B | Flat | All engines, all weights, KV and buffers on **DDR** | Engine comparison at the DDR ceiling |
| anchor | A and B | llama3.x.c, 1 socket, decode-heavy, HBM, run on both | Measures node-to-node difference before any HBM-vs-DDR claim crosses nodes |
| Cache node | Cache | All engines, 1 and 2 sockets, three workloads, out of the box (no binding), timing-first | HBM as transparent cache |

Every engine-vs-engine comparison therefore happens on the same node. Each node job must finish in under 3 hours.

## 3. Memory configurations per engine

Placement is **proven, not assumed** (section 6). Whatever an engine does when asked is itself a reported finding.

| Engine | 1 socket | 2 sockets |
| --- | --- | --- |
| llama3.x.c | `LLAMA_SOCKETS=1`, `LLAMA_PLACEMENT=hbm\|ddr` (engine-owned, verified by the engine) | `LLAMA_SOCKETS=2`, same placement knob; tensor parallel, each socket reads only its own shard |
| llama.cpp | No `--numa` (its `distribute` mode puts threads on CPU-less HBM nodes in Flat mode); CPU masks; non-mmap load; `numactl --cpunodebind=0 --membind=<HBM0\|DDR0>` | `numactl --cpunodebind=0,1 --interleave=<HBM0,HBM1\|DDR0,DDR1>`. No CPU tensor parallelism: one process, interleaved weights, so about half of all reads cross UPI |
| vLLM | TP=1, `VLLM_CPU_OMP_THREADS_BIND` one range on socket 0. Stock binds the rank's memory to its CPUs' NUMA node, which is **DDR in Flat mode**; 1S-HBM uses `numactl --membind=<HBM0>` (stock mechanism) | TP=2, one thread range per socket (55 cores per rank, one spare core for the frontend: its documented setup). **2S-DDR** = stock TP=2 on node B. **2S-HBM** = a clearly labelled **patched** variant: a one-line change mapping each rank's memory node to its socket's HBM node |

Stock vLLM behaviour on Flat nodes (memory lands on DDR unless forced) is reported as a finding in its own right; the
patched variant is never presented as stock.

**Cache baseline (decision D1).** "Cache-mode baseline, only DDR" is ambiguous. The 8B INT8 model (about 8.5 GB) plus an
8192-token KV cache fits in the 64 GB per-socket HBM cache, so after warm-up Cache mode should behave close to HBM, not
DDR. Both are run and labelled **"Flat-DDR (DDR only)"** (node B) and **"Cache (HBM as transparent cache)"** (Cache node).

## 4. Workloads

All batch 1, greedy, exact generation lengths (EOS ignored in benchmark mode, or proven not hit).

| Id | Shape | Metrics | Mechanism per engine |
| --- | --- | --- | --- |
| P | 2048-token prompt, 1 generated token | TTFT, prefill tok/s | native phase timers in each engine |
| D | about 128-token prompt, 128 generated tokens | decode tok/s, TPOT | same |
| L | 128 tokens decoded at 8192 context depth | long-context decode tok/s, KV cost | C engine `--warm-context`; llama-bench `-d 8192`; vLLM: 8192-token prompt with output 128, minus the 8192-prompt output-1 run |

An 8192-token prefill run is dropped unless time allows. The L mechanisms are verified once in the setup jobs before the
campaign relies on them.

**Uniform definitions (all engines).** TTFT from the G = 1 run. TPOT = (T(G) - T(1)) / (G - 1), end to end. Native timers
are reported next to these and never mixed with them.

**Exact-length token sequences (decision D4).** Every engine is timed on token sequences of the exact required length.
Token identity may differ between engines (llama-bench uses random tokens) because dense-model cost depends only on
length, not on token values. A separate **output sanity check** dumps 32 greedy tokens from a fixed prompt on each
engine; text need not match bit for bit but garbage output disqualifies the configuration.

## 5. Fairness rules

| Rule | How it is applied |
| --- | --- |
| Best documented configuration per engine | llama.cpp threads, `--no-mmap`, batch/ubatch, flash attention, KV type; vLLM TP, thread binding, `VLLM_CPU_KVCACHE_SPACE`, chunked-prefill size, tcmalloc/iomp preload as its docs recommend. Every setting is written to the manifest |
| Small tuning pass only | llama.cpp 1S decode thread sweep (2-3 points around the core count); best point kept, sweep recorded and reported |
| Thread counts (D6) | llama3.x.c: its default per socket; llama.cpp: swept (1S), all cores of the cpuset (2S); vLLM: 55 cores per rank. Owner may overrule |
| Prefix caching | Off in vLLM and **proven off** (second identical prompt shows full prefill cost, TTFT not collapsing) |
| AMX proven | `AMX_OPS_RETIRED.INT8` and `EXE.AMX_BUSY` counted in each engine's prefill window; an engine that falls back to AVX-512 VNNI is reported as such |
| Placement proven | `numastat -p` and `/proc/PID/numa_maps` per run, plus the HBM vs DDR CAS split from the counters |
| Format differences recorded | Q8_0 32-group (about 8.5 bit) vs per-channel W8A8 with a bf16 LM head; KV FP32 (llama3.x.c), F16 (llama.cpp), BF16 (vLLM). Bytes read per token compared to nominal model bytes, so efficiency is comparable across formats |
| Same-node comparisons | Engines are compared only against each other on one node |
| Failed cells | Recorded and the job continues; never silently dropped |

## 6. Instrumentation

System-wide `perf stat -a --per-socket -I 200 -x,` time series per run, one wall-clock **marker file** per run (load
start, prefill start, first token, end) used to cut load, prefill and decode windows reproducibly. Counter groups are split
across repeats to avoid multiplexing; enabled/running percentages are checked and reported.

| Repeat | Content | Events |
| --- | --- | --- |
| 1 | Memory tiers, cross-socket, power | `uncore_imc` `ddr_cas_rd`, `uncore_type_14` `hbm_cas_rd` (and write events where available), UPI rx/tx flits, RAPL pkg and ram |
| 2 | Compute and AMX, power | cycles, ref-cycles, instructions, `AMX_OPS_RETIRED.INT8`, `EXE.AMX_BUSY`, RAPL pkg and ram |
| 3 | No perf | Timing only; compared with repeats 1 and 2 to show perf overhead is negligible |

- Idle baseline of about 10 s per node, subtracted for energy per token.
- GB/s per tier per socket = ddr_cas_rd x 64 B, hbm_cas_rd x 32 B (calibrated, job 5892172) / window. Bandwidth is normalised to the measured rooflines: local HBM read
  about 552 GB/s per socket, local DDR about 262-268 GB/s per socket, remote HBM about 50 GB/s, Cache mode about 560 GB/s
  for an in-cache working set (see [`docs/hbm-roofline.md`](../hbm-roofline.md)).
- Bytes read per decoded token (derived from counters and TPOT) against nominal model bytes.
- RAPL: check whether HBM power sits inside the package domain on this CPU and say what was found.
- **Cache node (decision D2):** privileged perf counters are not available there, so it is **timing-first**; whatever
  unprivileged counters work are used, and the missing ones are listed in the validation record.

## 7. Matrix

Flat nodes: 3 engines x {1S, 2S} x {HBM on A, DDR on B} x 3 workloads x 3 repeats, with the vLLM variants of section 3.
Cache node: 3 engines x {1S, 2S} x 3 workloads, out of the box, repeats as time allows.

| Node | llama3.x.c | llama.cpp | vLLM (server configs) |
| --- | --- | --- | --- |
| A (HBM) | 1S, 2S; P, D, L | 1S, 2S; P, D, L (+ 1S thread sweep) | 1S-HBM (`numactl`), 2S-HBM (patched) |
| B (DDR) | 1S, 2S; P, D, L (+ anchor) | 1S, 2S; P, D, L (+ 1S thread sweep) | 1S-DDR (stock), 2S-DDR (stock) |
| Cache | 1S, 2S | 1S, 2S | TP=1, TP=2, stock |

vLLM is run as a server per configuration: started **once**, then all workloads and repeats are issued against it, because
start-up dominates its cost.

## 8. Time budget (estimate, per node job, limit 3 h)

| Block | Estimate |
| --- | --- |
| llama3.x.c, all cells | about 20 min |
| llama.cpp, all cells (including the 1S thread sweep) | about 20 min |
| vLLM, 2 server configs x (start-up about 5 min + runs about 10 min) | about 30 min |
| Environment record, model staging, idle baseline, placement checks | about 20 min |
| Margin | rest of the 3 h |

If a job projects over budget, cut in this order: 8192-token prefill, repeats of L, the llama.cpp tuning sweep. The harness
prints a per-cell estimate in `DRY_RUN=1` mode before anything is submitted.

## 9. Data layout and deliverables

Raw data lives **outside the repository** in a per-campaign results directory (never committed; no host names or cluster
paths in the repository):

| Item | Content |
| --- | --- |
| `environment.txt` | lscpu, `numactl -H`, memory-mode evidence, kernel, compilers, engine commits and versions, Python freeze |
| `manifest.json` | every cell: engine, configuration, environment, command line |
| per run | stdout and stderr, engine report, perf CSV, marker file, placement evidence |
| `runs.csv` | one row per run and phase, built by a script (never edited by hand) |

| Repo deliverable | Notes |
| --- | --- |
| `docs/plans/xengine-comparison.md` | this plan, written before the runs |
| `scripts/xengine.sbatch`, helpers in `scripts/lib/` | separate from the existing single-engine job; reuses its Flat-mode check, environment record, staging, `perf_attach`, `record_result`, `DRY_RUN=1` |
| `tools/xengine_summary.py` | builds `runs.csv` and the tables; named to avoid `tools/compare_engines.py`, which compares logits between configurations of this engine |
| `docs/xengine-results.md`, `docs/xengine-results-validation.md` | results and validation record: job ids, commits, labels measured/derived/modelled |

Working rules follow [`E-followups/agent-brief.md`](E-followups/agent-brief.md) section 0: branch
`feature/xengine-comparison`, conventional commits, builds and runs only in jobs, no host names, credentials or absolute
cluster paths in the repository, defaults of the C engine unchanged.

## 10. Questions the results document must answer

| # | Question | Evidence |
| --- | --- | --- |
| 1 | Prefill tok/s and TTFT, decode tok/s and TPOT, long-context decode per engine x sockets x memory tier | `runs.csv`, uniform and native definitions |
| 2 | Achieved decode bandwidth (GB/s and % of the 552 GB/s HBM or about 265 GB/s DDR roofline); bytes read per token vs nominal | repeat 1 counters |
| 3 | Which tier served the reads in each configuration (HBM/DDR CAS split): did the engine really use HBM | counters and `numa_maps` |
| 4 | Dual-socket scaling (2S/1S) per engine, UPI traffic per token, mechanism (tensor parallel vs NUMA policy) | UPI flits, timing |
| 5 | AMX use in prefill per engine | `AMX_OPS_RETIRED.INT8`, `EXE.AMX_BUSY` |
| 6 | Average power per phase and energy per token, HBM vs DDR vs Cache | RAPL, idle baseline |
| 7 | HBM speedup over DDR per engine (the headline for the thesis question) | node A vs node B, anchor-corrected |

Plus a short "surprises and caveats" section (including whether llama.cpp and vLLM ended up on HBM when asked, and the stock
vLLM Flat-mode behaviour) and a list of follow-ups. Claims use measured data only; inferred values are marked **derived** or
**modelled**.

## 11. Open decisions (the owner may overrule any of them)

| Id | Decision | Reason |
| --- | --- | --- |
| D1 | Run both "Flat-DDR (DDR only)" and "Cache (HBM as transparent cache)" | The "DDR-only Cache baseline" is ambiguous; the working set fits in the HBM cache |
| D2 | Cache node is timing-first with unprivileged counters only | Privileged perf counters are unavailable on Cache nodes |
| D3 | Scope fixed to one model, INT8, batch 1 | First look, a few hours of wall time, not an exhaustive study |
| D4 | Exact-length prompts, token identity may differ; separate 32-token greedy sanity dump | Dense cost depends on length only; llama-bench uses random tokens |
| D5 | vLLM reported stock and, for 2S-HBM, as a labelled patched variant | Stock binds ranks to DDR in Flat mode; the patch is one line and disclosed |
| D6 | Thread counts as in section 5 (llama.cpp swept on 1S, vLLM 55 cores per rank) | Best documented configuration per engine, small tuning pass only |

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| A pinned version does not build or run with AMX | Setup jobs confirm first; adjust the pin and record it |
| vLLM start-up or the patched variant fails | Cell recorded as failed; stock results still reported |
| Counter multiplexing | Events split over repeats; running percentages checked |
| Node-to-node difference | Anchor cell on A and B |
| Job over 3 h | Cut order in section 8 |
| An engine silently lands on DDR or falls back from AMX | Placement and AMX proofs per run; reported as a finding, not hidden |
