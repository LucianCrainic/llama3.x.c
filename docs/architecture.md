# Engine architecture

This document describes the engine as it is in the tree: what it supports, how a token flows through it, where memory
lives, and why each performance technique exists. Measured numbers are quoted with the results document they come from;
everything here was measured on CRESCO8 Flat-mode nodes (2 x Xeon Max 9480, 56 cores and 64 GB HBM2e per socket, NUMA
nodes 0/1 = DDR, 2/3 = HBM). The knob reference is [configuration.md](configuration.md); the cleanup audit that produced
this document is [audit-2026-11.md](audit-2026-11.md).

## 1. What the engine is

A dependency-free C11 inference engine for Llama 3.x text models (1B, 3B, 8B; GQA, RoPE scaling of 3.1/3.2, tied or untied
embeddings), derived from llama2.c. It runs one request at a time (single-stream generate or chat) on CPUs, and is tuned for
dual-socket Sapphire Rapids with HBM in Flat mode. It builds and passes its tests on macOS (arm64, scalar kernels and
emulated AVX-512/AMX) and on Linux x86-64.

Two executables share every source file except the forward pass:

| Executable | Checkpoint | Weights | Activations |
| --- | --- | --- | --- |
| `run` (`src/run.c`) | FP32 v3 (`tools/export.py`) | FP32; optional BF16 packed copy for prefill | FP32 |
| `runq` (`src/runq.c`) | Q8_0 v3 | int8 with one FP32 scale per group of `gs` inputs | int8 per group (W8A8, default) or FP32 (W8A32) |

The checkpoint format, tokenizer format, chat template and the build modes are specified in [phase2.md](phase2.md).

## 2. Status of every feature

**Default** runs without any knob. **Opt-in** is supported, tested and measured, and selected by a knob. **Diagnostic**
exists for tests and measurements only and is never a deployment setting.

| Feature | Status | Evidence |
| --- | --- | --- |
| Scalar reference kernels (original llama2.c operation order) | Default where AVX-512 is absent | `tests/kernels.c` (double-precision references) |
| AVX-512 GEMV / attention / norm kernels | Default on AVX-512 + VNNI CPUs | [kernels.md](kernels.md) |
| Persistent worker teams | The only backend | [F2](plans/00-foundations/F2-worker-teams.md): decode +4% to +11% against OpenMP; one-socket A/B in [audit-2026-11.md](audit-2026-11.md) §6 |
| Batched prefill (`LLAMA_PREFILL_BATCH=M`) | Default where the batched GEMM runs (AVX-512, AMX): section 2.1 | [batched-prefill.md](batched-prefill.md): 8B 2048-token TTFT 121 s to 10.5 s |
| AMX-INT8 prefill for Q8_0 (`LLAMA_KERNELS=amx`, `runq`) | Default for W8A8 batched prefill where AMX can run | [numa-tp-prefill-results.md](numa-tp-prefill-results.md) §7: 3.4x AVX-512 Q8 |
| AMX-BF16 prefill for FP32 (`LLAMA_PRECISION=bf16`, on AMX under `auto`) | Opt-in by user decision (BF16 never silent) | same, 3.6x the AVX-512 FP32 GEMM |
| Engine-owned, verified placement (`LLAMA_PLACEMENT=hbm\|ddr`) | Opt-in on one socket; two sockets always own their shards, in HBM by default on an HBM node | [numa-placement.md](numa-placement.md) |
| Two-socket tensor parallelism (`LLAMA_SOCKETS=2`) | Default on a node whose two sockets each have an HBM node | [tensor-parallel.md](tensor-parallel.md): 8B decode 1.87-1.99x one socket |
| Fused tiled prefill attention (`LLAMA_ATTN=tiled`) | Default for batched prefill | [attention-results.md](attention-results.md): 1.55x TTFT at 8B/8192 |
| Split-K prefill attention (`LLAMA_ATTN=split`) | Opt-in (reproduces runs before E4) | same |
| Split-K decode attention (`LLAMA_ATTN_DECODE=split`) | Default | [decode-attention-results.md](decode-attention-results.md) |
| GQA-group decode attention (`LLAMA_ATTN_DECODE=group`) | Opt-in, value gates missed; candidate for long contexts | same, §7 |
| W8A32 Q8 decode (`LLAMA_Q8_ACTIVATIONS=fp32`) | Opt-in | [kernels.md](kernels.md) |
| Transparent huge pages for owned regions (`LLAMA_THP=on`) | Opt-in; measured neutral | [numa-placement.md](numa-placement.md) |
| Profile (`LLAMA_PROFILE=1`), JSON report (`--json`), `LLAMA_TP_CHECK=1` | Diagnostic | section 9 |
| Head-major K/V cache (`LLAMA_KV_LAYOUT=head`) | Default since E5.4 (owner decision 2026-11-20); `rows` is opt-in and bitwise equal | [decode-attention-results.md](decode-attention-results.md) §8, §9: 1B decode at 32768 one socket 1.34x FP32, 1.65x W8A8 |
| AMX flash attention (E3) | Not implemented (postponed; the `LLAMA_ATTN=amx` placeholder was removed in F5) | [E-followups](plans/E-followups/README.md) |

Removed after measurement (setting their knobs is an error, see [audit-2026-11.md](audit-2026-11.md)): the packed4 FP32 decode
layout, the unfused and two-barrier decode reductions, the counter team join, the reduction trace, the OpenMP backend
(`LLAMA_BACKEND=omp`, `make OPENMP=1`), and the knobs of finished sweeps (GEMV schedule, attention row block, tiled queries
and order, attention workspace, GEMM K block; F5).

### 2.1 How the defaults are resolved

The defaults follow the decisions of the final campaign ([numa-tp-prefill-results.md](numa-tp-prefill-results.md) §7),
set on 20 November 2026. Each rule applies only while its own knob is unset, and the report records what was chosen.

| Choice | Rule when unset |
| --- | --- |
| Sockets | 2 when the discovered topology has two sockets with allowed CPUs, each with an HBM node in the job's allowed memory (Flat mode on a Xeon Max), and neither `LLAMA_TOPOLOGY` nor `LLAMA_PLACEMENT=file\|ddr` is set; else 1. A job bound to one socket (`numactl --cpunodebind`, a one-socket allocation) therefore keeps one |
| Placement | `hbm` with two sockets on such a node (each socket's shards in its own HBM, verified), `file` otherwise |
| Prefill batch | where the batched GEMM runs (AVX-512 or AMX, or `LLAMA_PREFILL_GEMM=on`): 128 rows for FP32 on one socket, 64 on two, 256 for W8A8 and AMX-BF16; 1 on the scalar kernels, with `LLAMA_PREFILL_GEMM=off`, and for W8A32 |
| Kernels | `auto`: AVX-512 where available, scalar otherwise. Batched prefill then takes AMX where it can run: AMX-INT8 for `runq` W8A8 (group size a multiple of 64), AMX-BF16 when `LLAMA_PRECISION=bf16` was asked for. An FP32 checkpoint without it never gets AMX or asks for the tile permission (BF16 stays a user decision) |

To reproduce a run made before these defaults, set `LLAMA_SOCKETS=1 LLAMA_PREFILL_BATCH=1 LLAMA_KERNELS=avx512` (and
`LLAMA_ATTN=split` for runs before E4, `LLAMA_KV_LAYOUT=rows` for runs before E5.4; the layout changes speed, never results). The campaign scripts do this job-wide (`scripts/lib/cresco8-common.sh`). Runs made
on OpenMP now run on worker teams of the same size: decode is 5-8% faster there and prefill within 2% (6.5% slower for 1B
FP32), see [audit-2026-11.md](audit-2026-11.md) §6.

## 3. Source map

```
src/app.c          CLI: options, prompt handling, prefill/decode loop, --warm-context, chat
src/report.c       the schema-v1 JSON performance report (--json)
src/run.c          FP32 engine: weights view, run state, forward (per token, batched GEMM, BF16)
src/runq.c         Q8_0 engine: the same structure with quantized weights and activations
src/tp.c           two-socket engine used by both executables when LLAMA_SOCKETS=2
src/prefill.c      prefill policy shared by both engines (batch size, GEMM on/off, precision guard)
src/runtime.c      checkpoint validation and mapping, logging, fail(), accounted allocator
src/shards.c       tensor manifest (file offset of every tensor) and the per-socket shard planner/loader
src/loader.c       weight placement (file|ddr|hbm), run-state arena, packed-weight store, HBM budget
src/numa.c         topology discovery from sysfs, bound regions (mmap+mbind), page verification (move_pages)
src/workers.c      persistent pinned worker teams, barriers, the cross-socket exchange
src/kernels_simd.c ISA selection, GEMV/attention/norm/RoPE kernels (scalar + AVX-512), scheduling
src/kernels.c      scalar reference rmsnorm/softmax/RoPE used by tests and the scalar path
src/gemm.c         batched GEMMs: FP32 (AVX-512), BF16 and INT8 on AMX tiles, Q8 VNNI, packing
src/tokenizer.c    Llama 3 BPE tokenizer (PCRE2 pre-tokenization when built with PCRE2=1), chat template
src/sampler.c      temperature / top-p sampling
src/profile.c      LLAMA_PROFILE phase timers
```

`include/*.h` holds one header per module; each header documents its contract.

## 4. Execution paths

### 4.1 Load

`engine_load_context` (in `run.c` / `runq.c`):

1. `kernels_init` reads the ISA and attention knobs and refuses retired knobs; `kernels_backend_init` discovers the
   topology, resolves the socket count (section 2.1) and, for `teams` (the default), starts the pinned worker pool (one
   team, or one per socket). `prefill_configure` then sets the prefill batch, and `runq` takes AMX-INT8 for it where it can.
2. `checkpoint_map` validates the header and maps the file read-only; `manifest_build` records the byte offset of every
   tensor. Every later view of a weight (mapped, owned, sharded, packed) is derived from this manifest.
3. One socket: `weights_open` either keeps the mapping (`file`) or copies the tensors into engine-owned regions bound to a
   NUMA node (`ddr`/`hbm`), then the run state and KV caches are allocated (`malloc_run_state`). With AMX-INT8 or BF16 the
   seven per-layer matrices get a packed copy in a bound region.
4. Two sockets: `tp_create` plans the shards, reserves each socket's regions on its own nodes, starts the socket-1
   coordinator thread, and both sockets load (and pack) their shards in parallel with `pread`; the file mapping is then
   dropped.

### 4.2 Prompt and generation

`run_sequence` in `app.c` splits the prompt into microbatches of at most `LLAMA_PREFILL_BATCH` tokens and calls
`engine_forward_batch` for each, asking for logits only on the last one; then it samples and calls `engine_forward` once
per generated token. `--warm-context N` first fills N positions untimed, so decode can be measured at a long context.

### 4.3 The forward pass

Every path computes the same Llama block: `x += Wo · attn(rope(Wq·n(x)), rope(Wk·n(x)), Wv·n(x))`, then
`x += W2 · (silu(W1·n(x)) * W3·n(x))`, with RMSNorm `n` and the K/V rows appended to the cache. The cache is head-major
(`[layer][kv_head][position][head_dim]`, section 6 item 8): the projections write K/V rows to a staging buffer, RoPE runs
on it, and `k_kv_store` copies each head's slice into place (`PHASE_KV_STORE`); attention goes through `k_attention_kv` /
`k_attention_batch_kv`. With `LLAMA_KV_LAYOUT=rows` (`[layer][position][kv_head][head_dim]`) the projections write
straight into the cache, as before E5.4.

| Path | Where | Projections | Attention | When |
| --- | --- | --- | --- | --- |
| Per-token decode, 1 socket | `forward_layer` | GEMV `k_matmul_*` (fused QKV, fused W1/W3+SwiGLU) | `k_attention` (split-K) | every decode step; prefill when M = 1 or the GEMM is off |
| Batched prefill, 1 socket | `forward_layer_batch` | GEMM: FP32 `k_gemm_f32`, BF16/AMX `k_gemm_bf16`, Q8 `k_gemm_q8` (portable, VNNI, AMX-INT8) | `k_attention_batch` (tiled) | prompt microbatches with M > 1 |
| Decode, 2 sockets | `tp.c` `forward_fp32` / `forward_q8` | GEMV on each socket's shards; fused Wo/W2 + flag reduction | per socket, its own heads | every decode step |
| Batched prefill, 2 sockets | `tp.c` `forward_batch` | GEMM on each socket's shards; barrier reduction of M x dim partials | per socket | prompt microbatches |

The per-token path is also the numerical oracle: the batched reference path (scalar kernels) is bitwise identical to it,
and every faster path is gated against it (section 9).

## 5. Memory management

**Accounting.** Every heap allocation goes through `memory_alloc(count, size, kind)` (zeroed, overflow-checked, tagged with a
`MemoryKind`), and every bound region is accounted with `memory_track`; the JSON report lists current and peak bytes per
kind. Only the coordinator thread allocates; workers never touch the accounting.

**Bound regions (`numa.c`).** A `NumaRegion` is one anonymous `mmap` that is bound to exactly one node with `mbind`
*before any page is touched*, so first touch cannot place a page elsewhere; a bind failure is fatal, never a silent
fallback. The team that will use the region writes it first (pages are created by the right socket's threads), weights are
then made read-only, and `region_verify` asks the kernel where every page lives (`move_pages` in query mode, full or
sampled). Under `LLAMA_PLACEMENT_STRICT` (default for `ddr`/`hbm`) a page on the wrong node fails the run. A registry
releases every region at exit, including on `fail()` paths. Off Linux the same API uses plain `mmap` (node -1).

**Weights.** `WeightStore` answers "where does the byte at file offset X live now": the mapping (`file`), or up to four
owned segments that keep each tensor's alignment modulo 64 (`ddr`, `hbm`; an untied input embedding stays in DDR to save HBM:
it is read one row per token). Two sockets always own their shards (`shards.c`): row shards of Wq/Wk/Wv/W1/W3 (each socket's
heads and hidden channels), column shards of Wo/W2 (packed to a dense stride through bounded scratch), copies of the norm
weights, and the vocabulary rows of the classifier (which also serve the embedding lookup when tied).

**Run state and KV cache.** In `file` mode activations and caches are heap allocations. In an owned mode `RunMemory`
sizes all activation buffers in a first pass, carves them from one `ROLE_WORK` region, and gives each KV cache its own
region whose pages appear when a position is first written. Two sockets keep everything per socket: its half of the KV
heads, its activations, and its exchange buffers.

**HBM budget.** Before reserving anything, the two-socket path checks that each socket's shards, packed copies, KV caches,
work and exchange regions fit `LLAMA_HBM_BUDGET_GIB` (56 GiB of the 64 GB per socket by default).

## 6. Maximizing HBM bandwidth (decode)

Single-stream decode reads every weight once per token, so tokens/s is HBM bandwidth divided by model bytes. The machine
ceilings ([hbm-roofline.md](hbm-roofline.md) §1): ~550-600 GB/s of local HBM read per socket with all 56 cores, ~1.1 TB/s
for both sockets reading their own HBM, and only 50-110 GB/s when a socket reads the other socket's HBM. Every technique
below follows from those three numbers.

1. **Weights in local HBM.** Either stage the checkpoint in `/dev/shm` under `numactl --membind=<hbm node>` (`file`), or
   let the engine own and verify it (`LLAMA_PLACEMENT=hbm`). Owned HBM equals staged HBM within 2.4% (A6); DDR halves decode
   (8B FP32 8.4 against 16.1 tok/s).
2. **Use the whole socket, one thread per physical core.** HBM bandwidth scales nearly linearly to 56 cores (DDR saturates
   at 28). Teams pin one worker per allowed physical core and verify the affinity; the report records it.
3. **Never read across UPI: one socket per half of the model.** Adding the second socket's threads to one model is
   *slower* (8.2 tok/s against 14.9 on one socket for 8B FP32) because half the reads go remote. Tensor parallelism gives
   each socket its own shards in its own HBM, so both sockets stream locally and the only cross-socket traffic is two
   `dim`-sized partial sums per layer (each socket reads 32 KiB from its peer per layer and token for 8B). Result: 8B FP32 30.6 tok/s against 16.4
   (1.87x), about 980 GB/s of weights, 82% of the 2 x 600 GB/s stream ceiling ([tensor-parallel.md](tensor-parallel.md)).
4. **GEMV kernels shaped for streaming.** Each worker takes one contiguous row range so the hardware prefetcher sees long
   sequential streams; four rows share every load of `x` and run independent FMA chains (`dots_f32_avx512`). Q8 kernels,
   which do more work per byte, add software prefetch at a measured distance (`q8_ahead`). QKV and W1/W3 run as one fused
   job, so `x` is read once and there is one dispatch instead of three.
5. **Absorb stragglers on big matrices.** For FP32 jobs of at least 96 MiB (the hybrid split) each member
   first streams a contiguous three quarters share and then claims small chunks of the rest, so a member slowed by a busy
   memory channel does not hold the whole team (+14% on 8B FP32 two sockets). Smaller jobs and Q8 stay static, where
   claiming measured 4-25% slower.
6. **Make synchronization nearly free.** A decode token runs several team jobs per layer (QKV, attention, Wo, W1/W3,
   W2: about 160 per 8B token), so dispatch and join costs multiply. Teams dispatch through a fan-out tree of group heads (`LLAMA_TEAM_FANOUT`, default 8), join through one flag
   line per member, and the two-socket Wo/W2 projection is fused with its reduction: every member flags its finished rows and
   adds its share of both sockets' partials as soon as those rows are ready, with no barrier. Together these took 8B Q8 two
   sockets from 87.3 to 95.5 tok/s; the reduce and wait time left is at most 0.022 ms per token
   ([tensor-parallel.md](tensor-parallel.md)).
7. **Pin each socket's leader to its own socket.** The thread that calls `team_run` is worker 0; `team_adopt_leader` pins
   the socket-1 coordinator to socket 1. Before this fix both leaders shared one CPU and two sockets were slower than one.
8. **Give long-context attention contiguous K/V streams.** At long contexts the K/V cache, not the weights, dominates
   the bytes a decode token reads. In the head-major cache each KV head's positions are one contiguous run of
   `position x head_dim` floats, so a split-K chunk streams a dense range instead of one `head_dim` slice (256 or 512 bytes) per position
   strided by the whole row of KV heads. Same bytes and same cache size, bitwise equal results; 1.49-2.86x in the
   attention microbenchmark at 8192 and 32768 tokens, 1.34x (FP32) and 1.65x (W8A8) for 1B decode at 32768 on one socket
   ([decode-attention-results.md](decode-attention-results.md) §8, §9).

The packed4 layout (four rows interleaved into one stream) reached 612 GB/s in a microbenchmark but only +3.2% end to end at
context 512 for 14 GB more HBM per socket; it was removed.

## 7. Batched prefill and AMX

Prefill processes M prompt tokens per weight pass, which turns each GEMV into a GEMM and moves prefill from
bandwidth-bound to compute-bound. The precision ladder (user decision: each rung explicit and gated):

| Rung | Kernel | Selected by |
| --- | --- | --- |
| AVX-512 FP32 GEMM | `k_gemm_f32`: register-blocked microkernel, K blocks of 1024 | `LLAMA_PREFILL_BATCH>1` on AVX-512 (`LLAMA_PREFILL_GEMM=auto`) |
| AMX-BF16 | `k_gemm_bf16`: activations converted to BF16 (round to nearest even) into an aligned panel; weights pre-packed in VNNI pairs; FP32 sums | `LLAMA_PRECISION=bf16 LLAMA_KERNELS=amx` (`run`) |
| AVX-512 VNNI Q8 | `k_gemm_q8` tiled VNNI, per-group int32 dots scaled into FP32 | `runq`, W8A8, `gs % 64 == 0` |
| AMX-INT8 | `k_gemm_q8` on weights packed in VNNI quads: exact int32 group dots on tiles, same Q8_0 semantics | `runq`, W8A8, `LLAMA_KERNELS=amx` |

AMX specifics (`gemm.c`): the tile-data permission is requested once with `arch_prctl(ARCH_REQ_XCOMP_PERM)` before any
worker can execute a tile instruction; each thread configures its tiles (16 rows x 64 bytes) and releases them on exit.
Weights are zero-padded to 16-row tiles and K to 32 (BF16) or 64 (INT8). The microkernel computes up to 32 x 32 outputs from
2 x 2 tiles. Packing runs on the worker team at load, into a bound region on the weights' node, and is counted in the HBM
budget; decode keeps reading the original FP32 or int8 weights. `LLAMA_KERNELS=amx` on an FP32 checkpoint without
`LLAMA_PRECISION=bf16` is refused, so FP32 never changes precision silently. The AMX code is compiled only on x86-64 with a
compiler that has the AMX intrinsics (GCC 11 or later); `tests/amx_emul.h` is a software tile model that runs the same code everywhere in tests.

Attention becomes the next cost at long prompts (41% of the 8B AMX forward at 2048 tokens, 74% at 8192). The tiled kernel
(`LLAMA_ATTN=tiled`) computes 16 query vectors per task against transposed key blocks in registers and streams V once per
block, merging chunks with a stable log-sum-exp; it cut 8B 8192-token TTFT from 25.1 s to 16.2 s on AMX-BF16.

Measured 8B prefill at a 2048-token prompt (time to first token): FP32 per token 121 s, AVX-512 GEMM 10.5 s, AMX-BF16 2.9 s on
one socket and 1.6 s on two; Q8 per token 41 s, AMX-INT8 2.8 s and 1.6 s ([numa-tp-prefill-results.md](numa-tp-prefill-results.md)).

## 8. Threads and scheduling

Kernels never create threads. A kernel describes its work as row ranges or segments and calls `parallel_rows` /
`kernels_parallel_segments`, which run it either on the calling thread's worker team (`kernels_set_team`,
`kernels_set_thread_team`) or, without a team (the unit tests), serially on the calling thread.

A team (`workers.c`) is a set of persistent threads created at load; the caller of `team_run` is worker 0 and does its own
share. Workers wait on a generation counter (spin for a bounded time, then `sched_yield`), run the job, and set their done
flag. With `LLAMA_SOCKETS=2` there are two teams; the main thread drives team 0 and a coordinator thread drives team 1. The two
coordinators run the same forward function on their own shards (`run_both` sends them a command) and meet only in the
exchange.

**The exchange** combines the two sockets' partial Wo/W2 outputs: `residual_s += partial_0 + partial_1` with the same
operand order on both sockets, so both residual copies stay bitwise identical (`LLAMA_TP_CHECK=1` verifies this after every
layer). Partials are double-buffered by generation, so a socket can write the next partial while the peer still reads the
previous one. Decode uses the per-row flag protocol (section 6, item 6); batched prefill uses the two-barrier protocol, whose
cost is amortized over M rows.

`fail()` sets a process-wide abort flag before exiting, which releases every spin-wait of the teams and the exchange, so a
failure on one thread cannot hang the others.

## 9. Correctness gates and tests

Every faster path is compared with a slower reference, and the gate depends on what may legitimately change:

| Comparison | Gate |
| --- | --- |
| Placement-only changes, batched reference path, packing, row-parallel norm/RoPE, scheduling variants | bitwise |
| SIMD kernels against double-precision references | `2e-5 * sum|w*x| + 1e-6` per output |
| FP32 GEMM, two sockets against one, against Hugging Face FP32 | max abs logit difference 1e-3, argmax 100% |
| BF16 against FP32 per token | max 2.0, rms 0.05, argmax 0.97 |
| W8A8 against W8A8 (any change of FP32 order requantizes) | statistical: rms 0.2, argmax 0.9, max 16 |

`make test` builds every engine and test binary and runs: kernel tests against references (native and on emulated AVX-512),
GEMM/AMX/BF16/Q8 tests (native and on the emulated tile model), manifest, shard, region, worker and sizing tests, and the
Python suites `tests/test_runtime.py`, `test_numa.py`, `test_tp.py` and `test_tools.py`, which run the real executables on
synthetic checkpoints (`tests/fixtures.py`) with virtual topologies (`LLAMA_TOPOLOGY=virtual:2xT`) and fake sysfs trees, so
the two-socket and placement logic is exercised on any machine. `make test-reference` compares against Hugging Face when
`torch` and a real model are available. `MODE=sanitize` and `MODE=tsan` build the same targets with sanitizers.

## 10. Cluster workflow

`scripts/cresco8-*.sbatch` are the campaign scripts (`STAGE=check|bench`, `PHASE=...`), sharing
`scripts/lib/cresco8-common.sh`; `tools/summarize_*.py` and `tools/campaign_tables.py` turn their JSON reports into the
results tables. Each results document cites the commit, job IDs and node it was produced on; to reproduce an old campaign,
check out that commit.
