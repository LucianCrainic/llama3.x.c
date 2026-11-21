# Configuration reference

Command-line options select the request; environment variables select how the engine executes it. Every variable is
validated at load (a bad value fails with a message), and every execution choice is recorded in the `--json` report. With
no variable set, the engine runs the default path of [architecture.md](architecture.md) §2; the defaults that depend on
the machine (sockets, placement, prefill batch, AMX) are resolved as in its §2.1.

## Command line (`run` and `runq`)

`run CHECKPOINT [options]` (FP32) or `runq CHECKPOINT [options]` (Q8_0); `-h` prints the list.

| Option | Meaning |
| --- | --- |
| `-z PATH` | tokenizer (`tokenizer.bin` by default; must match the model) |
| `-m generate\|chat`, `-i TEXT`, `--prompt-file PATH`, `-y TEXT`, `--chat-template` | mode and prompt |
| `-t FLOAT`, `-p FLOAT`, `-s INT` | temperature (0 = greedy), top-p, seed |
| `--context-length INT` | KV-cache capacity (default min(model context, 4096)) |
| `-n INT`, `--generation-length INT` | caps on forward passes and generated tokens |
| `--benchmark`, `--no-text`, `--json PATH` | non-interactive run, silent output, performance report |
| `--warm-context INT` | fill INT positions untimed before the prompt (measure decode at a long context) |
| `--log-level LEVEL` | `error`, `warn`, `info`, `debug`, `trace` |

## Execution knobs

### Kernels and precision

| Variable | Values (default first) | Meaning |
| --- | --- | --- |
| `LLAMA_KERNELS` | `auto`, `scalar`, `avx512`, `amx` | instruction set; `auto` picks AVX-512 when available, and AMX for batched prefill where it can run: AMX-INT8 for `runq` W8A8, AMX-BF16 with `LLAMA_PRECISION=bf16` (never for plain FP32); `amx` keeps AVX-512 for everything but the prefill GEMMs |
| `LLAMA_PRECISION` (`run` only) | `fp32`, `bf16` | `bf16` runs batched prefill on BF16-packed weights, on AMX where it can run (needs a batch of 2 or more, the default on AVX-512); decode stays FP32 |
| `LLAMA_Q8_ACTIVATIONS` (`runq` only) | `int8`, `fp32` | W8A8 (Q8_0 rule) or W8A32; W8A32 never uses the batched GEMM |
| `LLAMA_ATTN` | `tiled`, `split` | batched prefill attention (`tiled` falls back to `split` without AVX-512) |
| `LLAMA_ATTN_DECODE` | `split`, `group` | decode attention |
| `LLAMA_KV_LAYOUT` | `head`, `rows` | K/V cache layout: `head` is `[kv_head][position][head_dim]` per layer (default since E5.4), `rows` is `[position][kv_head][head_dim]`; results are bitwise equal, `head` is faster at long contexts |

### Prefill

| Variable | Values | Meaning |
| --- | --- | --- |
| `LLAMA_PREFILL_BATCH` | M | prompt tokens per forward pass (capped at the context). Default where the batched GEMM runs: 128 for FP32 on one socket, 64 on two, 256 for W8A8 and BF16; otherwise (scalar kernels, `LLAMA_PREFILL_GEMM=off`, W8A32) 1, token by token |
| `LLAMA_PREFILL_GEMM` | `auto`, `on`, `off` | batched GEMM (`auto`: on where AVX-512 or AMX runs) or the per-row reference |
| `LLAMA_PREFILL_LOGITS` | `last`, `all` | logits of the last prompt row only, or every row (diagnostic) |

### Threads, sockets and placement

| Variable | Values | Meaning |
| --- | --- | --- |
| `LLAMA_BACKEND` | `teams` | the only backend: persistent pinned worker teams. Any other value is an error; the OpenMP backend (`omp`) was removed ([audit](audit-2026-11.md) F2) |
| `LLAMA_THREADS` | N | team size (default: every allowed CPU for one team; each socket's allowed physical cores with two sockets) |
| `LLAMA_TOPOLOGY` | `auto`, `virtual:SxT`, `explicit:C0/H0/D0,C1/H1/D1` | socket/NUMA map: discovered, S fake unpinned sockets of T threads (tests), or explicit package/HBM/DDR nodes |
| `LLAMA_SOCKETS` | `2` on a dual-socket HBM node, else `1` | tensor parallelism over two sockets. Unset, 2 is chosen only on a discovered topology whose two sockets each have an allowed HBM node, with auto topology and `LLAMA_PLACEMENT` unset or `hbm` |
| `LLAMA_TEAM_FANOUT` | `8`, `0`, N <= 64 | group size of the dispatch tree (0 = flat) |
| `LLAMA_PLACEMENT` | `file` (one socket), `hbm` (two sockets on an HBM node) | `file\|hbm\|ddr\|hbm-only\|hybrid`. See [planned placement](#planned-placement) for strict and mixed configurations. |
| `LLAMA_PLACEMENT_STRICT` | `1` for explicit tier placement, else `0` | require page-placement verification. `hbm-only` and `hybrid` require strict placement. |
| `LLAMA_HBM_BUDGET_GIB` | `56` | per-socket HBM budget checked before reserving |
| `LLAMA_THP` | `off`, `on` | transparent huge pages for owned regions |
| `LLAMA_VERIFY_SAMPLES` | `0` (every page) or N | pages sampled per region by the placement check |
| `LLAMA_KV_VERIFY` | `1`, `0` | verify KV-cache page placement in the report |

### Diagnostics and tuning

| Variable | Meaning |
| --- | --- |
| `LLAMA_PROFILE=1` | per-phase times (and per socket, with the exchange split) at log level `info` and in the report |
| `LLAMA_TP_CHECK=1` | compare both sockets' residual copies after every layer (bitwise) |
| `LLAMA_TOKENS_OUT=FILE` | write prompt and generated token ids as JSON (generate mode) |
| `LLAMA_LOADER_CHUNK`, `LLAMA_LOADER_MAX_READ` | test-only: `pread` sizes of the owned loader, so the tests reach the multi-chunk and short-read paths on small fixtures |
| `LLAMA_TOPOLOGY_ALLOW_FAR_HBM=1` | accept an explicit map that points a socket at the other socket's HBM (A6 remote control only) |
| `LLAMA_SYSFS_ROOT`, `LLAMA_NUMA_FAULT` | test-only: fake sysfs tree, injected syscall errors |

### Retired

These selected experimental paths whose code was removed after measurement; setting any of them is an error, so an old
script cannot silently measure the default path under an old label. See [audit-2026-11.md](audit-2026-11.md).

| Variable | Why it was removed |
| --- | --- |
| `LLAMA_FP32_LAYOUT` | packed4 decode layout: +3.2% at context 512 only, for 14 GB more HBM per socket (B7, D2: not adopted) |
| `LLAMA_TP_FUSED`, `LLAMA_TP_REDUCE`, `LLAMA_TEAM_JOIN` | slower synchronization variants, superseded by the fused flag reduction and flag join (B7) |
| `LLAMA_TP_TRACE` | per-worker reduction trace of the finished B7 investigation |
| `LLAMA_BF16_DROP_FP32` | reserved for a BF16 decode that does not exist |
| `LLAMA_BACKEND` other than `teams` | the OpenMP backend; worker teams were faster or equal (F2) |
| `LLAMA_GEMV_SCHED` | finished sweep: FP32 GEMV is hybrid on jobs of 96 MiB or more and static below, Q8 is static (F5) |
| `LLAMA_ATTN_ROW_BLOCK`, `LLAMA_ATTN_TILED_QUERIES`, `LLAMA_ATTN_TILED_ORDER` | finished E0/E2 sweeps: 4 rows per split task, 16 query vectors per tiled task, plan order (F5) |
| `LLAMA_ATTN_WORKSPACE_MIB`, `LLAMA_GEMM_KC` | fixed at 32 MiB and the 1024 K block (F5) |

## Planned placement

On Linux Flat/Quadrant nodes, `LLAMA_PLACEMENT=hbm-only|ddr|hybrid` uses the same owned loader and startup planner.
`hbm-only` binds all planned data to socket-local HBM, including an untied input embedding. Strict `ddr` binds the
same arenas to DDR. Use `LLAMA_SOCKETS=1|2` explicitly for controlled comparisons. The `file` and `hbm` presets
retain their legacy behavior. Setting `LLAMA_PLACEMENT_STRICT=0` on `ddr` selects the legacy fallback behavior.

`hybrid` keeps work, logits, exchange, and norms in HBM and defaults untied embeddings, loader scratch, and packed
prefill weights to DDR. The two layer controls accept `auto` or an exact count of initial layers in HBM:

- `LLAMA_HYBRID_WEIGHTS_HBM_LAYERS`
- `LLAMA_HYBRID_KV_HBM_LAYERS`

`LLAMA_HYBRID_CLASSIFIER=auto|hbm|ddr` controls the original classifier and its tied embedding if applicable.
`LLAMA_HYBRID_PACKED=hbm|ddr` applies only with packed prefill. Auto commits exact layer counts first, then prioritizes
the classifier, original weights, and KV in that order. Each layer choice is symmetric across TP sockets.

`LLAMA_HBM_BUDGET_GIB` defaults to 56 per socket. `LLAMA_DDR_BUDGET_GIB` optionally lowers the DDR ceiling. Both
tiers account for rounded arena bytes and full configured KV capacity. Use `--plan-only` to print the plan as JSON
without reading weight payloads, starting teams, or loading a tokenizer. With `--json PATH`, the plan is in the report's
`placement_plan` field. Real executions record each arena's requested node and page-query result and verify active
KV ranges. Use `python3 tools/check_placement.py REPORT --planned` to check that evidence. A nonzero
`LLAMA_VERIFY_SAMPLES` permits stratified sampling. For active KV, at least the first and last page of every span
are checked, even when that exceeds the requested sample count. Add `--allow-sampled` to accept sampled reports.

Planned execution requiring every prompt position's logits must set `LLAMA_PREFILL_LOGITS=all` so the planner can
admit its output buffers and any BF16 classifier pack. An explicit `LLAMA_TOPOLOGY=virtual:2x3` can exercise the
same layouts and kernels without physical NUMA binding. Those reports carry `simulated: true`.

The generate-mode report includes `phase_epoch_ns` for aligning interval counters with prefill and decode. The
prefill window starts after any retained-context warm-up and ends before the prompt's KV verification. The decode
window begins just before the first decode forward and ends before final KV verification and teardown.
