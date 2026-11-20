# Configuration reference

Command-line options select the request; environment variables select how the engine executes it. Every variable is
validated at load (a bad value fails with a message), and every execution choice is recorded in the `--json` report. With
no variable set, the engine runs the default path of [architecture.md](architecture.md) §2; the defaults that depend on
the machine (sockets, backend, placement, prefill batch, AMX) are resolved as in its §2.1.

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
| `LLAMA_KERNELS` | `auto`, `scalar`, `avx512`, `amx` | instruction set; `auto` picks AVX-512 when available, and AMX-INT8 for `runq` W8A8 batched prefill where AMX can run (never AMX for an FP32 checkpoint); `amx` keeps AVX-512 for everything but the prefill GEMMs |
| `LLAMA_PRECISION` (`run` only) | `fp32`, `bf16` | `bf16` runs batched prefill on BF16-packed weights (needs a batch of 2 or more); decode stays FP32 |
| `LLAMA_Q8_ACTIVATIONS` (`runq` only) | `int8`, `fp32` | W8A8 (Q8_0 rule) or W8A32; W8A32 never uses the batched GEMM |
| `LLAMA_ATTN` | `tiled`, `split` | batched prefill attention (`tiled` falls back to `split` without AVX-512) |
| `LLAMA_ATTN_DECODE` | `split`, `group` | decode attention |
| `LLAMA_GEMV_SCHED` | `auto`, `static`, `dynamic`, `hybrid` | how a team splits GEMV rows (teams backend) |

### Prefill

| Variable | Values | Meaning |
| --- | --- | --- |
| `LLAMA_PREFILL_BATCH` | M | prompt tokens per forward pass (capped at the context). Default where the batched GEMM runs: 128 for FP32 on one socket, 64 on two, 256 for W8A8 and BF16; otherwise (scalar kernels, `LLAMA_PREFILL_GEMM=off`, W8A32) 1, token by token |
| `LLAMA_PREFILL_GEMM` | `auto`, `on`, `off` | batched GEMM (`auto`: on where AVX-512 or AMX runs) or the per-row reference |
| `LLAMA_PREFILL_LOGITS` | `last`, `all` | logits of the last prompt row only, or every row (diagnostic) |

### Threads, sockets and placement

| Variable | Values | Meaning |
| --- | --- | --- |
| `LLAMA_BACKEND` | `omp` (one socket), `teams` (two) | OpenMP regions, or persistent pinned worker teams. A build without `OPENMP=1` (the `make` default) runs `omp` serially: set `teams` on one socket |
| `LLAMA_THREADS` | N | team size (default: every allowed CPU for one team; each socket's allowed physical cores with two sockets) |
| `LLAMA_TOPOLOGY` | `auto`, `virtual:SxT`, `explicit:C0/H0/D0,C1/H1/D1` | socket/NUMA map: discovered, S fake unpinned sockets of T threads (tests), or explicit package/HBM/DDR nodes |
| `LLAMA_SOCKETS` | `2` on a dual-socket HBM node, else `1` | tensor parallelism over two sockets (runs on `teams`; `omp` is an error). Unset, 2 is chosen only on a discovered topology whose two sockets each have an allowed HBM node, with no `LLAMA_BACKEND=omp`, `LLAMA_TOPOLOGY` or `LLAMA_PLACEMENT=file\|ddr` |
| `LLAMA_TEAM_FANOUT` | `8`, `0`, N <= 64 | group size of the dispatch tree (0 = flat) |
| `LLAMA_PLACEMENT` | `file` (one socket), `hbm` (two sockets on an HBM node), `ddr` | where the weights live. Two sockets always copy their shards into owned regions: `hbm`/`ddr` bind them to each socket's node, `file` leaves them unbound (first touch by the socket's own team, which lands in DDR) |
| `LLAMA_PLACEMENT_STRICT` | `1` for `ddr`/`hbm`, else `0` | fail when a page is found on the wrong node or placement cannot be verified |
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
| `LLAMA_GEMM_KC`, `LLAMA_ATTN_WORKSPACE_MIB` | FP32 GEMM K block (1024), batched-attention workspace (32 MiB) |
| `LLAMA_ATTN_ROW_BLOCK`, `LLAMA_ATTN_TILED_QUERIES`, `LLAMA_ATTN_TILED_ORDER` | attention sweep knobs of E0/E2 (defaults 4, 16, `natural`) |
| `LLAMA_LOADER_CHUNK`, `LLAMA_LOADER_MAX_READ` | `pread` sizes of the owned loader |
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
