# llama3.x.c

A C11 inference engine for Llama 3.x text models (1B to 8B) on CPUs, tuned for dual-socket Intel Xeon Max (Sapphire
Rapids with HBM). It grew out of llama2.c and keeps its spirit: no third-party runtime libraries (PCRE2 for the tokenizer is
the one optional dependency), one process, one request.

What it does beyond a reference implementation:

- **AVX-512 kernels** for decode (GEMV, attention, norms) that stream weights at about 85% of a socket's HBM read bandwidth.
- **Batched prefill** on an AVX-512 FP32 GEMM, and on **AMX** tiles in BF16 (opt-in precision mode) or INT8 (Q8_0).
- **Engine-owned NUMA placement**: weights, KV cache and buffers bound to a chosen HBM or DDR node, verified page by page.
- **Two-socket tensor parallelism**: each socket holds half of every layer in its own HBM, so no weight crosses UPI;
  8B decode runs 1.87-1.99x faster than on one socket.

Measured on 8B (2 x Xeon Max 9480, Flat mode): FP32 decode 30.6 tok/s and Q8 decode 91.3 tok/s on two sockets; a
2048-token prompt reaches the first token in 1.6 s with AMX on two sockets, against 121 s token by token.

## Build

Needs a C11 compiler, `make`, `pkg-config` and the PCRE2-8 development files (`make PCRE2=0` builds without them, for
legacy tokenizers only).

```bash
make                     # build/release/run (FP32) and build/release/runq (Q8_0)
```

```bash
make test                # unit tests, emulated AVX-512/AMX tests and end-to-end tests on synthetic models
```

`MODE=debug|profile|sanitize|tsan` select other builds and `NATIVE=1` adds `-march=native`. The engine schedules its
kernels on its own worker teams (POSIX threads), so no OpenMP is needed. AMX code needs x86-64 and GCC 11 or later.

## Convert a model

Conversion needs Python with `tools/requirements.txt` and a local Hugging Face snapshot; the C binaries need neither.

```bash
python3 tools/export.py checkpoints/llama-fp32.bin --hf /path/to/snapshot --version 3
```

```bash
python3 tools/export.py checkpoints/llama-int8.bin --hf /path/to/snapshot --version 3 --int8
```

```bash
python3 tools/tokenizer.py --hf /path/to/snapshot -o checkpoints/tokenizer.bin
```

## Run

```bash
build/release/run checkpoints/llama-fp32.bin -z checkpoints/tokenizer.bin --chat-template -i 'Explain what a CPU does.' -t 0
```

The defaults follow the measurements: on a dual-socket Xeon Max node in Flat mode, a plain run uses both sockets with the
weights in HBM, batched prefill, and (for Q8_0) AMX-INT8 prefill, and the `--json` report records each choice:

```bash
build/release/runq checkpoints/llama-int8.bin -z checkpoints/tokenizer.bin --chat-template -i 'Explain what a CPU does.' -t 0 --json report.json
```

On any other machine the same command runs on one socket, on every allowed CPU (`LLAMA_THREADS` caps it).

Every option and environment variable is listed in [docs/configuration.md](docs/configuration.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | how the engine works: execution paths, memory, HBM techniques, AMX, tensor parallelism, feature status |
| [docs/configuration.md](docs/configuration.md) | command line and environment reference |
| [docs/phase2.md](docs/phase2.md) | checkpoint and tokenizer formats, build modes, reports |
| [docs/hbm-roofline.md](docs/hbm-roofline.md) | the machine's measured bandwidth and compute ceilings |
| [docs/numa-tp-prefill-results.md](docs/numa-tp-prefill-results.md) | final campaign (D2) and the decisions taken from it |
| [docs/audit-2026-11.md](docs/audit-2026-11.md) | code-quality audit and the simplification plan |
| [docs/plans/](docs/plans/README.md) | the phased plan that built placement, tensor parallelism and prefill, with every phase's record |

Results documents (`docs/*-results.md`, `docs/tensor-parallel.md`, `docs/batched-prefill.md`, `docs/numa-placement.md`,
`docs/kernels.md`) each cite the commit, node and Slurm jobs they were measured with.

## License

MIT, see [LICENSE](LICENSE).
