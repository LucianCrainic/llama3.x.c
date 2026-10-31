# CPU inference baseline

README.md is intentionally empty. See audit.md for removals and provenance.

## Layout and ownership

- src/run.c: FP32 weight views, activation/cache lifecycle, scalar numerical kernels.
- src/runq.c: Q8_0 views, quantization, dequantization and INT8 numerical kernels.
- src/runtime.c: checked allocations, checkpoint validation/mapping, logging, clock.
- src/kernels.c: shared RMSNorm and softmax.
- src/tokenizer.c and src/sampler.c: shared legacy tokenization and sampling.
- src/app.c and src/report.c: CLI, generation/chat, measurement and JSON export.
- include/: the small shared runtime and application interfaces.
- tools/: Python checkpoint/tokenizer converters and the model they import.
- tests/: synthetic fixtures, original-logit references, C unit and Python process tests.
- docs/: this guide, audit, validation record, and an actual synthetic report example.

Each executable links one precision implementation. Kernels retain their original
operation order; neither batched prefill nor hardware-specific optimization was
introduced. Shared infrastructure is compiled into each executable, without an
external runtime dependency. Only one model is supported per CLI process; INT8's
group size remains private to that implementation.

Weights borrow a read-only mmap of the checkpoint. FP32 creates no heap copy of
weights. INT8 preserves the original full FP32 embedding expansion; its size is
reported separately. Quantized descriptors own no mapped data. Activations,
quantization scratch, tokenizer, sampler and prompt buffers own heap allocations.
The KV layout is `[layer][position][kv_dim]`, with separate key and value arrays;
`kv_dim = (dim / n_heads) * n_kv_heads`. GQA query heads share KV heads.

runtime.c is the allocation/mapping boundary for future placement work. Memory
categories describe lifetimes and roles without NUMA policies. No HBM assumption,
libnuma dependency, migration, automatic binding or scheduling is present.
Normal completion frees all owned runtime structures and unmaps weights; fatal
initialization/runtime errors exit after diagnostics/reporting, letting the OS
reclaim partially initialized process resources. This is a CLI, not a recoverable
in-process library API.

## Builds

A C11 compiler, POSIX system interfaces and libm are sufficient. Python 3.8+ is
needed only for tests; Python ML packages are needed only for conversion.

Linux (GCC, or substitute CC=clang):

```sh
make CC=gcc
make test CC=gcc
make MODE=debug CC=gcc
make MODE=profile CC=gcc
make test MODE=sanitize CC=gcc
make test OPENMP=1 CC=gcc
```

macOS (Apple Clang):

```sh
make CC=clang
make test CC=clang
make MODE=debug CC=clang
make MODE=profile CC=clang
make test MODE=sanitize CC=clang
```

Executables are build/<mode>/run and runq. Release uses -O3 without fast-math;
profile uses -O2 -g -fno-omit-frame-pointer; debug uses -O0 -g3. Sanitize uses
AddressSanitizer and UndefinedBehaviorSanitizer. Flags are recorded in reports.
No build requires OpenMP. GCC on Linux supports OPENMP=1; Apple Clang needs an
external libomp installation and explicit flags, for example:

```sh
make CC=clang OPENMP=1 \
  OMP_CFLAGS='-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include' \
  OMP_LDFLAGS='-L/opt/homebrew/opt/libomp/lib -lomp'
```

That macOS OpenMP command is an example, not a validated installation recipe;
adjust the prefix to your installation. Optional NATIVE=1 adds -march=native.
CFLAGS/CPPFLAGS/LDFLAGS/LDLIBS and BUILD_DIR may be overridden. A configuration
stamp forces rebuilds when the compiler or flags change. Changing OPENMP in the
same directory replaces those binaries; use a distinct BUILD_DIR for experiments.

External affinity tools can launch the binaries normally, for example numactl
on Linux. Reports expose online CPUs, sched_getaffinity's CPU list, and the Linux
sysfs online NUMA node list when available. Failed/unsupported diagnostics are
null. These are observations, not memory-tier identification. OpenMP thread
counts and affinity policies remain controlled by external environment variables.

## Runtime and conversion

```sh
build/release/run model-fp32.bin -z tokenizer.bin -i 'Once upon a time' -n 128
build/release/runq model-int8.bin -z tokenizer.bin -m chat -y 'You are helpful.'
build/debug/run --help
```

Legacy flags -t/-p/-s/-n/-i/-z/-m/-y remain. -n is a maximum number of **forward
passes including prompt processing**, not output length; 0 selects checkpoint
context length. A short -n may process only part of the prompt and produce no
output. The generation-length option caps non-stop output tokens separately;
0 tokenizes but executes no forward passes. Context and -n still take priority.
The final prompt position can emit one output token even at the context limit.

--benchmark is non-interactive, defaults to prompt `Once upon a time`, seed 1,
and 128 output tokens (unless an explicit -n or generation length overrides that
default). It preserves EOS/EOT stopping and never pads a completed response.
Temperature 0 gives greedy decoding. Nonzero temperatures are reproducible with
the same seed/build/environment; cross-platform bitwise equality is not promised.
Seed 0 explicitly maps to state 1 because xorshift cannot use an all-zero state.

--prompt-file reads a whole file, including newlines; it cannot be combined with
-i/--prompt. Embedded NUL bytes are rejected. --no-text suppresses stdout.
--log-level accepts error, warn, info (default), debug, trace. Diagnostics and
interactive labels use stderr. Stdout contains prompt echo and generated text
for generation mode, and assistant response text for chat. Prompt echo is kept
for the original CLI behavior but excluded from generated token counts.

Chat preserves the original LLaMA 3 system/user/assistant token template and uses
dynamic buffers instead of fixed 32 KiB arrays. It stops cleanly on EOF or exit.
End-of-turn markers are cached between turns. Counts/timings aggregate turns;
interactive wait time is excluded from total_inference but included in end_to_end.
Use generation/benchmark mode for comparable latency experiments.

Provide a tokenizer matching the model; no tokenizer is now distributed.
Preserved conversion entry points:

```sh
python3 -m pip install -r tools/requirements.txt
python3 tools/tokenizer.py -t /path/to/tokenizer.model
python3 tools/export.py model-fp32.bin --meta-llama /path/to/original --version 0
python3 tools/export.py model-int8.bin --meta-llama /path/to/original --version 2
```

Tokenizer export writes alongside the .model input. Hugging Face import/export
also needs transformers. All original converter formats remain, but C inference
accepts only legacy v0 FP32 and ak42 v2 Q8_0; v1 and HF-format output are not read
by these binaries. Python conversion fixes include the previously undefined
variables in Tokenizer.encode, zero-block quantization, preserving positive
model vocabulary metadata during legacy export, and ensuring group size divides
both matrix input dimensions.

## Benchmarking and report schema

```sh
mkdir -p results
build/release/runq model-int8.bin -z tokenizer.bin \
  --benchmark --prompt-file prompt.txt --seed 42 -t 0 \
  --generation-length 128 --no-text --log-level warn --json results/baseline.json
python3 -m json.tool results/baseline.json
```

--json writes a separate file. It must not alias a checkpoint, tokenizer or
prompt input (including hardlinks/symlinks). Parent directories must exist. JSON
is schema version 1, with UTC completion/report timestamp; model path/config and
precision; OS/architecture; compiler/flags; parameters; token counts; seconds;
tokens/second; memory in bytes; completed/error status and stopping reason.
No checkpoint hash is computed, avoiding a full weight read during startup.
Paths identify the model but are not cryptographic model identities. UTF-8 is
preserved in JSON, while invalid bytes are escaped as U+00xx. Prompt content is
included for reproducibility; keep result files appropriately.

Ordinary fatal errors produce status=error when a usable --json path was
provided, and exit nonzero. Unavailable/unmeasured timing or rates are null,
not invented zeros or nonfinite JSON numbers. OS signals, abrupt termination,
and failures writing the report cannot guarantee a report. Initialization-only
errors do not claim measured inference. Failed report writes also exit nonzero.

All elapsed durations use CLOCK_MONOTONIC at nanosecond API resolution. Actual
clock resolution is platform-dependent. No warm-up pass is discarded. Mapping
is lazy: first-touch/page-fault cost belongs to the operation that incurs it.

| Field under timing_seconds | Boundary/definition |
| --- | --- |
| checkpoint_load | Open/stat/read/validate/map plus construction of weight views; INT8 includes its embedding expansion. |
| tokenizer_init | Vocabulary file parsing and allocations, excluding lazy lookup sorting. |
| runtime_init | Activation/KV/scratch allocation plus sampler initialization, excluding tokenizer and weight setup. |
| prompt_tokenization | Token buffer allocation and encode, including first-use vocabulary sort; chat includes template assembly. |
| prefill | Sum of forward-call durations for supplied prompt/template tokens, including the last prompt position. Sequential, never represented as batched. |
| decode | Sum of forward-call durations for previously sampled tokens; chat includes the stop-marker cache pass. Sampling and printing are excluded. |
| total_inference | From tokenization start through generation completion, including sampling and text output; chat sums active turns plus stop-marker caching, excluding input waits. |
| end_to_end | Main entry through final runtime cleanup before report serialization; includes file prompt reading and chat input waits. The file-prompt buffer is retained until reporting finishes. |
| time_to_first_token | Tokenization start to availability of the first non-stop output token, before printing it. Null if none. Chat refers to the turn producing the first output. Excludes model/tokenizer setup. |
| average_time_per_generated_token | total_inference / non-stop output count; includes prefill, tokenization, sampling and output overhead. Null for zero output. |
| average_decode_step | decode / actual decode forward passes. Null if none. |

Timing components are nested measurements: do **not** add total_inference to its
subcomponents, or add end_to_end to anything. Forward timings deliberately omit
sampling/printing overhead; total_inference includes them. Clock readings and
loop bookkeeping introduce small measurement overhead.

Token counts distinguish input_prompt (includes BOS and chat template),
processed_prompt (actual prompt forward passes), generated_output (non-stop
sampled tokens, including pieces filtered from display), sampled and sampled_stop,
forward_passes and decode_steps. Always:
`forward_passes = processed_prompt + decode_steps` and
`sampled = generated_output + sampled_stop`. The first output uses final-prompt
logits, so a three-token completion normally has two decode steps; an EOS after
those tokens can require a further decode step. Stops are not output tokens.

Throughput fields are prefill=processed_prompt/prefill seconds,
decode=decode_steps/decode seconds, overall_forward=forward_passes/total_inference,
and output=generated_output/total_inference. Ratios require positive measured
time and positive counts; otherwise null. Decode throughput is based on executed
passes, not output count. A zero-generation request has null prefill/decode/TTFT.

Memory fields distinguish mapped file size from explicit allocation payloads and
peak resident bytes. Category fields are simultaneous per-category allocation
peaks; explicit_allocations_peak is the maximum simultaneous total, not the sum
of category peaks. Allocator headers/alignment, libc, stdio, OpenMP internals and
stack memory are excluded from explicit payload counts. getrusage peak RSS is a
process-wide high-water mark (macOS bytes; Linux KiB converted to bytes), not a
sum of mappings and allocations. Mapping size is not resident memory.

A reproducible small report example can be regenerated without model downloads:

```sh
python3 tests/fixtures.py build/fixtures
build/release/run build/fixtures/fp32.bin -z build/fixtures/tokenizer.bin \
  --benchmark -i A -t 0 --generation-length 3 --no-text --json results/synthetic.json
```

benchmark-example.json records an actual such run. It is a schema example and
correctness fixture, not representative model-performance evidence.

## Compatibility boundaries

The two supported formats store seven 32-bit dimensions, no extensible model
metadata. Only little-endian 32-bit-int/float hosts are supported. FP32 v0 uses
vocabulary sign for classifier sharing and includes unused RoPE tables; Q8_0 v2
has a 256-byte header and separate unpadded scales per tensor. Invalid dimensions,
head divisibility/even head sizes, extent overflows, incomplete files, invalid
classifier flags and unsupported INT8 grouping/alignment are rejected. Group
size divides dim and hidden_dim, is bounded for int32 accumulation, and tensors
must keep FP32 scale alignment. Extra checkpoint bytes warn; tokenizer vocabulary
must match exactly. Files must remain unchanged while mapped. No file checksum
exists; payload corruption cannot always be identified, though nonfinite values
encountered in normalization/sampling/quantization fail with diagnostics.

Unresolved original model assumptions remain explicit: RoPE theta 500000,
RMSNorm epsilon 1e-5, adjacent rotary pairs, checkpoint context length,
BOS=128000/EOS=128001/EOT=128009, original LLaMA 3 chat header IDs and role tokens.
No RoPE scaling or LLaMA 3.1/3.2 compatibility is claimed. The C tokenizer keeps
its original UTF-8 lookup, byte fallback offset 3 and highest-score pair/triple
merging; it is **not** a verified implementation of tiktoken's regex/rank rules.
Its accuracy versus a real LLaMA tokenizer still needs model-specific validation.
Python ModelArgs/loaders also retain older defaults (including RoPE/context and
HF KV-head assumptions); format metadata cannot convey all architecture options.
Use the established Meta-original workflow, and validate every real conversion
before research measurements. This phase does not certify all existing import
paths as LLaMA 3.x compatible.
