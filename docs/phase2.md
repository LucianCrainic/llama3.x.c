# Phase 2: configuration-driven Llama 3 text inference

Status and executed checks are in [phase2-validation.md](phase2-validation.md).
README.md is intentionally empty; this document covers building, conversion,
running and the report schema. Kernels and their measured performance are in
[kernels.md](kernels.md).

## Architecture and limits

The FP32 and repository Q8_0 engines share validated metadata, RMSNorm epsilon
and RoPE implementation. Adjacent Q/K rotary pairs, causal single-token attention,
GQA, SwiGLU, KV layout and matrix operation order are preserved. HF Q/K projection
rows are unpermuted separately using query and KV head counts. Tied embeddings
are stored once; independent classifiers remain separate.

The intended targets are Llama 3 8B, Llama 3.1 8B, and Llama 3.2 1B/3B text models,
base and instruct. Code paths supporting an architecture are distinct from
validation of an actual released checkpoint. No larger, vision, bias-bearing,
partially rotary, independent-head-dimension, externally quantized or tool-calling
architecture is certified. Unknown RoPE types and incompatible shapes fail.

The current runtime requires `hidden_size = num_attention_heads * head_dim`,
an even head dimension, and a query head count divisible by KV head count.
Every extent is checked before weight views or activation allocation. INT8 group
size must divide both matrix input dimensions and keep scales aligned; this is
this repository's weights-and-activations Q8_0 implementation, not GGUF, GPTQ,
AWQ or another framework's INT8 format.

`--context-length N` controls KV capacity independently of supported model
context. Default capacity is min(model context, 4096); larger capacities must be
requested explicitly and cannot exceed metadata. The context does not change
RoPE parameters. `-n` still caps total forward passes including prompt processing;
`--generation-length` separately caps non-stop output. Prefill remains sequential.

## Builds and conversion

A C11/POSIX compiler, libm, pkg-config and PCRE2-8 development files are needed.
PCRE2 provides the official Unicode regex split without requiring Python at run
time. Its internal allocations are included in process RSS, while explicit
allocator counters cover only the runtime-owned payloads. Install the system's `pcre2-devel` or equivalent package. The default
`PCRE2=1` build supports versioned tokenizers; `make PCRE2=0` retains dependency-free
legacy inference and explicitly rejects new tokenizers. OpenMP remains optional.

```sh
make CC=gcc OPENMP=1
make test CC=gcc OPENMP=1
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements.txt
make test-reference PYTHON=.venv/bin/python
```

Use a local authorized HF snapshot with config.json, weights and tokenizer files.
No model downloads, credentials or weights belong in the repository.

```sh
.venv/bin/python tools/export.py checkpoints/model-fp32.bin --hf /models/snapshot --version 3
.venv/bin/python tools/export.py checkpoints/model-int8.bin --hf /models/snapshot --version 3 --int8
.venv/bin/python tools/tokenizer.py --hf /models/snapshot -o checkpoints/tokenizer.bin
build/release/run checkpoints/model-fp32.bin -z checkpoints/tokenizer.bin \
  --context-length 2048 --chat-template -i 'Explain what a CPU does.' -t 0 \
  --generation-length 128 --json results/example.json
```

Parent directories must exist. Conversion widens source weights to FP32, checks
all tensor shapes and finite values, preserves norm/attention/RoPE metadata, and
supports shared or independent outputs. HF models are shape-instantiated on the
meta device to avoid a second initialized model copy. FP16 C output is rejected;
FP32 and `--int8` are explicit alternatives. v3 is the default.

Legacy v0 FP32 and v2 Q8_0 loading remains available with their historical theta,
epsilon and special IDs. Export to old formats rejects scaled RoPE or nonlegacy
normalization/theta metadata rather than discarding it. Meta consolidated import
preserves tensor-derived FFN size and supplied metadata, but ambiguous
`use_scaled_rope` without an explicit scaling dictionary fails: use the HF
snapshot for Llama 3.1/3.2. v1 output is historical and has no C reader.

### Build modes

Python 3.8+ is needed only for tests; Python ML packages are needed only for
conversion and reference validation.

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
No build requires OpenMP or a CPU-specific flag: AVX-512 kernels are selected at
run time (see [kernels.md](kernels.md)). GCC on Linux supports OPENMP=1; Apple Clang needs an
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

## Running

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

## RoPE

For pair dimension j and head size H, f = theta^(-j/H), wavelength w = 2*pi/f.
Default scaling keeps f. Llama3 scaling uses the checkpoint's factor F,
low/high factors L/U and original context C:

- w < C/U: f is unchanged.
- w > C/L: f becomes f/F.
- Otherwise, s = (C/w - L)/(U-L), and f becomes (1-s)*f/F + s*f.

The same angle position*f rotates Q and K. Scaling factor is read from metadata;
it is not assumed to be 8 for every model. No dynamic context-dependent scaling
or change to theta substitutes for this algorithm. Reference implementation:
[Transformers v4.57.1 RoPE utilities](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py).

## Checkpoint v3 (little endian)

The header is exactly 256 bytes. Integer/float fields are 32 bits unless noted;
no C structure is dumped to disk. Payload size must match exactly, unlike legacy
files which tolerate trailing bytes with a warning.

| Byte offset | Field |
| --- | --- |
| 0, 4 | magic 0x616b3432, version 3 |
| 8..32 | dim, FFN dim, layers, query heads, KV heads, vocabulary, max context |
| 36, 40, 44 | precision (0 FP32 / 1 Q8_0), shared classifier (0/1), group size (0 for FP32) |
| 48, 52, 56 | RMSNorm epsilon, theta, RoPE type (0 default / 1 llama3) |
| 60, 64, 68, 72 | factor, low factor, high factor, original context |
| 76, 80 | max context repeated for consistency, head dimension |
| 84, 88, 92, 96 | BOS, EOS, EOT, EOM IDs; optional EOT/EOM use -1 |
| 100..227 | NUL-terminated model identifier, at most 127 UTF-8 bytes |
| 228..255 | reserved zeros, rejected if nonzero |

FP32 payload follows v0 ordering: embedding; all attention norms; all Wq, Wk,
Wv, Wo by tensor kind/layer; all FFN norms; all W1, W2, W3; final norm; optional
classifier. The obsolete legacy frequency tables are omitted.

Q8_0 payload follows v2 ordering: attention norms, FFN norms, final norm in FP32;
then embedding, Wq/Wk/Wv/Wo/W1/W2/W3 by kind/layer, optional classifier. Each matrix
stores flat int8 bytes followed by FP32 per-group scales, without padding.
Zero groups use zero scales/values. Activations are quantized during inference.

The format checks shape, header semantics, precision and total extent. It does
not contain a payload checksum; arbitrary finite weight bit flips cannot be
recognized solely from shape checks. Preserve hashes/manifests with artifacts
and never change files while mapped.

## Tokenizer and chat

Versioned tokenizers use the exact Llama 3 regex and smallest-rank adjacent byte
pair merging, with leftmost tie breaking and no cross-piece merges. Binary
lengths support every byte, including zero. Decoded pieces are written with their
lengths, preserving UTF-8 split across tokens. Prompt inputs must be valid UTF-8
without NUL (the existing CLI input contract). Literal special-token spellings
in user text are ordinary bytes; only the chat formatter inserts special IDs.

HF tokenizer.json conversion reverses the ByteLevel alphabet and reads special
IDs from added_tokens. Unsupported normalizers/pretokenizers and chat templates
fail. A model v3 requires a matching versioned tokenizer; a legacy vocabulary
with the same size cannot silently substitute for it.

Tokenizer header: magic 0x336b6f74, version, vocabulary count, base rank count,
max token byte length, BOS, EOS, EOT, EOM, start-header, end-header, and prefix
length (12 little-endian 32-bit words). Current version 2 adds prefix UTF-8 bytes
immediately after the header. Version 1 is accepted with zero prefix length.
Then each token in ID order stores FP32 rank equal to ID, uint32 byte length and
raw bytes. Base IDs are contiguous. Duplicate pieces/missing byte ranks fail.
The unversioned tokenizer reader is retained only for old checkpoints.

`--chat-template` provides one noninteractive instruct request, including in
benchmark mode. `-m chat` supports interactive turns. Role names and separators
are encoded with the selected vocabulary; special IDs come from its header.
The converter renders the official HF template for a basic system/user exchange
and stores its system prefix. Llama 3.1/3.2 knowledge/date headers therefore match
that snapshot. For reproducibility, the exported date defaults to `26 Jul 2024`;
`tools/tokenizer.py --chat-date '31 Oct 2026'` can explicitly change it. Pass the
same date to validate.py. This is prompt data, not the runtime wall clock.

Content is stripped according to Python's Unicode whitespace set. Empty system
messages are omitted unless the exported template supplies a prefix. EOS/EOT/EOM
terminate a response; interactive chat caches EOT before the next turn and exits
on other stop markers. Tool execution and advanced chat schemas remain excluded.
Meta's [tokenizer reference](https://github.com/meta-llama/llama-models/blob/main/models/llama3/tokenizer.py)
provides the base regex/message format; HF snapshot templates define the actual
model-specific system prefix.

## Correctness and reproducible experiments

`make test` retains Phase 1 fixed-logit, loader, sampler and CLI regressions.
`make test-reference` requires Python ML packages and compares small random
Transformers Llama models with GQA, head sizes 64/128, tied/untied projections,
custom epsilon, default and factors 8/32 RoPE. It compares all vocabulary logits
at multiple prefix lengths, greedy continuations, RoPE frequencies, malformed
v3 files, tiktoken multilingual/random prompts and basic chat templates. Synthetic
results under results/ are ignored and do not certify released models.

```sh
.venv/bin/python tools/validate.py --hf /models/snapshot \
  --checkpoint checkpoints/model-fp32.bin --tokenizer checkpoints/tokenizer.bin \
  --output results/correctness.json
```

The validation tool compares FP32 logits (default atol/rtol 1e-3), tokenizer IDs,
actual HF instruct templates, and greedy continuation in raw and instruct modes.
Q8_0 requires an explicitly chosen tolerance (`--precision int8 --max-abs ...
--rtol ...`); the absolute bound is the allclose component (atol), combined pointwise with
`rtol * abs(reference)`. It reports max absolute/RMS errors, top-1 agreement and token
sequence divergence separately. It never treats readable prose as numerical
validation. A failing result returns nonzero and blocks the benchmark script.

Reports retain schema 1 with additive model/head/RoPE/shared-weight/parameter-count
and hostname/kernel fields. `affinity_cpus` is the reporting thread's mask; with
OpenMP binding it may contain one CPU even when worker threads use the full
Slurm task allocation. Interpret it alongside the task CPU allocation, recorded
OMP_PLACES/OMP_PROC_BIND settings and `openmp_max_threads`. Model count excludes quantization scales and unused
legacy tables. Initialization, sequential prefill, executed decode passes and
TTFT follow the boundaries in [Benchmark reports](#benchmark-reports).

The Phase 2 Cache Mode campaign (results in
[phase2-validation.md](phase2-validation.md)) copied each converted checkpoint to
a temporary node-local file, recorded the copy time separately and verified its
SHA256. Validation and smoke runs warmed that file before repeated benchmarks.
This controlled filesystem effects without changing memory placement; no
cache-eviction or cold-cache claim was made. Memory-side cache visibility
established the mode, not HBM hit rates, bandwidth or explicit HBM allocation.
Flat-mode experiments stage checkpoints into `/dev/shm` under an explicit
`numactl` policy instead (see [kernels.md](kernels.md)).

## Benchmark reports

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

`build.kernels` names the active kernel set (`scalar` or `avx512`; see
[kernels.md](kernels.md)). With `LLAMA_PROFILE=1`, `profile_seconds` holds, per
stage (`prefill`, `decode`), the forward time, the forward pass count, wall time
per forward-pass phase, and `other` (forward time outside every phase);
otherwise it is null. Phase timings use the same clock and are nested inside
the stage's forward time.

A small report can be regenerated without model downloads:

```sh
python3 tests/fixtures.py build/fixtures
build/release/run build/fixtures/fp32.bin -z build/fixtures/tokenizer.bin \
  --benchmark -i A -t 0 --generation-length 3 --no-text --json results/synthetic.json
```
