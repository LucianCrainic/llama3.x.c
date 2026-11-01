# Phase 2: configuration-driven Llama 3 text inference

Status and executed checks are in [phase2-validation.md](phase2-validation.md).
README.md remains empty. Phase 1 metric definitions in [phase1.md](phase1.md)
still apply; this document supersedes its compatibility/conversion limitations.

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
`tools/diagnose_quantization.py` can compare a failed INT8 case against HF with
independent weight/input fake quantization, without changing that gate.

`cresco8-phase2.sbatch` runs only within Slurm. Supply the partition and a node
whose current mode has been checked; the script additionally requires ACPI
memory-side cache evidence on the allocated dual-Xeon-Max node. It records
hardware, affinity, scheduler allocation, compiler/flags, source hashes, Git base
and working diff, and Python packages. Uncommitted deployments are explicitly
identified by base commit plus source manifest, not represented as a clean commit.

Submit from a cluster checkout with authorized model files and a Python conversion
environment. Example placeholders must be replaced with verified local values:

```sh
mkdir -p results
export HF_MODEL=/models/snapshot ARTIFACTS=/workspace/converted/model
export OUT=/workspace/results/unique-run PYTHON=/path/to/python
sbatch --partition=VERIFIED_PARTITION --nodelist=VERIFIED_CACHE_NODE \
  --exclude=VERIFIED_FLAT_NODE --exclusive \
  -o results/job-%j.stdout -e results/job-%j.stderr scripts/cresco8-phase2.sbatch
```

Set `SMOKE_ONLY=1` for conversion/reference/smoke validation without a benchmark.
`WORKLOADS=long` can rerun only the longer workload; `REPEATS` defaults to three.

Default allocation: one node, eight CPU threads, 128 GiB requested memory,
two-hour cap; conventional GCC -O3/OpenMP without native/fast-math flags. The
script builds/tests, converts once, validates, executes a short greedy smoke
request, then short/long prompts with up to 128 outputs and three repeats each.
It writes text/stdout/stderr and JSON separately, propagates every failure, and
uses a unique results directory. Short/long lengths are measured, not assumed.
EOS may produce fewer than 128 outputs. The 2048 context capacity avoids a 128K
KV allocation. No memory binding, migration or HBM-specific kernel is introduced.

Reports retain schema 1 with additive model/head/RoPE/shared-weight/parameter-count
and hostname/kernel fields. `affinity_cpus` is the reporting thread's mask; with
OpenMP binding it may contain one CPU even when worker threads use the full
Slurm task allocation. Interpret it alongside the task CPU allocation, recorded
OMP_PLACES/OMP_PROC_BIND settings and `openmp_max_threads`. Model count excludes quantization scales and unused
legacy tables. Initialization, sequential prefill, executed decode passes and
TTFT retain Phase 1 boundaries. See phase1.md for precise definitions.
`tools/summarize.py results/run` emits means and sample standard deviations plus
raw paths and actual token counts. By default, the script copies the converted checkpoint to a temporary node-local
file, records copy time separately and verifies its SHA256. Validation/smoke warm
that file before repeated benchmarks. The temporary copy is removed at job exit;
the source artifact remains reusable. This controls filesystem effects without
changing memory placement. Set LOCAL_STAGE=0 to measure the shared-filesystem
path instead, with uncontrolled first-touch conditions. No cache eviction or
cold-cache claim is made. Memory-side cache visibility establishes the mode,
not HBM hit rates, measured bandwidth or explicit HBM allocation.
