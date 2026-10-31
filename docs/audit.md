# Phase 1 audit

Baseline: commit `10b7531`. Both single-file engines compiled with Apple Clang
using `-O2 -Wall -Wextra -Wno-unknown-pragmas -lm`. Each emitted four warnings:
signed layer comparison, unused decode argument, signed tokenizer comparison,
and ambiguous chat termination precedence. No real model checkpoints were
present. Numerical regression uses deterministic synthetic weights instead.

Removed Windows mmap shims/build script and the duplicated DLL engine/interface:
the supported surface is now the Linux/macOS FP32 and INT8 CLI. Removed the
example JPEG and old README content; README intentionally remains empty.
Preserved LICENSE and source attribution, including Meta's tokenizer notice.
Conversion code and its model definition moved together to tools/.

Removed the bundled tokenizer.bin: it has no standalone licensing/provenance
record, and is model-specific rather than a universal test asset. Supply a
matching tokenizer with -z, or export one from a licensed tokenizer.model using
tools/tokenizer.py. Tests create their own small synthetic vocabulary.

Observed defects included unchecked checkpoint extents/dimensions and integer
products, unchecked tokenizer lengths/allocations, insufficient buffer space for
triple merges, fixed-size chat buffers, and division by zero in all-zero INT8
quantization blocks. The original timer used wall time, omitted the first pass,
and mixed prompt and generated tokens in throughput.

The Python model also contains training helpers. These remain because removing
them is not necessary for the inference baseline and could affect external
checkpoint workflows. Converter formats v1 and Hugging Face export remain
available, but the C engines consume only legacy FP32 v0 and Q8_0 v2.
