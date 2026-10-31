# Phase 1 validation record

Validated on 2026-10-31. These are functional checks, not research benchmarks.

## Baseline

Original commit 10b7531 was built with Apple Clang -O2 -Wall -Wextra
-Wno-unknown-pragmas -lm, before edits. Both engines built, each with four
warnings recorded in audit.md. No real model checkpoint was available.

Before refactoring, a harness executed four forward passes at token IDs 65..68
and positions 0..3 on deterministic FP32 and Q8_0 fixtures. The first twelve
logits per position are stored in tests/baseline_logits.json. Refactored engines
are compared against these 48 values each with absolute tolerance 2e-6 or relative
tolerance 2e-5. This covers attention/KV state across positions, RoPE, normalization,
FFN, classifier and quantization. It does not establish real-language quality.

## Executed checks

Host: macOS Darwin 25.6 ARM64; Apple Clang 21.0.0; Python 3.14.2.

- Release, debug, profile and ASan/UBSan builds of both engines compile cleanly
  with -Wall -Wextra -Wpedantic.
- `make test`: 13 runtime test groups and two converter smoke groups pass.
- `make test MODE=sanitize`: the same 15 groups pass without ASan/UBSan reports.
- README size verified as zero bytes; git diff --check passes.

Linux: local Docker Desktop, Ubuntu 22.04 ARM64 image
mcr.microsoft.com/devcontainers/base:jammy, GCC 11.4.0, Python 3.10.12.
The repository was mounted read-only; binaries and fixtures used container /tmp.

- Release builds and the same 15 test groups pass without compiler warnings.
- GCC ASan/UBSan builds and all 15 groups pass with default sanitizer options.
- OpenMP release builds and all 15 groups pass with OMP_NUM_THREADS=2.
  This run exposed concurrent fatal-report writes on corrupt input; fatal
  reporting now elects a single thread with a C11 atomic flag, and the OpenMP
  suite was rerun successfully after the fix.

## Test coverage

The runtime groups use dependency-free synthetic fixtures with shared and
separate classifier weights, GQA, nonzero weights and all-zero quantization
blocks. They check original-logit agreement, RMSNorm/softmax, seeded sampling,
triple tokenizer merges, allocation arithmetic overflow, resource release on
normal completion, parsed JSON and Unicode, prompt files, no-text behavior,
zero/one-token outputs, partial prompts, first-sample and later EOS, memory sizes,
timing/rate consistency, malformed/truncated/missing model and tokenizer files,
invalid CLI options, nonfinite checkpoint payloads, long chat prompts/EOF,
and report/input alias protection. C assertions stay enabled in test builds.

The two converter smoke checks compile all Python source and execute the real
Tokenizer.encode wrapper with a stub backend to verify delegation/BOS/EOS.
Torch, NumPy and tiktoken were not installed; model/tokenizer conversion with
those packages was not executed. Existing converter loading paths need separate
real-checkpoint verification.

## Remaining validation limits

No real LLaMA checkpoint, model-output comparison, licensed reference tokenizer,
or language-quality test was available. Linux validation is ARM64, not the
future Intel Xeon Max target. HBM/NUMA policy, socket scaling and real-model
performance are outside this phase. Apple Clang OpenMP was not validated because
libomp is not installed. Sanitizers do not certify that arbitrary mapped weights
are semantically valid; payload formats have no checksum.
