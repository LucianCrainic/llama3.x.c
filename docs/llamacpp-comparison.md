# llama.cpp CPU comparison

This campaign compares the phase-2 FP32 C baseline with llama.cpp commit
`dde879bc4f954b809892fd0d660515f42536394a`, using the same official Llama
3.2 1B Instruct and Llama 3.1 8B Instruct snapshots. The C baseline is documented
in [phase2-validation.md](phase2-validation.md).

HF snapshot revisions: 1B `9213176726f574b556790deb65791e0c5aa438b6`;
8B `0e9e39f249a16976918f6564b8830bc894c89659`.

## Batched prefill results

Measured on 2026-11-01. Each row uses 128 output tokens and three repeats.
Throughput is mean ± sample standard deviation; RSS is the largest process
high-water mark. llama.cpp uses batch/ubatch 512.

| Model / threads | Input | Engine | Prefill tok/s | Decode tok/s | Peak RSS GiB |
| --- | ---: | --- | ---: | ---: | ---: |
| 3.2 1B / 8 | 86 | C baseline | 5.61305 ± 0.00135 | 5.65254 ± 0.00304 | 4.74 |
| 3.2 1B / 8 | 86 | llama.cpp | 224.431 ± 0.120 | 21.313 ± 0.002 | 4.82 |
| 3.2 1B / 8 | 491 | C baseline | 5.59277 ± 0.00217 | 5.55627 ± 0.00204 | 4.74 |
| 3.2 1B / 8 | 491 | llama.cpp | 248.836 ± 0.367 | 20.795 ± 0.007 | 4.85 |
| 3.1 8B / 32 | 86 | C baseline | 2.74196 ± 0.00236 | 2.75094 ± 0.00231 | 28.22 |
| 3.1 8B / 32 | 86 | llama.cpp | 88.595 ± 2.397 | 4.836 ± 0.068 | 30.53 |
| 3.1 8B / 32 | 491 | C baseline | 2.73742 ± 0.00039 | 2.69949 ± 0.00023 | 28.35 |
| 3.1 8B / 32 | 491 | llama.cpp | 63.859 ± 4.275 | 4.834 ± 0.007 | 30.59 |

Ratios of the measured means:

| Model / threads | Input | Prefill speedup | Decode speedup |
| --- | ---: | ---: | ---: |
| 3.2 1B / 8 | 86 | 39.98× | 3.77× |
| 3.2 1B / 8 | 491 | 44.49× | 3.74× |
| 3.1 8B / 32 | 86 | 32.31× | 1.76× |
| 3.1 8B / 32 | 491 | 23.33× | 1.79× |

Peak RSS rises by approximately 0.08–0.12 GiB for 1B and 2.25–2.31 GiB for
8B. The RSS comparison includes loader, residency, workspace, and runtime
differences; it is not simply a comparison of checkpoint file sizes.

## Sequential prefill diagnostic

Batch/ubatch 1 retains the C engine's sequential strategy. Each row again uses
three repeats and 128 outputs. All repeated runs and both batch settings
produced identical 128-token continuations for each model and workload.

| Model / threads | Input | Prefill tok/s | Decode tok/s | Peak RSS GiB | Prefill speedup over C |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.2 1B / 8 | 86 | 21.267 ± 0.021 | 21.285 ± 0.007 | 4.79 | 3.79× |
| 3.2 1B / 8 | 491 | 21.145 ± 0.025 | 20.776 ± 0.024 | 4.79 | 3.78× |
| 3.1 8B / 32 | 86 | 4.899 ± 0.006 | 4.893 ± 0.003 | 30.48 | 1.79× |
| 3.1 8B / 32 | 491 | 4.888 ± 0.003 | 4.792 ± 0.069 | 30.48 | 1.79× |

Sequential prefill is about 3.8× faster for 1B and 1.79× for 8B. These
measurements separate the remaining CPU kernel/runtime/build advantage from
the much larger additional benefit of processing the prompt in a batch.

## Method

- Use the same Intel Xeon CPU Max 9480 Cache Mode hardware and exclusive Slurm
  allocation policy. Verify the memory-side cache in sysfs before benchmarking.
  Use 8 threads for 1B and 32 for 8B, with core binding and no explicit NUMA
  memory placement.
- Convert each existing HF snapshot with llama.cpp's own `convert_hf_to_gguf.py
  --outtype f32`. Run CPU-only, with FP32 K/V, context capacity 2048, and flash
  attention disabled. No quantized or half-precision model is substituted.
- Build the native CPU backend with GCC 14, Release optimization, OpenMP,
  `GGML_NATIVE=ON`, and `GGML_BLAS=OFF`. The original C baseline uses GCC 11 and
  `-O3`; compiler and instruction-set choices differ and are part of the engine
  configurations being compared.
- Render the original prompts with the fixed `26 Jul 2024` HF chat template.
  Compare the complete HF, C, and llama.cpp prompt token sequences, including
  BOS, system prefix, chat headers, and generation header. Actual input lengths
  must be 86 and 491; equal token counts alone are insufficient.
- Greedy generation must reach exactly 128 non-stop output tokens naturally.
  Do not ignore or pad EOS. Time prefill separately from the 127 subsequent
  single-token forward passes. Exclude tokenization, sampling, and model loading
  from both throughput denominators, as in the C baseline.
- Run llama.cpp through a small public-API harness rather than synthetic
  `llama-bench` tokens. Batch/ubatch 512 measures batched prompt processing;
  batch/ubatch 1 additionally measures the C engine's sequential prefill
  strategy, including a vocabulary projection at every prompt position.
- Stage GGUF weights to node-local storage and verify checksums. For each model,
  workload, and batch setting, run one discarded warm-up process followed by
  three fresh measured processes with empty KV state. File cache is warm; no
  cache eviction is attempted. This is not a cold-start comparison.
- Report arithmetic means and sample standard deviations for throughput, and
  maximum process RSS across three repeats. Read Linux `ru_maxrss` and verify
  against GNU `/usr/bin/time -v`; use GNU time's final high-water mark for the
  aggregate. Include resident mapped weights, tokenizer, KV, workspace, and
  process/library overhead. Exclude the separate conversion and staging
  processes. GiB means 2^30 bytes.

The benchmark and preparation sources are
[llamacpp-compare.cpp](../scripts/llamacpp-compare.cpp),
[llamacpp-prepare.py](../scripts/llamacpp-prepare.py), and
[llamacpp-compare.sbatch](../scripts/llamacpp-compare.sbatch).
[summarize_llamacpp.py](../tools/summarize_llamacpp.py) validates and aggregates
the downloaded reports against `results/cresco8/phase2-summary.json`.

llama.cpp's [benchmark documentation](https://github.com/ggml-org/llama.cpp/blob/dde879bc4f954b809892fd0d660515f42536394a/tools/llama-bench/README.md)
describes its usual prompt-processing and token-generation tests. Here real
prompts and a single continuous KV history are used to match the C workload.

## Validation and provenance

All 24 measured reports pass the input-ID, output-count, affinity, precision,
context, and independent RSS checks. Both main benchmark jobs exit successfully.
Eight additional benchmark processes are discarded warm-ups. For each of the
four model/workload combinations, all repeats and both prefill modes produce
the same 128 output token IDs.

A separate sanity check runs the raw prefix `The capital of Italy is` and
compares eight greedy tokens against the saved, independently matching HF/C
reference from the original numerical validation. Both models match all eight
IDs with batch 512 and batch 1. This is a short-prefix conversion/runtime check;
it does not establish full-logit equivalence over the performance prompts.

Successful benchmark jobs are `5621106` (1B) and `5621108` (8B); the completed
reference check is `5621121`. Raw JSON, GNU time reports, stderr/performance
logs, actual job scripts, cache-mode evidence, model hashes, and build metadata
are saved locally under ignored `results/cresco8/llamacpp-comparison/`.
`summary.json` contains all eight aggregated configurations and speedups;
`sanity/sanity.json` contains the four reference checks. The measured C++ and
prompt-preparation source hashes match the workspace files.

## Interpretation limits

This is a comparison of the chosen FP32 CPU configurations, rather than a
search for either engine's best thread count or precision. Batched prefill also
avoids computing unused vocabulary logits at intermediate prompt positions;
its speedup includes that optimization as well as matrix batching. Matching
greedy policy does not guarantee identical continuations across floating-point
implementations. Repeats measure variation within one allocation, not across
days or nodes. Differences from the earlier C measurements can also include
allocation and compiler effects.
