# Phase 2 validation record

Phase 2 implementation and the first FP32 Cache Mode baselines are available.
Real 3.2-1B and 3.1-8B Instruct FP32 reference/smoke tests passed. Full acceptance
remains pending real 3B weights and further 8B INT8 qualification.

## Local checks (2026-10-31)

Host: Apple ARM64/macOS, Apple Clang 21, Python 3.12.12 for ML references and
Python 3.14.2 for dependency-free tests. PCRE2 10.47, PyTorch 2.15.0,
Transformers 4.57.1, tiktoken 0.12.0.

- `make test`: 14 runtime groups plus two converter groups pass.
- `make test-reference PYTHON=.venv/bin/python`: four integration groups pass.
- ASan/UBSan runtime and reference suites pass.
- Legacy `PCRE2=0` builds/tests pass (new-tokenizer tests require PCRE2).
- Empty README and `git diff --check` verified.

Synthetic Transformers models exercise 64/128-dimensional heads, GQA ratios
4/3, tied and untied classifiers, custom RMSNorm epsilon, default RoPE and
Llama3 factors 8/32. All vocabulary logits are compared at prefix lengths 1, 4
and 17. FP32 maximum absolute error across those cases: 1.847744e-6; maximum RMS
error: 4.280524e-7; top-1 agreement: 100%. Six-token greedy sequences agree.

Q8_0 synthetic maximum absolute error: 0.03519643; maximum RMS error: 0.00929740.
Minimum per-case top-1 agreement is 88.24%. Quantized output is deliberately
reported separately from FP32; equivalent greedy text is not promised.
These are random-model errors, not release-model accuracy estimates.

Additional checks cover malformed/unsupported/truncated v3 headers and payload
extents, precision mismatch, context caps and reported cache sizes, EOS/EOT/EOM
stopping in both engines, binary-safe BPE pieces, Unicode whitespace, and regex
boundaries against tiktoken on multilingual and 100 seeded randomized prompts.
Real cached 3.2-1B and 3.1-8B tokenizer metadata convert locally. Both match
tiktoken/HF on 42 ordinary prompts each and three fixed-date instruct templates
with empty, explicit and Unicode-trimmed system messages.
Literal special spellings are deliberately encoded as ordinary input bytes.

## Cluster environment and provenance

A Slurm-only environment probe (job 5612410) executed on cresco8-hbm01.
Verified hardware: two Intel Xeon Max 9480 sockets, 56 cores/socket, 112 online
CPUs, two visible DDR NUMA nodes. ACPI sysfs exposes two memory-side caches of
68,719,476,736 bytes each (128 GiB total), supporting the Cache Mode designation.
No experiment targeted the Flat Mode node. No HBM/DDR placement was applied.

Partition cresco8_hbm was read from current scheduler state. hbm13 was occupied
by existing jobs, which were left alone. GCC 11.4.1 and PCRE2 10.40 are available;
Intel oneAPI is listed among available modules, but the baseline uses GCC.
Build flags: C11, -Wall -Wextra -Wpedantic -O3 -fopenmp, no native/fast-math.
Python conversion environment: PyTorch 2.8.0+xpu (CPU execution), Transformers
4.55.0, tiktoken 0.10.0. No package or shared environment was modified.

Sources were cloned from develop at
`c57fd23fe247b88f2af9811d344898c686094124` and overlaid with an uncommitted
Phase 2 source archive. Those runs used **uncommitted source snapshots**. Every run saves
the full Git base hash, tracked diff, source SHA256 manifest, compiler recipe,
Python freeze, scheduler/environment data and artifact hashes. Use those files
to identify the precise tested snapshot; the final working tree may include later
validation/documentation fixes.

The first staging archive carried macOS metadata files; a corrected archive
excluded them. Linux/GCC/OpenMP Phase 1 regression tests then passed. An initial
conversion stopped with `Disk quota exceeded` on persistent storage. Only this
run's incomplete checkpoint was removed. Canonical cached HF weights were reused
read-only; subsequent generated conversion caches use ordinary scratch storage,
whose retention policy means they are disposable. A reference-harness OpenMP
issue was found: Torch narrowed the launcher mask
and C children inherited one CPU. Validation now restores the original task mask
and disables launcher binding. Direct Slurm benchmark launches do not inherit
the Python mask. Repeated shared-filesystem mmap waits were observed, so the
benchmark job stages a disposable node-local checkpoint, records copy time and
checks its hash before validation. This is filesystem preparation, not HBM/NUMA
placement. Those initialization/reference attempts stopped before performance
evaluation.

Raw reports are copied locally under ignored `results/cresco8/` and must be preserved separately for research.

## Compatibility matrix

| Configuration | FP32 | Repository Q8_0 | Modes / limits |
| --- | --- | --- | --- |
| Legacy Llama 3-format synthetic fixtures | Regression pass | Regression pass | Generation/chat/JSON; original tokenizer is unverified |
| Small default/scaled Llama architecture references | Numerical + greedy pass | Numerical tolerance pass | Synthetic GQA, RoPE, shared/independent outputs |
| Llama 3.2 1B Instruct, cached official snapshot | Short-prefix numerical + greedy pass; Cache Mode smoke pass | Error budget + greedy/smoke pass | Generation/instruct; full context not certified |
| Llama 3.2 3B base/instruct | Architecture implemented; real weights unavailable | Real weights unavailable | Pending release-model validation |
| Llama 3.1 8B Instruct, cached official snapshot | Short-prefix numerical + greedy pass; Cache Mode smoke pass | Initial FP32-deviation budget failed; further validation needed | Generation/instruct; full context not certified |
| Llama 3 8B and 3.1/3.2 base release weights | Real weights unavailable | Real weights unavailable | No new real-model support claim |
| Vision, larger models, other quantization formats | Unsupported | Unsupported | Explicitly out of scope |

## Actual correctness comparisons

| Official cached Instruct weights | Precision | Maximum absolute logit error | Maximum RMS error | Eight-token greedy comparison |
| --- | --- | ---: | ---: | --- |
| Llama 3.2 1B | FP32 | 8.296967e-5 | 1.389041e-5 | Raw-prefix and instruct both identical |
| Llama 3.1 8B | FP32 | 3.268123e-4 | 2.496521e-5 | Raw-prefix and instruct both identical |
| Llama 3.2 1B | Q8_0 | 0.9933014 | 0.1540874 | Both selected continuations identical |
| Llama 3.1 8B | Q8_0 | 1.411355 | 0.1774905 | Both selected continuations identical; budget failed |

Real-model comparisons use prefixes of 2, 9, 32 and 13 tokens, including Unicode.
FP32 tolerances are atol=1e-3, rtol=1e-3; all compared positions have identical
top-1 IDs. Q8_0 uses a separately declared initial error budget (atol=1.0,
rtol=0.05), not FP32 equivalence. For 1B Q8_0, one of 56 compared positions has
a different top-1 ID (minimum per-prompt agreement 92.31%). This measures
quantization deviations; it is not a perplexity or broad accuracy certification.
For 8B Q8_0, every compared top-1 ID and both short continuations agree, but the
pointwise allclose budget fails. The job exits nonzero before CLI smoke or
benchmarking. No threshold was relaxed to turn that result into a pass.

A follow-up diagnostic (job 5612450) applies independently fake-quantized weights
and Linear inputs to the established HF model using the same 64-element W8A8
rules. On BOS + Hello, that reference itself deviates from FP32 by max 1.410409
(RMS 0.171555), while C deviates by max 1.411300 (RMS 0.177490). C versus this
fake-quantized BLAS reference has max 0.302926, RMS 0.037744 and matching top-1
IDs. This indicates substantial inherent approximation loss, but does not prove
exact grouped-integer kernel equivalence: floating reductions and activation
rounding boundaries differ. Further per-layer/quantized-reference validation is
needed before qualifying 8B Q8_0. The original failed budget result is retained.
`tools/diagnose_quantization.py` reproduces this diagnostic; it never changes the
benchmark gate or its original quality budget.
The JSON labels `base`/`instruct` identify raw/instruct prompt formats using the
same supplied Instruct weights; no Base release weights were tested.

Tokenizer IDs and fixed-date HF templates match. Real 1B/8B smoke tests process
43 inputs; 1B reaches its 16-output cap, while 8B produces 11 outputs then EOT.
An initial 536-input repeated long request naturally stopped after seven 1B
outputs at EOT. It is retained as a diagnostic. The workload was replaced with
coherent reference material and a single detailed-answer request; no completed
response was padded. Synthetic v3 tests separately force EOS, EOT and EOM.
Interactive multi-turn chat has synthetic regression coverage; the real-model
instruct checks use noninteractive single-turn formatting.

## Actual FP32 Cache Mode baseline

Three successful runs per row. Values are mean +/- sample standard deviation;
RSS is the largest process high-water mark among those three runs. The 491-token
workload approximates the requested 512-token input; actual counts are preserved.
All selected runs generated exactly 128 output tokens.

| Model | Precision | Threads | Input | Output | Prefill tok/s | Decode tok/s | Peak RSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama 3.2 1B Instruct | FP32 | 8 | 86 | 128 | 5.61305 ± 0.00135 | 5.65254 ± 0.00304 | 4.74 |
| Llama 3.2 1B Instruct | FP32 | 8 | 491 | 128 | 5.59277 ± 0.00217 | 5.55627 ± 0.00204 | 4.74 |
| Llama 3.1 8B Instruct | FP32 | 32 | 86 | 128 | 2.74196 ± 0.00236 | 2.75094 ± 0.00231 | 28.22 |
| Llama 3.1 8B Instruct | FP32 | 32 | 491 | 128 | 2.73742 ± 0.00039 | 2.69949 ± 0.00023 | 28.35 |

The models use different configured thread counts, so these are separate
operating points, not a controlled model-scaling comparison. Both run on the
same verified Cache Mode node without explicit memory placement. KV capacity is
2048 positions: 128 MiB allocated for 1B and 512 MiB for 8B. Peak RSS includes
resident mapped weights and runtime state; mapped file size is not resident RAM.

Model parameter counts are 1,235,814,400 (1B, tied embeddings, head size 64,
GQA 32/8, scaling factor 32) and 8,030,261,248 (8B, independent classifier,
head size 128, GQA 32/8, scaling factor 8). FP32 checkpoint sizes are
4,943,257,856 and 32,121,045,248 bytes respectively.

Warm mapping/header/view initialization averages about 27 microseconds for 1B
and 29-32 microseconds for 8B. These are not disk-load measurements: separately
recorded node-local copies took 2.99 seconds (initial 1B campaign) and 19.18
seconds (8B). SHA256 values agree between source and staged copies. C mapping is
lazy and first-touch costs remain in prefill. Validation and smoke precede the
repeated benchmark; file-cache conditions are warm and no eviction was attempted.

Prefill and decode rates are close because prompt processing is sequential,
not batched. Longer prompts spend most inference time in prefill. The observed
three-run variation is small; this does not establish stability across days,
other allocations or thread configurations. No memory-bandwidth, compute-bound,
HBM allocation, hit-rate or energy conclusion is drawn.

Selected run provenance: 5612418 supplied the three successful 1B short JSON
reports (the campaign was then intentionally cancelled after the early-EOT
long diagnostic); 5612429 completed the revised 1B long campaign; 5612430
completed both 8B workloads. Q8_0 validation/smoke jobs are 5612436 (1B,
completed) and 5612437 (8B, failed quality gate before smoke). Q8_0 was not
benchmarked as a repeated performance configuration in this phase.

Raw reports and the selected machine-readable aggregate are under ignored
`results/cresco8/`; `phase2-summary.json` references all 12 selected raw reports.
Source archives for tested snapshots are saved there alongside source manifests.
Generated measurements and model weights are intentionally absent from Git.

## Pending acceptance criteria

- Real 1B/8B FP32 short-prefix reference, smoke, and three repeats of both
  workloads have passed. Longer-context numerical certification remains pending.
- Qualify 8B Q8_0 beyond the failed initial deviation budget and its two-token
  fake-quantized diagnostic. Its CLI smoke/performance runs remain blocked; 1B
  Q8_0 conversion, reference comparisons and smoke passed the declared budget.
- Obtain an authorized 3.2-3B checkpoint, plus base/Llama 3 release weights if
  full variant certification is needed.
- Validate long-context behavior beyond short reference prefixes; no full-128K
  correctness or performance experiment is claimed.
- Converted caches currently reside on scratch because persistent storage hit
  its quota. Preserve them in suitable long-term storage when space is available.

## Integration checks (2026-11-01)

The following checks were rerun for the Phase 2 commit and integration into
`develop`:

- Release: `make test` and `make test-reference PYTHON=.venv/bin/python` pass
  (14 runtime groups, two converter groups, four reference groups).
- ASan/UBSan: runtime, converter, and numerical reference suites pass using a
  separate sanitize build directory.
- Legacy `PCRE2=0`: runtime and converter suites pass; the versioned-tokenizer
  runtime group is skipped as intended.
- Both Slurm scripts pass Bash syntax checks; changed Python tools and reference
  tests compile. README remains empty, and the staged diff passes whitespace
  checks.

The commit history records the implementation and its merge. Earlier cluster
measurements retain their original source manifests and snapshot provenance;
committing the implementation does not turn those deployments into clean-commit
runs. Generated checkpoints, raw results, and local access configuration remain
outside version control.
