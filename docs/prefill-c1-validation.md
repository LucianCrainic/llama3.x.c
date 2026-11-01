# C1 local validation

Date: 1 November 2026. Branch: `codex/prefill-c1-batch-api`.
Base: `develop` at `446b680`; reused tensor-manifest implementation: `962dc11`.
Platform: macOS arm64, Apple Clang 21.0.0, `OPENMP=0`, scalar kernels.

## Results

| Check | Result |
| --- | --- |
| `make -j4 test` | Pass: 50,314 kernel checks, 20 runtime tests; tooling checks also pass |
| `make -j4 MODE=sanitize test` | Pass: AddressSanitizer/UndefinedBehaviorSanitizer, 20 runtime and 3 tooling tests |
| `make test-reference` with existing Python environment | Pass: 4 integration tests, Torch 2.15.0, Transformers 4.57.1 |
| Independent build of `develop` at `446b680` vs new batched probe | All 12 comparisons bitwise identical, max absolute logit difference 0, argmax agreement 100% |

The independent-build matrix covers shared and unshared classifiers, FP32, Q8
W8A8 and Q8 W8A32, 33-token prefixes, and M = 7 and 32. The baseline was built
from a temporary archive of the exact develop commit, without modifying another
worktree. Local machine-readable records are generated under
`results/prefill-c1-local/` (ignored by Git); temporary checkpoint paths in those
records are not persistent model artifacts.

The C API oracle additionally covers M = 1, 2, 3, 7 and 32; prompt lengths 1, 5,
31, 32, 33 and the full context; starting positions 0 and 7; all three logits
requests; and up to three decode steps after prefill. It overwrites an existing
cache between cases, checking that future KV entries cannot affect earlier
positions. Every output is compared with `memcmp`, and tracked allocations must
return to zero after free. Profile counters verify zero classifier calls for
`NONE`, one for a completed last-only prompt, and one per row for `ALL`.

CLI coverage checks identical output bytes and token accounting across policies,
zero generation, empty prompts, partial prompts, exact context limits, stop
tokens, and two chat turns including EOT caching. Invalid batch sizes, policies,
token IDs and overflowing/context-crossing ranges fail cleanly. The independent
Transformers fixtures cover four architectures, tied/untied weights, GQA, RoPE
scaling, head dimensions 64/128, lengths 1/4/17/33 and batch sizes 1/3/7/32.

## Remaining gates

AVX-512/OpenMP execution and real 1B/8B checkpoint comparisons have not run in
this worktree. No cluster job has been submitted and no speedup or decode
performance result is claimed. C1 remains pending its cluster gate; C2–C7,
including AMX enablement and precision modes, are not implemented here.

The next shared prerequisite is F2's worker-team interface, followed by C2's
GEMM. Reuse F2 when available; do not create an independent prefill scheduler.
F3's campaign helpers and the prefill Slurm campaign remain to be integrated.
