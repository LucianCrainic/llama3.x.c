# Remaining work: local code first, then the cluster

Written 8 November 2026 from the phase documents and the tree at `21de4c8` (`develop`). The phase documents stay the
source of truth for design; this file only lists what is still open, splits it into work that needs no cluster (§1) and
work that does (§2), and fixes the order. Update it, not the milestone prose, when an item closes.

Rule of the split: **§2 starts only when §1 has no code left to write.** A cluster job exercises code that already
passed locally, so a job is never spent on a bug the virtual topology would have shown.

## 0. What is actually missing

State at the end of §1 (8 November 2026; all of it on branch `feature/integ-d1-two-socket-prefill`, uncommitted):

| Area | State | Missing |
| --- | --- | --- |
| Foundations, A1-A5, B1-B6, C1-C6 | implemented, cluster `check` passed | The open gates of §2.1 only |
| D1 two-socket batched prefill | implemented for FP32, W8A8, AMX-BF16 and AMX-INT8; tested on the virtual topology and the software tile model; TSAN clean | `STAGE=check PHASE=D1` on the hardware (K4) |
| `--warm-context`, packed4 decode layout, sizing rule | implemented and tested locally | The AVX-512 packed4 kernel and the HBM budget check have only been compiled / unit-tested, not run on hardware (gated in B7 and D1) |
| Campaign scripts and tables | `PHASE=B7`, `D1`, `D2` written, C7 switched to `--warm-context`, summarizer tables, results skeletons; dry runs exit 0 | The runs (K5-K8) |
| A6, C7 | scripts written and dry-run | Cluster only |
| `LLAMA_TP_PARTITION=rows` | decided against (L8) | Only if B7 shows reductions plus waits above 20% |

## 1. Local subtasks (no cluster)

Each is one branch off `develop`, named `feature/<track>-<id>-<slug>`, and ends with `make test` green on macOS plus a
sanitizer run where it touches threads. Dependencies are in the third column; subtasks with no dependency can run in
parallel (they touch different files).

| ID | Subtask | Needs | Files | Done when |
| --- | --- | --- | --- | --- |
| **L1** (done 2026-11-08, uncommitted) | `--warm-context N`: prefill N tokens untimed, then time `-n` decode tokens; report `warm_context`, exclude warm prefill from `prefill_s`/TTFT | — | `src/app.c`, `src/report.c`, `include/app.h`, `tests/test_runtime.py` | Fixture test: timed decode excludes the warm prefill; report has the field; schema version handled by `tools/summarize_campaign.py` |
| **L2** (done 2026-11-08, uncommitted) | D1 core, FP32: `tp_forward_batch` (M rows per microbatch): embed rows, row-shard GEMMs (`PARTIAL` epilogue on Wo/W2), local-head batched attention with `pos0`/M, exchange buffers sized `M_max × dim`, final row gather through the B5 classifier. `engine_forward_batch` calls it when `t->tp` is set | — | `src/tp.c`, `src/run.c`, `include/tp.h`, `src/prefill.c` (sizing check against `ROLE_WORK`) | `compare_engines.py` one socket batched vs two sockets batched, `virtual:2x3`, M ∈ {1, 3, 7, 32}, prompts {1, 5, 33}, `all` logits, within fixture tolerance; `LLAMA_TP_CHECK=1` shows identical `X_s` after every layer and microbatch |
| **L3** (done 2026-11-08, uncommitted; W8A8 on VNNI/portable, W8A32 stays per token, AMX-INT8 moved to L4) | D1 Q8: same for `runq`. Full-row activation quantization for QKV/W1/W3, owned-slice quantization on the original group boundaries for Wo/W2 | L2 | `src/tp.c`, `src/runq.c`, `src/gemm.c` (callers only) | Same comparisons, W8A32 strict, W8A8 under the statistical gate of `f348684` |
| **L4** (done 2026-11-08, uncommitted; software tile model only, the HBM budget check is untested off the cluster) | D1 packed rungs: pack BF16 **and int8 (`pw`/`pw3`, for AMX-INT8)** per shard on the owning team (C5/C6 hooks, one packed region per socket), lift the `src/run.c:464` refusal, add the per-socket budget to setup | L2 | `src/tp.c`, `src/shards.c`, `src/loader.c`, `src/run.c` | Scalar BF16 reference: two sockets agree with one socket within the C5 budget; setup rejects a budget overrun with the packed bytes in the message |
| **L5** (done 2026-11-08, uncommitted) | D1 tests and sanitizers: exchange stress at `E = M_max × dim` under TSAN with delay injection, residual-copy check, M-sizing rule (§4.2 of D1) as a pure function with a unit test | L2 (L3, L4 as they land) | `tests/workers.c`, `tests/test_tp.py` | `make test` and `MODE=tsan` runs pass; the M-sizing function returns the three conditions for 8B-sized inputs |
| **L6** (done 2026-11-08, uncommitted; all dry runs exit 0, labels parse, `test_campaign_scripts_dry_run_and_their_labels_parse`) | Scripts: `PHASE=B7` in `cresco8-tp.sbatch` (matrix of B7 §4.1, using L1), `PHASE=D1` and `PHASE=D2` in `cresco8-prefill.sbatch`, and `phase_C7` switched to `--warm-context` | L1 (D1/D2 matrices need the L2-L4 knob names) | `scripts/cresco8-tp.sbatch`, `scripts/cresco8-prefill.sbatch`, `scripts/lib/cresco8-common.sh` | `DRY_RUN=1` exits 0, prints the launch counts, every label parses with `tools/summarize_campaign.py` |
| **L7** (done 2026-11-08, uncommitted; `tools/campaign_tables.py`, `test_campaign_tables_decode_prefill_socket_and_modeled`) | Summarizers: sockets, `warm_context`, per-socket profile split, modeled-vs-measured table for B7, crossover tables for D2 | L1 | `tools/summarize_campaign.py`, `tools/summarize_decode_diag.py` | Runs on synthetic v2 reports in `tests/test_tools.py` |
| **L8** (done 2026-11-08, uncommitted: repack built as `LLAMA_FP32_LAYOUT=packed4`, `rows` comparator dropped; AVX-512 kernel compiled for x86 but not run, its bitwise gate is in B7) | Decision: repack FP32 rows (`mini-gemv-packed4`) and the `rows` comparator. Recommendation: **do the repack before B7** (it moves the headline number B7 reports and D2 reuses), **drop `rows`** unless B7 shows synchronization above 20% | — | `src/shards.c`, `src/kernels_simd.c` only if the repack is accepted | A written decision in B7; if accepted, a bitwise-or-budget test against the current layout |
| **L9** (done 2026-11-08, uncommitted; eight files in `docs/`, `test_results_skeletons_match_the_summarizer`) | Results skeletons: `docs/numa-placement.md`, `docs/tensor-parallel.md`, `docs/batched-prefill.md`, `docs/numa-tp-prefill-results.md` (+ `-validation`), with the tables the summarizers will fill | L7 | `docs/` | Empty tables match the summarizer output columns |
| **L10** (done 2026-11-08) | Housekeeping: README milestone/phase tables, this file's §0, memory notes | last | `docs/plans/README.md`, memory | Matches the tree |

Suggested order: **L1** and **L2** first (independent, different files), then **L3/L4/L5** (L3 and L4 touch
different code paths in `tp.c` and can share one session only if they do not overlap), then **L6/L7** once the knob
names are fixed, then L8-L10. L8's decision should be made before L6 so the B7 matrix is written once.

## 2. Cluster stages (after §1)

Every submission needs the user's go-ahead for that job (plans README §6.5). Nodes are exclusive, so these run one at a
time; the order puts the cheap gates ahead of the long campaigns.

### 2.0 Pre-flight (done)

| Job | Result |
| --- | --- |
| 5704660 (`PHASE=none`, tree 51e8424) | FAILED in `test-kernels`: the OpenMP build split the packed4 GEMV mid-group; fixed in 0679af8 |
| 5704671 (`PHASE=none`, tree 73043e5) | PASSED in 36 minutes: `make test` green on the hardware, including the AVX-512 and real-AMX kernel and GEMM tests, `test-sizing`, `test-workers` (the 256 x 4096 exchange stress) and the 24 two-socket tests |

| 5704682 (B7 gates, tree 8310e50) | PASSED, 8 of 8, including packed4 against rows **bitwise** on the real 8B AVX-512 kernels |
| 5704683 / 5704685 (D1 gates + D3, trees 8310e50 / e763e08) | Two script problems and a W8A8 gate too tight for 8B found (see D1 results); gate revised |
| 5704689 (D1 gates + placement, tree cd31a67) | PASSED, 20 of 20 comparisons and 8 of 8 placement checks |

| 5704694 (full D1, tree 04e913b, 22 min) | PASSED; 8B two-socket AMX-BF16 1092 tok/s, 1.5-1.9x one socket; RMSNorm found to be 18% of the token |
| 5713378 (`make test` + full D1, tree 75761d4, 56 min) | PASSED; row-parallel RMSNorm/RoPE: 8B AMX-BF16 1354 tok/s (+24%), AMX-INT8 1341 (+26%), 1.78-1.93x one socket |

`GATES_ONLY=1` (B7, D1) runs only the gates and D1's placement checks, so the hardware-only paths (the AVX-512 packed4 bitwise gate of B7, AMX on two
sockets, the KV and packed-shard placement) can be checked before a campaign.

### 2.1 Check jobs (≤ 90 min each)

| ID | Job | Purpose | Closes |
| --- | --- | --- | --- |
| K1 | `cresco8-prefill.sbatch STAGE=check PHASE="C5 C6"` with `HF_GATE=1` | Hugging Face numbers for BF16; INT8 re-run with the corrected statistical gate; fix the C5 provisional gate from the data (1B short prompt is at 98.1% argmax against the 99% proposal) | C5, C6 status lines |
| K2 | `cresco8-tp.sbatch STAGE=check PHASE="B5 B6"` with `HF_GATE=1` | Hugging Face gates of B5 and B6 | B5, B6 |
| K3 | `cresco8-kernels.sbatch STAGE=check` vs `cresco8-placement.sbatch` check on one commit | F3 acceptance comparison | F3 |
| K4 | `cresco8-prefill.sbatch STAGE=check PHASE=D1` (about 90 min) | Two-socket prefill against the one-socket batched reference per rung, placement, M sweep | D1, milestone M4 |

K1-K3 do not depend on L2-L5 and can run as soon as the user approves them, even while §1 is in progress; the rule
above only keeps the **campaigns** (K5-K8) behind §1. K4 needs L2-L6.

### 2.2 Campaigns (hours each, split by model if needed)

| ID | Job | Time | Depends on |
| --- | --- | --- | --- |
| K5 | A6 placement (`PHASE=A6`, 121 launches, `--time` extended) | ~4 h | K3 |
| K6 | C7 one-socket prefill (`PHASE=C7`, `--time=06:00:00`) | ~5 h | K1, K5 (anon HBM; else staged, recorded) |
| K7 | B7 two-socket decode (`PHASE=B7`) | 4-6 h | K2, K5, L6, L8 |
| K8 | D2 final campaign (`PHASE=D2`) | 5-6 h | K4, K6, K7 |

After each: results doc plus its `-validation` companion, the phase status line, the milestone table, and a commit
citing date, node, compiler, job ID and commit (plans README §6.6).

## 3. Exit

M4 closes with K4, M5 with K8 and the decisions of D2 §7. Items outside this list (BF16 checkpoints, KV tiering,
batched decode, exchange/compute overlap, AMX attention) stay in the results doc as candidates for a later plan.
